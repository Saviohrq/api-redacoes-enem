"""Calcula as métricas de um experimento (ou compara dois) a partir do registro.

Uso:
  python -m experimentos.metricas --listar
  python -m experimentos.metricas --experimento "teste-gemini-...-k0"
  python -m experimentos.metricas --comparar "EXP_A" "EXP_B"     # bootstrap pareado (B - A)

As métricas de cada experimento são a média das repetições; os intervalos de
confiança de 95% vêm de reamostragem bootstrap das redações (1.000 reamostras).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from app.config import obter_config
from app.rag.corpus import COLS, carregar_redacoes
from app.registro import conectar

from .metricas_base import metricas_competencia, metricas_total

B = 1000


def carregar(exp: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Devolve Y [n,5], P [n,reps,5] (nan onde falhou) e os ids das redações."""
    con = conectar()
    linhas = con.execute("SELECT redacao_id, repeticao, resultado FROM chamadas "
                         "WHERE experimento=? AND sucesso=1 AND operacao='avaliar'", (exp,)).fetchall()
    con.close()
    if not linhas:
        raise SystemExit(f"Nenhum resultado para o experimento '{exp}'")
    ids = sorted({int(r[0]) for r in linhas})
    reps = sorted({r[1] for r in linhas})
    pos, prep = {i: n for n, i in enumerate(ids)}, {r: n for n, r in enumerate(reps)}
    P = np.full((len(ids), len(reps), 5), np.nan)
    for rid, rep, res in linhas:
        d = json.loads(res)
        P[pos[int(rid)], prep[rep]] = [d[c]["nota"] for c in COLS]
    Y = carregar_redacoes().set_index("id").loc[ids, COLS].to_numpy(float)
    return Y, P, ids


def calcular(Y: np.ndarray, P: np.ndarray) -> dict[str, dict[str, float]]:
    """Média das métricas sobre as repetições."""
    acum: dict[str, list[dict]] = {}
    for r in range(P.shape[1]):
        ok = ~np.isnan(P[:, r, 0])
        if ok.sum() < 2:
            continue
        y, p = Y[ok], P[ok, r]
        for c in range(5):
            acum.setdefault(COLS[c].upper(), []).append(metricas_competencia(y[:, c], p[:, c]))
        acum.setdefault("Total", []).append(metricas_total(y, p))
    return {k: pd.DataFrame(v).mean().to_dict() for k, v in acum.items()}


def bootstrap(Y, P, semente=42) -> dict[str, dict[str, tuple[float, float]]]:
    rng = np.random.default_rng(semente)
    amostras = [calcular(Y[idx], P[idx]) for idx in (rng.integers(0, len(Y), len(Y)) for _ in range(B))]
    out = {}
    for linha in amostras[0]:
        out[linha] = {m: tuple(np.nanpercentile([a[linha][m] for a in amostras], [2.5, 97.5]))
                      for m in amostras[0][linha]}
    return out


def tabela(exp: str) -> pd.DataFrame:
    Y, P, ids = carregar(exp)
    base, ic = calcular(Y, P), bootstrap(Y, P)
    linhas = []
    for linha, ms in base.items():
        for m, v in ms.items():
            lo, hi = ic[linha][m]
            linhas.append({"linha": linha, "metrica": m, "valor": v, "ic95_inf": lo, "ic95_sup": hi})
    df = pd.DataFrame(linhas)
    # variabilidade entre repetições: % de redações com a mesma nota em todas as repetições
    if P.shape[1] > 1:
        completos = ~np.isnan(P).any(axis=(1, 2))
        if completos.any():
            iguais = (P[completos] == P[completos][:, :1]).all(axis=1).mean(axis=0)
            for c in range(5):
                df.loc[len(df)] = {"linha": COLS[c].upper(), "metrica": "Estavel_entre_reps", "valor": iguais[c],
                                   "ic95_inf": np.nan, "ic95_sup": np.nan}
    print(f"\nExperimento: {exp}\nRedações: {len(ids)} | repetições: {P.shape[1]} | "
          f"chamadas válidas: {int((~np.isnan(P[:, :, 0])).sum())}")
    return df


def comparar(a: str, b: str) -> pd.DataFrame:
    Ya, Pa, ia = carregar(a)
    Yb, Pb, ib = carregar(b)
    comum = sorted(set(ia) & set(ib))
    sa, sb = [ia.index(i) for i in comum], [ib.index(i) for i in comum]
    Y, Pa, Pb = Ya[sa], Pa[sa], Pb[sb]
    rng = np.random.default_rng(42)
    difs: dict[tuple[str, str], list[float]] = {}
    base_a, base_b = calcular(Y, Pa), calcular(Y, Pb)
    for _ in range(B):
        idx = rng.integers(0, len(Y), len(Y))
        ma, mb = calcular(Y[idx], Pa[idx]), calcular(Y[idx], Pb[idx])
        for linha in ma:
            for m in ma[linha]:
                difs.setdefault((linha, m), []).append(mb[linha][m] - ma[linha][m])
    linhas = []
    for (linha, m), v in difs.items():
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        linhas.append({"linha": linha, "metrica": m, "A": base_a[linha][m], "B": base_b[linha][m],
                       "B_menos_A": base_b[linha][m] - base_a[linha][m], "ic95_inf": lo, "ic95_sup": hi,
                       "conclusivo": not (lo <= 0 <= hi)})
    print(f"\nA = {a}\nB = {b}\nRedações em comum: {len(comum)}")
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--experimento")
    g.add_argument("--comparar", nargs=2, metavar=("A", "B"))
    g.add_argument("--listar", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    pasta = obter_config().dados_dir / "resultados"
    pasta.mkdir(parents=True, exist_ok=True)
    if a.listar:
        con = conectar()
        q = ("SELECT experimento, provedor, modelo_resposta, count(*) chamadas, sum(sucesso) ok, "
             "sum(tokens_entrada) tok_in, sum(tokens_saida) tok_out, avg(latencia_ms) lat_media_ms "
             "FROM chamadas WHERE experimento IS NOT NULL GROUP BY experimento ORDER BY min(id)")
        print(pd.read_sql_query(q, con).to_string(index=False))
        return
    if a.experimento:
        df = tabela(a.experimento)
        arq = pasta / f"{a.experimento}.csv"
    else:
        df = comparar(*a.comparar)
        arq = pasta / f"comparacao__{a.comparar[0]}__{a.comparar[1]}.csv"
    print(df.round(3).to_string(index=False))
    df.to_csv(arq, index=False)
    print(f"\nSalvo em {arq}")


if __name__ == "__main__":
    main()
