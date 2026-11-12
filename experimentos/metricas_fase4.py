"""Fase 4: quanto o erro de transcrição altera a nota.

Compara, página a página e repetição a repetição, a nota dada à transcrição de referência (A)
com a nota dada à transcrição do modelo (B), pelo mesmo avaliador.

Como o avaliador não é determinístico (o Claude, por exemplo, não aceita temperatura 0), a
mesma página pode receber notas diferentes só por repetir a chamada. Por isso também se calcula
o "ruído do modelo": referência × referência em repetições diferentes. O efeito atribuível à
transcrição é a diferença entre as duas comparações (bootstrap pareado das páginas).

Uso:
  python -m experimentos.metricas_fase4 --referencia EXP_REF --transcricao EXP_TRANSC [--transcritor EXP_FASE3]
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from app.config import obter_config
from app.rag.corpus import COLS
from app.registro import conectar

from .metricas_base import metricas_competencia, metricas_total

B = 1000


def notas(exp: str) -> dict[tuple[str, int], list[int]]:
    con = conectar()
    linhas = con.execute("SELECT redacao_id, repeticao, resultado FROM chamadas WHERE experimento=? "
                         "AND operacao='avaliar' AND sucesso=1", (exp,)).fetchall()
    con.close()
    if not linhas:
        raise SystemExit(f"Nenhuma avaliação para '{exp}'")
    return {(p, r): [json.loads(res)[c]["nota"] for c in COLS] for p, r, res in linhas}


def pares(a: dict, b: dict, combinacoes: list[tuple[int, int]], paginas: list[str]):
    """Para cada combinação de repetições (ra, rb), devolve Y [n,5] e P [n,5] nas páginas em comum."""
    out = []
    for ra, rb in combinacoes:
        ok = [p for p in paginas if (p, ra) in a and (p, rb) in b]
        if len(ok) >= 2:
            out.append((ok, np.array([a[(p, ra)] for p in ok], float), np.array([b[(p, rb)] for p in ok], float)))
    return out


def resumir(lista) -> dict[str, float]:
    acum = []
    for _, Y, P in lista:
        m = {f"{c.upper()}_{k}": v for i, c in enumerate(COLS) for k, v in metricas_competencia(Y[:, i], P[:, i]).items()
             if k in ("MAE", "QWK")}
        m.update({f"Total_{k}": v for k, v in metricas_total(Y, P).items() if k in ("MAE", "QWK", "Discrepancia_INEP")})
        m["Total_vies"] = float((P.sum(1) - Y.sum(1)).mean())  # B - A: positivo = nota maior com a transcrição
        acum.append(m)
    return pd.DataFrame(acum).mean().to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referencia", required=True)
    ap.add_argument("--transcricao", required=True)
    ap.add_argument("--transcritor", default=None, help="experimento da Fase 3, para correlacionar CER e mudança de nota")
    a = ap.parse_args()

    ref, tr = notas(a.referencia), notas(a.transcricao)
    paginas = sorted({p for p, _ in ref} & {p for p, _ in tr})
    reps = sorted({r for _, r in ref})
    mesmas = [(r, r) for r in reps]
    entre_reps = list(itertools.combinations(reps, 2))

    def efeito(pg):
        return resumir(pares(ref, tr, mesmas, pg))

    def ruido(pg):
        return resumir(pares(ref, ref, entre_reps, pg))

    base_ef, base_ru = efeito(paginas), ruido(paginas) if entre_reps else {}
    rng = np.random.default_rng(42)
    amostras = []
    for _ in range(B):
        pg = list(rng.choice(paginas, len(paginas), replace=True))
        e = efeito(pg)
        r = ruido(pg) if entre_reps else {}
        amostras.append((e, r))

    linhas = []
    for m in base_ef:
        ve = [s[0][m] for s in amostras]
        lo, hi = np.nanpercentile(ve, [2.5, 97.5])
        linha = {"metrica": m, "transcricao_vs_ref": base_ef[m], "ic95_inf": lo, "ic95_sup": hi}
        if m in base_ru:
            dif = [s[0][m] - s[1][m] for s in amostras]
            dlo, dhi = np.nanpercentile(dif, [2.5, 97.5])
            linha.update({"ruido_ref_vs_ref": base_ru[m], "efeito_alem_do_ruido": base_ef[m] - base_ru[m],
                          "dif_ic95_inf": dlo, "dif_ic95_sup": dhi, "conclusivo": not (dlo <= 0 <= dhi)})
        linhas.append(linha)
    df = pd.DataFrame(linhas)

    pd.set_option("display.width", 220)
    print(f"\nReferência (A): {a.referencia}\nTranscrição (B): {a.transcricao}\n"
          f"Páginas em comum: {len(paginas)} | repetições: {reps}")
    print(df.round(3).to_string(index=False))

    if a.transcritor:
        from .metricas_transcricao import por_pagina
        cer = por_pagina(a.transcritor)["CER"]
        dif = pd.Series({p: np.mean([abs(sum(tr[(p, r)]) - sum(ref[(p, r)])) for r in reps
                                     if (p, r) in tr and (p, r) in ref]) for p in paginas})
        comum = cer.index.intersection(dif.index)
        rho = pd.Series(cer[comum]).corr(dif[comum], method="spearman")
        boots = []
        for _ in range(B):
            s = rng.choice(comum, len(comum), replace=True)
            boots.append(pd.Series(cer[s].to_numpy()).corr(pd.Series(dif[s].to_numpy()), method="spearman"))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        print(f"\nSpearman entre CER da página e |mudança da nota total|: {rho:.3f} [IC95 {lo:.3f}; {hi:.3f}] "
              f"({len(comum)} páginas)")
        df.loc[len(df)] = {"metrica": "Spearman_CER_x_absdif_total", "transcricao_vs_ref": rho,
                           "ic95_inf": lo, "ic95_sup": hi}

    pasta = obter_config().dados_dir / "resultados"
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / f"fase4__{a.transcricao}.csv".replace("[", "(").replace("]", ")")
    df.to_csv(arq, index=False)
    print(f"\nSalvo em {arq}")


if __name__ == "__main__":
    main()
