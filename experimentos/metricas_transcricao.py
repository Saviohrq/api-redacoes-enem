"""Fase 3: CER e WER das transcrições do BRESSAY, a partir do registro.

Uso:
  python -m experimentos.metricas_transcricao --experimento "bressay-transc-gemini-..."

Reporta, para cada experimento:
  - CER e WER médios por página (média das repetições de cada página), com IC 95% por
    bootstrap das páginas (1.000 reamostras);
  - CER e WER agregados (soma das edições / soma do tamanho das referências);
  - o mesmo, separando páginas com e sem trecho ilegível na referência.
As regras de normalização estão em experimentos/bressay.py.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from app.config import obter_config
from app.registro import conectar

from .bressay import contar_ilegiveis, normalizar_hipotese, normalizar_referencia, referencia_bruta
from .metricas_base import distancia_edicao

B = 1000


def por_pagina(exp: str) -> pd.DataFrame:
    con = conectar()
    linhas = con.execute("SELECT redacao_id, repeticao, resultado FROM chamadas WHERE experimento=? "
                         "AND operacao='transcrever' AND sucesso=1", (exp,)).fetchall()
    con.close()
    if not linhas:
        raise SystemExit(f"Nenhuma transcrição para o experimento '{exp}'")
    refs: dict[str, tuple[str, int]] = {}
    registros = []
    for pagina, rep, texto in linhas:
        if pagina not in refs:
            bruta = referencia_bruta(pagina)
            refs[pagina] = (normalizar_referencia(bruta), contar_ilegiveis(bruta))
        ref, ileg = refs[pagina]
        hip = normalizar_hipotese(texto or "")
        ec = distancia_edicao(ref, hip)
        ew = distancia_edicao(ref.split(), hip.split())
        registros.append({"pagina": pagina, "repeticao": rep, "ilegiveis": ileg,
                          "edicoes_car": ec, "n_car": len(ref), "edicoes_pal": ew, "n_pal": len(ref.split())})
    df = pd.DataFrame(registros)
    df["CER"] = df.edicoes_car / df.n_car.clip(lower=1)
    df["WER"] = df.edicoes_pal / df.n_pal.clip(lower=1)
    return df.groupby("pagina").agg(CER=("CER", "mean"), WER=("WER", "mean"), ilegiveis=("ilegiveis", "first"),
                                    edicoes_car=("edicoes_car", "mean"), n_car=("n_car", "first"),
                                    edicoes_pal=("edicoes_pal", "mean"), n_pal=("n_pal", "first"),
                                    repeticoes=("repeticao", "count"))


def resumo(pp: pd.DataFrame, rotulo: str, semente: int = 42) -> list[dict]:
    if pp.empty:
        return []
    rng = np.random.default_rng(semente)
    idx = [rng.integers(0, len(pp), len(pp)) for _ in range(B)]
    out = []
    for m in ("CER", "WER"):
        v = pp[m].to_numpy()
        lo, hi = np.percentile([v[i].mean() for i in idx], [2.5, 97.5])
        out.append({"grupo": rotulo, "metrica": f"{m}_medio_pagina", "valor": v.mean(),
                    "ic95_inf": lo, "ic95_sup": hi, "paginas": len(pp)})
    out.append({"grupo": rotulo, "metrica": "CER_agregado", "valor": pp.edicoes_car.sum() / pp.n_car.sum(),
                "ic95_inf": np.nan, "ic95_sup": np.nan, "paginas": len(pp)})
    out.append({"grupo": rotulo, "metrica": "WER_agregado", "valor": pp.edicoes_pal.sum() / pp.n_pal.sum(),
                "ic95_inf": np.nan, "ic95_sup": np.nan, "paginas": len(pp)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experimento", required=True)
    a = ap.parse_args()
    pp = por_pagina(a.experimento)
    linhas = resumo(pp, "todas") + resumo(pp[pp.ilegiveis == 0], "sem_ilegivel") + \
        resumo(pp[pp.ilegiveis > 0], "com_ilegivel")
    df = pd.DataFrame(linhas)
    pd.set_option("display.width", 200)
    print(f"\nExperimento: {a.experimento}\nPáginas: {len(pp)} | "
          f"repetições por página: {pp.repeticoes.min()}–{pp.repeticoes.max()}")
    print(df.round(4).to_string(index=False))
    pasta = obter_config().dados_dir / "resultados"
    pasta.mkdir(parents=True, exist_ok=True)
    df.to_csv(pasta / f"{a.experimento}.csv", index=False)
    pp.to_csv(pasta / f"{a.experimento}__por_pagina.csv")
    print(f"\nSalvo em {pasta / (a.experimento + '.csv')}")


if __name__ == "__main__":
    main()
