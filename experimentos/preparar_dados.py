"""Gera os splits do Essay-BR e as amostras fixas usadas nos experimentos.

Uso:  python -m experimentos.preparar_dados [--tamanho 250] [--semente 42]

Saídas (vão para o GitHub, para reprodutibilidade):
  dados/splits/{treino,validacao,teste}.txt   ids das redações de cada conjunto
  dados/amostras/teste.txt                    amostra estratificada do teste
  dados/amostras/validacao.txt                amostra pequena para ajustar o prompt
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from app.rag.corpus import carregar_redacoes, gerar_splits, pasta_splits

FAIXAS = [0, 200, 400, 600, 800, 1001]  # faixas da nota total


def amostra_estratificada(df: pd.DataFrame, tamanho: int, semente: int, minimo: int) -> list[int]:
    """Alocação proporcional por faixa de nota total, com um mínimo por faixa
    (para que as faixas raras, como 0-199 e 800-1000, apareçam)."""
    faixa = pd.cut(df["score"], FAIXAS, right=False)
    grupos = {f: g for f, g in df.groupby(faixa, observed=True)}
    cota = {f: max(min(minimo, len(g)), round(tamanho * len(g) / len(df))) for f, g in grupos.items()}
    while sum(cota.values()) > tamanho:  # ajusta tirando da maior faixa
        maior = max(cota, key=cota.get)
        cota[maior] -= 1
    rng = np.random.default_rng(semente)
    ids = []
    for f, g in grupos.items():
        ids += rng.choice(g["id"].to_numpy(), size=min(cota[f], len(g)), replace=False).tolist()
    return sorted(int(i) for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamanho", type=int, default=250)
    ap.add_argument("--tamanho-validacao", type=int, default=30)
    ap.add_argument("--minimo-por-faixa", type=int, default=15)
    ap.add_argument("--semente", type=int, default=42)
    a = ap.parse_args()

    df = carregar_redacoes()
    splits = gerar_splits()
    pasta = pasta_splits()
    pasta.mkdir(parents=True, exist_ok=True)
    for nome, ids in splits.items():
        (pasta / f"{nome}.txt").write_text("\n".join(map(str, ids)))
        print(f"{nome}: {len(ids)} redações")

    pasta_am = pasta.parent / "amostras"
    pasta_am.mkdir(exist_ok=True)
    por_id = df.set_index("id", drop=False)
    teste = amostra_estratificada(por_id.loc[splits["teste"]], a.tamanho, a.semente, a.minimo_por_faixa)
    val = amostra_estratificada(por_id.loc[splits["validacao"]], a.tamanho_validacao, a.semente, 3)
    (pasta_am / "teste.txt").write_text("\n".join(map(str, teste)))
    (pasta_am / "validacao.txt").write_text("\n".join(map(str, val)))
    (pasta_am / "LEIA-ME.txt").write_text(
        f"Amostras geradas por experimentos/preparar_dados.py com semente={a.semente}, "
        f"tamanho={a.tamanho}, minimo_por_faixa={a.minimo_por_faixa}, faixas da nota total={FAIXAS[:-1]}.\n")

    am = por_id.loc[teste]
    print(f"\nAmostra de teste: {len(teste)} redações (semente {a.semente})")
    print(pd.cut(am["score"], FAIXAS, right=False).value_counts().sort_index().to_string())
    print("\nDistribuição por competência na amostra (contagem por nível):")
    print(pd.DataFrame({c: am[c].value_counts() for c in ["c1", "c2", "c3", "c4", "c5"]})
          .reindex([0, 40, 80, 120, 160, 200]).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
