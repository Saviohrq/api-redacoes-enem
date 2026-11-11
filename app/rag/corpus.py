"""Leitura do Essay-BR estendido e divisão treino/validação/teste.

O id de cada redação é a posição da linha (0, 1, 2, ...) no arquivo
extended-corpus/extended_essay-br.csv do repositório lplnufpi/essay-br.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..config import obter_config

COLS = ["c1", "c2", "c3", "c4", "c5"]
SEMENTE_SPLIT_OFICIAL = 230  # a mesma de build_dataset.py do repositório do Essay-BR


def _pasta() -> Path:
    p = Path(obter_config().essaybr_dir) / "extended-corpus"
    if not p.exists():
        raise FileNotFoundError(
            f"Corpus não encontrado em {p}. Rode: git clone https://github.com/lplnufpi/essay-br dados/essay-br")
    return p


def _texto(paragrafos: str) -> str:
    try:
        lista = ast.literal_eval(paragrafos)
    except (ValueError, SyntaxError):
        return str(paragrafos)
    return "\n\n".join(p.strip() for p in lista if str(p).strip())


@lru_cache
def carregar_redacoes() -> pd.DataFrame:
    df = pd.read_csv(_pasta() / "extended_essay-br.csv")
    df.insert(0, "id", range(len(df)))
    df["texto"] = df["essay"].map(_texto)
    for c in COLS + ["score"]:
        df[c] = df[c].astype(int)
    return df.drop(columns=["essay"])


@lru_cache
def carregar_propostas() -> pd.DataFrame:
    p = pd.read_csv(_pasta() / "prompts.csv")
    p["id"] = p["id"].astype(int)
    p["textos_motivadores"] = p["description"].map(_texto)
    return p.set_index("id")


def gerar_splits() -> dict[str, list[int]]:
    """Reproduz a divisão 70/15/15 do build_dataset.py oficial (random_state=230).

    Atenção: apesar do nome da função original, a divisão oficial NÃO é
    estratificada (o parâmetro stratify não é passado ao train_test_split).
    """
    from sklearn.model_selection import train_test_split

    df = carregar_redacoes()
    treino, resto = train_test_split(df, test_size=0.3, random_state=SEMENTE_SPLIT_OFICIAL)
    val, teste = train_test_split(resto, test_size=0.5, random_state=SEMENTE_SPLIT_OFICIAL)
    return {"treino": sorted(treino["id"]), "validacao": sorted(val["id"]), "teste": sorted(teste["id"])}


def pasta_splits() -> Path:
    return Path(obter_config().dados_dir) / "splits"


def ids_do_split(nome: str) -> list[int]:
    arq = pasta_splits() / f"{nome}.txt"
    if not arq.exists():
        raise FileNotFoundError(f"{arq} não existe. Rode: python -m experimentos.preparar_dados")
    return [int(x) for x in arq.read_text().split()]
