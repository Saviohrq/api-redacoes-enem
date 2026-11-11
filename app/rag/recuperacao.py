"""Recuperação de redações corrigidas do acervo (split de treino do Essay-BR).

- modo "similares": RAG por embeddings (similaridade do cosseno);
- modo "aleatorios": controle com k redações sorteadas (mesma quantidade de
  exemplos, sem a similaridade), para separar o efeito da recuperação do
  efeito de simplesmente haver exemplos no prompt.
"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import obter_config
from .corpus import COLS, carregar_propostas, carregar_redacoes, ids_do_split

PREFIXO = "query: "  # modelos E5: prefixo "query: " nos dois lados para similaridade simétrica


def _arquivos() -> tuple[Path, Path]:
    d = Path(obter_config().dados_dir) / "indice"
    return d / "embeddings.npy", d / "ids.txt"


@lru_cache
def _modelo_embeddings():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(obter_config().embeddings_model)


def embutir(textos: list[str], mostrar_progresso: bool = False) -> np.ndarray:
    m = _modelo_embeddings()
    return m.encode([PREFIXO + t for t in textos], normalize_embeddings=True, batch_size=16,
                    show_progress_bar=mostrar_progresso).astype(np.float32)


def construir_indice() -> int:
    df = carregar_redacoes().set_index("id")
    ids = ids_do_split("treino")
    emb = embutir(df.loc[ids, "texto"].tolist(), mostrar_progresso=True)
    arq_emb, arq_ids = _arquivos()
    arq_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(arq_emb, emb)
    arq_ids.write_text("\n".join(map(str, ids)))
    (arq_emb.parent / "modelo.txt").write_text(obter_config().embeddings_model)
    _indice.cache_clear()
    return len(ids)


@lru_cache
def _indice() -> tuple[np.ndarray, np.ndarray]:
    arq_emb, arq_ids = _arquivos()
    if not arq_emb.exists():
        raise FileNotFoundError("Índice não construído. Rode: python -m experimentos.construir_indice")
    return np.load(arq_emb), np.array([int(x) for x in arq_ids.read_text().split()])


def _candidatos(ids: np.ndarray, excluir: set[int], proposta_id: int | None) -> np.ndarray:
    mask = ~np.isin(ids, list(excluir)) if excluir else np.ones(len(ids), bool)
    if proposta_id is not None:
        df = carregar_redacoes().set_index("id")
        mask &= df.loc[ids, "prompt"].to_numpy() == proposta_id
    return mask


def recuperar(texto: str, k: int, modo: str, *, semente: int = 42, redacao_id: int | None = None,
              excluir_ids: list[int] | None = None, proposta_id: int | None = None) -> list[int]:
    if k <= 0:
        return []
    emb, ids = _indice()
    excluir = set(excluir_ids or [])
    if redacao_id is not None:
        excluir.add(redacao_id)
    mask = _candidatos(ids, excluir, proposta_id)
    if modo == "aleatorios":
        # semente fixa por redação: o mesmo sorteio em todas as repetições e modelos
        rng = random.Random(f"{semente}-{redacao_id if redacao_id is not None else texto}")
        return rng.sample(ids[mask].tolist(), k)
    q = embutir([texto])[0]
    sims = emb @ q
    sims[~mask] = -np.inf
    ordem = np.argsort(-sims)[:k]
    return ids[ordem].tolist()


def exemplos_por_ids(ids: list[int]) -> list[dict]:
    df = carregar_redacoes().set_index("id")
    props = carregar_propostas()
    out = []
    for i in ids:
        lin = df.loc[i]
        tema = props.loc[lin["prompt"], "title"] if lin["prompt"] in props.index else ""
        out.append({"id": i, "texto": lin["texto"], "tema": tema, "notas": {c: int(lin[c]) for c in COLS}})
    return out
