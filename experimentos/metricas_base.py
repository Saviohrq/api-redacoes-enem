"""Métricas de concordância (notas) e de transcrição (CER/WER)."""
from __future__ import annotations

import numpy as np

NIVEIS_COMP = np.arange(0, 201, 40)      # 6 níveis
NIVEIS_TOTAL = np.arange(0, 1001, 40)    # 26 níveis


def qwk(y: np.ndarray, p: np.ndarray, niveis: np.ndarray = NIVEIS_COMP) -> float:
    """Kappa ponderado quadrático com o conjunto de níveis fixo (Equação do artigo)."""
    n = len(niveis)
    iy = np.searchsorted(niveis, y)
    ip = np.searchsorted(niveis, p)
    O = np.zeros((n, n))
    np.add.at(O, (iy, ip), 1)
    w = (np.subtract.outer(np.arange(n), np.arange(n)) ** 2) / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den > 0 else float("nan")


def pearson(y, p) -> float:
    if np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def metricas_competencia(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    d = np.abs(y - p)
    return {"MAE": float(d.mean()), "MSE": float(((y - p) ** 2).mean()), "Pearson": pearson(y, p),
            "QWK": qwk(y, p), "Exata": float((d == 0).mean()), "Adjacente": float((d <= 40).mean())}


def metricas_total(Y: np.ndarray, P: np.ndarray) -> dict[str, float]:
    """Y e P: matrizes [n, 5] com as notas por competência."""
    yt, pt = Y.sum(1), P.sum(1)
    d = np.abs(yt - pt)
    disc = (np.abs(Y - P) > 80).any(1) | (d > 100)
    return {"MAE": float(d.mean()), "MSE": float(((yt - pt) ** 2).mean()), "Pearson": pearson(yt, pt),
            "QWK": qwk(yt, pt, NIVEIS_TOTAL), "Discrepancia_INEP": float(disc.mean())}


# ---------- transcrição ----------

def distancia_edicao(a, b) -> int:
    """Levenshtein (substituição, remoção e inserção com custo 1) entre duas sequências."""
    if len(a) < len(b):
        a, b = b, a
    anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i]
        for j, cb in enumerate(b, 1):
            atual.append(min(anterior[j] + 1, atual[j - 1] + 1, anterior[j - 1] + (ca != cb)))
        anterior = atual
    return anterior[-1]


def normalizar(texto: str) -> str:
    """Normaliza espaços e quebras de linha; preserva maiúsculas, acentos e pontuação."""
    return " ".join(texto.split())


def cer(referencia: str, hipotese: str) -> float:
    r, h = normalizar(referencia), normalizar(hipotese)
    return distancia_edicao(r, h) / max(len(r), 1)


def wer(referencia: str, hipotese: str) -> float:
    r, h = normalizar(referencia).split(), normalizar(hipotese).split()
    return distancia_edicao(r, h) / max(len(r), 1)
