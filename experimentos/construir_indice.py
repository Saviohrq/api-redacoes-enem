"""Gera os embeddings do acervo de recuperação (split de treino do Essay-BR).

Uso:  python -m experimentos.construir_indice
Na primeira vez baixa o modelo de embeddings (~1 GB). Em CPU leva alguns minutos.
"""
from app.config import obter_config
from app.rag.recuperacao import construir_indice

if __name__ == "__main__":
    n = construir_indice()
    print(f"Índice com {n} redações de treino, modelo {obter_config().embeddings_model}")
