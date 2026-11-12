"""Leitura do BRESSAY (nível de página), normalização das marcações e sorteio da amostra.

O BRESSAY (Neto et al., ICDAR 2024; Zenodo, DOI 10.5281/zenodo.11637681, CC BY 4.0)
fica em dados/bressay/ (fora do Git), extraído do zip oficial:
  dados/bressay/sets/{training,validation,test}.txt
  dados/bressay/data/pages/<id>/<id>.png e <id>.txt

Uso:  python -m experimentos.bressay [--tamanho 100] [--semente 42]
Saída (vai para o GitHub): dados/amostras/bressay_teste.txt

Regras de normalização (aplicadas antes do CER/WER):
  1. trechos riscados (--texto--, --xxx--, ##--...--##, $$--...--$$): removidos da referência;
  2. inserções acima/abaixo da linha (##texto##, $$texto$$): mantém o texto, tira as marcas;
  3. ilegíveis (@@???@@): removidos da referência (contados à parte);
  4. hifenização no fim da linha é desfeita e as quebras de linha viram espaço,
     na referência e na transcrição do modelo.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from app.config import obter_config

ILEGIVEL = re.compile(r"@@\?\?\?@@")
RISCADO = re.compile(r"--.*?--")
MARCAS_POSICAO = re.compile(r"##|\$\$")
HIFEN_FIM_DE_LINHA = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")


def pasta_bressay() -> Path:
    p = Path(obter_config().dados_dir) / "bressay"
    if not (p / "sets").exists():
        raise FileNotFoundError(f"BRESSAY não encontrado em {p}. Extraia o zip oficial para essa pasta.")
    return p


def ids_do_conjunto(nome: str = "test") -> list[str]:
    return (pasta_bressay() / "sets" / f"{nome}.txt").read_text(encoding="utf-8").split()


def caminho_imagem(pagina: str) -> Path:
    return pasta_bressay() / "data" / "pages" / pagina / f"{pagina}.png"


def referencia_bruta(pagina: str) -> str:
    return (pasta_bressay() / "data" / "pages" / pagina / f"{pagina}.txt").read_text(encoding="utf-8")


def contar_ilegiveis(texto: str) -> int:
    return len(ILEGIVEL.findall(texto))


def juntar_linhas(texto: str) -> str:
    """Desfaz a hifenização no fim da linha e troca quebras de linha e espaços repetidos por um espaço."""
    texto = texto.replace("\r\n", "\n")
    texto = HIFEN_FIM_DE_LINHA.sub(r"\1\2", texto)
    return " ".join(texto.split())


def normalizar_referencia(texto: str) -> str:
    texto = ILEGIVEL.sub(" ", texto)
    texto = RISCADO.sub(" ", texto)          # cobre também ##--...--## e $$--...--$$
    texto = MARCAS_POSICAO.sub("", texto)    # inserções: mantém o texto
    return juntar_linhas(texto)


def normalizar_hipotese(texto: str) -> str:
    return juntar_linhas(texto)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamanho", type=int, default=100)
    ap.add_argument("--semente", type=int, default=42)
    a = ap.parse_args()

    teste = sorted(ids_do_conjunto("test"))
    rng = np.random.default_rng(a.semente)
    amostra = sorted(rng.choice(teste, size=min(a.tamanho, len(teste)), replace=False).tolist())
    destino = Path(obter_config().dados_dir) / "amostras" / "bressay_teste.txt"
    destino.write_text("\n".join(amostra), encoding="utf-8")

    ileg = [contar_ilegiveis(referencia_bruta(p)) for p in amostra]
    chars = [len(normalizar_referencia(referencia_bruta(p))) for p in amostra]
    print(f"Conjunto de teste oficial: {len(teste)} páginas")
    print(f"Amostra: {len(amostra)} páginas (semente {a.semente}) -> {destino}")
    print(f"Páginas com trecho ilegível: {sum(1 for n in ileg if n)} ({sum(ileg)} trechos)")
    print(f"Caracteres por página após normalização: média {np.mean(chars):.0f}, "
          f"mín {min(chars)}, máx {max(chars)}")


if __name__ == "__main__":
    main()
