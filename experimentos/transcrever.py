"""Fase 3: transcreve as páginas da amostra do BRESSAY e grava tudo em dados/registro.sqlite.

Exemplos:
  python -m experimentos.transcrever --provedor gemini --limite 3 --repeticoes 1 --experimento teste-transc
  python -m experimentos.transcrever --provedor anthropic --paralelo 4
  python -m experimentos.transcrever --provedor ollama

Pode interromper (Ctrl+C) e rodar de novo: o que já deu certo é pulado.
Métricas: python -m experimentos.metricas_transcricao --experimento "NOME"
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config import obter_config
from app.provedores import obter_provedor
from app.registro import conectar
from app.servicos import transcrever

from .bressay import caminho_imagem


def ja_feitos(experimento: str) -> set[tuple[str, int]]:
    con = conectar()
    try:
        return set(con.execute("SELECT redacao_id, repeticao FROM chamadas WHERE experimento=? "
                               "AND operacao='transcrever' AND sucesso=1", (experimento,)).fetchall())
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provedor", required=True, choices=["anthropic", "gemini", "ollama", "simulado"])
    ap.add_argument("--modelo", default=None, help="padrão: o do .env")
    ap.add_argument("--repeticoes", type=int, default=1)
    ap.add_argument("--limite", type=int, default=None, help="usa só as N primeiras páginas (teste rápido)")
    ap.add_argument("--paralelo", type=int, default=1, help="chamadas simultâneas (cuidado com a cota)")
    ap.add_argument("--experimento", default=None, help="nome da rodada (padrão: gerado a partir da configuração)")
    a = ap.parse_args()

    modelo = obter_provedor(a.provedor, a.modelo).modelo
    exp = a.experimento or f"bressay-transc-{a.provedor}-{modelo}"
    arq = obter_config().dados_dir / "amostras" / "bressay_teste.txt"
    paginas = arq.read_text(encoding="utf-8").split()[: a.limite]
    feitos = ja_feitos(exp)
    tarefas = [(p, r) for r in range(1, a.repeticoes + 1) for p in paginas if (p, r) not in feitos]
    print(f"Experimento: {exp}\n{len(paginas)} páginas x {a.repeticoes} repetições; faltam {len(tarefas)} chamadas")

    def uma(pagina: str, rep: int):
        imagem = caminho_imagem(pagina).read_bytes()
        return transcrever(imagem, "image/png", a.provedor, a.modelo, experimento=exp,
                           redacao_id=pagina, repeticao=rep)

    erros = 0
    with ThreadPoolExecutor(max_workers=a.paralelo) as ex:
        futuros = {ex.submit(uma, p, r): (p, r) for p, r in tarefas}
        for n, f in enumerate(as_completed(futuros), 1):
            p, r = futuros[f]
            try:
                res = f.result()
                print(f"[{n}/{len(tarefas)}] página {p} rep {r}: {len(res.texto)} caracteres "
                      f"({res.meta.latencia_ms} ms)")
            except Exception as e:  # noqa: BLE001
                erros += 1
                print(f"[{n}/{len(tarefas)}] página {p} rep {r}: ERRO {e}", file=sys.stderr)
    print(f"\nConcluído com {erros} erro(s). Rode de novo para refazer as que falharam.")
    print(f"Métricas: python -m experimentos.metricas_transcricao --experimento \"{exp}\"")


if __name__ == "__main__":
    main()
