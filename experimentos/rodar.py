"""Roda uma configuração sobre a amostra fixa e grava tudo em dados/registro.sqlite.

Exemplos:
  # Fase 1 (k=0), um comando por modelo
  python -m experimentos.rodar --provedor gemini
  python -m experimentos.rodar --provedor anthropic
  python -m experimentos.rodar --provedor ollama
  # Fase 2: k=3 com redações similares (RAG) e controle com 3 sorteadas
  python -m experimentos.rodar --provedor gemini --k 3 --modo similares
  python -m experimentos.rodar --provedor gemini --k 3 --modo aleatorios

Pode interromper (Ctrl+C) e rodar de novo: o que já deu certo é pulado.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config import obter_config
from app.esquemas import ConfigRecuperacao, PedidoAvaliacao, Proposta
from app.provedores import obter_provedor
from app.rag.corpus import carregar_propostas, carregar_redacoes
from app.registro import conectar
from app.servicos import avaliar


def nome_padrao(a, modelo: str) -> str:
    modo = f"-{a.modo}" if a.k else ""
    tm = "" if a.textos_motivadores else "-semTM"
    return f"{a.amostra}-{a.provedor}-{modelo}-k{a.k}{modo}{tm}"


def ja_feitos(experimento: str) -> set[tuple[str, int]]:
    con = conectar()
    try:
        return set(con.execute("SELECT redacao_id, repeticao FROM chamadas WHERE experimento=? AND sucesso=1",
                               (experimento,)).fetchall())
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provedor", required=True, choices=["anthropic", "gemini", "ollama", "simulado"])
    ap.add_argument("--modelo", default=None, help="padrão: o do .env")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--modo", default="similares", choices=["similares", "aleatorios"])
    ap.add_argument("--mesma-proposta", action="store_true")
    ap.add_argument("--repeticoes", type=int, default=3)
    ap.add_argument("--amostra", default="teste", choices=["teste", "validacao"])
    ap.add_argument("--limite", type=int, default=None, help="usa só as N primeiras redações (teste rápido)")
    ap.add_argument("--paralelo", type=int, default=1, help="chamadas simultâneas (cuidado com a cota)")
    ap.add_argument("--sem-textos-motivadores", dest="textos_motivadores", action="store_false")
    ap.add_argument("--experimento", default=None, help="nome da rodada (padrão: gerado a partir da configuração)")
    a = ap.parse_args()

    modelo = obter_provedor(a.provedor, a.modelo).modelo
    exp = a.experimento or nome_padrao(a, modelo)
    arq = obter_config().dados_dir / "amostras" / f"{a.amostra}.txt"
    ids = [int(x) for x in arq.read_text().split()][: a.limite]
    df = carregar_redacoes().set_index("id")
    props = carregar_propostas()
    feitos = ja_feitos(exp)
    tarefas = [(i, r) for r in range(1, a.repeticoes + 1) for i in ids if (str(i), r) not in feitos]
    print(f"Experimento: {exp}\n{len(ids)} redações x {a.repeticoes} repetições; faltam {len(tarefas)} chamadas")

    def uma(i: int, rep: int):
        lin = df.loc[i]
        pid = int(lin["prompt"])
        prop = Proposta(titulo=str(props.loc[pid, "title"]), textos_motivadores=props.loc[pid, "textos_motivadores"])
        pedido = PedidoAvaliacao(texto=lin["texto"], proposta=prop, provedor=a.provedor, modelo=a.modelo,
                                 recuperacao=ConfigRecuperacao(k=a.k, modo=a.modo, mesma_proposta=a.mesma_proposta),
                                 incluir_textos_motivadores=a.textos_motivadores, redacao_id=i, proposta_id=pid)
        return avaliar(pedido, experimento=exp, repeticao=rep)

    erros = 0
    with ThreadPoolExecutor(max_workers=a.paralelo) as ex:
        futuros = {ex.submit(uma, i, r): (i, r) for i, r in tarefas}
        for n, f in enumerate(as_completed(futuros), 1):
            i, r = futuros[f]
            try:
                res = f.result()
                print(f"[{n}/{len(tarefas)}] redação {i} rep {r}: total {res.nota_total} "
                      f"({res.meta.latencia_ms} ms)")
            except Exception as e:  # noqa: BLE001
                erros += 1
                print(f"[{n}/{len(tarefas)}] redação {i} rep {r}: ERRO {e}", file=sys.stderr)
    print(f"\nConcluído com {erros} erro(s). Rode de novo para refazer as que falharam.")
    print(f"Métricas: python -m experimentos.metricas --experimento \"{exp}\"")


if __name__ == "__main__":
    main()
