"""Fase 4: efeito da transcrição sobre a nota (amostra do BRESSAY).

Para cada página da amostra, o mesmo modelo avalia:
  - a transcrição de referência (normalizada, com parágrafos)  -> experimento "bressay-aval-ref-<prov>-<modelo>"
  - cada transcrição da Fase 3 feita por um modelo             -> "bressay-aval-transc[<prov_t>-<modelo_t>]-<prov>-<modelo>"
Modo "único" = transcritor e avaliador iguais (reaproveita as transcrições da Fase 3);
modo "separado" = transcritor diferente do avaliador.

O BRESSAY não tem tema nem textos motivadores; as duas avaliações usam o mesmo texto de tema
(TEMA), então a falta de tema afeta as duas igualmente.

Uso:
  python -m experimentos.fase4 --provedor anthropic --transcritor "bressay-transc-anthropic-claude-haiku-5-5"
  python -m experimentos.fase4 --provedor anthropic --transcritor ... --limite 2 --repeticoes 1   (teste)
Métricas: python -m experimentos.metricas_fase4 --referencia "EXP_REF" --transcricao "EXP_TRANSC"
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config import obter_config
from app.esquemas import ConfigRecuperacao, PedidoAvaliacao, Proposta
from app.provedores import obter_provedor
from app.registro import conectar
from app.servicos import avaliar

from .bressay import referencia_bruta, texto_para_avaliacao

TEMA = "Não informado. Avalie a abordagem do tema considerando o tema que o próprio texto desenvolve."


def ja_feitos(experimento: str) -> set[tuple[str, int]]:
    con = conectar()
    try:
        return set(con.execute("SELECT redacao_id, repeticao FROM chamadas WHERE experimento=? "
                               "AND operacao='avaliar' AND sucesso=1", (experimento,)).fetchall())
    finally:
        con.close()


def transcricoes(experimento: str) -> dict[tuple[str, int], str]:
    con = conectar()
    try:
        linhas = con.execute("SELECT redacao_id, repeticao, resultado FROM chamadas WHERE experimento=? "
                             "AND operacao='transcrever' AND sucesso=1", (experimento,)).fetchall()
    finally:
        con.close()
    if not linhas:
        raise SystemExit(f"Nenhuma transcrição encontrada para '{experimento}' (rode a Fase 3 antes)")
    return {(p, r): t for p, r, t in linhas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provedor", required=True, choices=["anthropic", "gemini", "ollama", "simulado"])
    ap.add_argument("--modelo", default=None, help="padrão: o do .env")
    ap.add_argument("--transcritor", required=True, help="experimento da Fase 3 cujas transcrições serão avaliadas")
    ap.add_argument("--repeticoes", type=int, default=3)
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--paralelo", type=int, default=1)
    ap.add_argument("--sem-referencia", action="store_true", help="não avalia a transcrição de referência")
    a = ap.parse_args()

    modelo = obter_provedor(a.provedor, a.modelo).modelo
    paginas = (obter_config().dados_dir / "amostras" / "bressay_teste.txt").read_text(encoding="utf-8").split()
    paginas = paginas[: a.limite]
    transc = transcricoes(a.transcritor)
    nome_t = a.transcritor.removeprefix("bressay-transc-")
    exp_ref = f"bressay-aval-ref-{a.provedor}-{modelo}"
    exp_tr = f"bressay-aval-transc[{nome_t}]-{a.provedor}-{modelo}"

    tarefas = []  # (experimento, página, repetição, texto)
    if not a.sem_referencia:
        feitos = ja_feitos(exp_ref)
        for r in range(1, a.repeticoes + 1):
            for p in paginas:
                if (p, r) not in feitos:
                    tarefas.append((exp_ref, p, r, texto_para_avaliacao(referencia_bruta(p))))
    feitos = ja_feitos(exp_tr)
    for r in range(1, a.repeticoes + 1):
        for p in paginas:
            if (p, r) not in feitos and (p, r) in transc:
                tarefas.append((exp_tr, p, r, transc[(p, r)]))
    print(f"Referência: {exp_ref}\nTranscrição: {exp_tr}\n{len(paginas)} páginas; faltam {len(tarefas)} chamadas")

    def uma(exp: str, pagina: str, rep: int, texto: str):
        pedido = PedidoAvaliacao(texto=texto, proposta=Proposta(titulo=TEMA), provedor=a.provedor, modelo=a.modelo,
                                 recuperacao=ConfigRecuperacao(k=0), incluir_textos_motivadores=False,
                                 redacao_id=pagina)
        return avaliar(pedido, experimento=exp, repeticao=rep)

    erros = 0
    with ThreadPoolExecutor(max_workers=a.paralelo) as ex:
        futuros = {ex.submit(uma, *t): t[:3] for t in tarefas}
        for n, f in enumerate(as_completed(futuros), 1):
            exp, p, r = futuros[f]
            tipo = "ref" if exp == exp_ref else "transc"
            try:
                res = f.result()
                print(f"[{n}/{len(tarefas)}] {tipo} página {p} rep {r}: total {res.nota_total}")
            except Exception as e:  # noqa: BLE001
                erros += 1
                print(f"[{n}/{len(tarefas)}] {tipo} página {p} rep {r}: ERRO {e}", file=sys.stderr)
    print(f"\nConcluído com {erros} erro(s). Rode de novo para refazer as que falharam.")
    print(f"Métricas: python -m experimentos.metricas_fase4 --referencia \"{exp_ref}\" --transcricao \"{exp_tr}\" "
          f"--transcritor \"{a.transcritor}\"")


if __name__ == "__main__":
    main()
