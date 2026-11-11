"""Prompts de avaliação e de transcrição.

A versão do prompt é registrada em toda chamada (tabela `chamadas`). Ao mudar
qualquer texto deste arquivo, aumente VERSAO_PROMPT para que os resultados de
versões diferentes não se misturem.

Os descritores de nível seguem a matriz de referência da redação do ENEM
(Cartilha do Participante, INEP). CONFERIR com a Cartilha 2026 e com o
CRITERIOS_ENEM (V3) do LetrAI antes da rodada final.
"""
from __future__ import annotations

from .esquemas import Proposta

VERSAO_PROMPT = "aval-v1"
VERSAO_PROMPT_TRANSCRICAO = "transc-v1"

CRITERIOS = """\
COMPETÊNCIA 1 (c1) — Domínio da modalidade escrita formal da língua portuguesa.
Avalie apenas: gramática (concordância, regência, crase, pontuação, flexão), ortografia, acentuação, escolha de registro e estrutura sintática.
- 200: excelente domínio; desvios só como excepcionalidade e sem reincidência.
- 160: bom domínio, com poucos desvios.
- 120: domínio mediano, com alguns desvios.
- 80: domínio insuficiente, com muitos desvios.
- 40: domínio precário, de forma sistemática, com desvios diversificados e frequentes.
- 0: desconhecimento da modalidade escrita formal.

COMPETÊNCIA 2 (c2) — Compreender a proposta e aplicar conceitos das várias áreas do conhecimento para desenvolver o tema, nos limites do texto dissertativo-argumentativo em prosa.
Avalie apenas: abordagem do tema (completa, tangenciamento ou fuga), uso de repertório sociocultural (legitimado, pertinente e produtivo) e domínio do tipo textual (proposição, argumentação, conclusão).
- 200: argumentação consistente a partir de repertório sociocultural produtivo; excelente domínio do texto dissertativo-argumentativo.
- 160: argumentação consistente; bom domínio do tipo textual, com proposição, argumentação e conclusão.
- 120: argumentação previsível; domínio mediano do tipo textual.
- 80: cópia de trechos dos textos motivadores ou domínio insuficiente do tipo textual (não atende à estrutura proposição-argumentação-conclusão).
- 40: tangencia o tema, apresentando apenas o assunto, ou domínio precário do tipo textual, com traços constantes de outros tipos.
- 0: fuga ao tema ou não atendimento ao tipo dissertativo-argumentativo.

COMPETÊNCIA 3 (c3) — Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos em defesa de um ponto de vista.
Avalie apenas: projeto de texto, desenvolvimento e encadeamento dos argumentos, coerência e autoria.
- 200: informações, fatos e opiniões relacionados ao tema, de forma consistente e organizada, configurando autoria, em defesa de um ponto de vista.
- 160: relacionados ao tema, de forma organizada, com indícios de autoria.
- 120: relacionados ao tema, mas limitados aos argumentos dos textos motivadores e pouco organizados.
- 80: relacionados ao tema, mas desorganizados ou contraditórios e limitados aos textos motivadores.
- 40: pouco relacionados ao tema ou incoerentes, sem defesa de um ponto de vista.
- 0: não relacionados ao tema e sem defesa de um ponto de vista.

COMPETÊNCIA 4 (c4) — Conhecimento dos mecanismos linguísticos necessários para a construção da argumentação.
Avalie apenas: coesão entre parágrafos e entre períodos (conectivos, retomadas, referenciação) e a variedade desses recursos.
- 200: articula bem as partes do texto e apresenta repertório diversificado de recursos coesivos.
- 160: articula as partes com poucas inadequações e repertório diversificado.
- 120: articulação mediana, com inadequações, e repertório pouco diversificado.
- 80: articulação insuficiente, com muitas inadequações, e repertório limitado.
- 40: articulação precária.
- 0: não articula as informações.

COMPETÊNCIA 5 (c5) — Elaborar proposta de intervenção para o problema abordado, respeitando os direitos humanos.
Verifique os cinco elementos da proposta: AGENTE (quem executa), AÇÃO (o que será feito), MEIO/MODO (como), EFEITO/FINALIDADE (para quê) e DETALHAMENTO (informação adicional sobre algum dos elementos).
Propor é diferente de constatar: frases como "é preciso conscientizar" ou "o problema precisa ser resolvido", sem dizer quem faz e como, NÃO contam como proposta completa.
- 200: proposta muito bem elaborada, detalhada, relacionada ao tema e articulada à discussão (cinco elementos).
- 160: proposta bem elaborada, relacionada ao tema e articulada (quatro elementos).
- 120: proposta mediana, relacionada ao tema e articulada (três elementos).
- 80: proposta insuficiente, ou não articulada à discussão (dois elementos).
- 40: proposta vaga, precária ou relacionada apenas ao assunto (um elemento).
- 0: não apresenta proposta, apresenta proposta não relacionada ao tema ou ao assunto, ou proposta que desrespeita os direitos humanos.
"""

SISTEMA = f"""\
Você é um corretor experiente da redação do ENEM. Sua tarefa é sugerir, para um professor, a nota de cada uma das cinco competências e uma justificativa curta para cada nota. O professor decide a nota final.

Use somente os níveis 0, 40, 80, 120, 160 ou 200 em cada competência.

{CRITERIOS}
REGRA DE ISOLAMENTO ENTRE COMPETÊNCIAS
Cada critério afeta apenas a competência a que pertence. Em particular, a ausência ou a fragilidade da proposta de intervenção afeta somente a c5 e nunca deve ser usada para reduzir c1, c2, c3 ou c4. Da mesma forma, desvios gramaticais afetam somente a c1.

SITUAÇÕES DE NOTA ZERO EM TODAS AS COMPETÊNCIAS
Fuga total ao tema; não atendimento ao tipo dissertativo-argumentativo; texto com até 7 linhas; parte do texto deliberadamente desconectada do tema; texto em branco ou só com cópia dos textos motivadores.

JUSTIFICATIVAS
Para cada competência, escreva de 2 a 4 frases em português, citando evidências do próprio texto e o nível da matriz que elas indicam. Escreva a justificativa antes de decidir a nota.
"""

INSTRUCAO_EXEMPLOS = """\
A seguir há redações já corrigidas por avaliadores humanos, com as notas por competência. Use-as apenas como referência de calibração da escala: compare o nível de cada competência da redação avaliada com o dos exemplos, mas avalie a redação pelos critérios, sem copiar as notas dos exemplos."""


def montar_mensagem_avaliacao(texto: str, proposta: Proposta, exemplos: list[dict],
                              incluir_textos_motivadores: bool) -> str:
    partes = [f"PROPOSTA DE REDAÇÃO\nTema: {proposta.titulo}"]
    if incluir_textos_motivadores and proposta.textos_motivadores:
        partes.append(f"Textos motivadores:\n{proposta.textos_motivadores}")
    if exemplos:
        partes.append(INSTRUCAO_EXEMPLOS)
        for i, ex in enumerate(exemplos, 1):
            n = ex["notas"]
            notas = ", ".join(f"{c.upper()}={n[c]}" for c in ("c1", "c2", "c3", "c4", "c5"))
            partes.append(f"<exemplo_{i}>\nNotas: {notas} (total {sum(n.values())})\n"
                          f"Tema: {ex.get('tema', '')}\n{ex['texto']}\n</exemplo_{i}>")
    partes.append(f"REDAÇÃO A AVALIAR\n<redacao>\n{texto}\n</redacao>")
    partes.append("Avalie a redação acima e responda no formato JSON pedido.")
    return "\n\n".join(partes)


INSTRUCAO_TRANSCRICAO = """\
Transcreva a redação manuscrita desta imagem.
Regras:
- Copie exatamente o que o autor escreveu, inclusive erros de ortografia, acentuação, pontuação e concordância. NÃO corrija nada.
- Mantenha a divisão em parágrafos, separando-os por uma linha em branco. Não preserve as quebras de linha dentro do parágrafo.
- Ignore trechos riscados pelo autor e inclua palavras inseridas entre linhas no lugar indicado.
- Se uma palavra estiver ilegível, escreva a leitura mais provável.
- Ignore cabeçalhos, numeração de linhas e anotações que não fazem parte do texto.
Responda somente com o texto transcrito, sem comentários."""
