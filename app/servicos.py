"""Operações da API: avaliar, transcrever e pré-análise (transcrever + avaliar)."""
from __future__ import annotations

import json

from pydantic import ValidationError

from . import prompts
from .esquemas import (Avaliacao, Metadados, PedidoAvaliacao, RespostaAvaliacao, RespostaPreAnalise,
                       RespostaTranscricao, esquema_json_avaliacao)
from .provedores import ErroProvedor, obter_provedor
from .registro import registrar


class ErroAvaliacao(RuntimeError):
    pass


def _interpretar(texto: str) -> Avaliacao:
    bruto = texto.strip()
    if bruto.startswith("```"):
        bruto = bruto.strip("`").removeprefix("json").strip()
    dados = json.loads(bruto)
    for c in ("c1", "c2", "c3", "c4", "c5"):
        dados[c]["nota"] = int(dados[c]["nota"])
    return Avaliacao.model_validate(dados)


def avaliar(pedido: PedidoAvaliacao, *, experimento: str | None = None, repeticao: int | None = None) -> RespostaAvaliacao:
    rec = pedido.recuperacao
    ids_ex: list[int] = []
    exemplos: list[dict] = []
    if rec.k > 0:
        from .rag.recuperacao import exemplos_por_ids, recuperar
        ids_ex = recuperar(pedido.texto, rec.k, rec.modo, semente=rec.semente, redacao_id=pedido.redacao_id,
                           excluir_ids=pedido.excluir_ids,
                           proposta_id=pedido.proposta_id if rec.mesma_proposta else None)
        exemplos = exemplos_por_ids(ids_ex)

    usuario = prompts.montar_mensagem_avaliacao(pedido.texto, pedido.proposta, exemplos,
                                                pedido.incluir_textos_motivadores)
    provedor = obter_provedor(pedido.provedor, pedido.modelo)
    config = {"recuperacao": rec.model_dump(), "incluir_textos_motivadores": pedido.incluir_textos_motivadores}
    base = dict(operacao="avaliar", experimento=experimento, repeticao=repeticao,
                redacao_id=None if pedido.redacao_id is None else str(pedido.redacao_id),
                provedor=provedor.nome, modelo_pedido=provedor.modelo, versao_prompt=prompts.VERSAO_PROMPT,
                ids_exemplos=ids_ex)
    try:
        r = provedor.gerar_json(prompts.SISTEMA, usuario, esquema_json_avaliacao())
    except Exception as e:  # noqa: BLE001
        registrar(**base, config=config, sucesso=False, erro=f"{type(e).__name__}: {e}")
        raise ErroAvaliacao(f"Falha ao chamar {provedor.nome}: {e}") from e
    config["geracao"] = r.parametros
    campos = dict(**base, config=config, modelo_resposta=r.modelo, tokens_entrada=r.tokens_entrada,
                  tokens_saida=r.tokens_saida, latencia_ms=r.latencia_ms, resposta_bruta=r.texto)
    try:
        aval = _interpretar(r.texto)
    except (ValueError, KeyError, TypeError, ValidationError) as e:
        registrar(**campos, sucesso=False, erro=f"Resposta fora do esquema: {e}")
        raise ErroAvaliacao(f"Resposta fora do esquema: {e}") from e
    id_reg = registrar(**campos, sucesso=True, resultado=aval.model_dump_json())
    return RespostaAvaliacao(
        avaliacao=aval, nota_total=aval.total, ids_exemplos=ids_ex,
        meta=Metadados(id_registro=id_reg, provedor=provedor.nome, modelo=r.modelo,
                       versao_prompt=prompts.VERSAO_PROMPT, latencia_ms=r.latencia_ms,
                       tokens_entrada=r.tokens_entrada, tokens_saida=r.tokens_saida))


def transcrever(imagem: bytes, mime: str, provedor_nome: str, modelo: str | None = None, *,
                experimento: str | None = None, redacao_id: str | None = None,
                repeticao: int | None = None) -> RespostaTranscricao:
    provedor = obter_provedor(provedor_nome, modelo)
    if not provedor.aceita_imagem:
        raise ErroAvaliacao(f"O modelo {provedor.modelo} não aceita imagem")
    base = dict(operacao="transcrever", experimento=experimento, redacao_id=redacao_id, repeticao=repeticao,
                provedor=provedor.nome, modelo_pedido=provedor.modelo,
                versao_prompt=prompts.VERSAO_PROMPT_TRANSCRICAO)
    try:
        r = provedor.transcrever(imagem, mime, prompts.INSTRUCAO_TRANSCRICAO)
    except Exception as e:  # noqa: BLE001
        registrar(**base, sucesso=False, erro=f"{type(e).__name__}: {e}")
        raise ErroAvaliacao(f"Falha ao transcrever com {provedor.nome}: {e}") from e
    texto = r.texto.strip()
    id_reg = registrar(**base, config={"geracao": r.parametros}, modelo_resposta=r.modelo,
                       tokens_entrada=r.tokens_entrada, tokens_saida=r.tokens_saida, latencia_ms=r.latencia_ms,
                       resposta_bruta=r.texto, sucesso=True, resultado=texto)
    return RespostaTranscricao(texto=texto, meta=Metadados(
        id_registro=id_reg, provedor=provedor.nome, modelo=r.modelo,
        versao_prompt=prompts.VERSAO_PROMPT_TRANSCRICAO, latencia_ms=r.latencia_ms,
        tokens_entrada=r.tokens_entrada, tokens_saida=r.tokens_saida))


def pre_analise(imagem: bytes, mime: str, pedido_base: PedidoAvaliacao, modo: str,
                provedor_transcricao: str | None = None, modelo_transcricao: str | None = None,
                **contexto) -> RespostaPreAnalise:
    """Modo 'unico': o mesmo modelo transcreve e avalia (duas chamadas, para permitir medir o CER).
    Modo 'separado': um modelo transcreve e outro avalia."""
    if modo == "unico":
        provedor_transcricao, modelo_transcricao = pedido_base.provedor, pedido_base.modelo
    elif not provedor_transcricao:
        raise ErroAvaliacao("No modo 'separado', informe provedor_transcricao")
    t = transcrever(imagem, mime, provedor_transcricao, modelo_transcricao, **contexto)
    pedido = pedido_base.model_copy(update={"texto": t.texto})
    a = avaliar(pedido, experimento=contexto.get("experimento"), repeticao=contexto.get("repeticao"))
    return RespostaPreAnalise(modo=modo, transcricao=t, avaliacao=a)


def avaliacao_erro(e: Exception) -> str:
    return str(e) if isinstance(e, (ErroAvaliacao, ErroProvedor)) else f"{type(e).__name__}: {e}"
