"""Provedor Anthropic (Claude), via SDK oficial `anthropic`.

Observação para o artigo: os modelos atuais do Claude (ex.: Haiku 5.5) não aceitam
temperatura; o raciocínio (thinking) fica ligado e é controlado pelo `effort`.
"""
from __future__ import annotations

import base64

import anthropic

from .base import ErroProvedor, Provedor, RespostaLLM, com_retentativas, cronometro


class ProvedorAnthropic(Provedor):
    nome = "anthropic"

    def __init__(self, modelo: str, api_key: str, effort: str = "medium", tentativas: int = 6):
        super().__init__(modelo)
        if not api_key:
            raise ErroProvedor("ANTHROPIC_API_KEY não definida no .env")
        # o SDK já repete 429/5xx; somamos poucas tentativas extras
        self.cliente = anthropic.Anthropic(api_key=api_key, max_retries=tentativas)
        self.effort = effort

    def _chamar(self, sistema: str | None, conteudo, formato: dict | None) -> RespostaLLM:
        parar = cronometro()
        output_config: dict = {"effort": self.effort}
        if formato:
            output_config["format"] = formato
        kwargs = dict(model=self.modelo, max_tokens=16000, output_config=output_config,
                      messages=[{"role": "user", "content": conteudo}])
        if sistema:
            kwargs["system"] = sistema
        r = self.cliente.messages.create(**kwargs)
        if r.stop_reason == "refusal":
            raise ErroProvedor(f"Recusa do modelo: {getattr(r, 'stop_details', None)}")
        if r.stop_reason == "max_tokens":
            raise ErroProvedor("Resposta truncada (max_tokens)")
        texto = "".join(b.text for b in r.content if b.type == "text")
        return RespostaLLM(texto=texto, modelo=r.model, latencia_ms=parar(),
                           tokens_entrada=r.usage.input_tokens, tokens_saida=r.usage.output_tokens,
                           parametros={"effort": self.effort, "max_tokens": 16000})

    def gerar_json(self, sistema: str, usuario: str, esquema: dict) -> RespostaLLM:
        return com_retentativas(
            lambda: self._chamar(sistema, usuario, {"type": "json_schema", "schema": esquema}),
            lambda e: 5.0 if isinstance(e, anthropic.APIConnectionError) else None, 2)

    def transcrever(self, imagem: bytes, mime: str, instrucao: str) -> RespostaLLM:
        conteudo = [
            {"type": "image", "source": {"type": "base64", "media_type": mime,
                                         "data": base64.standard_b64encode(imagem).decode()}},
            {"type": "text", "text": instrucao},
        ]
        return self._chamar(None, conteudo, None)
