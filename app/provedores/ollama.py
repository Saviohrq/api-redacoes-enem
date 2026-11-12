"""Provedor de modelo aberto (Qwen2.5-VL) executado localmente pelo Ollama (API HTTP em localhost:11434)."""
from __future__ import annotations

import base64

import httpx

from .base import ErroProvedor, Provedor, RespostaLLM, com_retentativas, cronometro


# Limite de saída: evita respostas que entram em repetição e nunca terminam
# (uma avaliação normal usa ~500 a 1.000 tokens).
MAX_TOKENS_SAIDA = 4096


def _espera(e: Exception) -> float | None:
    if isinstance(e, httpx.ConnectError):
        return 5.0
    if isinstance(e, httpx.ReadTimeout):
        return None  # estourou o tempo: repetir costuma estourar de novo, então desiste logo
    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500:
        return 5.0
    return None


class ProvedorOllama(Provedor):
    nome = "ollama"

    def __init__(self, modelo: str, url: str, tentativas: int = 3):
        super().__init__(modelo)
        self.url = url.rstrip("/")
        self.tentativas = tentativas

    def _chamar(self, mensagens: list[dict], formato: dict | None) -> RespostaLLM:
        parar = cronometro()
        corpo = {"model": self.modelo, "messages": mensagens, "stream": False,
                 "options": {"temperature": 0, "seed": 42, "num_ctx": 16384, "num_predict": MAX_TOKENS_SAIDA}}
        if formato:
            corpo["format"] = formato
        r = httpx.post(f"{self.url}/api/chat", json=corpo, timeout=600)
        r.raise_for_status()
        d = r.json()
        if d.get("done_reason") == "length":
            raise ErroProvedor(f"Resposta truncada (num_predict={MAX_TOKENS_SAIDA})")
        return RespostaLLM(texto=d["message"]["content"], modelo=d.get("model", self.modelo), latencia_ms=parar(),
                           tokens_entrada=d.get("prompt_eval_count"), tokens_saida=d.get("eval_count"),
                           parametros=corpo["options"])

    def gerar_json(self, sistema: str, usuario: str, esquema: dict) -> RespostaLLM:
        msgs = [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}]
        return com_retentativas(lambda: self._chamar(msgs, esquema), _espera, self.tentativas)

    def transcrever(self, imagem: bytes, mime: str, instrucao: str) -> RespostaLLM:
        msgs = [{"role": "user", "content": instrucao, "images": [base64.b64encode(imagem).decode()]}]
        return com_retentativas(lambda: self._chamar(msgs, None), _espera, self.tentativas)
