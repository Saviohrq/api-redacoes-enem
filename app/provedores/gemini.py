"""Provedor Google Gemini, via SDK `google-genai`."""
from __future__ import annotations

import re

from google import genai
from google.genai import errors, types

from .base import ErroProvedor, Provedor, RespostaLLM, com_retentativas, cronometro

# Mesmo ajuste do protótipo LetrAI: evita bloqueio falso em temas sensíveis do ENEM.
SEGURANCA = [
    types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH")
    for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
              "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
]


def _espera(e: Exception) -> float | None:
    if isinstance(e, errors.APIError) and getattr(e, "code", None) in (429, 500, 502, 503, 504):
        m = re.search(r"retry(?:Delay)?['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(e), re.I)
        return float(m.group(1)) if m else 10.0
    return None


class ProvedorGemini(Provedor):
    nome = "gemini"

    def __init__(self, modelo: str, api_key: str, tentativas: int = 6):
        super().__init__(modelo)
        if not api_key:
            raise ErroProvedor("GEMINI_API_KEY não definida no .env")
        self.cliente = genai.Client(api_key=api_key)
        self.tentativas = tentativas

    def _chamar(self, contents, config: types.GenerateContentConfig) -> RespostaLLM:
        parar = cronometro()
        r = self.cliente.models.generate_content(model=self.modelo, contents=contents, config=config)
        if not r.text:
            motivo = r.candidates[0].finish_reason if r.candidates else r.prompt_feedback
            raise ErroProvedor(f"Resposta vazia do Gemini: {motivo}")
        u = r.usage_metadata
        return RespostaLLM(texto=r.text, modelo=r.model_version or self.modelo, latencia_ms=parar(),
                           tokens_entrada=getattr(u, "prompt_token_count", None),
                           tokens_saida=getattr(u, "candidates_token_count", None),
                           parametros={"temperature": 0, "seed": 42})

    def gerar_json(self, sistema: str, usuario: str, esquema: dict) -> RespostaLLM:
        config = types.GenerateContentConfig(
            system_instruction=sistema, temperature=0, seed=42, safety_settings=SEGURANCA,
            response_mime_type="application/json", response_json_schema=esquema)
        return com_retentativas(lambda: self._chamar(usuario, config), _espera, self.tentativas)

    def transcrever(self, imagem: bytes, mime: str, instrucao: str) -> RespostaLLM:
        config = types.GenerateContentConfig(temperature=0, seed=42, safety_settings=SEGURANCA)
        contents = [types.Part.from_bytes(data=imagem, mime_type=mime), instrucao]
        return com_retentativas(lambda: self._chamar(contents, config), _espera, self.tentativas)
