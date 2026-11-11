"""Provedor falso, sem custo e sem rede: serve para testar a API e os scripts."""
from __future__ import annotations

import hashlib
import json

from .base import Provedor, RespostaLLM


class ProvedorSimulado(Provedor):
    nome = "simulado"

    def __init__(self, modelo: str = "simulado-v1"):
        super().__init__(modelo)

    def gerar_json(self, sistema: str, usuario: str, esquema: dict) -> RespostaLLM:
        h = hashlib.sha256(usuario.encode()).digest()
        saida = {f"c{i}": {"justificativa": "Avaliação simulada.", "nota": str(40 * (2 + h[i] % 3))}
                 for i in range(1, 6)}
        return RespostaLLM(texto=json.dumps(saida), modelo=self.modelo, latencia_ms=0,
                           tokens_entrada=len(usuario) // 4, tokens_saida=60)

    def transcrever(self, imagem: bytes, mime: str, instrucao: str) -> RespostaLLM:
        return RespostaLLM(texto="Texto transcrito simulado.", modelo=self.modelo, latencia_ms=0)
