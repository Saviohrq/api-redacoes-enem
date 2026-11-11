"""Interface única de provedores de LLM.

Cada provedor implementa duas operações com a mesma assinatura:
- gerar_json: recebe instruções + texto e devolve JSON no esquema pedido (avaliação);
- transcrever: recebe uma imagem e devolve o texto transcrito.
A lógica de prompt, RAG e validação fica fora dos provedores (app/servicos.py),
de modo que trocar de LLM é só trocar de configuração.
"""
from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RespostaLLM:
    texto: str
    modelo: str
    latencia_ms: int
    tokens_entrada: int | None = None
    tokens_saida: int | None = None
    parametros: dict[str, Any] = field(default_factory=dict)


class ErroProvedor(RuntimeError):
    """Erro definitivo do provedor (bloqueio, recusa, resposta truncada etc.)."""


class Provedor(ABC):
    nome: str
    aceita_imagem: bool = True

    def __init__(self, modelo: str):
        if not modelo:
            raise ErroProvedor(f"Nenhum modelo configurado para o provedor '{self.nome}'. Veja o arquivo .env.")
        self.modelo = modelo

    @abstractmethod
    def gerar_json(self, sistema: str, usuario: str, esquema: dict) -> RespostaLLM: ...

    @abstractmethod
    def transcrever(self, imagem: bytes, mime: str, instrucao: str) -> RespostaLLM: ...


def com_retentativas(fn: Callable[[], RespostaLLM], eh_temporario: Callable[[Exception], float | None],
                     tentativas: int) -> RespostaLLM:
    """Repete a chamada em erros temporários (cota, sobrecarga), com espera crescente.

    `eh_temporario` devolve quantos segundos esperar (ou None se o erro não for temporário).
    """
    for i in range(tentativas):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            espera = eh_temporario(e)
            if espera is None or i == tentativas - 1:
                raise
            espera = max(espera, min(60.0, 2 ** i)) + random.uniform(0, 1)
            time.sleep(espera)
    raise AssertionError("inalcançável")


def cronometro() -> Callable[[], int]:
    inicio = time.perf_counter()
    return lambda: int((time.perf_counter() - inicio) * 1000)
