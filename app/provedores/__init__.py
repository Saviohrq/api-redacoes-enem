from __future__ import annotations

from functools import lru_cache

from ..config import obter_config
from .base import ErroProvedor, Provedor, RespostaLLM


@lru_cache(maxsize=16)
def obter_provedor(nome: str, modelo: str | None = None) -> Provedor:
    """Fábrica de provedores: o modelo vem do .env, a menos que o pedido informe outro."""
    cfg = obter_config()
    if nome == "anthropic":
        from .anthropic_claude import ProvedorAnthropic
        return ProvedorAnthropic(modelo or cfg.anthropic_model, cfg.anthropic_api_key, cfg.anthropic_effort)
    if nome == "gemini":
        from .gemini import ProvedorGemini
        return ProvedorGemini(modelo or cfg.gemini_model, cfg.gemini_api_key, cfg.max_tentativas)
    if nome == "ollama":
        from .ollama import ProvedorOllama
        return ProvedorOllama(modelo or cfg.ollama_model, cfg.ollama_url)
    if nome == "simulado":
        from .simulado import ProvedorSimulado
        return ProvedorSimulado(modelo or "simulado-v1")
    raise ErroProvedor(f"Provedor desconhecido: {nome}")


__all__ = ["obter_provedor", "Provedor", "RespostaLLM", "ErroProvedor"]
