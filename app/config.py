"""Configuração lida do arquivo .env (ou de variáveis de ambiente)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

RAIZ = Path(__file__).resolve().parent.parent


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=RAIZ / ".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_effort: str = "medium"

    gemini_api_key: str = ""
    gemini_model: str = ""

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:7b"

    essaybr_dir: Path = RAIZ / "dados" / "essay-br"
    embeddings_model: str = "intfloat/multilingual-e5-base"
    dados_dir: Path = RAIZ / "dados"
    banco: Path = RAIZ / "dados" / "registro.sqlite"

    max_tentativas: int = 6


@lru_cache
def obter_config() -> Config:
    return Config()
