"""Lista os modelos Gemini disponíveis para a sua chave (use um nome fixo no .env)."""
from google import genai

from app.config import obter_config

if __name__ == "__main__":
    cliente = genai.Client(api_key=obter_config().gemini_api_key)
    for m in cliente.models.list():
        if "generateContent" in (m.supported_actions or []):
            print(m.name.removeprefix("models/"))
