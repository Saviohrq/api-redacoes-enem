"""API de pré-análise de redações do ENEM com LLMs e RAG.

Rodar:  uvicorn app.main:app --reload
Documentação interativa: http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import prompts
from .esquemas import (ConfigRecuperacao, NomeProvedor, PedidoAvaliacao, Proposta, RespostaAvaliacao,
                       RespostaPreAnalise, RespostaTranscricao)
from .provedores import ErroProvedor
from .servicos import ErroAvaliacao, avaliar, pre_analise, transcrever

app = FastAPI(
    title="API de pré-análise de redações do ENEM",
    version="0.1.0",
    description="Sugere notas e justificativas por competência (C1 a C5) com LLMs de provedores "
                "intercambiáveis e recuperação de redações corrigidas (RAG). A nota é uma sugestão: "
                "quem decide é o professor.",
)

TIPOS_IMAGEM = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def _erro(e: Exception) -> HTTPException:
    if isinstance(e, FileNotFoundError):
        return HTTPException(503, str(e))
    if isinstance(e, ErroProvedor):
        return HTTPException(400, str(e))
    return HTTPException(502, str(e))


async def _ler_imagem(arquivo: UploadFile) -> tuple[bytes, str]:
    mime = arquivo.content_type or ""
    if mime not in TIPOS_IMAGEM:
        raise HTTPException(415, f"Formato não aceito: {mime}. Use PNG, JPEG, WEBP ou GIF.")
    dados = await arquivo.read()
    if len(dados) > 20 * 1024 * 1024:
        raise HTTPException(413, "Imagem maior que 20 MB")
    return dados, mime


@app.get("/saude")
def saude():
    return {"status": "ok", "versao_prompt": prompts.VERSAO_PROMPT,
            "versao_prompt_transcricao": prompts.VERSAO_PROMPT_TRANSCRICAO}


@app.post("/avaliar", response_model=RespostaAvaliacao)
def rota_avaliar(pedido: PedidoAvaliacao):
    """Texto + proposta → nota sugerida e justificativa por competência."""
    try:
        return avaliar(pedido)
    except (ErroAvaliacao, ErroProvedor, FileNotFoundError) as e:
        raise _erro(e) from e


@app.post("/transcrever", response_model=RespostaTranscricao)
async def rota_transcrever(imagem: UploadFile = File(...), provedor: NomeProvedor = Form("gemini"),
                           modelo: Optional[str] = Form(None)):
    """Imagem da redação manuscrita → texto transcrito."""
    dados, mime = await _ler_imagem(imagem)
    try:
        return transcrever(dados, mime, provedor, modelo)
    except (ErroAvaliacao, ErroProvedor) as e:
        raise _erro(e) from e


@app.post("/pre-analise", response_model=RespostaPreAnalise)
async def rota_pre_analise(
    imagem: UploadFile = File(...),
    tema: str = Form(...),
    textos_motivadores: Optional[str] = Form(None),
    modo: Literal["unico", "separado"] = Form("unico"),
    provedor: NomeProvedor = Form("gemini", description="Provedor da avaliação (e da transcrição, no modo único)"),
    modelo: Optional[str] = Form(None),
    provedor_transcricao: Optional[NomeProvedor] = Form(None, description="Só no modo separado"),
    modelo_transcricao: Optional[str] = Form(None),
    k: int = Form(0, ge=0, le=10),
    modo_exemplos: Literal["similares", "aleatorios"] = Form("similares"),
):
    """Imagem → transcrição → avaliação, encadeando /transcrever e /avaliar."""
    if modo == "separado" and not provedor_transcricao:
        raise HTTPException(422, "No modo 'separado', informe provedor_transcricao")
    dados, mime = await _ler_imagem(imagem)
    pedido = PedidoAvaliacao(texto="-", proposta=Proposta(titulo=tema, textos_motivadores=textos_motivadores),
                             provedor=provedor, modelo=modelo,
                             recuperacao=ConfigRecuperacao(k=k, modo=modo_exemplos))
    try:
        return pre_analise(dados, mime, pedido, modo, provedor_transcricao, modelo_transcricao)
    except (ErroAvaliacao, ErroProvedor, FileNotFoundError) as e:
        raise _erro(e) from e
