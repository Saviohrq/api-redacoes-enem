"""Formatos de entrada e saída da API (validados pelo Pydantic)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

NOTAS_VALIDAS = (0, 40, 80, 120, 160, 200)
COMPETENCIAS = ("c1", "c2", "c3", "c4", "c5")
Nota = Literal[0, 40, 80, 120, 160, 200]
NomeProvedor = Literal["anthropic", "gemini", "ollama", "simulado"]


class AvaliacaoCompetencia(BaseModel):
    nota: Nota
    justificativa: str


class Avaliacao(BaseModel):
    c1: AvaliacaoCompetencia
    c2: AvaliacaoCompetencia
    c3: AvaliacaoCompetencia
    c4: AvaliacaoCompetencia
    c5: AvaliacaoCompetencia

    @property
    def total(self) -> int:
        return sum(getattr(self, c).nota for c in COMPETENCIAS)

    def notas(self) -> dict[str, int]:
        return {c: getattr(self, c).nota for c in COMPETENCIAS}


def esquema_json_avaliacao() -> dict:
    """Esquema JSON enviado ao LLM.

    As notas vão como texto com enum ("0", "40", ...) porque nem todo provedor
    aceita enum de inteiros no modo de saída estruturada; a conversão para
    inteiro e a validação final são feitas por `Avaliacao`.
    """
    comp = {
        "type": "object",
        "properties": {
            "justificativa": {"type": "string"},
            "nota": {"type": "string", "enum": [str(n) for n in NOTAS_VALIDAS]},
        },
        "required": ["justificativa", "nota"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {c: comp for c in COMPETENCIAS},
        "required": list(COMPETENCIAS),
        "additionalProperties": False,
    }


class Proposta(BaseModel):
    titulo: str = Field(..., description="Tema / título da proposta de redação")
    textos_motivadores: Optional[str] = None


class ConfigRecuperacao(BaseModel):
    k: int = Field(0, ge=0, le=10, description="Quantidade de redações corrigidas inseridas no prompt (0 = zero-shot)")
    modo: Literal["similares", "aleatorios"] = Field(
        "similares", description="'similares' = RAG por embeddings; 'aleatorios' = controle com k redações sorteadas"
    )
    mesma_proposta: bool = Field(False, description="Restringe a busca a redações da mesma proposta")
    semente: int = 42


class PedidoAvaliacao(BaseModel):
    texto: str = Field(..., min_length=1)
    proposta: Proposta
    provedor: NomeProvedor = "gemini"
    modelo: Optional[str] = Field(None, description="Sobrescreve o modelo do .env para este pedido")
    recuperacao: ConfigRecuperacao = ConfigRecuperacao()
    incluir_textos_motivadores: bool = True
    # usados pelos experimentos: impedem que a própria redação (ou outras) entrem como exemplo
    redacao_id: Optional[int | str] = None  # int no Essay-BR; texto no BRESSAY (ex.: "0635-026")
    proposta_id: Optional[int] = None
    excluir_ids: list[int] = []


class Metadados(BaseModel):
    id_registro: Optional[int] = None
    provedor: str
    modelo: str
    versao_prompt: str
    latencia_ms: int
    tokens_entrada: Optional[int] = None
    tokens_saida: Optional[int] = None


class RespostaAvaliacao(BaseModel):
    avaliacao: Avaliacao
    nota_total: int
    ids_exemplos: list[int] = []
    meta: Metadados


class RespostaTranscricao(BaseModel):
    texto: str
    meta: Metadados


class RespostaPreAnalise(BaseModel):
    modo: Literal["unico", "separado"]
    transcricao: RespostaTranscricao
    avaliacao: RespostaAvaliacao
