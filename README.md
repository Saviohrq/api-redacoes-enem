# API de pré-análise de redações do ENEM (TCC)

API em FastAPI que sugere, para cada competência do ENEM (C1 a C5), uma nota (0, 40, 80, 120, 160 ou 200) e uma justificativa, usando LLMs de provedores intercambiáveis (Gemini, Claude e Qwen2.5-VL, este aberto e rodando localmente) e recuperação de redações corrigidas (RAG). A nota é uma **sugestão**: quem decide é o professor.

| Operação | O que faz |
|---|---|
| `POST /avaliar` | texto + proposta → notas e justificativas por competência |
| `POST /transcrever` | imagem da redação manuscrita → texto |
| `POST /pre-analise` | imagem → transcrição → avaliação (modo `unico` ou `separado`) |
| `GET /saude` | verifica se a API está no ar e mostra a versão do prompt |

Toda chamada a um modelo fica registrada em `dados/registro.sqlite` (modelo exato, data, versão do prompt, configuração de RAG, tokens, tempo, resposta completa).

## Estrutura

```
app/
  main.py            rotas da API
  servicos.py        avaliar / transcrever / pré-análise
  prompts.py         prompt de avaliação (critérios INEP) e de transcrição, com versão
  esquemas.py        formatos de entrada/saída e esquema JSON das notas
  registro.py        registro das chamadas em SQLite
  provedores/        interface única + Gemini, Anthropic (Claude), Ollama (Qwen2.5-VL) e "simulado"
  rag/               leitura do Essay-BR, splits e recuperação (similares ou aleatórios)
experimentos/
  preparar_dados.py  splits do Essay-BR + amostra fixa de teste (semente 42)
  construir_indice.py embeddings do acervo de treino
  rodar.py           roda uma configuração sobre a amostra (retoma de onde parou)
  metricas.py        MAE, MSE, Pearson, QWK, concordância exata/adjacente, discrepância INEP, bootstrap
tests/               testes automáticos (rodam sem chave e sem internet)
```

## Instalação (Windows / PowerShell)

Requer Python 3.11 ou mais novo. Todos os comandos abaixo são rodados **dentro da pasta do repositório** (`api-redacoes-enem`).

```powershell
git clone https://github.com/Saviohrq/api-redacoes-enem
cd api-redacoes-enem
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
notepad .env        # preencha as chaves e os modelos
```

Corpus Essay-BR (estendido), dentro da pasta `dados`:

```powershell
git clone https://github.com/lplnufpi/essay-br dados\essay-br
```

Teste rápido (não gasta nada):

```powershell
python -m pytest -q
```

## Rodar a API

```powershell
uvicorn app.main:app --reload
```

Abra http://localhost:8000/docs para testar as rotas pelo navegador. Exemplo de corpo para `/avaliar`:

```json
{
  "texto": "Texto da redação...",
  "proposta": {"titulo": "Tema da redação", "textos_motivadores": "opcional"},
  "provedor": "gemini",
  "recuperacao": {"k": 3, "modo": "similares"}
}
```

`provedor` pode ser `gemini`, `anthropic`, `ollama` ou `simulado` (falso, para testes). Para o modelo aberto, instale o [Ollama](https://ollama.com) e rode `ollama pull qwen2.5vl:7b` antes.

## Experimentos

```powershell
# 1) uma vez: splits, amostra de teste (250 redações) e índice do RAG
python -m experimentos.preparar_dados
python -m experimentos.construir_indice

# 2) teste curto antes da rodada de verdade (5 redações, 1 repetição)
python -m experimentos.rodar --provedor gemini --limite 5 --repeticoes 1 --experimento teste-curto

# 3) Fase 1 — comparação entre modelos (k = 0)
python -m experimentos.rodar --provedor gemini
python -m experimentos.rodar --provedor anthropic
python -m experimentos.rodar --provedor ollama

# 4) Fase 2 — efeito das redações recuperadas (para cada modelo)
python -m experimentos.rodar --provedor gemini --k 3 --modo similares     # RAG
python -m experimentos.rodar --provedor gemini --k 3 --modo aleatorios    # controle: 3 sorteadas

# 5) resultados
python -m experimentos.metricas --listar
python -m experimentos.metricas --experimento "NOME_DO_EXPERIMENTO"
python -m experimentos.metricas --comparar "EXP_k0" "EXP_k3_similares"   # bootstrap pareado
```

Se a cota acabar ou der erro, é só rodar o mesmo comando de novo: o que já deu certo é pulado. `--paralelo 4` acelera quando a cota permite.

## Decisões registradas (para a Metodologia)

- **Splits**: os mesmos do `build_dataset.py` do Essay-BR (70/15/15, `random_state=230`). Observação: o script oficial **não** estratifica (o parâmetro `stratify` não é passado), apesar do nome da função.
- **Amostra de teste**: 250 redações do split de teste, alocação proporcional por faixa de nota total (0-199, 200-399, ..., 800-1000) com mínimo de 15 por faixa, semente 42. Ids em `dados/amostras/teste.txt`.
- **Acervo do RAG**: só o split de treino. A redação avaliada nunca entra como exemplo.
- **Embeddings**: `intfloat/multilingual-e5-base` (código aberto, multilíngue), similaridade do cosseno.
- **Controle da Fase 2**: k redações sorteadas do treino, com semente fixa por redação (mesmo sorteio para todos os modelos e repetições).
- **Notas**: o esquema JSON restringe cada nota aos seis níveis válidos; a resposta é validada antes de ser aceita.
- **Repetições**: 3 por configuração; as métricas são a média das repetições, com IC de 95% por bootstrap (1.000 reamostras das redações).
- **Temperatura**: 0 no Gemini e no Qwen2.5-VL. O Claude Haiku 5.5 não aceita temperatura; usa `effort` (registrado).
- **Pré-análise modo único**: duas chamadas ao mesmo modelo (transcrição, depois avaliação), o que permite medir o CER também nesse modo.
- **Modelo aberto**: o plano era o Llama 3.2 Vision 11B, mas o Ollama 0.40 não carrega mais a arquitetura `mllama`. Foi trocado pelo Qwen2.5-VL 7B (Alibaba): aberto, multimodal (necessário para a transcrição), de um terceiro fornecedor e próximo em tamanho.
