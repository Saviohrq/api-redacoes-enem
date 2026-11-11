# dados/

Só `splits/` e `amostras/` vão para o GitHub (ids das redações, para reprodutibilidade).
O resto é gerado localmente e fica fora do Git:

- `essay-br/` — clone de https://github.com/lplnufpi/essay-br
- `indice/` — embeddings do acervo de treino (`python -m experimentos.construir_indice`)
- `registro.sqlite` — todas as chamadas aos modelos (faça backup dele!)
- `resultados/` — CSVs gerados por `experimentos.metricas`
