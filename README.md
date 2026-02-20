# Credit Default Prediction (Logistic Regression)

Pipeline simples e reproduzível de Machine Learning para prever inadimplência (default_30d) usando Logistic Regression.

## O que tem aqui
- Geração de dados simulados (para rodar sem depender de dataset externo)
- Limpeza e split de dados
- Treino (Pipeline: StandardScaler + LogisticRegression)
- Avaliação (ROC AUC + métricas)
- CI com GitHub Actions + Artifacts (metrics.json e run_info.json)

## Como rodar (end-to-end)
python -m src.run_all

## Como fazer uma previsão
python -m src.predict --renda 4000 --limite 8000 --utilizacao 0.8 --atrasos_12m 2 --idade_conta 24

## Onde ver as métricas no GitHub
Repo → Actions → último run (verde) → Artifacts → baixar “metrics”
- reports/metrics/metrics.json
- reports/metrics/run_info.json

## Saídas locais
- Modelo: models/model.pkl
- Métricas: reports/metrics/metrics.json e reports/metrics/run_info.json