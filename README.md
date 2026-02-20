# Credit Default Prediction (Logistic Regression)

Pipeline simples de ML para prever inadimplência (default_30d) com:
- limpeza de dados
- treino (Logistic Regression)
- avaliação (ROC AUC + métricas)
- CI com GitHub Actions

## Como rodar localmente

python -m src.generate_data
python -m src.data_prep
python -m src.train
python -m src.evaluate

## Saídas
- Modelo: models/model.pkl
- Métricas: reports/metrics/metrics.json