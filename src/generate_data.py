import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

n = 2000

renda = np.random.normal(5000, 1500, n).clip(1000)
limite = np.random.normal(8000, 2000, n).clip(2000)
utilizacao = np.random.beta(2, 5, n)
atrasos_12m = np.random.poisson(1.5, n)
idade_conta = np.random.randint(6, 120, n)

# Regra para gerar probabilidade de default
score = (
    0.0003 * renda * -1 +
    0.0002 * limite * -1 +
    2.5 * utilizacao +
    0.8 * atrasos_12m +
    -0.01 * idade_conta
)

prob_default = 1 / (1 + np.exp(-score))

default = np.random.binomial(1, prob_default)

df = pd.DataFrame({
    "renda": renda,
    "limite": limite,
    "utilizacao": utilizacao,
    "atrasos_12m": atrasos_12m,
    "idade_conta": idade_conta,
    "default_30d": default
})

Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/credit_data.csv", index=False)

print("Dataset criado em data/raw/credit_data.csv")