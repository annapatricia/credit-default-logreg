import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/credit_data.csv")
OUT_DIR = Path("data/processed")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tipos
    numeric_cols = ["renda", "limite", "utilizacao", "atrasos_12m", "idade_conta", "default_30d"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Remover linhas sem target
    df = df.dropna(subset=["default_30d"])

    # Preencher missing com mediana
    features = ["renda", "limite", "utilizacao", "atrasos_12m", "idade_conta"]
    df[features] = df[features].fillna(df[features].median(numeric_only=True))

    # Regras simples de sanidade
    df["utilizacao"] = df["utilizacao"].clip(0, 1)
    df["atrasos_12m"] = df["atrasos_12m"].clip(lower=0)
    df["idade_conta"] = df["idade_conta"].clip(lower=0)

    return df

def main():
    df = pd.read_csv(RAW_PATH)
    df = clean_data(df)

    X = df[["renda", "limite", "utilizacao", "atrasos_12m", "idade_conta"]]
    y = df["default_30d"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    print("Dados processados salvos em data/processed/")

if __name__ == "__main__":
    main()