import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")

def main():
    X_train = pd.read_csv(PROC_DIR / "X_train.csv")
    y_train = pd.read_csv(PROC_DIR / "y_train.csv").squeeze("columns")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=42))
    ])

    pipe.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "model.pkl")

    print("Modelo salvo em models/model.pkl")

if __name__ == "__main__":
    main()