import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report
import joblib

PROC_DIR = Path("data/processed")
MODEL_PATH = Path("models/model.pkl")
METRICS_DIR = Path("reports/metrics")

def main():
    X_test = pd.read_csv(PROC_DIR / "X_test.csv")
    y_test = pd.read_csv(PROC_DIR / "y_test.csv").squeeze("columns")

    model = joblib.load(MODEL_PATH)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "roc_auc": auc,
        "precision_class_1": report["1"]["precision"],
        "recall_class_1": report["1"]["recall"],
        "f1_class_1": report["1"]["f1-score"]
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Métricas salvas em reports/metrics/metrics.json")
    print(f"ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    main()