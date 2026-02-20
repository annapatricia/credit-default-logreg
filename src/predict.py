import argparse
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path("models/model.pkl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renda", type=float, required=True)
    parser.add_argument("--limite", type=float, required=True)
    parser.add_argument("--utilizacao", type=float, required=True)
    parser.add_argument("--atrasos_12m", type=int, required=True)
    parser.add_argument("--idade_conta", type=int, required=True)

    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)

    data = pd.DataFrame([{
        "renda": args.renda,
        "limite": args.limite,
        "utilizacao": args.utilizacao,
        "atrasos_12m": args.atrasos_12m,
        "idade_conta": args.idade_conta
    }])

    prob = model.predict_proba(data)[0, 1]

    print(f"Probabilidade de default: {prob:.4f}")

if __name__ == "__main__":
    main()