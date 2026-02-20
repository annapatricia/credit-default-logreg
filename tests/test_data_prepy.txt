import pandas as pd
from src.data_prep import clean_data

def test_clean_data_outputs_expected_columns():
    df = pd.DataFrame({
        "renda": [5000, None],
        "limite": [8000, 9000],
        "utilizacao": [1.2, -0.1],
        "atrasos_12m": [2, None],
        "idade_conta": [24, 36],
        "default_30d": [1, 0]
    })

    out = clean_data(df)

    expected = {"renda","limite","utilizacao","atrasos_12m","idade_conta","default_30d"}
    assert expected.issubset(set(out.columns))
    assert out["utilizacao"].between(0, 1).all()
    assert (out["atrasos_12m"] >= 0).all()
    assert (out["idade_conta"] >= 0).all()
    assert out.isna().sum().sum() == 0