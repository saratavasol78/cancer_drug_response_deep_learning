import joblib
import pandas as pd

def transform_row(row_dict, features_path="models/features.json",
                  imputer_path="models/imputer.joblib",
                  scaler_path="models/scaler.joblib"):
    import json
    with open(features_path, encoding="utf-8") as f:
        features = json.load(f)
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    row = pd.DataFrame([row_dict]).reindex(columns=features)
    return scaler.transform(imputer.transform(row))
