import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

MODEL = tf.keras.models.load_model("models/final_model.keras")
IMPUTER = joblib.load("models/imputer.joblib")
SCALER = joblib.load("models/scaler.joblib")

with open("models/features.json", encoding="utf-8") as f:
    FEATURES = json.load(f)

def predict(row_dict):
    row = pd.DataFrame([row_dict]).reindex(columns=FEATURES)
    x = SCALER.transform(IMPUTER.transform(row))
    return float(MODEL.predict(x, verbose=0).ravel()[0])
