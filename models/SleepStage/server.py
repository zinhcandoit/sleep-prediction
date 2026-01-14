from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import joblib
import pandas as pd
from collections import deque

from sleepStages import SleepPrediction

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("sleep_model.pth", map_location=device)

meta = checkpoint["meta"]
FEATURE_COLS = meta["feature_cols"]
T = meta["T"]

model = SleepPrediction(
    input_dim=checkpoint["input_dim"],
    hidden1=128,
    hidden2=64,
    num_classes=5
).to(device)

model.load_state_dict(checkpoint["model_state"])
model.eval()

scaler = joblib.load("scaler.pkl")

buffer = deque(maxlen=T)

# INPUT SCHEMA

class SensorInput(BaseModel):
    hr: float
    ax: float
    ay: float
    az: float

app = FastAPI()

#FEATURE ENGINEERING

def customize_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # diff accel
    df[['dax','day','daz']] = (
        df[['ax','ay','az']]
        .diff()
        .fillna(0)
    )

    df['dacc_mag'] = np.sqrt(
        df['dax']**2 + df['day']**2 + df['daz']**2
    )

    # RR
    df['RR'] = 60000.0 / df['HR'].replace(0, np.nan)
    df['RR'] = df['RR'].ffill().bfill()

    df['RR_diff'] = df['RR'].diff().fillna(0)

    win = T

    df['HRV_RMSSD'] = (
        df['RR_diff']
        .rolling(win, min_periods=1)
        .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
    )

    df['acc_mean_10'] = (
        df['dacc_mag']
        .rolling(win, min_periods=1)
        .mean()
    )

    df['acc_std_10'] = (
        df['dacc_mag']
        .rolling(win, min_periods=1)
        .std()
        .fillna(0)
    )

    return df


@app.post("/push")
def push_data(data: SensorInput):

    buffer.append({
        "HR": data.hr,
        "ax": data.ax,
        "ay": data.ay,
        "az": data.az
    })

    if len(buffer) < 10:
        return {
            "status": "buffering",
            "buffer_size": len(buffer)
        }

    df = pd.DataFrame(buffer)

    df_feat = customize_features(df)

    X = df_feat[FEATURE_COLS].values
    X = scaler.transform(X)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(X)
        pred = logits.argmax(dim=1).item()

    return {
        "status": "ok",
        "prediction": int(pred)
    }

@app.get("/model-info")
def model_info():
    return {
        "window_size": T,
        "features": FEATURE_COLS,
        "num_classes": 5
    }

# uvicorn server:app --host 0.0.0.0 --port 8000