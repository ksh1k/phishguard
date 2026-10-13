"""FastAPI app: POST /predict + static UI."""
import json
import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from features import explain, extract_features, normalize_for_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "model.pkl")
METRICS_PATH = os.path.join(HERE, "model", "metrics.json")

app = FastAPI(title="PhishGuard")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(503, "Model not trained yet. Run train.py.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class PredictRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


def verdict_for(score: int) -> str:
    if score >= 70:
        return "likely_phishing"
    if score >= 40:
        return "suspicious"
    return "safe"


def score_url(url: str) -> dict:
    b = get_bundle()
    names, model, med = b["feature_names"], b["model"], b["benign_median"]
    feats = extract_features(normalize_for_model(url))
    x = np.array([[feats[n] for n in names]], dtype=float)
    # Occlusion: how much does the phishing probability drop if one feature is set to the benign
    # median? Row 0 is the original URL; row i+1 has feature i replaced. One batched call.
    batch = np.repeat(x, len(names) + 1, axis=0)
    for i in range(len(names)):
        batch[i + 1, i] = med[i]
    probs = model.predict_proba(batch)[:, 1]
    p = float(probs[0])
    contribs = sorted(((n, p - float(probs[i + 1])) for i, n in enumerate(names)), key=lambda t: abs(t[1]), reverse=True)
    top = [
        {"feature": n, "value": feats[n], "impact": round(d, 4),
         "direction": "phishing" if d > 0 else "safe",
         "explanation": explain(n, feats[n], float(med[names.index(n)]), d > 0)}
        for n, d in contribs[:3] if abs(d) > 0
    ]
    score = round(p * 100)
    return {"score": score, "verdict": verdict_for(score), "top_features": top}


@app.post("/predict")
def predict(req: PredictRequest):
    return score_url(req.url)


@app.get("/info")
def info():
    if not os.path.exists(METRICS_PATH):
        return {}
    return json.load(open(METRICS_PATH))


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
