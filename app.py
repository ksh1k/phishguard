"""FastAPI app: POST /predict + static UI.

Scoring pipeline: normalise URL -> character n-gram model -> stacked gradient boosting (engineered
features + n-gram probability) -> Platt calibration -> score and verdict.
"""
import json
import os
import re

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from features import extract_features, normalize_for_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "model.pkl")
METRICS_PATH = os.path.join(HERE, "model", "metrics.json")
TOKEN = re.compile(r"[a-z0-9]+")

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


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def char_contributions(pipe, url: str):
    """Per-character share of the n-gram model's log-odds (excluding the intercept).

    The logistic regression is linear in the TF-IDF vector, so each n-gram's contribution is
    (its TF-IDF value) x (its learned coefficient). Each contribution is split across the
    occurrences of that n-gram in the URL and then across the characters it covers.
    """
    tfidf, lr = pipe.named_steps["tfidf"], pipe.named_steps["lr"]
    s = url.lower()
    x = tfidf.transform([url]).tocoo()
    contrib = {int(j): float(v * lr.coef_[0][j]) for j, v in zip(x.col, x.data)}
    occurrences = {}
    for n in range(3, 6):
        for i in range(len(s) - n + 1):
            j = tfidf.vocabulary_.get(s[i:i + n])
            if j in contrib:
                occurrences.setdefault(j, []).append((i, n))
    per_char = np.zeros(len(s))
    for j, spots in occurrences.items():
        share = contrib[j] / len(spots)
        for i, n in spots:
            per_char[i:i + n] += share / n
    return s, per_char


def fragment_contributions(pipe, url: str, k: int = 3) -> list:
    """Top-k alphanumeric URL tokens by absolute contribution, as (token, log-odds contribution)."""
    s, per_char = char_contributions(pipe, url)
    totals = {}
    for m in TOKEN.finditer(s):
        totals[m.group()] = totals.get(m.group(), 0.0) + float(per_char[m.start():m.end()].sum())
    ranked = sorted(totals.items(), key=lambda t: abs(t[1]), reverse=True)
    return [(tok, c) for tok, c in ranked[:k] if abs(c) > 1e-6]


def score_url(url: str) -> dict:
    b = get_bundle()
    normalized = normalize_for_model(url)
    feats = extract_features(normalized)
    p_ngram = float(b["ngram"].predict_proba([normalized])[0, 1])
    x = np.array([[feats[n] for n in b["feature_names"]] + [p_ngram]], dtype=float)
    p_stacked = float(b["stacked"].predict_proba(x)[0, 1])
    p = float(b["platt"].predict_proba([[logit(p_stacked)]])[0, 1])
    t = b["thresholds"]
    verdict = "likely_phishing" if p >= t["likely_phishing"] else "suspicious" if p >= t["suspicious"] else "safe"
    top = [
        {"feature": "url_fragment", "value": tok, "impact": round(c, 4),
         "direction": "phishing" if c > 0 else "safe",
         "explanation": f'The fragment "{tok}" is typical of ' + ("phishing" if c > 0 else "legitimate") + " URLs in the training data"}
        for tok, c in fragment_contributions(b["ngram"], normalized)
    ]
    return {"score": round(p * 100), "verdict": verdict, "top_features": top}


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
