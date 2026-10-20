"""FastAPI app: POST /predict + static UI.

Scoring pipeline: normalise URL -> character n-gram model -> stacked gradient boosting (engineered
features + n-gram probability) -> Platt calibration -> score and verdict.

Explanations are faithful to the stacked model: each of its 32 inputs (31 engineered features plus
the n-gram model's probability) is set in turn to the typical legitimate value, and the inputs that
move the model's log-odds most are reported. The n-gram input is explained further by the URL
tokens that drive the text model.
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

from features import explain, extract_features, normalize_for_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "model.pkl")
METRICS_PATH = os.path.join(HERE, "model", "metrics.json")
TOKEN = re.compile(r"[a-z0-9]+")
GENERIC_TOKENS = {"http", "https", "www"}  # carry no meaning of their own in an explanation
MIN_IMPACT = 0.05  # ignore inputs that move the log-odds by less than this

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


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def token_contributions(pipe, url: str) -> dict:
    """Contribution of each alphanumeric URL token to the n-gram model's log-odds.

    The logistic regression is linear in the TF-IDF vector, so an n-gram's contribution is its TF-IDF
    value times its learned coefficient. Only n-grams lying entirely inside a token count towards it,
    so a token's contribution never depends on its neighbours (the same token has the same sign in
    every URL). N-grams that straddle a boundary, such as "com/w", are left out of the per-token view.
    """
    tfidf, lr = pipe.named_steps["tfidf"], pipe.named_steps["lr"]
    s = url.lower()
    row = tfidf.transform([url]).tocsr()
    weight = {int(j): float(v * lr.coef_[0][j]) for j, v in zip(row.indices, row.data)}
    spots = {}
    for n in range(3, 6):
        for i in range(len(s) - n + 1):
            j = tfidf.vocabulary_.get(s[i:i + n])
            if j in weight:
                spots.setdefault(j, []).append((i, n))
    totals = {}
    for m in TOKEN.finditer(s):
        a, b = m.span()
        total = sum(weight[j] / len(sp) for j, sp in spots.items() for i, n in sp if i >= a and i + n <= b)
        totals[m.group()] = totals.get(m.group(), 0.0) + total
    return totals


def stacked_impacts(b, x: np.ndarray) -> np.ndarray:
    """Drop in the stacked model's log-odds when each input in turn is set to the legitimate reference."""
    ref = np.asarray(b["reference"], dtype=float)
    batch = np.repeat(x, len(ref) + 1, axis=0)
    for i in range(len(ref)):
        batch[i + 1, i] = ref[i]
    z = logit(b["stacked"].predict_proba(batch)[:, 1])
    return z[0] - z[1:]


def explanations(b, feats: dict, p_ngram: float, x: np.ndarray, normalized: str, k: int = 3) -> list:
    names = list(b["feature_names"]) + ["url_text"]
    impacts = stacked_impacts(b, x)
    ref = np.asarray(b["reference"], dtype=float)
    tokens = None
    out = []
    for i in np.argsort(-np.abs(impacts))[:k]:
        d = float(impacts[i])
        if abs(d) < MIN_IMPACT:
            continue
        toward = d > 0
        if names[i] == "url_text":
            tokens = tokens if tokens is not None else token_contributions(b["ngram"], normalized)
            ranked = sorted(((t, c) for t, c in tokens.items() if t not in GENERIC_TOKENS and (c > 0) == toward and abs(c) > 1e-3),
                            key=lambda t: abs(t[1]), reverse=True)
            examples = [t for t, _ in ranked[:2]]
            text = "The URL text looks like " + ("phishing" if toward else "legitimate") + " URLs"
            if examples:
                text += ", e.g. " + ", ".join(f'"{t}"' for t in examples)
            out.append({"feature": "url_text", "value": round(p_ngram, 3), "impact": round(d, 3),
                        "direction": "phishing" if toward else "safe", "explanation": text, "examples": examples})
        else:
            out.append({"feature": names[i], "value": feats[names[i]], "impact": round(d, 3),
                        "direction": "phishing" if toward else "safe",
                        "explanation": explain(names[i], feats[names[i]], float(ref[i]), toward)})
    return out


def score_url(url: str) -> dict:
    b = get_bundle()
    normalized = normalize_for_model(url)
    feats = extract_features(normalized)
    p_ngram = float(b["ngram"].predict_proba([normalized])[0, 1])
    x = np.array([[feats[n] for n in b["feature_names"]] + [p_ngram]], dtype=float)
    p_stacked = float(b["stacked"].predict_proba(x)[0, 1])
    p = float(b["platt"].predict_proba([[float(logit(p_stacked))]])[0, 1])
    t = b["thresholds"]
    verdict = "likely_phishing" if p >= t["likely_phishing"] else "suspicious" if p >= t["suspicious"] else "safe"
    return {"score": round(p * 100), "verdict": verdict,
            "top_features": explanations(b, feats, p_ngram, x, normalized)}


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
