import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app as appmod
from features import FEATURE_NAMES, feature_vector

PHISH = ["http://secure-login.verify-paypal.tk/update", "http://192.168.1.5/signin", "http://a-b-c.xyz/account/confirm?x=1"]
GOOD = ["https://google.com", "https://wikipedia.org", "https://github.com"]


@pytest.fixture(autouse=True)
def tiny_model():
    X = np.array([feature_vector(u) for u in PHISH + GOOD], dtype=float)
    y = [1] * len(PHISH) + [0] * len(GOOD)
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    appmod._bundle = {"model": rf, "feature_names": FEATURE_NAMES, "benign_median": np.median(X[len(PHISH):], axis=0)}
    yield
    appmod._bundle = None


client = TestClient(appmod.app)


def test_predict_contract():
    r = client.post("/predict", json={"url": PHISH[0]})
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"score", "verdict", "top_features"}
    assert 0 <= d["score"] <= 100
    assert d["verdict"] in {"safe", "suspicious", "likely_phishing"}
    assert len(d["top_features"]) <= 3
    for f in d["top_features"]:
        assert {"feature", "explanation", "direction"} <= set(f)


def test_phish_scores_higher_than_safe():
    assert client.post("/predict", json={"url": PHISH[1]}).json()["score"] > client.post("/predict", json={"url": GOOD[0]}).json()["score"]


def test_validation():
    assert client.post("/predict", json={"url": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_index_served():
    assert client.get("/").status_code == 200
