import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app as appmod
from features import FEATURE_NAMES, feature_vector, normalize_for_model

PHISH = ["http://secure-login.verify-paypal.tk/update", "http://192.168.1.5/signin", "http://a-b-c.xyz/account/confirm?x=1",
         "http://amaz0n-verify.pages.dev/login", "http://free-gift.weebly.com/claim", "http://x7k2.tk/bank/login.php"]
GOOD = ["https://google.com/search?q=cats", "https://wikipedia.org/wiki/Cat", "https://github.com/python/cpython",
        "https://docs.python.org/3/library/", "https://nytimes.com/section/world", "https://mozilla.org/en-US/firefox/"]


@pytest.fixture(autouse=True)
def tiny_bundle():
    urls = [normalize_for_model(u) for u in PHISH + GOOD]
    y = np.array([1] * len(PHISH) + [0] * len(GOOD))
    ngram = Pipeline([("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True)),
                      ("lr", LogisticRegression(C=10, solver="liblinear"))]).fit(urls, y)
    p_ng = ngram.predict_proba(urls)[:, 1]
    X = np.column_stack([[feature_vector(u) for u in urls], p_ng])
    stacked = HistGradientBoostingClassifier(max_iter=20, min_samples_leaf=2, random_state=0).fit(X, y)
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    platt = LogisticRegression(C=1e6).fit(logit(stacked.predict_proba(X)[:, 1]), y)
    appmod._bundle = {"ngram": ngram, "stacked": stacked, "platt": platt, "feature_names": FEATURE_NAMES,
                      "thresholds": {"likely_phishing": 0.7, "suspicious": 0.4}}
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
        assert {"feature", "value", "impact", "direction", "explanation"} <= set(f)
        assert f["direction"] in {"phishing", "safe"}


def test_phish_scores_higher_than_safe():
    assert client.post("/predict", json={"url": PHISH[1]}).json()["score"] > client.post("/predict", json={"url": GOOD[0]}).json()["score"]


def test_fragments_come_from_the_url():
    url = "http://amaz0n-verify.pages.dev/login"
    for f in client.post("/predict", json={"url": url}).json()["top_features"]:
        assert f["value"] in url.lower()


def test_attribution_conserves_log_odds():
    b = appmod.get_bundle()
    for url in (normalize_for_model(PHISH[0]), normalize_for_model(GOOD[0])):
        _, per_char = appmod.char_contributions(b["ngram"], url)
        intercept = float(b["ngram"].named_steps["lr"].intercept_[0])
        assert abs(per_char.sum() + intercept - float(b["ngram"].decision_function([url])[0])) < 1e-9


def test_validation():
    assert client.post("/predict", json={"url": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_index_served():
    assert client.get("/").status_code == 200
