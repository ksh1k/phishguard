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
    z = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    platt = LogisticRegression(C=1e6).fit(z(stacked.predict_proba(X)[:, 1]), y)
    appmod._bundle = {"ngram": ngram, "stacked": stacked, "platt": platt, "feature_names": FEATURE_NAMES,
                      "thresholds": {"likely_phishing": 0.7, "suspicious": 0.4},
                      "reference": np.median(X[y == 0], axis=0)}
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


def test_explanation_never_blames_scheme_tokens():
    for url in PHISH + GOOD:
        for f in client.post("/predict", json={"url": url}).json()["top_features"]:
            assert not set(f.get("examples", [])) & appmod.GENERIC_TOKENS
            assert '"https"' not in f["explanation"] and '"http"' not in f["explanation"] and '"www"' not in f["explanation"]


def test_same_token_has_the_same_sign_in_every_url():
    # The complaint this guards against: "com" looking suspicious in one URL and safe in another.
    b = appmod.get_bundle()
    a = appmod.token_contributions(b["ngram"], normalize_for_model("https://python.org/a"))
    c = appmod.token_contributions(b["ngram"], normalize_for_model("http://x.python.org/b/c?q=1"))
    assert a["python"] != 0 and c["python"] != 0
    assert np.sign(a["python"]) == np.sign(c["python"])


def test_impacts_match_recomputing_the_model():
    b = appmod.get_bundle()
    url = normalize_for_model(PHISH[3])
    feats = appmod.extract_features(url)
    p_ng = float(b["ngram"].predict_proba([url])[0, 1])
    x = np.array([[feats[n] for n in b["feature_names"]] + [p_ng]], dtype=float)
    impacts = appmod.stacked_impacts(b, x)
    z0 = float(appmod.logit(b["stacked"].predict_proba(x)[0, 1]))
    for i in (0, 5, len(impacts) - 1):
        x2 = x.copy()
        x2[0, i] = b["reference"][i]
        assert abs(impacts[i] - (z0 - float(appmod.logit(b["stacked"].predict_proba(x2)[0, 1])))) < 1e-9


def test_validation():
    assert client.post("/predict", json={"url": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_index_served():
    assert client.get("/").status_code == 200
