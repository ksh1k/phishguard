"""Sanity check of the deployed model on data from outside the training pool.
Usage: python independent_check.py [--model model/model.pkl] [--out model/independent.json]
(needs a trained model, data/phishtank.csv and data/tranco.csv)

Phishing side: data/phishtank.csv (PhishTank "online-valid", downloaded 2026-10-07). Any URL whose
registered domain appears anywhere in the training pool (train.load_data(): the Mendeley data plus the
homepage supplement) is dropped, so every URL scored here is on a domain the model has never seen.
PhishTank is also the upstream of the Mendeley phishing URLs, so this is domain-disjoint and later in
time, but not a completely different source.

Legitimate side: a Tranco sample disjoint from the training sample, with domains in the training pool
dropped, scored as bare homepages and reported by popularity bucket. There is no independent source of
legitimate URLs *with paths*, so this side only measures false alarms on homepages.

Writes model/independent.json by default.
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

import train
from features import extract_features, feature_vector, normalize_for_model

HERE = os.path.dirname(os.path.abspath(__file__))
REGRESSION_URLS = [  # cases found by hand-testing the first versions
    "https://boredbutton.com", "https://www.youtube.com/watch?v=GkmRKFtNddA", "https://www.wikipedia.org/",
    "https://github.com/ksh1k/phishguard", "https://www.google.com/search?q=cats", "https://example.com/",
    "http://secure-paypal-login.verify-account.tk/update/confirm?id=48213", "https://amaz0n-account-verify.pages.dev/login",
    "http://192.168.4.17/signin/banking.php",
]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def score_batch(b, urls) -> np.ndarray:
    """Calibrated phishing probability for each URL (same pipeline as app.score_url, vectorised)."""
    urls = [normalize_for_model(u) for u in urls]
    X = np.array([feature_vector(u) for u in urls], dtype=float)
    p_ng = b["ngram"].predict_proba(urls)[:, 1]
    p_st = b["stacked"].predict_proba(np.column_stack([X, p_ng]))[:, 1]
    return b["platt"].predict_proba(logit(p_st).reshape(-1, 1))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "model", "model.pkl"))
    ap.add_argument("--out", default=os.path.join(HERE, "model", "independent.json"))
    a = ap.parse_args()
    b = joblib.load(a.model)
    hi, lo = b["thresholds"]["likely_phishing"], b["thresholds"]["suspicious"]
    pool = train.load_data()
    seen = set(pool["group"])
    out = {"thresholds": {"likely_phishing": hi, "suspicious": lo}, "training_pool_domains": len(seen)}

    # ---- phishing: PhishTank URLs on domains never seen in training ----
    pt = pd.read_csv(os.path.join(HERE, "data", "phishtank.csv"))["url"].dropna().astype(str)
    pt = pt.map(normalize_for_model).drop_duplicates()
    group = pt.map(train.registered_domain)
    keep = (group != "") & ~group.isin(seen)
    pt, group = pt[keep].reset_index(drop=True), group[keep].reset_index(drop=True)
    p = score_batch(b, pt.tolist())
    bare = pt.map(train.is_bare).to_numpy()
    top_domain = group.value_counts().index[0]
    out["phishing"] = {
        "n_urls": int(len(pt)), "n_domains": int(group.nunique()), "largest_domain": top_domain,
        "largest_domain_share": float((group == top_domain).mean()),
        "recall_likely_phishing": float((p >= hi).mean()), "recall_suspicious_or_worse": float((p >= lo).mean()),
        "recall_excluding_largest_domain": float((p[(group != top_domain).to_numpy()] >= hi).mean()),
        "recall_bare_homepages": float((p[bare] >= hi).mean()), "n_bare": int(bare.sum()),
        "recall_urls_with_paths": float((p[~bare] >= hi).mean()), "n_with_paths": int((~bare).sum()),
    }

    # ---- legitimate: Tranco homepages not used in training ----
    t = train.tranco_shuffled()
    t = t.iloc[train.TRANCO_TRAIN:].copy()
    t["group"] = t["domain"].map(lambda d: ".".join(d.lower().split(".")[-2:]))
    t = t[~t["group"].isin(seen)].head(train.TRANCO_INDEPENDENT).reset_index(drop=True)
    urls = train.homepage_urls(t["domain"], 7, 0.875)
    q = score_batch(b, urls)
    bucket = pd.cut(t["rank"], [0, 10_000, 100_000, 1_000_001], labels=["rank <= 10k", "rank 10k-100k", "rank 100k-1M"])
    out["legitimate_homepages"] = {
        "n": int(len(t)), "false_alarm_rate_likely_phishing": float((q >= hi).mean()),
        "false_alarm_rate_suspicious_or_worse": float((q >= lo).mean()),
        "false_alarm_rate_by_popularity": {str(k): float((q[(bucket == k).to_numpy()] >= hi).mean()) for k in bucket.cat.categories},
    }

    # ---- hand-found regression cases ----
    r = score_batch(b, REGRESSION_URLS)
    out["regression_cases"] = {u: int(round(s * 100)) for u, s in zip(REGRESSION_URLS, r)}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
