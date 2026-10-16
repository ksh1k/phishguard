"""Train the deployable model. Usage: python train.py [--out model] [--limit N]

Expects data/mendeley.csv (columns URL, Label; Label 1 = phishing).

Trains the stacked model (character n-gram logistic regression + gradient boosting on engineered
features), calibrated and thresholded on a validation split, and evaluates it once on a held-out set
of domains. See modeling.py for the protocol and evaluate.py for the repeated-split comparison.
"""
import argparse
import json
import os
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from features import FEATURE_NAMES, feature_vector, normalize_for_model
from modeling import fit_pipeline, predict_all, summarize

HERE = os.path.dirname(os.path.abspath(__file__))


def registered_domain(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return ".".join(host.split(".")[-2:])


def load_data(limit: int = 0, seed: int = 42) -> pd.DataFrame:
    """Load data/mendeley.csv, deduplicated and www-normalised, with a registered-domain group column."""
    df = pd.read_csv(os.path.join(HERE, "data", "mendeley.csv")).rename(columns={"URL": "url", "Label": "label"})
    if limit:
        df = df.sample(limit, random_state=seed)
    df["url"] = df["url"].astype(str).map(normalize_for_model)
    df["group"] = df["url"].map(registered_domain)
    df = df[df["group"] != ""].drop_duplicates("url")
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "model"))
    ap.add_argument("--limit", type=int, default=0, help="subsample rows (smoke tests only)")
    a = ap.parse_args()

    df = load_data(a.limit, a.seed)
    print("class balance:", df["label"].value_counts().to_dict(), flush=True)
    urls = df["url"].to_numpy()
    X = np.array([feature_vector(u) for u in urls], dtype=float)
    y, groups = df["label"].to_numpy(), df["group"].to_numpy()

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=a.seed).split(X, y, groups))
    fit = fit_pipeline(X[tr], urls[tr], y[tr], groups[tr], seed=a.seed)
    probs = predict_all(fit, X[te], urls[te])

    test = {name: summarize(y[te], p, fit["thresholds"][name]["likely_phishing"])
            for name, p in probs.items() if name in fit["thresholds"]}
    for name, m in test.items():
        print(name, {k: round(v, 4) for k, v in m.items()}, flush=True)

    os.makedirs(a.out, exist_ok=True)
    bundle = {
        "ngram": fit["ngram"], "stacked": fit["stacked"], "platt": fit["platt"],
        "feature_names": FEATURE_NAMES, "thresholds": fit["thresholds"]["stacked_platt"],
    }
    joblib.dump(bundle, os.path.join(a.out, "model.pkl"), compress=3)
    info = {
        "model": "stacked: character n-gram logistic regression + gradient boosting, Platt-calibrated",
        "n_total": int(len(df)), "n_phishing": int(y.sum()), "n_benign": int((y == 0).sum()),
        "n_fit": fit["n_fit"], "n_validation": fit["n_val"], "n_test": int(len(te)),
        "params": fit["params"], "thresholds": fit["thresholds"]["stacked_platt"],
        "test_metrics": test, "feature_names": FEATURE_NAMES,
    }
    json.dump(info, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
