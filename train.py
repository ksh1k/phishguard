"""Train the phishing classifier. Usage: python train.py

Expects data/mendeley.csv (columns URL, Label; Label 1 = phishing).
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlsplit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit

from features import FEATURE_NAMES, feature_vector, normalize_for_model

HERE = os.path.dirname(os.path.abspath(__file__))


def registered_domain(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return ".".join(host.split(".")[-2:])


def load_data() -> pd.DataFrame:
    """Load data/mendeley.csv (columns URL, Label; 1 = phishing), deduplicated and www-normalised."""
    df = pd.read_csv(os.path.join(HERE, "data", "mendeley.csv")).rename(columns={"URL": "url", "Label": "label"})
    df["url"] = df["url"].astype(str).map(normalize_for_model)
    df["group"] = df["url"].map(registered_domain)
    df = df[df["group"] != ""].drop_duplicates("url")
    return df.reset_index(drop=True)


def metrics(y, pred) -> dict:
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = load_data()
    print("class balance:\n", df["label"].value_counts().to_string())
    X = np.array([feature_vector(u) for u in df["url"]], dtype=float)
    y = df["label"].to_numpy()
    groups = df["group"].to_numpy()
    # Split by registered domain so no site appears in both train and test.
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=a.seed).split(X, y, groups))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    cv = GroupKFold(n_splits=3)
    F2 = make_scorer(fbeta_score, beta=2)  # recall-weighted: missing a phish is worse than a false alarm
    candidates = {
        "random_forest": (
            RandomForestClassifier(random_state=a.seed, n_jobs=-1, class_weight="balanced"),
            {"n_estimators": [200, 400], "max_depth": [20, None], "min_samples_leaf": [1, 3]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=a.seed),
            {"max_iter": [200, 400], "learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63]},
        ),
    }
    # Model selection uses grouped CV on the training split only, so the holdout stays untouched.
    fitted, results, cv_f2 = {}, {}, {}
    for name, (est, params) in candidates.items():
        g = GridSearchCV(est, params, scoring=F2, cv=cv, n_jobs=-1).fit(Xtr, ytr, groups=groups[tr])
        fitted[name] = g
        cv_f2[name] = float(g.best_score_)
        results[name] = metrics(yte, g.best_estimator_.predict(Xte))
        print(name, "cv_f2=%.4f" % cv_f2[name], g.best_params_)
        print("  holdout", {k: round(v, 4) for k, v in results[name].items()})
    best = max(cv_f2, key=cv_f2.get)
    print("selected:", best)
    chosen = fitted[best].best_estimator_

    benign_median = np.median(Xtr[ytr == 0], axis=0)
    bundle = {"model": chosen, "feature_names": FEATURE_NAMES, "benign_median": benign_median}
    os.makedirs(os.path.join(HERE, "model"), exist_ok=True)
    joblib.dump(bundle, os.path.join(HERE, "model", "model.pkl"), compress=3)
    info = {
        "model": type(chosen).__name__, "selected": best, "cv_f2": cv_f2,
        "best_params": {k: v.best_params_ for k, v in fitted.items()},
        "n_total": int(len(df)), "n_phishing": int(y.sum()), "n_benign": int((y == 0).sum()),
        "n_test": int(len(yte)), "holdout_metrics": results, "feature_names": FEATURE_NAMES,
    }
    json.dump(info, open(os.path.join(HERE, "model", "metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
