"""Train the deployable model. Usage: python train.py [--out model] [--limit N] [--no-homepages]

Expects data/mendeley.csv (columns URL, Label; Label 1 = phishing) and, for the homepage supplement,
data/tranco.csv (rank,domain; no header).

Trains the stacked model (character n-gram logistic regression + gradient boosting on engineered
features), calibrated and thresholded on a validation split, and evaluates it once on a held-out set
of domains. See modeling.py for the protocol and evaluate.py for the repeated-split comparison.

Why the homepage supplement exists: in the Mendeley data only about 3% of legitimate URLs are bare
homepages (no path or query) against about 46% of phishing URLs, so "bare homepage" alone predicted
phishing 93% of the time and ordinary sites such as https://example.com scored as phishing. The
supplement adds legitimate bare homepages so that having no path says nothing about the label.
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
from modeling import fit_pipeline, predict_all, summarize, rates_at
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
TRANCO_TRAIN = 25_000        # legitimate homepages sampled uniformly over the whole Tranco ranking
TRANCO_INDEPENDENT = 5_000   # a disjoint sample, kept out of training (see independent_check.py)


def registered_domain(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return ".".join(host.split(".")[-2:])


def is_bare(url: str) -> bool:
    """True for a homepage: no path (other than '/'), no query."""
    try:
        p = urlsplit(url)
    except ValueError:
        return False
    return p.path in ("", "/") and not p.query


def tranco_shuffled(seed: int = 42) -> pd.DataFrame:
    t = pd.read_csv(os.path.join(HERE, "data", "tranco.csv"), header=None, names=["rank", "domain"])
    return t.sample(frac=1, random_state=seed).reset_index(drop=True)


def homepage_urls(domains, seed: int, https_share: float) -> list:
    """Bare homepage URLs for domains, with the scheme mix and optional trailing slash seen in real data."""
    rng = np.random.default_rng(seed)
    return [("https://" if rng.random() < https_share else "http://") + d + ("/" if rng.random() < 0.5 else "")
            for d in domains]


def load_data(limit: int = 0, seed: int = 42, homepages: bool = True) -> pd.DataFrame:
    """Load data/mendeley.csv, deduplicated and www-normalised, with a registered-domain group column.

    With homepages=True also adds legitimate bare homepages: Tranco domains sampled over the whole
    ranking, and the homepages (scheme://host) of the legitimate sites already in the data, which
    carry realistic subdomains.
    """
    df = pd.read_csv(os.path.join(HERE, "data", "mendeley.csv")).rename(columns={"URL": "url", "Label": "label"})
    if limit:
        df = df.sample(limit, random_state=seed)
    df["url"] = df["url"].astype(str).map(normalize_for_model)
    df["source"] = "mendeley"
    if homepages:
        legit = df[df["label"] == 0]
        https_share = float(legit["url"].str.startswith("https://").mean())
        origins = legit["url"].map(lambda u: f"{urlsplit(u).scheme}://{urlsplit(u).netloc}" if urlsplit(u).netloc else None).dropna().unique()
        extra = [pd.DataFrame({"url": homepage_urls([o.split("://", 1)[1] for o in origins], seed, https_share), "label": 0,
                               "source": "legit_origin"})]
        if os.path.exists(os.path.join(HERE, "data", "tranco.csv")):
            n = max(1, int(TRANCO_TRAIN * (limit / 149_726))) if limit else TRANCO_TRAIN
            domains = tranco_shuffled(seed)["domain"].head(n)
            extra.append(pd.DataFrame({"url": homepage_urls(domains, seed + 1, https_share), "label": 0, "source": "tranco"}))
        df = pd.concat([df] + extra, ignore_index=True)
        df["url"] = df["url"].map(normalize_for_model)
    df["group"] = df["url"].map(registered_domain)
    df["bare"] = df["url"].map(is_bare)
    df = df[df["group"] != ""].drop_duplicates("url")
    return df.reset_index(drop=True)


def slice_metrics(y, p, thr, mask) -> dict:
    """AUC, recall and false-positive rate on one slice of the test set (e.g. bare homepages only)."""
    out = {"n": int(mask.sum())}
    if len(set(y[mask])) == 2:
        out["roc_auc"] = float(roc_auc_score(y[mask], p[mask]))
        out["recall"], out["fpr"] = rates_at(y[mask], p[mask], thr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "model"))
    ap.add_argument("--limit", type=int, default=0, help="subsample rows (smoke tests only)")
    ap.add_argument("--no-homepages", action="store_true", help="train on the Mendeley data only")
    a = ap.parse_args()

    df = load_data(a.limit, a.seed, homepages=not a.no_homepages)
    print("class balance:", df["label"].value_counts().to_dict(), "| sources:", df["source"].value_counts().to_dict(), flush=True)
    print("P(phishing | bare homepage) = %.3f, P(phishing | has path) = %.3f"
          % (df[df.bare]["label"].mean(), df[~df.bare]["label"].mean()), flush=True)
    urls = df["url"].to_numpy()
    X = np.array([feature_vector(u) for u in urls], dtype=float)
    y, groups, bare = df["label"].to_numpy(), df["group"].to_numpy(), df["bare"].to_numpy()

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=a.seed).split(X, y, groups))
    fit = fit_pipeline(X[tr], urls[tr], y[tr], groups[tr], seed=a.seed)
    probs = predict_all(fit, X[te], urls[te])

    test = {name: summarize(y[te], p, fit["thresholds"][name]["likely_phishing"])
            for name, p in probs.items() if name in fit["thresholds"]}
    thr = fit["thresholds"]["stacked_platt"]["likely_phishing"]
    slices = {"bare_homepages": slice_metrics(y[te], probs["stacked_platt"], thr, bare[te]),
              "urls_with_paths": slice_metrics(y[te], probs["stacked_platt"], thr, ~bare[te])}
    for name, m in test.items():
        print(name, {k: round(v, 4) for k, v in m.items()}, flush=True)
    print("slices (stacked_platt):", json.dumps(slices), flush=True)

    os.makedirs(a.out, exist_ok=True)
    bundle = {
        "ngram": fit["ngram"], "stacked": fit["stacked"], "platt": fit["platt"],
        "feature_names": FEATURE_NAMES, "thresholds": fit["thresholds"]["stacked_platt"], "reference": fit["reference"],
    }
    joblib.dump(bundle, os.path.join(a.out, "model.pkl"), compress=3)
    info = {
        "model": "stacked: character n-gram logistic regression + gradient boosting, Platt-calibrated",
        "n_total": int(len(df)), "n_phishing": int(y.sum()), "n_benign": int((y == 0).sum()),
        "sources": {k: int(v) for k, v in df["source"].value_counts().items()},
        "homepage_supplement": not a.no_homepages,
        "n_fit": fit["n_fit"], "n_validation": fit["n_val"], "n_test": int(len(te)),
        "params": fit["params"], "thresholds": fit["thresholds"]["stacked_platt"],
        "test_metrics": test, "test_slices": slices, "feature_names": FEATURE_NAMES,
    }
    json.dump(info, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
