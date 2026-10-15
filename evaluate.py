"""Model comparison and evaluation. Usage: python evaluate.py [--seeds 0 1 2 3 4]

Compares three models on host-grouped holdouts (no registered domain appears in both train and test):
  hgb      gradient boosting on the 31 engineered features (the shipped model's family)
  ngram    TF-IDF over character 3-5-grams of the URL + logistic regression (no hand-written features)
  stacked  hgb on the engineered features plus the n-gram model's out-of-fold probability

Reports threshold-free metrics (ROC-AUC, average precision), recall at a fixed 5% false-positive rate,
prevalence-adjusted precision, calibration error, a per-platform breakdown, and the spread over several
different grouped splits. Writes model/evaluation.json and model/evaluation.png.
"""
import argparse
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline

import train
from features import feature_vector

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_FPR = 0.05
PREVALENCES = (0.5, 0.1, 0.01)


def make_ngram(c: float = 4.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=400_000)),
        ("lr", LogisticRegression(C=c, solver="liblinear", max_iter=300)),
    ])


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def summarize(y, p) -> dict:
    fpr, tpr, thr = roc_curve(y, p)
    i = max(np.searchsorted(fpr, TARGET_FPR, side="right") - 1, 0)
    out = {
        "roc_auc": float(roc_auc_score(y, p)),
        "avg_precision": float(average_precision_score(y, p)),
        f"recall_at_{int(TARGET_FPR * 100)}pct_fpr": float(tpr[i]),
        "threshold_at_fpr": float(thr[i]),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
    }
    for prev in PREVALENCES:  # precision at the 5%-FPR operating point if only `prev` of URLs were phishing
        out[f"precision_at_{prev:g}_prevalence"] = float(tpr[i] * prev / (tpr[i] * prev + fpr[i] * (1 - prev)))
    return out


def fit_models(Xtr, ytr, urls_tr, groups_tr, hgb_params, c):
    ng = make_ngram(c).fit(urls_tr, ytr)
    oof = cross_val_predict(make_ngram(c), urls_tr, ytr, groups=groups_tr, cv=GroupKFold(n_splits=3),
                            method="predict_proba", n_jobs=3)[:, 1]
    hgb = HistGradientBoostingClassifier(random_state=42, **hgb_params).fit(Xtr, ytr)
    stacked = HistGradientBoostingClassifier(random_state=42, **hgb_params).fit(np.column_stack([Xtr, oof]), ytr)
    return hgb, ng, stacked


def predict_all(models, Xte, urls_te):
    hgb, ng, stacked = models
    p_ng = ng.predict_proba(urls_te)[:, 1]
    return {"hgb": hgb.predict_proba(Xte)[:, 1], "ngram": p_ng,
            "stacked": stacked.predict_proba(np.column_stack([Xte, p_ng]))[:, 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()

    df = train.load_data()
    urls = df["url"].to_numpy()
    X = np.array([feature_vector(u) for u in urls], dtype=float)
    y, groups = df["label"].to_numpy(), df["group"].to_numpy()
    hgb_params = json.load(open(os.path.join(HERE, "model", "metrics.json")))["best_params"]["hist_gradient_boosting"]

    def split(seed):
        return next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups))

    # ---- Primary split (same one train.py uses): tune the n-gram C on training data only ----
    tr, te = split(42)
    grid = GridSearchCV(make_ngram(), {"lr__C": [1.0, 4.0, 16.0]}, scoring="roc_auc",
                        cv=GroupKFold(n_splits=3), n_jobs=3).fit(urls[tr], y[tr], groups=groups[tr])
    c = float(grid.best_params_["lr__C"])
    print("n-gram C =", c, "cv auc = %.4f" % grid.best_score_, flush=True)
    models = fit_models(X[tr], y[tr], urls[tr], groups[tr], hgb_params, c)
    probs = predict_all(models, X[te], urls[te])
    primary = {k: summarize(y[te], p) for k, p in probs.items()}
    for k, v in primary.items():
        print(k, {m: round(x, 4) for m, x in v.items()}, flush=True)

    # ---- Per-platform breakdown on the primary holdout ----
    g_te, y_te = groups[te], y[te]
    phish_counts = {g: int(((g_te == g) & (y_te == 1)).sum()) for g in set(g_te[y_te == 1])}
    top = sorted(phish_counts, key=phish_counts.get, reverse=True)[:3]
    platform = {}
    for name, p in probs.items():
        thr = primary[name]["threshold_at_fpr"]
        rest = ~np.isin(g_te, top)
        platform[name] = {
            "recall_by_top_phishing_domain": {g: float((p[(g_te == g) & (y_te == 1)] >= thr).mean()) for g in top},
            "roc_auc_excluding_top_domains": float(roc_auc_score(y_te[rest], p[rest])),
        }
    platform["top_domains_phishing_urls_in_test"] = {g: phish_counts[g] for g in top}
    print("platform:", json.dumps(platform, indent=1), flush=True)

    # ---- Variance over several different grouped splits ----
    repeats = {k: [] for k in probs}
    for seed in a.seeds:
        s_tr, s_te = split(seed)
        m = fit_models(X[s_tr], y[s_tr], urls[s_tr], groups[s_tr], hgb_params, c)
        for k, p in predict_all(m, X[s_te], urls[s_te]).items():
            repeats[k].append(summarize(y[s_te], p))
        print("seed", seed, {k: round(v[-1]["roc_auc"], 4) for k, v in repeats.items()}, flush=True)
    spread = {}
    for k, runs in repeats.items():
        spread[k] = {m: {"mean": float(np.mean([r[m] for r in runs])), "std": float(np.std([r[m] for r in runs]))}
                     for m in ("roc_auc", "avg_precision", f"recall_at_{int(TARGET_FPR * 100)}pct_fpr", "precision_at_0.01_prevalence")}
    print("spread:", json.dumps(spread, indent=1), flush=True)

    json.dump({"ngram_C": c, "primary_split_seed": 42, "target_fpr": TARGET_FPR, "primary": primary,
               "per_platform": platform, "repeated_splits": {"seeds": a.seeds, "summary": spread}},
              open(os.path.join(HERE, "model", "evaluation.json"), "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, p in probs.items():
        f, t, _ = roc_curve(y_te, p)
        ax[0].plot(f, t, label=f"{k} (AUC {primary[k]['roc_auc']:.3f})")
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax[1].plot(rc, pr, label=f"{k} (AP {primary[k]['avg_precision']:.3f})")
        edges = np.linspace(0, 1, 11)
        idx = np.minimum((p * 10).astype(int), 9)
        pts = [(p[idx == b].mean(), y_te[idx == b].mean()) for b in range(10) if (idx == b).sum() > 20]
        ax[2].plot(*zip(*pts), marker="o", label=k)
    ax[0].axvline(TARGET_FPR, ls=":", c="grey")
    ax[0].set(title="ROC (dotted: 5% FPR)", xlabel="false-positive rate", ylabel="recall")
    ax[1].set(title="Precision-recall (test set is 33% phishing)", xlabel="recall", ylabel="precision")
    ax[2].plot([0, 1], [0, 1], "k:")
    ax[2].set(title="Calibration", xlabel="predicted probability", ylabel="observed phishing rate")
    for x in ax:
        x.legend()
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "model", "evaluation.png"), dpi=130)


if __name__ == "__main__":
    main()
