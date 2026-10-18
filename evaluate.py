"""Model comparison over several domain-grouped splits. Usage: python evaluate.py [--seeds 42 0 1 2 3 4]

For every outer split (test = 20% of registered domains) the full modeling.fit_pipeline protocol runs
on the training domains only: hyperparameters are tuned, calibration is fitted and decision thresholds
are chosen on a validation split of those training domains. Test labels are never used for any choice.

Models compared:
  hgb            gradient boosting on the 31 engineered features
  ngram          TF-IDF of character 3-5-grams + logistic regression (no hand-written features)
  stacked        hgb on the engineered features + the n-gram model's out-of-fold probability
  stacked_platt  the stacked model after Platt calibration (isotonic reported for comparison)

Writes model/evaluation.json and model/evaluation.png (figure drawn from the first seed).
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

import train
from features import feature_vector
from modeling import TARGET_FPR, ece, fit_pipeline, predict_all, rates_at, summarize
from sklearn.model_selection import GroupShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ("hgb", "ngram", "stacked", "stacked_platt")
KEYS = ("roc_auc", "avg_precision", "recall", "fpr", "oracle_recall_at_5pct_fpr", "precision_at_0.01_prevalence", "ece", "brier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3, 4])
    ap.add_argument("--out", default=os.path.join(HERE, "model"))
    a = ap.parse_args()

    df = train.load_data()
    urls = df["url"].to_numpy()
    X = np.array([feature_vector(u) for u in urls], dtype=float)
    y, groups = df["label"].to_numpy(), df["group"].to_numpy()

    runs = {m: [] for m in MODELS}
    iso_ece, first = [], None
    for seed in a.seeds:
        print(f"=== outer split seed {seed} ===", flush=True)
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups))
        fit = fit_pipeline(X[tr], urls[tr], y[tr], groups[tr], seed=seed)
        probs = predict_all(fit, X[te], urls[te])
        for m in MODELS:
            runs[m].append(summarize(y[te], probs[m], fit["thresholds"][m]["likely_phishing"]))
        iso_ece.append(ece(y[te], probs["stacked_isotonic"]))
        print({m: round(runs[m][-1]["roc_auc"], 4) for m in MODELS}, flush=True)
        if first is None:
            first = {"seed": seed, "y": y[te], "groups": groups[te], "probs": probs, "fit": fit}

    summary = {m: {k: {"mean": float(np.mean([r[k] for r in rs])), "std": float(np.std([r[k] for r in rs]))} for k in KEYS}
               for m, rs in runs.items()}
    summary["stacked_isotonic"] = {"ece": {"mean": float(np.mean(iso_ece)), "std": float(np.std(iso_ece))}}

    # Per-platform breakdown on the first split, using the validation-chosen threshold.
    y0, g0, probs0, fit0 = first["y"], first["groups"], first["probs"], first["fit"]
    counts = {g: int(((g0 == g) & (y0 == 1)).sum()) for g in set(g0[y0 == 1])}
    top = sorted(counts, key=counts.get, reverse=True)[:3]
    rest = ~np.isin(g0, top)
    platform = {"seed": first["seed"], "phishing_urls_in_test": {g: counts[g] for g in top}}
    for m in MODELS:
        thr = fit0["thresholds"][m]["likely_phishing"]
        platform[m] = {"recall_by_domain": {g: float((probs0[m][(g0 == g) & (y0 == 1)] >= thr).mean()) for g in top},
                       "roc_auc_excluding_top_domains": float(roc_auc_score(y0[rest], probs0[m][rest]))}

    result = {"seeds": a.seeds, "target_fpr": TARGET_FPR, "per_split": runs, "summary": summary, "per_platform": platform}
    os.makedirs(a.out, exist_ok=True)
    json.dump(result, open(os.path.join(a.out, "evaluation.json"), "w"), indent=2)
    print(json.dumps({"summary": summary, "per_platform": platform}, indent=1), flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for m in MODELS:
        p = probs0[m]
        f, t, _ = roc_curve(y0, p)
        ax[0].plot(f, t, label=f"{m} (AUC {roc_auc_score(y0, p):.3f})")
        pr, rc, _ = precision_recall_curve(y0, p)
        ax[1].plot(rc, pr, label=f"{m} (AP {average_precision_score(y0, p):.3f})")
        idx = np.minimum((p * 10).astype(int), 9)
        pts = [(p[idx == b].mean(), y0[idx == b].mean()) for b in range(10) if (idx == b).sum() > 20]
        ax[2].plot(*zip(*pts), marker="o", label=m)
    ax[0].axvline(TARGET_FPR, ls=":", c="grey")
    ax[0].set(title=f"ROC, split seed {first['seed']} (dotted: 5% FPR)", xlabel="false-positive rate", ylabel="recall")
    ax[1].set(title="Precision-recall (test set is about a third phishing)", xlabel="recall", ylabel="precision")
    ax[2].plot([0, 1], [0, 1], "k:")
    ax[2].set(title="Calibration", xlabel="predicted probability", ylabel="observed phishing rate")
    for x in ax:
        x.legend()
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "evaluation.png"), dpi=130)


if __name__ == "__main__":
    main()
