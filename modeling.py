"""Shared modelling code for train.py and evaluate.py.

Protocol for one fit (everything below happens inside the training data it is given):
  1. split off a validation set by registered domain (no domain in both fit and validation data)
  2. tune the n-gram model's C and the gradient-boosting hyperparameters on the validation set
  3. stack: gradient boosting on the engineered features + the n-gram model's out-of-fold probability
  4. calibrate the stacked model (Platt scaling) on the validation set
  5. choose decision thresholds on the validation set (target false-positive rates)
The outer test set is never touched, so test metrics use thresholds chosen without seeing test labels.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import Pipeline

TARGET_FPR = 0.05      # "likely phishing" operating point
SUSPICIOUS_FPR = 0.10  # looser "suspicious" operating point
PREVALENCES = (0.5, 0.1, 0.01)
C_GRID = (1.0, 4.0)
HGB_GRID = [{"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": 300} for lr in (0.05, 0.1) for leaves in (31, 63)]


def make_ngram(c: float = 1.0) -> Pipeline:
    """TF-IDF over character 3-5-grams of the URL + logistic regression. No hand-written features."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=300_000)),
        ("lr", LogisticRegression(C=c, solver="liblinear", max_iter=300)),
    ])


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def threshold_at_fpr(y, p, target: float) -> float:
    fpr, _, thr = roc_curve(y, p)
    return float(thr[max(np.searchsorted(fpr, target, side="right") - 1, 0)])


def rates_at(y, p, thr: float):
    pred = p >= thr
    return float((pred & (y == 1)).sum() / (y == 1).sum()), float((pred & (y == 0)).sum() / (y == 0).sum())


def summarize(y, p, thr: float) -> dict:
    """Threshold-free metrics plus recall/FPR at a threshold chosen elsewhere (validation data)."""
    recall, fpr = rates_at(y, p, thr)
    out = {
        "roc_auc": float(roc_auc_score(y, p)), "avg_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "ece": ece(y, p),
        "threshold": thr, "recall": recall, "fpr": fpr,
        "oracle_recall_at_5pct_fpr": rates_at(y, p, threshold_at_fpr(y, p, TARGET_FPR))[0],
    }
    for prev in PREVALENCES:  # precision at this operating point if only `prev` of URLs were phishing
        denom = recall * prev + fpr * (1 - prev)
        out[f"precision_at_{prev:g}_prevalence"] = float(recall * prev / denom) if denom else 0.0
    return out


def domain_weights(groups, y, cap):
    """Sample weights that stop a few huge domains dominating training.

    Within each (registered domain, label) pair, the URLs share a total weight of at most `cap`, so a
    domain with 4,000 phishing URLs counts like one with `cap`. Weights are rescaled to mean 1.
    cap=None gives all-ones weights (no reweighting).
    """
    if cap is None:
        return np.ones(len(y))
    keys = np.char.add(np.asarray(groups, dtype=str), np.where(np.asarray(y) == 1, "|1", "|0"))
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    w = np.minimum(1.0, cap / counts[inverse])
    return w / w.mean()


def fit_pipeline(X, urls, y, groups, seed: int = 42, log=print, domain_cap=None) -> dict:
    """Fit and tune all models on (X, urls, y). See module docstring for the protocol.

    With domain_cap set, training and hyperparameter selection use domain_weights(); calibration and
    thresholds still use the unweighted validation data, so scores reflect the natural mix.
    """
    fit_i, val_i = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
    Xf, Xv, yf, yv, uf, uv, gf = X[fit_i], X[val_i], y[fit_i], y[val_i], urls[fit_i], urls[val_i], groups[fit_i]
    wf, wv = domain_weights(gf, yf, domain_cap), domain_weights(groups[val_i], yv, domain_cap)

    best_auc, ngram, c_best = -1.0, None, None
    for c in C_GRID:
        m = make_ngram(c).fit(uf, yf, lr__sample_weight=wf)
        auc = roc_auc_score(yv, m.predict_proba(uv)[:, 1], sample_weight=wv)
        log(f"  n-gram C={c}: val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, ngram, c_best = auc, m, c
    p_ng_v = ngram.predict_proba(uv)[:, 1]
    oof = cross_val_predict(make_ngram(c_best), uf, yf, groups=gf, cv=GroupKFold(n_splits=3),
                            method="predict_proba", n_jobs=3, params={"lr__sample_weight": wf})[:, 1]

    def tune(Xa, Xb):
        best = (-1.0, None, None)
        for params in HGB_GRID:
            m = HistGradientBoostingClassifier(random_state=seed, **params).fit(Xa, yf, sample_weight=wf)
            ap = average_precision_score(yv, m.predict_proba(Xb)[:, 1], sample_weight=wv)
            if ap > best[0]:
                best = (ap, params, m)
        return best

    ap_h, params_h, hgb = tune(Xf, Xv)
    ap_s, params_s, stacked = tune(np.column_stack([Xf, oof]), np.column_stack([Xv, p_ng_v]))
    log(f"  hgb val AP {ap_h:.4f} {params_h}\n  stacked val AP {ap_s:.4f} {params_s}")

    val_probs = {"hgb": hgb.predict_proba(Xv)[:, 1], "ngram": p_ng_v,
                 "stacked": stacked.predict_proba(np.column_stack([Xv, p_ng_v]))[:, 1]}
    platt = LogisticRegression(C=1e6).fit(logit(val_probs["stacked"]).reshape(-1, 1), yv)
    iso = IsotonicRegression(out_of_bounds="clip").fit(val_probs["stacked"], yv)
    val_probs["stacked_platt"] = platt.predict_proba(logit(val_probs["stacked"]).reshape(-1, 1))[:, 1]
    thresholds = {k: {"likely_phishing": threshold_at_fpr(yv, p, TARGET_FPR), "suspicious": threshold_at_fpr(yv, p, SUSPICIOUS_FPR)}
                  for k, p in val_probs.items()}
    # Reference input for explanations: the typical legitimate URL (medians of the 31 features and of the
    # out-of-fold n-gram probability over the legitimate fit rows).
    reference = np.append(np.median(Xf[yf == 0], axis=0), np.median(oof[yf == 0]))
    return {"hgb": hgb, "ngram": ngram, "stacked": stacked, "platt": platt, "isotonic": iso, "thresholds": thresholds,
            "reference": reference, "params": {"ngram_C": c_best, "hgb": params_h, "stacked": params_s, "domain_cap": domain_cap}, "n_fit": len(fit_i), "n_val": len(val_i)}


def predict_all(fit: dict, X, urls) -> dict:
    p_ng = fit["ngram"].predict_proba(urls)[:, 1]
    p_st = fit["stacked"].predict_proba(np.column_stack([X, p_ng]))[:, 1]
    return {"hgb": fit["hgb"].predict_proba(X)[:, 1], "ngram": p_ng, "stacked": p_st,
            "stacked_platt": fit["platt"].predict_proba(logit(p_st).reshape(-1, 1))[:, 1],
            "stacked_isotonic": fit["isotonic"].predict(p_st)}
