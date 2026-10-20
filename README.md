# PhishGuard

A machine-learning phishing URL detector. Paste a URL, get a risk score from 0 to 100, a verdict, and the three inputs that moved the model most. It works from the URL string alone: no page fetching, no DNS or WHOIS lookups.

> **Status:** model, API and UI work locally and are evaluated honestly (see below). Not deployed yet; a `Dockerfile` for Hugging Face Spaces is included.

This is a machine-learning project first and a detector second. The focus is on method: leakage-free splits, nested tuning, validation-chosen thresholds, calibration, model comparison across repeated splits, an independent check on data from outside the training pool, and honest reporting of where it fails.

## How it works

```
URL string
   |
   v
normalize_for_model()        add missing scheme, drop leading "www."
   |
   +--------------------------------------+
   v                                      v
character 3-5-gram TF-IDF          31 engineered lexical features
+ logistic regression              (lengths, counts, entropy, TLD/keyword/host flags)
   |                                      |
   | out-of-fold probability              |
   +------------------+-------------------+
                      v
        gradient boosting (stacked)
                      v
           Platt calibration  ->  score 0-100
                      v
   verdict from thresholds chosen on validation data
```

- **Text model:** logistic regression over TF-IDF of character 3-5-grams of the raw URL. No hand-written features.
- **Engineered model:** 31 lexical features (`features.py`): sizes and counts, IP/port/punycode/`@` checks, subdomain count, digit ratios, token and digit-run lengths, hostname and path entropy, plus small lookup lists (suspicious TLDs, shorteners, keywords, free-hosting platforms, brand names).
- **Stacking:** gradient boosting on the engineered features plus the text model's out-of-fold probability, so it never sees a prediction made on its own training row.
- **Calibration:** Platt scaling fitted on a validation split.
- **Verdicts:** "Likely phishing" and "Suspicious" are thresholds chosen on validation data at 5% and 10% false-positive rates. With the shipped model that is a score of **31+** for likely phishing and **14-30** for suspicious; the UI shows these cutoffs under every result. The score is the estimated chance of phishing *on this training mix* (35% phishing), not on real browsing traffic.
- **Explanations** are faithful to the stacked model: each of its 32 inputs is set in turn to the typical legitimate value, and the inputs that move the log-odds most are reported. When the text model's input is among them, it is illustrated with the URL tokens that drive the text model, counting only n-grams *inside* a token, so a token has the same sign in every URL. `http`, `https` and `www` are never shown as reasons. Tests check that reported impacts match recomputing the model and that a token's sign does not depend on its neighbours.

## Data

- **[Phishing URL dataset, Mendeley Data](https://data.mendeley.com/datasets/3jddhy2f6s)**, downloaded 2026-10-07: 128,228 URLs after removing duplicates (54,759 phishing, 73,469 legitimate). Legitimate URLs come from Common Crawl (pages, almost always with paths), phishing URLs from PhishTank.
- **Legitimate homepage supplement:** 24,993 bare homepages from the [Tranco](https://tranco-list.eu) top-1M list, sampled uniformly over the whole ranking (so mostly obscure sites), plus the homepages of the 4,250 legitimate sites already in the Mendeley data (which bring realistic subdomains).

Total: **157,471 URLs** (54,759 phishing, 102,712 legitimate). Raw files are not committed; see [data/README.md](data/README.md).

Why the supplement exists: in the Mendeley data alone, 93% of bare homepages (URLs with no path or query) are phishing, because the legitimate side was collected from crawled pages. The first models learned "bare homepage = phishing" and flagged ordinary sites. This was found by hand-testing (`https://boredbutton.com` scored 69, "likely phishing"). With the supplement, 44.6% of bare homepages are phishing, so having no path no longer decides the verdict.

## Evaluation protocol

- **Splits by registered domain.** No site appears in both train and test.
- **Nested.** For every outer split, hyperparameters, calibration and decision thresholds are chosen on a validation split drawn from the training domains. Test labels are never used for any choice.
- **Repeated.** The whole pipeline runs on 6 different outer splits (seeds 42, 0, 1, 2, 3, 4). Single splits are unreliable here because a few large domains dominate whichever test set they land in.
- **Sliced.** Results are reported separately for bare homepages and for URLs with a path or query.
- **Independent check** (`independent_check.py`): PhishTank URLs downloaded on 2026-10-07 and a held-out Tranco sample, keeping only domains that appear nowhere in the training pool.

## Results

### Repeated splits (mean ± std over 6 outer splits; threshold chosen on validation for a 5% false-positive target)

| Model | ROC-AUC | Recall | False-positive rate | AUC, bare homepages | AUC, URLs with paths |
|---|---|---|---|---|---|
| Engineered features only | 0.900 ± 0.070 | 63.0% | 5.2% | 0.925 | 0.891 |
| Character n-grams only | 0.915 ± 0.025 | 64.5% | 5.3% | 0.902 | 0.930 |
| **Stacked + Platt (shipped)** | **0.945 ± 0.019** | **79.5% ± 8.5** | 6.1% ± 2.1 | **0.945** | **0.949** |

![ROC, precision-recall and calibration curves](model/evaluation.png)

The stacked model has the highest AUC on 5 of the 6 splits (on seed 4 both single models beat it) and the smallest spread. The text model generalises better than the engineered features to phishing on platforms not seen in training. On one split, a large group of phishing URLs on `adobe.com` was caught 0% of the time by the engineered model and 85% by the stacked model.

These numbers are lower than the previous version's (AUC 0.971 ± 0.012), but **the test set is harder now**: it includes legitimate homepages, which the previous test set almost entirely lacked. The independent check below is the fair before/after comparison.

### Independent check: domains never seen in training

| | Previous model (no homepage data) | Shipped model |
|---|---|---|
| Legitimate homepages flagged "likely phishing" (5,000 unseen Tranco sites) | **62.1%** | **0.7%** |
| Legitimate homepages flagged "suspicious" or worse | 95.6% | 2.3% |
| Phishing caught, "likely phishing" (21,437 PhishTank URLs on 12,113 unseen domains) | 72.6% | 67.1% |
| ... of which phishing URLs with paths | 62.9% | 70.0% |
| ... of which phishing bare homepages | 89.5% | 62.0% |
| Phishing caught, "suspicious" or worse | | 78.1% |

The previous model's higher recall on bare phishing homepages came from flagging nearly every homepage. False alarms on legitimate homepages are about 0.7% for popular and obscure sites alike (0.0% for the top 10k, 0.7% for ranks 100k-1M).

## Limitations

- **Look-alike bare domains get through.** Unseen phishing that is a bare registered domain with no subdomain (like a made-up `paypa1-secure.com`) is caught only **16%** of the time, against 84% for bare phishing homepages with subdomains (`something.pages.dev`). All the legitimate homepages added from Tranco are bare registered domains, so "clean registered domain" became a signal of safety. Telling a legitimate bare domain from a look-alike using the text alone is hard.
- **Known false alarms on ordinary sites.** Of 14 legitimate URLs I typed by hand, 11 were safe, but `store.steampowered.com` scored 31 ("likely phishing"), and `bbc.co.uk/news` (22) and `stackoverflow.com/questions` (28) were "suspicious". Long domain names and short paths push scores up.
- **Calibration is not solved.** Across splits, Platt scaling barely changes calibration error (0.063 to 0.057), and on the seed-42 split its curve is uneven in the 30-70% range (see the right panel above). The score is a ranking signal first and a probability second.
- **Precision collapses at realistic prevalence.** At the 5% false-positive operating point, precision is about 13% if only 1% of URLs are phishing.
- **The independent check is not fully independent.** PhishTank is also the upstream of the Mendeley phishing URLs (the check is domain-disjoint and later in time), and the legitimate side is Tranco, the same source as the homepage supplement. There is no independent source of legitimate URLs with paths.
- **Lexical only.** Page content, domain age and reputation are never inspected.
- **Not a security product.** Treat the score as a signal, not a verdict.

## Development notes

| Version | Result | What was wrong / what changed |
|---|---|---|
| PhishTank + Tranco only, 16 features | 99.5% accuracy | Tranco has bare domains, PhishTank URLs have paths. The model learned "has a path = phishing" and scored `wikipedia.org` 100/100. |
| PhiUSIIL (UCI) | not trained | All legitimate URLs are `https://www.<domain>` with no path: the same kind of shortcut. |
| ISCX-URL-2016 feature CSVs | not usable | Pre-computed numeric features, no URL strings. |
| Mendeley, host-only features | 90.5% accuracy | Worked, but threw away path information. |
| Mendeley, 31 engineered features, one split | 87.6% accuracy, 91% recall | Optimistic: one favourable split, threshold chosen after the fact. Repeated splits showed AUC from 0.80 to 0.96. |
| Mendeley, stacked n-grams + features, nested protocol | AUC 0.971 ± 0.012 | Strong on held-out data, but flagged 62% of real-world homepages as phishing. Found by hand-testing `boredbutton.com`. Explanations also blamed meaningless tokens like `https` and `com`. |
| **+ legitimate homepage supplement, faithful explanations** | **AUC 0.945 ± 0.019, 0.7% homepage false alarms** | Current model. |

## Run locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # on macOS/Linux: .venv/bin/python
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m uvicorn app:app --port 8000
```

Open http://localhost:8000.

To retrain, place the data files in `data/` (see [data/README.md](data/README.md)) and run `python train.py` (about 10 minutes). To reproduce the comparison, run `pip install -r requirements-eval.txt` and then `python evaluate.py` (writes `model/evaluation.json` and `model/evaluation.png`) and `python independent_check.py` (writes `model/independent.json`).

## API

`POST /predict` with `{ "url": "https://amaz0n-account-verify.pages.dev/login" }` returns:

```json
{
  "score": 100,
  "verdict": "likely_phishing",
  "top_features": [
    {"feature": "url_text", "value": 1.0, "impact": 6.763, "direction": "phishing", "explanation": "The URL text looks like phishing URLs, e.g. \"pages\", \"verify\"", "examples": ["pages", "verify"]},
    {"feature": "hostname_length", "value": 31, "impact": 1.854, "direction": "phishing", "explanation": "The domain name is unusually long"},
    {"feature": "num_dots", "value": 2, "impact": 0.846, "direction": "phishing", "explanation": "The hostname contains many dots"}
  ]
}
```

`impact` is how far the stacked model's log-odds would drop if that input took its typical legitimate value. Verdicts are `safe`, `suspicious` or `likely_phishing`. `GET /info` returns the training summary from `model/metrics.json`.

## Project layout

```
features.py           URL -> 31 engineered features; URL normalisation
modeling.py           n-gram model, stacking, tuning, calibration, thresholds (shared protocol)
train.py              builds the dataset, trains and saves the deployable model, reports held-out metrics
evaluate.py           repeated grouped-split comparison of all models, with homepage/path slices
independent_check.py  scores PhishTank and Tranco data from outside the training pool
app.py                FastAPI app: /predict, /info, serves the UI; faithful explanations
static/               single-page frontend (HTML/CSS/JS, mobile-friendly)
tests/                pytest: feature extraction, API contract, explanation correctness
model/                trained model, metrics, evaluation results and figure
data/                 dataset notes (raw data not committed)
Dockerfile            for Hugging Face Spaces / any container host
```

## Built by

Kaushik Vootukur
