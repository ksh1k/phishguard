# PhishGuard

A machine-learning phishing URL detector. Paste a URL, get a calibrated risk score from 0 to 100 and the three URL fragments that pushed the score. It works from the URL string alone: no page fetching, no DNS or WHOIS lookups.

> **Status:** model, API and UI work locally and are evaluated honestly (see below). Not deployed yet; a `Dockerfile` for Hugging Face Spaces is included.

This is a machine-learning project first and a detector second. The focus is on method: leakage-free splits, nested tuning, validation-chosen thresholds, calibration, model comparison across repeated splits, and honest reporting of where it fails.

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

- **Text model:** logistic regression over TF-IDF of character 3-5-grams of the raw URL. No hand-written features, so it learns what it learns from the data.
- **Engineered model:** 31 lexical features (`features.py`): sizes and counts, IP/port/punycode/`@` checks, subdomain count, digit ratios, token and digit-run lengths, hostname and path entropy, plus small lookup lists (suspicious TLDs, shorteners, keywords, free-hosting platforms, brand names).
- **Stacking:** the engineered model also receives the text model's out-of-fold probability as an input, so it never sees a prediction made on its own training row.
- **Calibration:** Platt scaling fitted on a validation split, so the score behaves like a probability (under the training data's class mix, see Limitations).
- **Verdicts:** "likely phishing" and "suspicious" are decision thresholds chosen on validation data at 5% and 20% false-positive rates.
- **Explanations:** the text model is linear, so each n-gram's contribution to the log-odds is its TF-IDF value times its learned coefficient. Contributions are spread over the characters they cover and summed per URL token. This is exact for the text model (a test checks it conserves the model's log-odds), but it explains the text model only, not the engineered features.

## Data

[Phishing URL dataset, Mendeley Data](https://data.mendeley.com/datasets/3jddhy2f6s), downloaded 2026-10-07. Per its description, legitimate URLs come from Common Crawl snapshot CC-MAIN-2024-33 and phishing URLs from PhishTank. After removing exact duplicates: **128,228 URLs** (54,759 phishing, 73,469 legitimate). The raw file is not committed; see [data/README.md](data/README.md).

## Evaluation protocol

- **Splits by registered domain.** No site appears in both train and test. Random splits would leak, because this dataset has many URLs per site.
- **Nested.** For every outer split, hyperparameters (n-gram `C`, gradient-boosting settings), calibration and decision thresholds are all chosen on a validation split drawn from the *training* domains. Test labels are never used for any choice.
- **Repeated.** The whole pipeline is run on 6 different outer splits (seeds 42, 0, 1, 2, 3, 4). One split is not trustworthy here: a single platform (`pages.dev`) makes up about a third of the phishing URLs in the seed-42 test set, so results depend a lot on which domains land where.

## Results

Mean ± standard deviation over the 6 outer splits. "Recall" and "FPR" are at the threshold chosen on validation data for a 5% false-positive target.

| Model | ROC-AUC | Avg. precision | Recall | False-positive rate | Calibration error |
|---|---|---|---|---|---|
| Engineered features only | 0.921 ± 0.039 | 0.846 ± 0.100 | 66.5% ± 10.2 | 5.6% ± 1.7 | 0.081 |
| Character n-grams only | 0.950 ± 0.035 | 0.902 ± 0.060 | 77.8% ± 15.2 | 6.5% ± 1.8 | 0.109 |
| **Stacked (shipped)** | **0.971 ± 0.012** | **0.931 ± 0.041** | **80.5% ± 16.1** | 5.3% ± 2.0 | 0.055 |
| Stacked + Platt calibration (shipped) | 0.971 ± 0.012 | 0.931 ± 0.041 | 80.5% ± 16.1 | 5.3% ± 2.0 | **0.049 ± 0.036** |

![ROC, precision-recall and calibration curves](model/evaluation.png)

**The deployed model** (`train.py`, seed-42 split, trained on 81,564 URLs, validated on 23,284, tested on 23,380 unseen-domain URLs): AUC 0.974, 89.9% of phishing caught at 6.2% false alarms, calibration error 0.014. This is one of the better splits; the table above is the representative number.

What the comparison shows:

- **The text model beats my hand-built features.** On every split but one, the character n-gram model has higher AUC than the 31 engineered features, with no lists or feature design at all.
- **Stacking is the most stable.** AUC std drops to 0.012, versus 0.035-0.039 for either model alone, and it has the highest AUC on 5 of 6 splits (seed 1 is a tie).
- **The engineered features generalise worse to new platforms.** On the seed-42 test set, recall on phishing hosted on `ipfs.io` was 4% for the engineered model and 100% for the text and stacked models; on `webcindario.com` it was 42% versus 98-99%. The text model picks up patterns (such as IPFS path hashes) that no engineered feature covered.
- **The hand-written lists add little.** In an earlier ablation on the engineered model, removing all six list-based features changed F1 by about one point (0.830 to 0.818). The model mostly relies on statistical signals: hostname length, token length, and hostname and path entropy.
- **Calibration helps on average only modestly.** Platt scaling cut calibration error from 0.055 to 0.049 across splits (large variance); the big drop on the shipped split (0.062 to 0.014) is not typical. Isotonic regression was no better (0.052).

## Limitations

- **Precision collapses at realistic prevalence.** At the 5% false-positive operating point, precision is about 14% if only 1% of URLs are phishing (and about 1.5% at 0.1%). Real traffic is far less phishing-heavy than this dataset (43% phishing), so a score of 90 here does not mean a 90% chance in the wild.
- **High variance across splits.** Recall at the validation-chosen threshold ranges widely (80.5% ± 16.1), and the realised false-positive rate (5.3% ± 2.0) drifts from its 5% target. Treat single-split numbers, including the deployed model's, with caution.
- **Dataset quirks leak into explanations.** `https` pushes the score toward phishing (about 90% of this dataset's phishing URLs use it). Brand names such as `paypal` and `google` push toward safe, because legitimate URLs with those names fill the data, so embedding a brand name in a phishing URL gets a small discount. Link shorteners score high because they appear in the phishing data.
- **One dataset, one time window.** Nothing has been tested on an independent source or on a later period. The model may partly learn which hosting platforms this dataset's phishing came from.
- **Lexical only.** A phishing URL on a clean-looking domain will not be caught; page content, domain age and reputation are never inspected.
- **Not a security product.** Treat the score as a signal, not a verdict.

## Development notes

The first datasets and setups gave misleadingly high scores and were discarded:

| Attempt | Result | Problem |
|---|---|---|
| PhishTank + Tranco top-1M, 16 features | 99.5% accuracy | Tranco has bare domains, PhishTank URLs have paths. The model learned "has a path = phishing" and scored `wikipedia.org` 100/100. |
| PhiUSIIL (UCI) | not trained | All legitimate URLs are `https://www.<domain>` with no path or query: the same kind of shortcut. |
| ISCX-URL-2016 feature CSVs | not usable | Pre-computed numeric features, no URL strings. |
| Mendeley, host-only features | 90.5% accuracy | Worked, but threw away path information. |
| Mendeley, engineered features, one split | 87.6% accuracy, 91% recall | Optimistic: one favourable split, threshold chosen after the fact. Repeated splits showed AUC from 0.80 to 0.96. |
| **Mendeley, stacked, nested protocol, 6 splits** | **AUC 0.971 ± 0.012** | Current model and the numbers reported above. |

## Run locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # on macOS/Linux: .venv/bin/python
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m uvicorn app:app --port 8000
```

Open http://localhost:8000.

To retrain, place `mendeley.csv` in `data/` (see [data/README.md](data/README.md)) and run `python train.py` (a few minutes). To reproduce the comparison, run `pip install -r requirements-eval.txt` and then `python evaluate.py` (roughly 10-15 minutes for all 6 splits; it writes `model/evaluation.json` and `model/evaluation.png`).

## API

`POST /predict`

```json
{ "url": "https://amaz0n-account-verify.pages.dev/login" }
```

```json
{
  "score": 99,
  "verdict": "likely_phishing",
  "top_features": [
    { "feature": "url_fragment", "value": "login", "impact": 0.7004, "direction": "phishing",
      "explanation": "The fragment \"login\" is typical of phishing URLs in the training data" },
    { "feature": "url_fragment", "value": "pages", "impact": 0.6194, "direction": "phishing",
      "explanation": "The fragment \"pages\" is typical of phishing URLs in the training data" },
    { "feature": "url_fragment", "value": "https", "impact": 0.5116, "direction": "phishing",
      "explanation": "The fragment \"https\" is typical of phishing URLs in the training data" }
  ]
}
```

(Actual output from the shipped model. `impact` is the fragment's contribution to the text model's log-odds. Verdicts are "safe", "suspicious" or "likely_phishing".)

`GET /info` returns the training summary from `model/metrics.json`.

## Project layout

```
features.py     URL -> 31 engineered features; URL normalisation
modeling.py     n-gram model, stacking, tuning, calibration, thresholds (shared protocol)
train.py        trains and saves the deployable model, reports held-out metrics
evaluate.py     repeated grouped-split comparison of all models
app.py          FastAPI app: /predict, /info, serves the UI
static/         single-page frontend (HTML/CSS/JS, mobile-friendly)
tests/          pytest: feature extraction + API contract + explanation correctness
model/          trained model, metrics, evaluation results and figure
data/           dataset notes (raw data not committed)
Dockerfile      for Hugging Face Spaces / any container host
```

## Built by

Kaushik Vootukur
