# PhishGuard

A machine-learning phishing URL detector. Paste a URL, get a risk score from 0 to 100 and a plain-English explanation of the three features that drove the verdict. The model works from the URL string alone: no page fetching, no DNS or WHOIS lookups.

> **Status: v1.** A working model, API and UI, evaluated on a single train/test split. Not deployed yet (a `Dockerfile` for Hugging Face Spaces is included). Repeated-split evaluation and model comparisons are the next steps, and this README will be rewritten with them.

## How it works

```
URL string
   |
   v
normalize_for_model()   add missing scheme, drop leading "www."
   |
   v
extract_features()      31 lexical features (features.py)
   |
   v
gradient boosting       predict_proba -> score 0-100
   |
   v
occlusion explanation   set each feature to the benign median, see how far the score drops;
                        report the top 3 as plain English
```

| Verdict | Score |
|---|---|
| Safe | 0-39 |
| Suspicious | 40-69 |
| Likely phishing | 70-100 |

## Features

All computed from the URL text (`features.py`, pure functions, unit-tested): sizes and counts (lengths, dots, hyphens, digits, path depth, query parameters), host signals (IP address, `@`, subdomain count, HTTPS, port, punycode, digit ratio), randomness (Shannon entropy of the hostname and path), and small lookup lists (suspicious TLDs, shorteners, keywords, free-hosting platforms, brand names).

## Data

[Phishing URL dataset, Mendeley Data](https://data.mendeley.com/datasets/3jddhy2f6s), downloaded 2026-10-07. Per its description, legitimate URLs come from Common Crawl snapshot CC-MAIN-2024-33 and phishing URLs from PhishTank. After removing exact duplicates: **128,228 URLs** (54,759 phishing, 73,469 legitimate). The raw file is not committed; see [data/README.md](data/README.md).

## Results (v1, single split)

One holdout of 23,380 URLs, split by registered domain so no site appears in both train and test. Model selection used 3-fold grouped cross-validation on the training split, scored by F2 (which weights recall) because missing a phishing URL is worse than a false alarm.

| Model | Accuracy | Precision | Recall (phishing) | F1 |
|---|---|---|---|---|
| **Gradient boosting (shipped)** | 87.6% | 76.1% | **91.1%** | 83.0% |
| Random forest | 86.6% | 79.0% | 81.0% | 80.0% |

Full numbers and hyperparameters are in `model/metrics.json`.

## Caveats at this stage

- **A single split is not enough.** Which domains land in the test set matters a lot in this dataset (one platform, `pages.dev`, is about a third of the phishing URLs in this test set). These numbers may be optimistic; repeated splits are the next step.
- **Selection criterion changed after the fact.** I first selected on F1, where the two models tied, then switched to F2 after seeing the holdout results.
- **Precision is 76%:** about 1 in 4 flagged URLs is a false alarm.
- **Bare homepages look suspicious.** `https://www.wikipedia.org/` scores 58, because bare-domain URLs are common among this dataset's phishing URLs.
- **One dataset, lexical features only.** Page content, domain age and reputation are never inspected.
- **Not a security product.** Treat the score as a signal, not a verdict.

## Development notes

| Attempt | Result | Problem |
|---|---|---|
| PhishTank + Tranco top-1M, 16 features | 99.5% accuracy | Tranco has bare domains, PhishTank URLs have paths. The model learned "has a path = phishing" and scored `wikipedia.org` 100/100. |
| PhiUSIIL (UCI) | not trained | All legitimate URLs are `https://www.<domain>` with no path or query: the same kind of shortcut. |
| ISCX-URL-2016 feature CSVs | not usable | Pre-computed numeric features, no URL strings. |
| Mendeley, host-only features | 90.5% accuracy | Worked, but threw away path information. |
| **Mendeley, full URLs, 31 features** | 87.6% accuracy, 91.1% recall | Current model. |

## Run locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # on macOS/Linux: .venv/bin/python
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m uvicorn app:app --port 8000
```

Open http://localhost:8000. To retrain, place `mendeley.csv` in `data/` (see [data/README.md](data/README.md)) and run `python train.py`.

## API

`POST /predict` with `{ "url": "..." }` returns `{ "score": 0-100, "verdict": "safe" | "suspicious" | "likely_phishing", "top_features": [...] }`. `GET /info` returns the training summary from `model/metrics.json`.

## Project layout

```
features.py     URL -> 31 features; URL normalisation
train.py        grouped split, model search, evaluation, saves model/
app.py          FastAPI app: /predict, /info, serves the UI
static/         single-page frontend (HTML/CSS/JS, mobile-friendly)
tests/          pytest: feature extraction + API contract
model/          trained model + metrics
data/           dataset notes (raw data not committed)
Dockerfile      for Hugging Face Spaces / any container host
```

## Built by

Kaushik Vootukur
