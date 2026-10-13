# Data

Raw datasets are not committed (see `.gitignore`). To reproduce training, put this file here:

**`mendeley.csv`** — columns `URL`, `Label` (`1` = phishing, `0` = legitimate), 149,726 rows.
Source: "Phishing URL dataset", Mendeley Data, https://data.mendeley.com/datasets/3jddhy2f6s
(per its description, legitimate URLs come from Common Crawl snapshot CC-MAIN-2024-33 and phishing URLs from PhishTank).
Downloaded 2026-10-07.

`train.py` removes exact duplicate URLs (leaving 128,228) and strips a leading `www.`.

## Datasets tried and not used

| Dataset | Why not |
|---|---|
| PhishTank + Tranco top-1M | Tranco gives bare domains only, phishing URLs have paths: the model learned "has a path = phishing" (99.5% accuracy, useless in practice). |
| PhiUSIIL (UCI) | All 134,850 legitimate URLs are `https://www.<domain>` with no path or query (100% / 100% / 0%): same shortcut, plus `www.`/https leakage. |
| ISCX-URL-2016 feature CSVs | Pre-computed numeric features, no URL strings. |
