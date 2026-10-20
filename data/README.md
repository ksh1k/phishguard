# Data

Raw datasets are not committed (see `.gitignore`). To reproduce training, put these files here.

## `mendeley.csv` (training data)

Columns `URL`, `Label` (`1` = phishing, `0` = legitimate), 149,726 rows.
Source: "Phishing URL dataset", Mendeley Data, https://data.mendeley.com/datasets/3jddhy2f6s
(per its description, legitimate URLs come from Common Crawl snapshot CC-MAIN-2024-33 and phishing URLs from PhishTank).
Downloaded 2026-10-07.

`train.py` removes exact duplicate URLs (leaving 128,228) and strips a leading `www.`.

**Known problem:** only about 3% of its legitimate URLs are bare homepages (no path or query), against about 46% of its phishing URLs. A model trained on it alone learns "bare homepage = phishing" (93% of bare homepages in the data are phishing) and flags ordinary sites such as `https://boredbutton.com`. It also contains only about 4,400 distinct legitimate hosts.

## `tranco.csv` (homepage supplement and independent test)

Tranco top-1M list (https://tranco-list.eu), columns `rank,domain` with no header. Downloaded 2026-10-07.

`train.py` adds legitimate bare homepages built from it: 25,000 domains sampled uniformly over the whole ranking (so most are obscure, not famous), formed as `http(s)://domain[/]` with the same http/https mix as the legitimate URLs. It also adds the homepage (`scheme://host`) of every legitimate site already in the Mendeley data, which brings realistic subdomains. With the supplement, 44.6% of bare homepages are phishing instead of 93%.

A disjoint 5,000-domain sample is held out for `independent_check.py`.

## `phishtank.csv` (independent test only)

PhishTank "online-valid" export, 69,906 rows, downloaded 2026-10-07. Never used for training. `independent_check.py` scores only URLs on registered domains that appear nowhere in the training pool.

## Datasets tried and not used

| Dataset | Why not |
|---|---|
| PhishTank + Tranco (as the only data) | Tranco gives bare domains only, phishing URLs have paths: the model learned "has a path = phishing" (99.5% accuracy, useless in practice). |
| PhiUSIIL (UCI) | All 134,850 legitimate URLs are `https://www.<domain>` with no path or query (100% / 100% / 0%): same shortcut, plus `www.`/https leakage. |
| ISCX-URL-2016 feature CSVs | Pre-computed numeric features, no URL strings. |
