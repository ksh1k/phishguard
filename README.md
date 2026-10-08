# PhishGuard

A machine-learning phishing URL detector: paste a URL, get a risk score (0-100) and a plain-English explanation of what drove the verdict. It works from the URL text alone, with no page fetching or network lookups.

**Status: work in progress.** The code is being added to this repository in stages, so some parts described here are not pushed yet.

## Planned

- Lexical URL feature extraction (pure, unit-tested functions)
- Gradient-boosted classifier trained on labelled phishing and benign URLs
- FastAPI service with a `POST /predict` endpoint
- One-page web UI showing the score and the top contributing features
- Docker deployment

A full write-up with results and limitations will replace this page when the project is complete.

Built by Kaushik Vootukur.
