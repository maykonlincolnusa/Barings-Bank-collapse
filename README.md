# Barings Bank Collapse Fraud & OpRisk Platform

This repository contains a modular prototype that recreates Barings-style rogue trading and operational control failures using only public references and synthetic data. It ingests curated public facts, reconstructs a timeline, simulates hidden-loss trading behavior, engineers domain features, trains an ensemble of fraud/anomaly models, exposes predictions through FastAPI, and presents the results in Streamlit.

## What the prototype covers

- Public-source registry with provenance for Barings reports, parliamentary material, central bank commentary, and press coverage.
- Timeline extraction and event normalization focused on Nick Leeson, account `88888`, control failures, and the Kobe earthquake shock.
- Synthetic trading, positions, cashflow, reconciliation, and audit logs for `healthy_desk`, `mild_anomaly`, `rogue_trader`, and `collapse`.
- Fraud feature engineering aimed at Barings-like indicators such as hidden-loss usage, PnL versus cash mismatch, funding spikes, reconciliation breaks, and front/back office overlap.
- Ensemble modeling with supervised, unsupervised, and sequence anomaly components.
- Risk scoring, explainable factors, narrative alerts, audit logs, JWT auth, role checks, rate limiting, and optional encrypted artifact storage.
- REST API and Streamlit dashboard for analysts and auditors.

## Repository layout

```text
app/
  api/
  dashboard/
  data/
    raw/
    clean/
    synthetic/
    schemas/
  ingestion/
  timeline/
  features/
  models/
  simulation/
  security/
  utils/
  tests/
  docs/
  configs/
  reports/
sources.csv
model_card.md
data_card.md
```

## Quick start

1. Create and activate a Python 3.12 environment.
2. Install the package:

```bash
pip install -e .
```

3. Copy `.env.example` to `.env` and adjust the secrets if needed.
4. Train a demo model and create sample alerts:

```bash
python -m app.api.bootstrap
```

5. Start the API:

```bash
uvicorn app.api.main:app --reload
```

6. In another shell, start the dashboard:

```bash
streamlit run app/dashboard/app.py
```

## Demo workflow

1. Obtain a token:

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"admin\",\"password\":\"admin123\"}"
```

2. Simulate a collapse scenario:

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d "{\"scenario\":\"collapse\",\"days\":90,\"seed\":7}"
```

3. Train the ensemble:

```bash
curl -X POST http://localhost:8000/train \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d "{\"scenarios\":[\"healthy_desk\",\"mild_anomaly\",\"rogue_trader\",\"collapse\"],\"days_per_scenario\":120}"
```

4. Score a scenario:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d "{\"scenario\":\"rogue_trader\",\"days\":60,\"seed\":13}"
```

## Public-source design

The platform ships with a curated `sources.csv` registry and a small normalized fact corpus in [app/data/raw/public/barings_public_extracts.json](/workspace/app/data/raw/public/barings_public_extracts.json). The ingestion layer is written so public PDFs/HTML can be added later without changing the downstream pipeline. The included records preserve provenance, dates, URL, reliability, and short paraphrased notes rather than copyrighted bulk text.

## Architecture

Architecture, data flow, threat model, and timeline diagrams live in [app/docs/architecture.md](/workspace/app/docs/architecture.md), [app/docs/threat_model.md](/workspace/app/docs/threat_model.md), and [app/docs/timeline.md](/workspace/app/docs/timeline.md).

## Testing

```bash
pytest
```

## Notes

- `SHAP`, `MLflow`, and TensorFlow-based sequence autoencoders are optional. The prototype falls back to deterministic local explainability, file-based experiment logs, and PCA reconstruction if those libraries are not installed.
- Raw public artifacts are treated as immutable. Derived tables, feature tables, models, and alerts are versioned under `app/data/clean`, `app/data/synthetic`, and `app/reports`.

