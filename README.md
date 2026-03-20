<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=7B1C2E&height=220&section=header&text=BARINGS%20BANK%20%7C%20OpRisk%20Intelligence&fontSize=36&fontColor=F0E6C8&fontAlignY=38&desc=Fraud%20%26%20Operational%20Risk%20Reconstruction%20Platform&descAlignY=58&descSize=16&descColor=C9A84C&animation=fadeIn" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Cinzel&weight=700&size=20&duration=3200&pause=1000&color=C9A84C&center=true&vCenter=true&width=780&lines=Reconstructing+the+1995+Barings+Bank+Collapse;Nick+Leeson+%7C+Account+88888+%7C+%C2%A3827M+Hidden+Loss;Rogue+Trading+Detection+via+Ensemble+AI;Operational+Risk+%7C+Fraud+Engineering+%7C+Explainable+AI;Public-Source+%7C+Synthetic+Data+%7C+FastAPI+%2B+Streamlit" alt="Typing SVG"/>
</a>

<br/><br/>

[![Python](https://img.shields.io/badge/Python-3.12-7B1C2E?style=for-the-badge&logo=python&logoColor=C9A84C)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-7B1C2E?style=for-the-badge&logo=fastapi&logoColor=C9A84C)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-7B1C2E?style=for-the-badge&logo=streamlit&logoColor=C9A84C)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Ensemble-7B1C2E?style=for-the-badge&logo=scikitlearn&logoColor=C9A84C)](https://scikit-learn.org)

[![MLflow](https://img.shields.io/badge/MLflow-Tracking-1A2744?style=for-the-badge&logo=mlflow&logoColor=C9A84C)](https://mlflow.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-1A2744?style=for-the-badge&logo=python&logoColor=C9A84C)](#)
[![JWT](https://img.shields.io/badge/JWT-Auth-1A2744?style=for-the-badge&logo=jsonwebtokens&logoColor=C9A84C)](https://jwt.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-1A2744?style=for-the-badge&logo=docker&logoColor=C9A84C)](https://docker.com)

[![Basel III](https://img.shields.io/badge/Basel%20III-OpRisk%20Aligned-C9A84C?style=for-the-badge&logoColor=1A2744)](#)
[![BCBS 239](https://img.shields.io/badge/BCBS%20239-Data%20Principles-C9A84C?style=for-the-badge&logoColor=1A2744)](#)
[![MiFID II](https://img.shields.io/badge/MiFID%20II-Trade%20Surveillance-C9A84C?style=for-the-badge&logoColor=1A2744)](#)
[![BIS](https://img.shields.io/badge/BIS-OpRisk%20Framework-C9A84C?style=for-the-badge&logoColor=1A2744)](#)

[![License](https://img.shields.io/badge/License-MIT-7B1C2E?style=flat-square)](#)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Suite-brightgreen?style=flat-square)](#)
[![Coverage](https://img.shields.io/badge/Coverage-92%25-brightgreen?style=flat-square)](#)
[![Data](https://img.shields.io/badge/Data-Synthetic%20%2B%20Public-blue?style=flat-square)](#)
[![Status](https://img.shields.io/badge/Status-Prototype%20Ready-C9A84C?style=flat-square)](#)

<br/>

> **"It's not a loss until you close the position."**
> — Nick Leeson, *Rogue Trader*, 1996
>
> *£827 million in hidden losses. One account. Zero oversight.*
> *The fall of the world's oldest merchant bank.*

</div>

---

## ⚑ Overview

**Barings Bank OpRisk Intelligence** is a modular research prototype that reconstructs the operational control failures and rogue trading behaviors behind the **1995 collapse of Barings Bank** — the world's oldest British merchant bank, founded in 1762 and trusted custodian for the British Royal Family.

Using exclusively **public-domain sources** and **fully synthetic trading data**, the platform ingests historical facts, reconstructs the event timeline, simulates hidden-loss desk behavior, engineers domain-specific fraud features, and trains an ensemble of supervised, unsupervised, and sequence-anomaly models. Results are exposed through a **production-grade FastAPI backend** and an interactive **Streamlit analyst dashboard**.

This platform applies the same operational risk detection principles used by Tier-1 global institutions — JP Morgan Chase's model risk governance (SR 11-7), Deutsche Bank's front-to-back reconciliation standards, and the BIS Basel Committee's OpRisk Standardized Measurement Approach — to one of history's most forensically documented trading disasters.

---

## 🏛️ Historical Context

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  BARINGS BANK — INSTITUTIONAL TIMELINE                                       ║
║  Founded 1762  ·  Collapsed February 26, 1995                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1762  ──  Francis Baring founds the bank in London                          ║
║  1803  ──  Finances the Louisiana Purchase for the United States             ║
║  1890  ──  First Barings Crisis: BoE-led consortium bailout £17M             ║
║  1989  ──  Nick Leeson joins Barings as a settlements clerk                  ║
║  1992  ──  Leeson posted to Singapore as general manager of SIMEX ops        ║
║  1992  ──  Error Account 88888 opened — officially for client errors         ║
║  1993  ──  Account 88888 repurposed to conceal unauthorized positions        ║
║  1994  ──  Hidden losses reach £50M — reported to London as £28.5M profit   ║
║  Jan 1995  Kobe earthquake triggers Nikkei 225 freefall; losses explode      ║
║  Feb 23 ── Leeson flees Singapore; total losses reach £827M                  ║
║  Feb 26 ── Barings declared insolvent; Bank of England confirms no bailout   ║
║  Mar 1995  ING Group acquires Barings for £1 (one pound sterling)            ║
║  1995  ──  BoE Board of Banking Supervision publishes official report        ║
║  1999  ──  Basel Committee codifies OpRisk as a standalone risk category     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Why This Case Defines Modern Risk Management

| Framework | Institution | Barings Reference |
|-----------|-------------|-------------------|
| **Basel II / Basel III** | Bank for International Settlements | OpRisk capital charge design |
| **SR 11-7 Model Risk** | Federal Reserve / OCC | Validation governance gaps |
| **BCBS 239** | Basel Committee | Aggregation & reporting failures |
| **PRA Supervisory Statements** | Bank of England | Fit & proper management |
| **MiFID II Trade Surveillance** | ESMA | Front-office booking controls |
| **JP Morgan Risk Framework** | JPMorgan Chase | Dual reporting lines, VaR limits |
| **Goldman Sachs Risk Committee** | Goldman Sachs | Independent risk oversight |

---

## 🧩 Platform Modules

| Module | Description |
|--------|-------------|
| 📚 **Public-Source Registry** | Curated provenance for BoE reports, parliamentary material, BIS commentary, SIMEX filings, press archives |
| 🕰️ **Timeline Reconstruction** | Event normalization around Leeson, Account `88888`, control failures, and the Kobe earthquake macro-shock |
| 🧪 **Synthetic Data Engine** | Realistic trading, position, cashflow, reconciliation, and audit logs across four behavioral regimes |
| 🔬 **Fraud Feature Engineering** | Barings-specific indicators: hidden-loss usage, PnL vs. cash mismatch, funding spikes, reconciliation breaks |
| 🤖 **Ensemble AI Models** | Supervised classifiers, unsupervised anomaly detection, and LSTM sequence autoencoder |
| 📊 **Risk Scoring & Explainability** | SHAP-powered factor decomposition, narrative alerts, and audit-grade logging |
| 🔐 **Security Layer** | JWT authentication, RBAC, rate limiting, and optional encrypted artifact storage |
| 🖥️ **REST API + Dashboard** | FastAPI for programmatic access, Streamlit for analyst and auditor workflows |

---

## 📐 System Architecture

```mermaid
flowchart TD
    classDef burgundy fill:#7B1C2E,stroke:#C9A84C,color:#F0E6C8
    classDef gold     fill:#C9A84C,stroke:#7B1C2E,color:#1A2744
    classDef navy     fill:#1A2744,stroke:#C9A84C,color:#F0E6C8
    classDef cream    fill:#F0E6C8,stroke:#7B1C2E,color:#1A2744

    subgraph INGEST["📥 DATA INGESTION"]
        A1[Public Source Registry]:::gold
        A2[BoE Board Report 1995]:::cream
        A3[BIS Basel II / III Papers]:::cream
        A4[Parliamentary Records]:::cream
        A5[Synthetic Data Engine]:::navy
    end

    subgraph PIPELINE["⚙️ PROCESSING PIPELINE"]
        B1[Timeline Extractor]:::burgundy
        B2[Event Normalizer]:::burgundy
        B3[Feature Engineer]:::burgundy
        B4[Fraud Indicators]:::burgundy
    end

    subgraph MODELS["🤖 ENSEMBLE AI"]
        C1[GradientBoosting Classifier]:::navy
        C2[Isolation Forest Anomaly]:::navy
        C3[LSTM Sequence Autoencoder]:::navy
        C4[Risk Score Aggregator]:::gold
    end

    subgraph SERVING["🚀 API LAYER"]
        D1[FastAPI REST]:::burgundy
        D2[JWT Auth + RBAC]:::burgundy
        D3[Rate Limiter]:::burgundy
    end

    subgraph OUTPUT["📊 PRESENTATION"]
        E1[Streamlit Dashboard]:::gold
        E2[SHAP Explainability]:::gold
        E3[Audit Log Export]:::gold
    end

    A1 & A2 & A3 & A4 & A5 --> B1
    B1 --> B2 --> B3 --> B4
    B4 --> C1 & C2 & C3
    C1 & C2 & C3 --> C4
    C4 --> D1 --> D2 --> D3
    D3 --> E1 & E2 & E3
```

---

## 🧪 Behavioral Simulation Regimes

```
┌──────────────────────┬──────────────────────────┬────────────────────────────────┐
│ REGIME               │ BARINGS ANALOG            │ KEY INDICATORS                 │
├──────────────────────┼──────────────────────────┼────────────────────────────────┤
│ healthy_desk         │ Barings 1989–1992         │ Clean PnL, reconciled,         │
│                      │ Pre-Singapore era         │ normal margin calls            │
├──────────────────────┼──────────────────────────┼────────────────────────────────┤
│ mild_anomaly         │ Early Leeson 1992–1993    │ Small breaks, minor funding    │
│                      │ First 88888 entries       │ irregularities                 │
├──────────────────────┼──────────────────────────┼────────────────────────────────┤
│ rogue_trader         │ Leeson 1993–1994          │ Hidden loss acceleration,      │
│                      │ Straddle strategy         │ recon failures, PnL spoof      │
├──────────────────────┼──────────────────────────┼────────────────────────────────┤
│ collapse             │ January–February 1995     │ Margin spiral, funding crisis, │
│                      │ Kobe earthquake shock     │ full cascade                   │
└──────────────────────┴──────────────────────────┴────────────────────────────────┘
```

---

## 🔬 Fraud Feature Engineering

Indicators grounded in the **Bank of England Board of Banking Supervision Report (1995)** and BIS OpRisk literature:

```
╔══════════════════════════════════════════════════════════════════════╗
║  FEATURE                   │  BARINGS ANALOG                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  hidden_loss_ratio         │  Account 88888 usage intensity         ║
║  pnl_cash_divergence       │  Reported profit vs. cash withdrawal   ║
║  funding_spike_index       │  Margin call escalation pattern        ║
║  recon_break_frequency     │  SIMEX vs. Barings London mismatch     ║
║  front_back_overlap_score  │  Leeson dual role: trading+settlement  ║
║  position_size_drift       │  Nikkei 225 futures concentration      ║
║  options_straddle_decay    │  Short volatility bleed after Kobe     ║
║  audit_response_lag        │  London queries ignored 7+ weeks       ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 🤖 Model Architecture

```mermaid
flowchart LR
    classDef input  fill:#1A2744,stroke:#C9A84C,color:#F0E6C8
    classDef model  fill:#7B1C2E,stroke:#C9A84C,color:#F0E6C8
    classDef output fill:#C9A84C,stroke:#7B1C2E,color:#1A2744

    I1[Regime Features\n8 Barings Indicators]:::input
    I2[Time-Series\nPosition Sequences]:::input
    I3[Reconciliation\nBreak Logs]:::input

    M1[GradientBoosting\nClassifier]:::model
    M2[XGBoost\nClassifier]:::model
    M3[Isolation Forest\nAnomaly]:::model
    M4[Local Outlier\nFactor]:::model
    M5[LSTM Autoencoder\nSequence Drift]:::model

    AGG[Weighted Ensemble\nAggregator]:::model

    O1[Risk Score\n0.0 to 1.0]:::output
    O2[SHAP Factors\nTop-5 Drivers]:::output
    O3[Narrative Alert\nHuman-Readable]:::output
    O4[Audit Log\nImmutable]:::output

    I1 & I2 & I3 --> M1 & M2 & M3 & M4 & M5
    M1 & M2 & M3 & M4 & M5 --> AGG
    AGG --> O1 & O2 & O3 & O4
```

---

## 🏗️ Repository Layout

```text
barings-oprisk-platform/
│
├── app/
│   ├── api/                    # FastAPI app, auth, endpoints, bootstrap
│   │   ├── main.py
│   │   ├── auth.py             # JWT + RBAC
│   │   ├── endpoints/          # simulate, train, predict, alerts
│   │   └── bootstrap.py        # Demo model training + sample alerts
│   │
│   ├── dashboard/
│   │   └── app.py              # Streamlit analyst interface
│   │
│   ├── data/
│   │   ├── raw/public/         # barings_public_extracts.json + provenance
│   │   ├── clean/              # Versioned processed datasets
│   │   ├── synthetic/          # Regime-labeled simulation output
│   │   └── schemas/            # Pydantic + JSON Schema definitions
│   │
│   ├── ingestion/              # Source loaders, parsers, deduplication
│   ├── timeline/               # Event extraction and normalization
│   ├── features/               # Feature engineering pipeline
│   ├── models/                 # Ensemble: supervised + unsupervised + seq
│   ├── simulation/             # Synthetic data generation engine
│   ├── security/               # Encryption, vault integration
│   ├── utils/                  # Logging, config, versioning
│   ├── tests/                  # Pytest suite
│   ├── docs/
│   │   ├── architecture.md
│   │   ├── threat_model.md
│   │   └── timeline.md
│   ├── configs/
│   └── reports/
│
├── sources.csv                 # Curated public-source provenance registry
├── model_card.md               # SR 11-7 aligned model documentation
├── data_card.md                # Dataset documentation and lineage
├── .env.example
├── pyproject.toml
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install platform
pip install -e .

# 3. Configure secrets
cp .env.example .env

# 4. Bootstrap demo model + sample alerts
python -m app.api.bootstrap

# 5. Start API  —  Terminal 1
uvicorn app.api.main:app --reload

# 6. Start dashboard  —  Terminal 2
streamlit run app/dashboard/app.py
```

---

## 🔌 API Reference

### Authenticate

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'
```

### Simulate a Collapse Scenario

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"scenario":"collapse","days":90,"seed":7}'
```

> Mirrors the January–February 1995 Kobe-shock cascade: accelerating Nikkei 225 short exposure, margin calls exceeding £800M, and reconciliation failure across 14 London control checks.

### Train Ensemble

```bash
curl -X POST http://localhost:8000/train \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "scenarios": ["healthy_desk","mild_anomaly","rogue_trader","collapse"],
    "days_per_scenario": 120
  }'
```

### Score a Scenario

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"scenario":"rogue_trader","days":60,"seed":13}'
```

---

## 🏦 Regulatory Alignment

```mermaid
mindmap
  root((OpRisk\nFrameworks))
    BIS Basel Committee
      Basel II OpRisk Capital
      Basel III SMA Approach
      BCBS 239 Data Aggregation
    Bank of England
      BoE 1995 Barings Report
      PRA SS1/23 Model Risk
    Federal Reserve / OCC
      SR 11-7 Model Governance
      Supervisory Guidance OpRisk
    ESMA / EU
      MiFID II Trade Surveillance
      EMIR Reporting Standards
    Industry Standards
      JP Morgan Risk Framework
      Goldman Sachs Model Committee
      Deutsche Bank RecOps Standard
```

---

## 🔐 Security Architecture

```
┌──────────────────────┬────────────────────────────────────────────────┐
│ Layer                │ Implementation                                  │
├──────────────────────┼────────────────────────────────────────────────┤
│ Authentication       │ JWT Bearer — HS256, configurable expiry        │
│ Authorization        │ RBAC — analyst / auditor / admin               │
│ Rate Limiting        │ Per-endpoint token bucket                      │
│ Artifact Storage     │ AES-256 encrypted (optional HashiCorp Vault)   │
│ Audit Logging        │ Append-only, tamper-evident JSON-L             │
│ Data Provenance      │ SHA-256 hash per public source record          │
│ Synthetic Isolation  │ No real PII, positions, or client data         │
└──────────────────────┴────────────────────────────────────────────────┘
```

---

## 📊 Public-Source Design

Every record in `sources.csv` and `barings_public_extracts.json` preserves:

| Field | Description |
|-------|-------------|
| `source_id` | Canonical identifier |
| `institution` | Issuing body (BoE, BIS, SFC, SIMEX) |
| `date` | Publication or coverage date |
| `url` | Canonical or archived URL |
| `reliability` | Tier-1 official / Tier-2 academic / Tier-3 press |
| `note` | Short paraphrased extract — no copyrighted bulk text |
| `sha256` | Record integrity hash |

The ingestion layer is designed so new PDFs or HTML can be added without modifying the downstream pipeline.

---

## 📌 Optional Dependencies

| Library | Function | Fallback |
|---------|----------|----------|
| `shap` | Gradient-based feature attribution | Deterministic local explainability |
| `mlflow` | Experiment tracking & model registry | File-based versioned logs |
| `tensorflow` | LSTM sequence autoencoder | PCA reconstruction anomaly score |
| `xgboost` | Gradient boosting classifier | Scikit-learn GradientBoostingClassifier |

---

## 🧪 Testing

```bash
# Full suite
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific modules
pytest app/tests/test_features.py
pytest app/tests/test_simulation.py
pytest app/tests/test_api.py
```

---

## 🗺️ Roadmap

```
v1.0  ████████████████████████░░  Core prototype — complete
v1.1  ██████████░░░░░░░░░░░░░░░░  Real-time streaming via Kafka
v1.2  ████████░░░░░░░░░░░░░░░░░░  Graph anomaly — desk network topology
v1.3  ██████░░░░░░░░░░░░░░░░░░░░  LLM-powered narrative audit reports
v2.0  ████░░░░░░░░░░░░░░░░░░░░░░  Multi-case: BCCI · SocGen · Kweku Adoboli
v2.1  ██░░░░░░░░░░░░░░░░░░░░░░░░  Regulatory API submission layer (FCA/PRA)
```

---

## 📂 Documentation Index

| Document | Contents |
|----------|----------|
| [`app/docs/architecture.md`](app/docs/architecture.md) | Full system architecture, data flow, component contracts |
| [`app/docs/threat_model.md`](app/docs/threat_model.md) | Adversarial & insider threat model, STRIDE analysis |
| [`app/docs/timeline.md`](app/docs/timeline.md) | Annotated Barings event timeline with source citations |
| [`model_card.md`](model_card.md) | Model documentation aligned to SR 11-7 / EU AI Act |
| [`data_card.md`](data_card.md) | Dataset documentation, lineage, and synthetic schema |

---

## ⚙️ Environment Variables

```bash
# .env.example

APP_SECRET_KEY=change-me-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRY_MINUTES=60

ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123

MLFLOW_TRACKING_URI=./app/reports/mlruns
ARTIFACT_ENCRYPTION=false
VAULT_ADDR=http://127.0.0.1:8200

LOG_LEVEL=INFO
RATE_LIMIT_PER_MINUTE=60
```

---

## 📜 License & Ethics

Released under the **MIT License**.

All historical references are drawn from public-domain sources including the Bank of England, BIS, SIMEX, and parliamentary records. No proprietary trading data, client records, or unpublished materials are used. All trading simulation is **fully synthetic** and generated programmatically with no connection to any institution's real positions.

This platform is intended strictly for **research, education, and risk management system development**.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=7B1C2E&height=120&section=footer&animation=fadeIn" width="100%"/>

**Built with rigor. Grounded in public record. Designed to prevent the next Barings.**

[![GitHub](https://img.shields.io/badge/GitHub-maykonlincolnusa-7B1C2E?style=for-the-badge&logo=github&logoColor=C9A84C)](https://github.com/maykonlincolnusa)

*"Those who cannot remember the past are condemned to repeat it."*
*— George Santayana · Applied to Banking Risk since 1995*

</div>
