from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = ROOT_DIR / "app"
DATA_DIR = APP_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
SCHEMA_DIR = DATA_DIR / "schemas"
REPORTS_DIR = APP_DIR / "reports"
DOCS_DIR = APP_DIR / "docs"
CONFIGS_DIR = APP_DIR / "configs"
SOURCE_REGISTRY_PATH = ROOT_DIR / "sources.csv"
PUBLIC_EXTRACTS_PATH = RAW_DATA_DIR / "public" / "barings_public_extracts.json"
MARKET_DATA_PATH = RAW_DATA_DIR / "public" / "market_nikkei_jgb_1995.csv"
TIMELINE_PATH = CLEAN_DATA_DIR / "timeline.json"
FEATURE_CATALOG_PATH = CLEAN_DATA_DIR / "feature_catalog.json"
LATEST_FEATURES_PATH = CLEAN_DATA_DIR / "latest_features.csv"
MODEL_BUNDLE_PATH = REPORTS_DIR / "model_bundle.joblib"
TRAINING_REPORT_PATH = REPORTS_DIR / "training_report.json"
ALERT_STORE_PATH = REPORTS_DIR / "alerts.json"
AUDIT_LOG_PATH = REPORTS_DIR / "audit_log.jsonl"
EXPERIMENT_LOG_PATH = REPORTS_DIR / "experiments.jsonl"
ENCRYPTED_ARTIFACT_PATH = REPORTS_DIR / "model_bundle.encrypted"


def ensure_directories() -> None:
    for directory in [RAW_DATA_DIR, CLEAN_DATA_DIR, SYNTHETIC_DATA_DIR, SCHEMA_DIR, REPORTS_DIR, DOCS_DIR, CONFIGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()

