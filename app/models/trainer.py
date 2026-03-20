from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.features.engine import training_columns
from app.models.sequence import SequenceAutoencoder
from app.security.crypto import maybe_encrypt_artifact
from app.utils.io import append_jsonl, write_json
from app.utils.paths import EXPERIMENT_LOG_PATH, MODEL_BUNDLE_PATH, TRAINING_REPORT_PATH


class ModelTrainer:
    def train(self, feature_frame: pd.DataFrame) -> dict[str, Any]:
        feature_cols = training_columns(feature_frame)
        X = feature_frame[feature_cols].astype(float)
        y = feature_frame["fraud_label"].astype(int)
        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, feature_frame.index, test_size=0.25, random_state=42, stratify=stratify
        )

        logistic = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        forest = RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42, class_weight="balanced_subsample")
        isolation = IsolationForest(random_state=42, contamination=max(0.05, float(y.mean()) if len(y) else 0.05))
        sequence = SequenceAutoencoder(sequence_length=8)

        logistic.fit(X_train, y_train)
        forest.fit(X_train, y_train)
        isolation.fit(X_train)
        sequence.fit(feature_frame.loc[train_idx], feature_cols)

        test_frame = feature_frame.loc[test_idx].copy()
        scores = self.score(
            test_frame,
            {"feature_cols": feature_cols, "logistic": logistic, "forest": forest, "isolation": isolation, "sequence": sequence},
        )
        roc_auc = float(roc_auc_score(y_test, scores["risk_score"])) if y_test.nunique() > 1 else 0.0
        report = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(feature_frame)),
            "feature_count": len(feature_cols),
            "metrics": {
                "roc_auc": roc_auc,
                "precision": float(precision_score(y_test, (scores["risk_score"] >= 0.65).astype(int), zero_division=0)),
                "recall": float(recall_score(y_test, (scores["risk_score"] >= 0.65).astype(int), zero_division=0)),
            },
            "sequence_backend": "tensorflow_lstm" if sequence.tensorflow_enabled else "pca_reconstruction",
        }

        logistic.fit(X, y)
        forest.fit(X, y)
        isolation.fit(X)
        sequence.fit(feature_frame, feature_cols)
        bundle = {
            "trained_at": report["trained_at"],
            "feature_cols": feature_cols,
            "logistic": logistic,
            "forest": forest,
            "isolation": isolation,
            "sequence": sequence,
            "metrics": report["metrics"],
            "training_stats": {
                "median": X.median().to_dict(),
                "iqr": (X.quantile(0.75) - X.quantile(0.25)).replace(0.0, 1.0).to_dict(),
                "feature_means": X.mean().to_dict(),
            },
        }
        joblib.dump(bundle, MODEL_BUNDLE_PATH)
        maybe_encrypt_artifact(MODEL_BUNDLE_PATH)
        write_json(TRAINING_REPORT_PATH, report)
        append_jsonl(EXPERIMENT_LOG_PATH, report)
        try:
            import mlflow  # type: ignore

            mlflow.set_tracking_uri(f"file:{MODEL_BUNDLE_PATH.parent / 'mlruns'}")
            with mlflow.start_run(run_name="barings-ensemble"):
                mlflow.log_params({"feature_count": len(feature_cols), "sequence_backend": report["sequence_backend"]})
                mlflow.log_metrics(report["metrics"])
        except Exception:
            pass
        return report

    def load_bundle(self) -> dict[str, Any]:
        return joblib.load(MODEL_BUNDLE_PATH)

    def score(self, feature_frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
        feature_cols = bundle["feature_cols"]
        X = feature_frame[feature_cols].astype(float)
        logistic_score = bundle["logistic"].predict_proba(X)[:, 1]
        forest_score = bundle["forest"].predict_proba(X)[:, 1]
        isolation_raw = -bundle["isolation"].score_samples(X)
        sequence_raw = bundle["sequence"].score(feature_frame, feature_cols)
        isolation_score = self._normalize(isolation_raw)
        sequence_score = self._normalize(sequence_raw)
        risk_score = 0.25 * logistic_score + 0.35 * forest_score + 0.2 * isolation_score + 0.2 * sequence_score

        scored = feature_frame.copy()
        scored["score_logistic"] = logistic_score
        scored["score_forest"] = forest_score
        scored["score_isolation"] = isolation_score
        scored["score_sequence"] = sequence_score
        scored["risk_score"] = risk_score
        scored["risk_band"] = scored["risk_score"].map(self._risk_band)
        scored["prediction"] = (scored["risk_score"] >= 0.65).astype(int)
        return scored

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return values
        lower = float(values.min())
        upper = float(values.max())
        if np.isclose(lower, upper):
            return np.zeros_like(values)
        return (values - lower) / (upper - lower)

    @staticmethod
    def _risk_band(score: float) -> str:
        if score >= 0.8:
            return "critical"
        if score >= 0.65:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

