from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd

from app.api.schemas import PredictRequest, SimulationRequest, TrainRequest
from app.features.engine import build_features
from app.ingestion.public_sources import load_source_registry
from app.models.explainer import explain_prediction
from app.models.trainer import ModelTrainer
from app.simulation.engine import BaringsSimulationEngine, SimulationConfig
from app.timeline.builder import build_timeline
from app.utils.io import append_jsonl, read_json, read_jsonl, write_json
from app.utils.paths import ALERT_STORE_PATH, AUDIT_LOG_PATH, MODEL_BUNDLE_PATH, TRAINING_REPORT_PATH


class PlatformService:
    def __init__(self) -> None:
        self.simulator = BaringsSimulationEngine()
        self.trainer = ModelTrainer()
        build_timeline()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "time": datetime.now(timezone.utc).isoformat(),
            "model_available": MODEL_BUNDLE_PATH.exists(),
        }

    def metrics(self) -> dict[str, Any]:
        report = read_json(TRAINING_REPORT_PATH, default={})
        alerts = read_json(ALERT_STORE_PATH, default=[])
        return {
            "trained_at": report.get("trained_at"),
            "model_metrics": report.get("metrics", {}),
            "total_alerts": len(alerts),
            "critical_alerts": sum(1 for alert in alerts if alert.get("risk_band") == "critical"),
        }

    def sources(self) -> list[dict[str, Any]]:
        frame = load_source_registry().copy()
        frame["date"] = frame["date"].dt.date.astype(str)
        return frame.to_dict(orient="records")

    def timeline(self) -> list[dict[str, Any]]:
        return build_timeline()

    def simulate(self, request: SimulationRequest, actor: dict[str, Any]) -> dict[str, Any]:
        config = SimulationConfig(**request.model_dump())
        result = self.simulator.simulate(config)
        features = build_features(result)
        self._audit("simulate", actor, result.dataset_id, {"scenario": request.scenario, "days": request.days})
        return {
            **result.to_payload(),
            "feature_rows": len(features),
            "high_risk_days": int(features["fraud_label"].sum()),
        }

    def train(self, request: TrainRequest, actor: dict[str, Any]) -> dict[str, Any]:
        frames = []
        for scenario in request.scenarios:
            for seed_offset in range(request.runs_per_scenario):
                config = SimulationConfig(scenario=scenario, days=request.days_per_scenario, seed=42 + seed_offset * 11)
                result = self.simulator.simulate(config)
                frames.append(build_features(result, persist=(scenario == request.scenarios[0] and seed_offset == 0)))
        training_frame = pd.concat(frames, ignore_index=True)
        report = self.trainer.train(training_frame)
        scored = self.trainer.score(training_frame, self.trainer.load_bundle())
        alerts = self._store_alerts(scored.nlargest(20, "risk_score"))
        self._audit("train", actor, "model_bundle", {"rows": len(training_frame), "scenarios": request.scenarios, "alerts_seeded": len(alerts)})
        return {"report": report, "seeded_alerts": len(alerts)}

    def predict(self, request: PredictRequest, actor: dict[str, Any]) -> dict[str, Any]:
        bundle = self.trainer.load_bundle()
        if request.observations is not None:
            feature_frame = pd.DataFrame(request.observations)
            feature_frame["date"] = pd.to_datetime(feature_frame["date"])
        else:
            scenario = request.scenario or "rogue_trader"
            result = self.simulator.simulate(SimulationConfig(scenario=scenario, days=request.days, seed=request.seed))
            feature_frame = build_features(result, persist=False)

        scored = self.trainer.score(feature_frame, bundle)
        alerts = self._store_alerts(scored)
        self._audit("predict", actor, alerts[0]["alert_id"] if alerts else "no-alert", {"rows": len(scored), "scenario": request.scenario})
        return {
            "count": len(alerts),
            "predictions": alerts,
            "summary": {
                "max_risk_score": max((alert["risk_score"] for alert in alerts), default=0.0),
                "critical": sum(1 for alert in alerts if alert["risk_band"] == "critical"),
                "high_or_worse": sum(1 for alert in alerts if alert["risk_band"] in {"high", "critical"}),
            },
        }

    def predict_batch(self, request: list[PredictRequest], actor: dict[str, Any]) -> dict[str, Any]:
        outputs = [self.predict(item, actor) for item in request]
        return {"items": outputs, "total_predictions": sum(item["count"] for item in outputs)}

    def explain(self, alert_id: str, actor: dict[str, Any]) -> dict[str, Any]:
        alerts = read_json(ALERT_STORE_PATH, default=[])
        for alert in alerts:
            if alert["alert_id"] == alert_id:
                self._audit("explain", actor, alert_id, {})
                return alert
        raise KeyError(alert_id)

    def audit(self, object_id: str, actor: dict[str, Any]) -> list[dict[str, Any]]:
        events = [entry for entry in read_jsonl(AUDIT_LOG_PATH) if entry.get("object_id") == object_id]
        self._audit("audit_lookup", actor, object_id, {"matches": len(events)})
        return events

    def _store_alerts(self, scored: pd.DataFrame) -> list[dict[str, Any]]:
        bundle = self.trainer.load_bundle()
        existing = read_json(ALERT_STORE_PATH, default=[])
        new_alerts = []
        for _, row in scored.iterrows():
            explanation = explain_prediction(row, bundle)
            new_alerts.append(
                {
                    "alert_id": f"ALERT-{uuid4().hex[:12]}",
                    "alert_key": row["alert_key"],
                    "date": row["date"].date().isoformat(),
                    "entity_id": row["entity_id"],
                    "risk_score": round(float(row["risk_score"]), 4),
                    "risk_band": row["risk_band"],
                    "prediction": int(row["prediction"]),
                    "top_features": explanation["top_features"],
                    "narrative": explanation["narrative"],
                    "scores": {
                        "logistic": round(float(row["score_logistic"]), 4),
                        "forest": round(float(row["score_forest"]), 4),
                        "isolation": round(float(row["score_isolation"]), 4),
                        "sequence": round(float(row["score_sequence"]), 4),
                    },
                }
            )
        write_json(ALERT_STORE_PATH, existing + new_alerts)
        return new_alerts

    def _audit(self, action: str, actor: dict[str, Any], object_id: str, detail: dict[str, Any]) -> None:
        append_jsonl(
            AUDIT_LOG_PATH,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "actor": actor["username"],
                "role": actor["role"],
                "object_id": object_id,
                "detail": detail,
            },
        )

