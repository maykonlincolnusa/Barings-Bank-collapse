from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.utils.io import read_json
from app.utils.paths import FEATURE_CATALOG_PATH


FEATURE_REASON_MAP = {
    "hidden_loss_total": "Large hidden-loss accumulation suggests off-book concealment.",
    "secret_account_fraction": "Frequent use of account 88888-like booking patterns is abnormal.",
    "pnl_cash_gap": "Reported PnL diverges materially from realized cash movements.",
    "margin_call_amount": "Margin pressure is rising faster than a healthy desk should require.",
    "funding_spike_3d": "Head-office funding transfers are spiking to support losses.",
    "front_back_same_user": "Front and back office duties appear concentrated in one user.",
    "control_break_score": "Operational control breakdown signals are elevated.",
    "reconciliation_breaks": "Reconciliation exceptions remain unresolved.",
    "exposure_growth_3d": "Gross exposure is growing too quickly.",
    "pnl_zscore_5d": "PnL behavior deviates sharply from the recent baseline.",
}


def explain_prediction(row: pd.Series, bundle: dict[str, Any], top_n: int = 5) -> dict[str, Any]:
    feature_cols = bundle["feature_cols"]
    local_frame = pd.DataFrame([row[feature_cols].astype(float)])
    explanation = _shap_explanation(local_frame, bundle)
    if explanation is None:
        explanation = _surrogate_explanation(row, bundle, top_n=top_n)
    top_features = explanation[:top_n]
    narrative = " ".join(FEATURE_REASON_MAP.get(item["feature"], item["description"]) for item in top_features[:3])
    return {"top_features": top_features, "narrative": narrative.strip()}


def _shap_explanation(local_frame: pd.DataFrame, bundle: dict[str, Any]) -> list[dict[str, Any]] | None:
    try:
        import shap  # type: ignore
    except Exception:
        return None

    try:
        explainer = shap.TreeExplainer(bundle["forest"])
        values = explainer.shap_values(local_frame)[1][0]
    except Exception:
        return None

    catalog = {item["name"]: item for item in read_json(FEATURE_CATALOG_PATH, default=[])}
    ordered = np.argsort(np.abs(values))[::-1]
    results = []
    for idx in ordered:
        feature = local_frame.columns[idx]
        meta = catalog.get(feature, {"description": feature})
        results.append(
            {
                "feature": feature,
                "impact": float(values[idx]),
                "value": float(local_frame.iloc[0, idx]),
                "description": meta["description"],
            }
        )
    return results


def _surrogate_explanation(row: pd.Series, bundle: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
    medians = bundle["training_stats"]["median"]
    iqr = bundle["training_stats"]["iqr"]
    forest_importance = bundle["forest"].feature_importances_
    logistic_coef = np.abs(bundle["logistic"].named_steps["model"].coef_[0])
    combined_importance = forest_importance + logistic_coef / max(logistic_coef.sum(), 1e-9)
    catalog = {item["name"]: item for item in read_json(FEATURE_CATALOG_PATH, default=[])}
    impacts = []
    for idx, feature in enumerate(bundle["feature_cols"]):
        scale = float(iqr.get(feature, 1.0) or 1.0)
        baseline = float(medians.get(feature, 0.0))
        deviation = (float(row[feature]) - baseline) / scale
        impact = deviation * float(combined_importance[idx])
        meta = catalog.get(feature, {"description": feature})
        impacts.append(
            {
                "feature": feature,
                "impact": impact,
                "value": float(row[feature]),
                "description": meta["description"],
            }
        )
    impacts.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return impacts[:top_n]

