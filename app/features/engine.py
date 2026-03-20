from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.simulation.engine import SimulationResult
from app.utils.io import write_frame, write_json
from app.utils.paths import FEATURE_CATALOG_PATH, LATEST_FEATURES_PATH


def feature_catalog() -> list[dict[str, Any]]:
    return [
        {"name": "reported_pnl_total", "description": "Daily PnL reported by the desk.", "lineage": "cashflow_events.reported_pnl_total"},
        {"name": "actual_pnl_total", "description": "Economic PnL before concealment.", "lineage": "cashflow_events.actual_pnl_total"},
        {"name": "hidden_loss_total", "description": "Loss amount shifted away from reported PnL.", "lineage": "cashflow_events.hidden_loss_total"},
        {"name": "pnl_cash_gap", "description": "Mismatch between reported PnL and realized cash movement.", "lineage": "reported_pnl_total - realized_cash_delta"},
        {"name": "gross_exposure_total", "description": "Total gross exposure across instruments.", "lineage": "position_records.gross_exposure sum"},
        {"name": "exposure_growth_3d", "description": "Three-day percentage growth in gross exposure.", "lineage": "gross_exposure_total.pct_change(3)"},
        {"name": "secret_account_fraction", "description": "Share of trades booked in account 88888.", "lineage": "trade_records.secret_account_flag mean"},
        {"name": "late_entry_rate", "description": "Late entries relative to trade count.", "lineage": "reconciliation_logs.late_entries / trade_count"},
        {"name": "backdated_entry_rate", "description": "Backdated entries relative to trade count.", "lineage": "reconciliation_logs.backdated_entries / trade_count"},
        {"name": "reconciliation_breaks", "description": "Unresolved reconciliation breaks.", "lineage": "reconciliation_logs.reconciliation_breaks"},
        {"name": "unresolved_break_value", "description": "Estimated value stuck in unresolved breaks.", "lineage": "reconciliation_logs.unresolved_break_value"},
        {"name": "margin_call_amount", "description": "Daily margin call demand.", "lineage": "cashflow_events.margin_call_amount"},
        {"name": "funding_transfer", "description": "Funding sent from head office.", "lineage": "cashflow_events.funding_transfer"},
        {"name": "funding_spike_3d", "description": "Acceleration in external funding.", "lineage": "funding_transfer.diff(3)"},
        {"name": "pnl_zscore_5d", "description": "Five-day z-score of actual PnL.", "lineage": "rolling z-score"},
        {"name": "after_kobe_shock", "description": "Indicator for post-17 Jan 1995 observations.", "lineage": "date >= 1995-01-17"},
        {"name": "front_back_same_user", "description": "Control flag for same person across trading and settlement.", "lineage": "audit_logs.front_back_same_user"},
        {"name": "signoff_gap", "description": "Required sign-offs minus actual sign-offs.", "lineage": "2 - audit_logs.signoff_count"},
        {"name": "audit_ignored", "description": "Ignored audit recommendation indicator.", "lineage": "audit_logs.audit_recommendation_ignored"},
        {"name": "trade_after_hours_share", "description": "Share of trades booked after 18:00.", "lineage": "trade_hour > 18"},
    ]


def build_features(result: SimulationResult, persist: bool = True) -> pd.DataFrame:
    trades = result.trade_records.copy()
    positions = result.position_records.copy()
    cash = result.cashflow_events.copy()
    recon = result.reconciliation_logs.copy()
    audit = result.audit_logs.copy()

    for frame in [trades, positions, cash, recon, audit]:
        frame["date"] = pd.to_datetime(frame["date"])

    trade_agg = (
        trades.groupby(["date", "entity_id"], as_index=False)
        .agg(
            trade_count=("instrument", "count"),
            secret_account_fraction=("secret_account_flag", "mean"),
            trade_after_hours_share=("trade_hour", lambda s: float((s > 18).mean())),
            unauthorized_trade_fraction=("authorized_flag", lambda s: 1.0 - float(s.mean())),
        )
    )
    position_agg = (
        positions.groupby(["date", "entity_id"], as_index=False)
        .agg(
            gross_exposure_total=("gross_exposure", "sum"),
            net_exposure_total=("net_exposure", "sum"),
            leverage_proxy_mean=("leverage_proxy", "mean"),
        )
    )
    features = cash.merge(trade_agg, on=["date", "entity_id"]).merge(position_agg, on=["date", "entity_id"]).merge(recon, on=["date", "entity_id", "fraud_label"]).merge(audit, on=["date", "entity_id", "fraud_label"])
    features = features.sort_values(["entity_id", "date"]).reset_index(drop=True)
    features["pnl_cash_gap"] = features["reported_pnl_total"] - features["realized_cash_delta"]
    features["late_entry_rate"] = features["late_entries"] / features["trade_count"].clip(lower=1)
    features["backdated_entry_rate"] = features["backdated_entries"] / features["trade_count"].clip(lower=1)
    features["signoff_gap"] = (2 - features["signoff_count"]).clip(lower=0)
    features["after_kobe_shock"] = (features["date"] >= pd.Timestamp("1995-01-17")).astype(int)
    features["audit_ignored"] = features["audit_recommendation_ignored"]
    features["control_break_score"] = 0.35 * features["front_back_same_user"] + 0.2 * features["signoff_gap"] + 0.25 * features["late_entry_rate"] + 0.2 * features["backdated_entry_rate"]
    features["profit_smoothing_score"] = features["hidden_loss_total"] / (features["actual_pnl_total"].abs() + 1.0)
    features["hidden_loss_ratio"] = features["hidden_loss_total"] / (features["gross_exposure_total"] + 1.0)

    for col in ["gross_exposure_total", "funding_transfer", "actual_pnl_total"]:
        features[f"{col}_rolling_mean_5d"] = features.groupby("entity_id")[col].transform(lambda s: s.rolling(5, min_periods=1).mean())

    features["exposure_growth_3d"] = features.groupby("entity_id")["gross_exposure_total"].pct_change(3).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features["funding_spike_3d"] = features.groupby("entity_id")["funding_transfer"].diff(3).fillna(0.0)
    rolling_mean = features.groupby("entity_id")["actual_pnl_total"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    rolling_std = features.groupby("entity_id")["actual_pnl_total"].transform(lambda s: s.rolling(5, min_periods=2).std()).replace(0.0, np.nan)
    features["pnl_zscore_5d"] = ((features["actual_pnl_total"] - rolling_mean) / rolling_std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    features["alert_key"] = features.apply(lambda row: f"{row['entity_id']}::{row['date'].date().isoformat()}", axis=1)

    if persist:
        write_json(FEATURE_CATALOG_PATH, feature_catalog())
        export = features.copy()
        export["date"] = export["date"].dt.date.astype(str)
        write_frame(LATEST_FEATURES_PATH, export)
    return features


def training_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "entity_id",
        "alert_key",
        "fraud_label",
        "audit_recommendation_ignored",
        "signoff_count",
        "late_entries",
        "backdated_entries",
        "realized_cash_delta",
        "trade_count",
    }
    return [col for col in frame.columns if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])]

