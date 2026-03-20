from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ingestion.public_sources import load_market_data
from app.utils.io import write_frame, write_json
from app.utils.paths import SYNTHETIC_DATA_DIR


SCENARIO_DEFAULTS: dict[str, dict[str, float]] = {
    "healthy_desk": {
        "exposure_growth_rate": 0.01,
        "concealment_strength": 0.0,
        "control_weakness_level": 0.05,
        "market_shock_sensitivity": 0.8,
        "detection_delay": 1.0,
    },
    "mild_anomaly": {
        "exposure_growth_rate": 0.03,
        "concealment_strength": 0.15,
        "control_weakness_level": 0.25,
        "market_shock_sensitivity": 1.1,
        "detection_delay": 0.7,
    },
    "rogue_trader": {
        "exposure_growth_rate": 0.08,
        "concealment_strength": 0.55,
        "control_weakness_level": 0.8,
        "market_shock_sensitivity": 1.75,
        "detection_delay": 0.45,
    },
    "collapse": {
        "exposure_growth_rate": 0.12,
        "concealment_strength": 0.75,
        "control_weakness_level": 0.95,
        "market_shock_sensitivity": 2.2,
        "detection_delay": 0.2,
    },
}


@dataclass(slots=True)
class SimulationConfig:
    scenario: str = "rogue_trader"
    days: int = 90
    seed: int = 42
    start_date: str = "1994-12-01"
    entity_id: str = "BFS_SINGAPORE"
    trader_id: str = "NICK_LEESON"
    desk_id: str = "EQUITY_DERIVATIVES_APAC"
    exposure_growth_rate: float | None = None
    concealment_strength: float | None = None
    control_weakness_level: float | None = None
    market_shock_sensitivity: float | None = None
    detection_delay: float | None = None

    def resolved(self) -> "SimulationConfig":
        defaults = SCENARIO_DEFAULTS[self.scenario]
        return SimulationConfig(
            scenario=self.scenario,
            days=self.days,
            seed=self.seed,
            start_date=self.start_date,
            entity_id=self.entity_id,
            trader_id=self.trader_id,
            desk_id=self.desk_id,
            exposure_growth_rate=self.exposure_growth_rate if self.exposure_growth_rate is not None else defaults["exposure_growth_rate"],
            concealment_strength=self.concealment_strength if self.concealment_strength is not None else defaults["concealment_strength"],
            control_weakness_level=self.control_weakness_level if self.control_weakness_level is not None else defaults["control_weakness_level"],
            market_shock_sensitivity=self.market_shock_sensitivity if self.market_shock_sensitivity is not None else defaults["market_shock_sensitivity"],
            detection_delay=self.detection_delay if self.detection_delay is not None else defaults["detection_delay"],
        )


@dataclass
class SimulationResult:
    dataset_id: str
    config: SimulationConfig
    output_dir: Path
    trade_records: pd.DataFrame
    position_records: pd.DataFrame
    cashflow_events: pd.DataFrame
    reconciliation_logs: pd.DataFrame
    audit_logs: pd.DataFrame
    market_data: pd.DataFrame

    def to_payload(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "scenario": self.config.scenario,
            "output_dir": str(self.output_dir),
            "rows": {
                "trade_records": len(self.trade_records),
                "position_records": len(self.position_records),
                "cashflow_events": len(self.cashflow_events),
                "reconciliation_logs": len(self.reconciliation_logs),
                "audit_logs": len(self.audit_logs),
            },
            "parameters": asdict(self.config),
        }


class BaringsSimulationEngine:
    def simulate(self, config: SimulationConfig) -> SimulationResult:
        config = config.resolved()
        rng = np.random.default_rng(config.seed)
        market = self._market_window(config.days, config.start_date)
        shock_date = pd.Timestamp("1995-01-17")
        dataset_id = f"{config.scenario}_{config.seed}_{config.days}"
        output_dir = SYNTHETIC_DATA_DIR / dataset_id
        output_dir.mkdir(parents=True, exist_ok=True)

        trade_rows: list[dict[str, Any]] = []
        position_rows: list[dict[str, Any]] = []
        cash_rows: list[dict[str, Any]] = []
        recon_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        exposures = {"NIKKEI_FUT": 55_000_000.0, "JGB_FUT": -18_000_000.0, "NIKKEI_SHORT_VOL": -8_000_000.0}
        fraud_onset = max(5, int(config.days * config.detection_delay))

        for idx, market_row in enumerate(market.itertuples(index=False)):
            date = pd.Timestamp(market_row.date)
            is_fraud_window = int(config.scenario in {"rogue_trader", "collapse"} and idx >= fraud_onset)
            if config.scenario == "mild_anomaly":
                is_fraud_window = int(idx >= fraud_onset and idx % 3 == 0)

            growth_factor = 1.0 + config.exposure_growth_rate + rng.normal(0.0, 0.02)
            shock_multiplier = 1.0 + config.market_shock_sensitivity * max(0.0, market_row.volatility_proxy - 0.25)
            same_user = int(rng.random() < config.control_weakness_level)
            signoff_count = max(0, int(round(2 - 2 * config.control_weakness_level + rng.normal(0.0, 0.3))))
            late_entries = int(abs(rng.normal(1 + 8 * config.control_weakness_level, 1.2)))
            backdated_entries = int(abs(rng.normal(0.5 + 4 * config.control_weakness_level, 0.8)))
            recon_breaks = int(abs(rng.normal(1 + 7 * config.control_weakness_level, 1.4)))

            total_actual_pnl = 0.0
            total_reported_pnl = 0.0
            total_hidden_loss = 0.0

            for instrument, prior_exposure in list(exposures.items()):
                direction_noise = rng.normal(1.0, 0.08)
                exposure = prior_exposure * growth_factor * direction_noise
                if instrument == "JGB_FUT":
                    pnl_driver = -(market_row.jgb_futures - 142.0) * 85_000
                elif instrument == "NIKKEI_SHORT_VOL":
                    pnl_driver = -market_row.volatility_proxy * 25_000_000
                else:
                    pnl_driver = market_row.nikkei_return * exposure * 1.6

                if date >= shock_date:
                    pnl_driver *= shock_multiplier

                actual_pnl = pnl_driver + rng.normal(0.0, 450_000)
                hidden_loss = max(0.0, -actual_pnl) * config.concealment_strength * (0.15 + 0.85 * is_fraud_window)
                reported_pnl = actual_pnl + hidden_loss
                account_id = "88888" if hidden_loss > 0 and rng.random() < max(0.15, config.concealment_strength) else "AUTH_MAIN"
                qty = abs(exposure) / max(market_row.nikkei_close, 1)

                trade_rows.append(
                    {
                        "date": date.date().isoformat(),
                        "entity_id": config.entity_id,
                        "desk_id": config.desk_id,
                        "trader_id": config.trader_id,
                        "instrument": instrument,
                        "account_id": account_id,
                        "authorized_flag": int(account_id != "88888"),
                        "secret_account_flag": int(account_id == "88888"),
                        "quantity": round(qty, 2),
                        "price": round(float(market_row.nikkei_close if "NIKKEI" in instrument else market_row.jgb_futures), 4),
                        "actual_pnl": round(actual_pnl, 2),
                        "reported_pnl": round(reported_pnl, 2),
                        "hidden_loss": round(hidden_loss, 2),
                        "trade_hour": int(np.clip(rng.normal(18 + 4 * same_user, 3.2), 0, 23)),
                        "fraud_label": is_fraud_window,
                    }
                )
                position_rows.append(
                    {
                        "date": date.date().isoformat(),
                        "entity_id": config.entity_id,
                        "instrument": instrument,
                        "net_exposure": round(exposure, 2),
                        "gross_exposure": round(abs(exposure), 2),
                        "leverage_proxy": round(abs(exposure) / 5_000_000, 4),
                        "fraud_label": is_fraud_window,
                    }
                )
                exposures[instrument] = exposure
                total_actual_pnl += actual_pnl
                total_reported_pnl += reported_pnl
                total_hidden_loss += hidden_loss

            margin_call_amount = max(0.0, -total_actual_pnl) * (0.12 + market_row.volatility_proxy)
            funding_transfer = margin_call_amount * (0.2 + config.control_weakness_level) if is_fraud_window else margin_call_amount * 0.12
            unresolved_break_value = recon_breaks * (45_000 + 60_000 * config.control_weakness_level) * (1 + market_row.volatility_proxy)

            cash_rows.append(
                {
                    "date": date.date().isoformat(),
                    "entity_id": config.entity_id,
                    "reported_pnl_total": round(total_reported_pnl, 2),
                    "actual_pnl_total": round(total_actual_pnl, 2),
                    "margin_call_amount": round(margin_call_amount, 2),
                    "funding_transfer": round(funding_transfer, 2),
                    "realized_cash_delta": round(total_actual_pnl - funding_transfer, 2),
                    "hidden_loss_total": round(total_hidden_loss, 2),
                    "fraud_label": is_fraud_window,
                }
            )
            recon_rows.append(
                {
                    "date": date.date().isoformat(),
                    "entity_id": config.entity_id,
                    "reconciliation_breaks": recon_breaks,
                    "late_entries": late_entries,
                    "backdated_entries": backdated_entries,
                    "unresolved_break_value": round(unresolved_break_value, 2),
                    "fraud_label": is_fraud_window,
                }
            )
            audit_rows.append(
                {
                    "date": date.date().isoformat(),
                    "entity_id": config.entity_id,
                    "front_back_same_user": same_user,
                    "signoff_count": signoff_count,
                    "audit_recommendation_ignored": int(is_fraud_window and rng.random() < config.control_weakness_level),
                    "fraud_label": is_fraud_window,
                }
            )

        result = SimulationResult(
            dataset_id=dataset_id,
            config=config,
            output_dir=output_dir,
            trade_records=pd.DataFrame(trade_rows),
            position_records=pd.DataFrame(position_rows),
            cashflow_events=pd.DataFrame(cash_rows),
            reconciliation_logs=pd.DataFrame(recon_rows),
            audit_logs=pd.DataFrame(audit_rows),
            market_data=market,
        )
        self._persist(result)
        return result

    def _market_window(self, days: int, start_date: str) -> pd.DataFrame:
        market = load_market_data().copy()
        market = market[market["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
        if len(market) >= days:
            return market.iloc[:days].reset_index(drop=True)

        last = market.iloc[-1].copy()
        extra_rows = []
        for _ in range(days - len(market)):
            last = last.copy()
            last["date"] = pd.Timestamp(last["date"]) + pd.Timedelta(days=1)
            last["nikkei_return"] = float(np.clip(np.random.normal(-0.002, 0.008), -0.04, 0.03))
            last["nikkei_close"] = float(max(14_500, last["nikkei_close"] * (1 + last["nikkei_return"])))
            last["jgb_futures"] = float(last["jgb_futures"] + np.random.normal(0.08, 0.12))
            last["usd_jpy"] = float(last["usd_jpy"] + np.random.normal(0.03, 0.18))
            last["gbp_jpy"] = float(last["gbp_jpy"] + np.random.normal(0.06, 0.25))
            last["volatility_proxy"] = float(np.clip(last["volatility_proxy"] + np.random.normal(0.002, 0.01), 0.2, 0.52))
            extra_rows.append(last.to_dict())
        if extra_rows:
            market = pd.concat([market, pd.DataFrame(extra_rows)], ignore_index=True)
        return market.iloc[:days].reset_index(drop=True)

    def _persist(self, result: SimulationResult) -> None:
        write_frame(result.output_dir / "trade_records.csv", result.trade_records)
        write_frame(result.output_dir / "position_records.csv", result.position_records)
        write_frame(result.output_dir / "cashflow_events.csv", result.cashflow_events)
        write_frame(result.output_dir / "reconciliation_logs.csv", result.reconciliation_logs)
        write_frame(result.output_dir / "audit_logs.csv", result.audit_logs)
        write_frame(result.output_dir / "market_data.csv", result.market_data)
        write_json(result.output_dir / "metadata.json", result.to_payload())

