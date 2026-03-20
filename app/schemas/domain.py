from datetime import datetime, date
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class InstrumentType(str, Enum):
    NIKKEI_FUTURE = "NIKKEI_FUTURE"
    JGB_FUTURE = "JGB_FUTURE"
    NIKKEI_OPTION = "NIKKEI_OPTION"

class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class TradeRecord(BaseModel):
    id: str
    timestamp: datetime
    entity_id: str
    account_id: str
    instrument: InstrumentType
    side: TradeSide
    quantity: int
    price: float
    is_authorized: bool = True
    is_settled: bool = True
    notes: Optional[str] = None

class PositionRecord(BaseModel):
    date: date
    entity_id: str
    account_id: str
    instrument: InstrumentType
    net_quantity: int
    avg_price: float
    unrealized_pnl: float

class MarketDataRecord(BaseModel):
    date: date
    instrument: InstrumentType
    close_price: float
    volatility: float

class CashFlowType(str, Enum):
    MARGIN_CALL = "MARGIN_CALL"
    FUNDING_REQUEST = "FUNDING_REQUEST"
    INTERNAL_TRANSFER = "INTERNAL_TRANSFER"

class CashFlowEvent(BaseModel):
    id: str
    date: date
    entity_id: str
    from_account: str
    to_account: str
    amount: float
    flow_type: CashFlowType
    reason: str

class SimulationScenario(str, Enum):
    HEALTHY = "HEALTHY"
    MILD_ANOMALY = "MILD_ANOMALY"
    ROGUE_TRADER = "ROGUE_TRADER"

class SimulationParams(BaseModel):
    start_date: date = date(1994, 1, 1)
    end_date: date = date(1995, 3, 1)
    scenario: SimulationScenario = SimulationScenario.ROGUE_TRADER
    kobe_earthquake_date: date = date(1995, 1, 17)
    rogue_account_id: str = "88888"
    trader_entity_id: str = "NL_001"

class Alert(BaseModel):
    id: str
    timestamp: datetime
    entity_id: str
    risk_score: float
    risk_band: str # LOW, MEDIUM, HIGH, CRITICAL
    top_features: Dict[str, float]
    reason: str
