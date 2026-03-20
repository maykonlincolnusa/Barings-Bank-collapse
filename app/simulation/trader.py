import uuid
from typing import Tuple
from datetime import timedelta
import pandas as pd
import numpy as np

from app.schemas.domain import (
    TradeRecord, PositionRecord, CashFlowEvent, 
    SimulationParams, InstrumentType, TradeSide, CashFlowType, SimulationScenario
)

class FraudSimulator:
    def __init__(self, params: SimulationParams, market_data: pd.DataFrame):
        self.params = params
        self.market_data = market_data
        self.market_dict = self._build_market_dict()
        self.np_random = np.random.RandomState(42)

    def _build_market_dict(self):
        # Convert market_data df to nested dict: {date: {instrument: price}}
        res = {}
        for _, row in self.market_data.iterrows():
            dt = row["date"]
            if dt not in res:
                res[dt] = {}
            res[dt][row["instrument"]] = row["close_price"]
        return res

    def simulate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        trades = []
        positions = []
        cashflows = []
        
        # Current state
        pos_net_qty = {InstrumentType.NIKKEI_FUTURE: 0, InstrumentType.JGB_FUTURE: 0}
        avg_price = {InstrumentType.NIKKEI_FUTURE: 0.0, InstrumentType.JGB_FUTURE: 0.0}
        
        # Secret account state mapping
        secret_pos_net_qty = {InstrumentType.NIKKEI_FUTURE: 0, InstrumentType.JGB_FUTURE: 0}
        secret_avg_price = {InstrumentType.NIKKEI_FUTURE: 0.0, InstrumentType.JGB_FUTURE: 0.0}
        
        dates = sorted(list(self.market_dict.keys()))
        
        for dt in dates:
            mkt_dt = self.market_dict[dt]
            if InstrumentType.NIKKEI_FUTURE not in mkt_dt or InstrumentType.JGB_FUTURE not in mkt_dt:
                continue
                
            nikkei_price = mkt_dt[InstrumentType.NIKKEI_FUTURE]
            jgb_price = mkt_dt[InstrumentType.JGB_FUTURE]
            
            # 1. Normal trading (Arbitrage / Client orders)
            qty_n = self.np_random.randint(10, 50)
            qty_j = int(qty_n * (nikkei_price / jgb_price * 0.1)) # rough hedge
            
            trades.append(self._create_trade(dt, "MAIN", InstrumentType.NIKKEI_FUTURE, TradeSide.BUY, qty_n, nikkei_price, True))
            trades.append(self._create_trade(dt, "MAIN", InstrumentType.JGB_FUTURE, TradeSide.SELL, qty_j, jgb_price, True))
            
            pos_net_qty[InstrumentType.NIKKEI_FUTURE] += qty_n
            pos_net_qty[InstrumentType.JGB_FUTURE] -= qty_j
            
            # 2. Rogue trading behavior (if activated)
            # Nick Leeson was shorting vol and taking directional unhedged long Nikkei bets
            if self.params.scenario in [SimulationScenario.ROGUE_TRADER, SimulationScenario.MILD_ANOMALY]:
                # Escalate hidden trades post Kobe EQ
                multiplier = 15 if dt >= self.params.kobe_earthquake_date else 2
                secret_qty_n = self.np_random.randint(100, 500) * multiplier
                secret_qty_j = self.np_random.randint(100, 500) * multiplier
                
                # Concealed in Account 88888
                tr_n = self._create_trade(dt, self.params.rogue_account_id, InstrumentType.NIKKEI_FUTURE, TradeSide.BUY, secret_qty_n, nikkei_price, False)
                tr_j = self._create_trade(dt, self.params.rogue_account_id, InstrumentType.JGB_FUTURE, TradeSide.SELL, secret_qty_j, jgb_price, False) # short JGB
                
                # Lack of segregation of duties - marking his own trades as settled!
                tr_n.is_settled = True
                tr_j.is_settled = True
                
                trades.extend([tr_n, tr_j])
                secret_pos_net_qty[InstrumentType.NIKKEI_FUTURE] += secret_qty_n
                secret_pos_net_qty[InstrumentType.JGB_FUTURE] -= secret_qty_j
                
                # Hidden losses trigger margin calls that are disguised as funding requests
                if dt.day % 5 == 0:
                    cf = CashFlowEvent(
                        id=str(uuid.uuid4()),
                        date=dt,
                        entity_id=self.params.trader_entity_id,
                        from_account="LONDON_HQ",
                        to_account=self.params.rogue_account_id,
                        amount=secret_qty_n * nikkei_price * 0.05 * self.np_random.uniform(0.8, 1.2),
                        flow_type=CashFlowType.FUNDING_REQUEST,
                        reason="Margin for client trades (Concealed)"
                    )
                    cashflows.append(cf)
            
            # Save daily position records (Calculating simplified P&L approximation)
            main_pnl = float(pos_net_qty[InstrumentType.NIKKEI_FUTURE] * (nikkei_price - 19000)) 
            secret_pnl = float(secret_pos_net_qty[InstrumentType.NIKKEI_FUTURE] * (nikkei_price - 19500))
            
            positions.append(self._create_pos(dt, "MAIN", InstrumentType.NIKKEI_FUTURE, pos_net_qty[InstrumentType.NIKKEI_FUTURE], nikkei_price, main_pnl))
            positions.append(self._create_pos(dt, "MAIN", InstrumentType.JGB_FUTURE, pos_net_qty[InstrumentType.JGB_FUTURE], jgb_price, 0))
            
            if self.params.scenario != SimulationScenario.HEALTHY:
                positions.append(self._create_pos(dt, self.params.rogue_account_id, InstrumentType.NIKKEI_FUTURE, secret_pos_net_qty[InstrumentType.NIKKEI_FUTURE], nikkei_price, secret_pnl))
                positions.append(self._create_pos(dt, self.params.rogue_account_id, InstrumentType.JGB_FUTURE, secret_pos_net_qty[InstrumentType.JGB_FUTURE], jgb_price, 0))
                
        df_trades = pd.DataFrame([t.model_dump() for t in trades])
        df_pos = pd.DataFrame([p.model_dump() for p in positions])
        df_cf = pd.DataFrame([c.model_dump() for c in cashflows])
        
        return df_trades, df_pos, df_cf

    def _create_trade(self, dt, acct, inst, side, qty, px, is_auth) -> TradeRecord:
        return TradeRecord(
            id=str(uuid.uuid4()),
            timestamp=pd.to_datetime(dt) + timedelta(hours=self.np_random.randint(8, 18)),
            entity_id=self.params.trader_entity_id,
            account_id=acct,
            instrument=inst,
            side=side,
            quantity=qty,
            price=px,
            is_authorized=is_auth,
            is_settled=is_auth,
            notes="Client hedge" if is_auth else "Error account parking"
        )
        
    def _create_pos(self, dt, acct, inst, net_qty, current_px, pnl) -> PositionRecord:
        return PositionRecord(
            date=dt,
            entity_id=self.params.trader_entity_id,
            account_id=acct,
            instrument=inst,
            net_quantity=net_qty,
            avg_price=current_px,
            unrealized_pnl=pnl    
        )
