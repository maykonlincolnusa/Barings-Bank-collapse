import numpy as np
import pandas as pd
from datetime import timedelta
from typing import List
from ..schemas.domain import MarketDataRecord, InstrumentType, SimulationParams

class MarketSimulator:
    def __init__(self, params: SimulationParams):
        self.params = params
        self.seed = 42
        
    def generate_market_data(self) -> pd.DataFrame:
        """
        Simulate Nikkei and JGB prices.
        Includes a massive drop for the Kobe earthquake on Jan 17, 1995.
        """
        np.random.seed(self.seed)
        date_range = pd.date_range(self.params.start_date, self.params.end_date, freq='B')
        
        # Nikkei 225 Baseline (~19000 mid-90s)
        nikkei_base = 19500.0
        nikkei_vol = 0.015 # daily vol
        
        # JGB Baseline (in points, e.g., 110.00)
        jgb_base = 110.0
        jgb_vol = 0.005
        
        nikkei_prices = [nikkei_base]
        jgb_prices = [jgb_base]
        
        for i in range(1, len(date_range)):
            current_date = date_range[i].date()
            
            # Normal random walk
            n_shock = np.random.normal(0, nikkei_vol)
            j_shock = np.random.normal(0, jgb_vol) - (n_shock * 0.2) # Negative correlation
            
            # Kobe Earthquake Shock (Jan 17 1995 -> 1000 point drop over a week)
            if self.params.kobe_earthquake_date <= current_date <= self.params.kobe_earthquake_date + timedelta(days=7):
                n_shock -= 0.02 # 2% drop per day for a week ~14% total drop
                j_shock += 0.005 # JGBs rally
                
            nikkei_prices.append(nikkei_prices[-1] * (1 + n_shock))
            jgb_prices.append(jgb_prices[-1] * (1 + j_shock))
            
        data = []
        for i, dt in enumerate(date_range):
            data.append(MarketDataRecord(
                date=dt.date(),
                instrument=InstrumentType.NIKKEI_FUTURE,
                close_price=round(nikkei_prices[i], 2),
                volatility=nikkei_vol * np.sqrt(252)
            ))
            data.append(MarketDataRecord(
                date=dt.date(),
                instrument=InstrumentType.JGB_FUTURE,
                close_price=round(jgb_prices[i], 2),
                volatility=jgb_vol * np.sqrt(252)
            ))
            
        return pd.DataFrame([d.model_dump() for d in data])
