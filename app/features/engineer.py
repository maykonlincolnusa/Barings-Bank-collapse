import pandas as pd
import numpy as np

class FeaturePipeline:
    def __init__(self, rogue_account_id: str = "88888"):
        self.rogue_account_id = rogue_account_id
        
    def build_features(self, trades: pd.DataFrame, positions: pd.DataFrame, cashflows: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates daily aggregated features across trades, positions, and cashflows for ML models.
        """
        if len(trades) == 0 or len(positions) == 0:
            return pd.DataFrame()
            
        trades['date'] = pd.to_datetime(trades['timestamp']).dt.date
        
        # Ensure enums/strings are correctly typed for matching
        if 'is_secret' not in trades.columns:
            trades['is_secret'] = trades['account_id'] == self.rogue_account_id
        
        # Self-settlement: front/back office separation failure (Nick Leeson ran both)
        trades['is_self_settled'] = (~trades['is_authorized']) & trades['is_settled']
        trades['volume'] = trades['quantity'] * trades['price']
        
        daily_trades = trades.groupby(['date', 'entity_id']).apply(
            lambda x: pd.Series({
                'total_volume': x['volume'].sum(),
                'secret_volume': x.loc[x['is_secret'], 'volume'].sum(),
                'self_settled_vol': x.loc[x['is_self_settled'], 'volume'].sum(),
            })
        ).reset_index()
        
        daily_trades['secret_account_ratio'] = daily_trades['secret_volume'] / daily_trades['total_volume'].clip(lower=1)
        daily_trades['self_settled_ratio'] = daily_trades['self_settled_vol'] / daily_trades['total_volume'].clip(lower=1)
        
        # 2. Position & P&L Features
        # positions['instrument'] might be enum or string. We check string representation.
        is_nikkei = positions['instrument'].astype(str).str.contains('NIKKEI')
        is_jgb = positions['instrument'].astype(str).str.contains('JGB')
        
        positions['nikkei_exposure'] = np.where(is_nikkei, positions['net_quantity'], 0)
        positions['jgb_exposure'] = np.where(is_jgb, positions['net_quantity'], 0)
        
        daily_pos = positions.groupby(['date', 'entity_id']).agg(
            total_pnl=('unrealized_pnl', 'sum'),
            total_nikkei_exposure=('nikkei_exposure', 'sum'),
            total_jgb_exposure=('jgb_exposure', 'sum') 
        ).reset_index()
        
        # Temporal position features
        daily_pos = daily_pos.sort_values(['entity_id', 'date'])
        daily_pos['exposure_growth_1d'] = daily_pos.groupby('entity_id')['total_nikkei_exposure'].diff().fillna(0)
        daily_pos['pnl_change_1d'] = daily_pos.groupby('entity_id')['total_pnl'].diff().fillna(0)
        
        # 3. Cashflow Features (Funding requests vs PnL mismatch)
        if len(cashflows) > 0:
            # cashflows flow_type can be enum or str
            is_funding = cashflows['flow_type'].astype(str).str.contains('FUNDING_REQUEST')
            cf_funding = cashflows[is_funding]
            if len(cf_funding) > 0:
                daily_cf = cf_funding.groupby(['date', 'entity_id'])['amount'].sum().reset_index()
                daily_cf.rename(columns={'amount': 'daily_funding_requested'}, inplace=True)
            else:
                daily_cf = pd.DataFrame(columns=['date', 'entity_id', 'daily_funding_requested'])
        else:
            daily_cf = pd.DataFrame(columns=['date', 'entity_id', 'daily_funding_requested'])
            
        # Merge all datasets into one analytical feature store table
        features = pd.merge(daily_pos, daily_trades, on=['date', 'entity_id'], how='left')
        features = pd.merge(features, daily_cf, on=['date', 'entity_id'], how='left')
        
        features.fillna({
            'daily_funding_requested': 0, 
            'secret_account_ratio': 0, 
            'self_settled_ratio': 0,
            'total_volume': 0,
            'secret_volume': 0,
            'self_settled_vol': 0
        }, inplace=True)
        
        # 4. Complex derived indicator
        # Mismatch: asking for massive funding while P&L change isn't explaining it
        features['funding_pnl_mismatch'] = features['daily_funding_requested'] - features['pnl_change_1d'].clip(upper=0).abs()
        
        return features
