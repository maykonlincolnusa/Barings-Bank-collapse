import pytest
from fastapi.testclient import TestClient
import pandas as pd

from app.api.main import app
from app.simulation.engine import BaringsSimulationEngine, SimulationConfig
from app.features.engine import build_features

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "total_alerts" in response.json()

def test_simulation_engine_healthy():
    engine = BaringsSimulationEngine()
    config = SimulationConfig(scenario="healthy", days=10, seed=42)
    result = engine.simulate(config)
    
    assert result.scenario == "healthy"
    assert len(result.trades) > 0
    
def test_simulation_engine_rogue():
    engine = BaringsSimulationEngine()
    config = SimulationConfig(scenario="rogue_trader", days=10, seed=42)
    result = engine.simulate(config)
    
    assert result.scenario == "rogue_trader"
    assert any("88888" in trade["account_id"] for trade in result.trades), "Expected hidden trades in account 88888"

def test_feature_engineering():
    engine = BaringsSimulationEngine()
    config = SimulationConfig(scenario="rogue_trader", days=10, seed=42)
    result = engine.simulate(config)
    
    features = build_features(result, persist=False)
    
    assert not features.empty
    assert "fraud_label" in features.columns
    assert "secret_account_fraction" in features.columns
