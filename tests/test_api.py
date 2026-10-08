import pytest
from fastapi.testclient import TestClient


def _use_local_test_feed(monkeypatch):
    from app.data import storage
    from app.data.base import DataProviderFactory

    provider = DataProviderFactory.get("crypto")
    monkeypatch.setattr(
        provider,
        "get_live_data",
        lambda symbol, timeframe: storage.load_candles("crypto", symbol, timeframe),
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.delenv("API_KEY", raising=False)
    from app.database import session as db_session

    db_session.get_database_url.cache_clear()
    db_session.get_engine.cache_clear()

    from main import create_app

    return TestClient(create_app())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "versions" in body


def test_assets(client):
    r = client.get("/assets")
    assert r.status_code == 200
    body = r.json()
    assert "BTCUSDT" in body["assets"]["crypto"]
    assert "EURUSD" in body["assets"]["forex"]
    assert "XAUUSD" in body["assets"]["commodities"]


def test_timeframes(client):
    r = client.get("/timeframes")
    assert r.status_code == 200
    names = [t["name"] for t in r.json()["timeframes"]]
    assert names == ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
    assert r.json()["multi_timeframe_context"]["1m"] == ["5m", "15m", "1h"]


def test_predict_rejects_unknown_market(client):
    r = client.post("/predict", json={"market": "stocks", "symbol": "AAPL", "timeframe": "1m"})
    assert r.status_code == 422  # pydantic literal validation


def test_predict_without_model_returns_400(client, tmp_storage, synthetic_candles):
    r = client.post("/predict", json={"market": "crypto", "symbol": "BTCUSDT", "timeframe": "1m"})
    assert r.status_code == 400
    body = r.json()
    assert body.get("error") == "ModelNotFoundError" or "scripts.train" in body.get("detail", "")


def test_predict_with_trained_model(client, trained_system, monkeypatch):
    _use_local_test_feed(monkeypatch)
    r = client.post("/predict", json={"market": "crypto", "symbol": "BTCUSDT", "timeframe": "1m"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["signal"] in {"BUY", "SELL", "WAIT"}
    assert len(body["forecast"]) == 10
    assert body["model"]["model_id"]


def test_train_endpoint_returns_live_progress_and_report(client, monkeypatch):
    from app.api import routes_training
    from types import SimpleNamespace

    def fake_train(*_args, progress_callback=None, **_kwargs):
        progress_callback({
            "percentage": 60,
            "phase": "walk_forward",
            "message": "Fitted horizon 6/10",
            "completed": 6,
            "total": 10,
        })
        return SimpleNamespace(to_dict=lambda: {"model_id": "test-model"})

    monkeypatch.setattr(routes_training, "train", fake_train)
    response = client.post(
        "/train",
        json={"market": "forex", "symbol": "EURUSD", "timeframe": "1m"},
    )
    assert response.status_code == 202
    training_id = response.json()["training_id"]

    result = client.get(f"/train/{training_id}")
    assert result.status_code == 200
    job = result.json()
    assert job["status"] == "succeeded"
    assert job["progress"]["percentage"] == 100
    assert job["report"]["model_id"] == "test-model"


def test_backtest_endpoint(client, trained_system):
    r = client.post(
        "/backtest",
        json={"market": "crypto", "symbol": "BTCUSDT", "timeframe": "1m",
              "horizon": 1, "min_probability": 0.3},
    )
    assert r.status_code == 200
    assert "trading_metrics" in r.json()


def test_model_status_endpoint(client, tmp_storage, synthetic_candles):
    r = client.get("/model-status?market=crypto&symbol=BTCUSDT&timeframe=1m")
    assert r.status_code == 200
    assert r.json()["trained"] is False


def test_models_list_endpoint(client, trained_system):
    r = client.get("/models")
    assert r.status_code == 200
    rows = r.json()["models"]
    assert any(m["symbol"] == "BTCUSDT" for m in rows)


def test_api_key_enforcement(tmp_path, monkeypatch, trained_system):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'api2.db'}")
    monkeypatch.setenv("API_KEY", "secret123")
    from app.database import session as db_session

    db_session.get_database_url.cache_clear()
    db_session.get_engine.cache_clear()
    from main import create_app

    _use_local_test_feed(monkeypatch)
    c = TestClient(create_app())
    r = c.post("/predict", json={"market": "crypto", "symbol": "BTCUSDT", "timeframe": "1m"})
    assert r.status_code == 401
    r = c.post(
        "/predict",
        json={"market": "crypto", "symbol": "BTCUSDT", "timeframe": "1m"},
        headers={"X-API-Key": "secret123"},
    )
    assert r.status_code == 200, r.text
    # health stays open
    assert c.get("/health").status_code == 200
