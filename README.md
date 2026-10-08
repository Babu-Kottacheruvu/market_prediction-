# Global Market Prediction Platform

Predict the **next 10 candles** (direction, probability, expected return / high / low /
volatility / confidence) for **Forex, Crypto and Commodities** with XGBoost and
walk-forward validation.

**Prediction and signals only — the platform never places, routes or executes orders.**

---

## 1. What this is

| Layer | Implementation |
|---|---|
| Models | XGBoost classifier (direction) + regressor (return) per horizon, 10 horizons |
| Validation | Walk-forward (train → validation → test → roll forward), embargo of `horizon × timeframe` between train and test |
| Features | ~108 columns: TA (RSI/MACD/ATR/BB/Stoch/ADX/CCI…), regime, higher-timeframe context joined on **closed candles only**, optional cross-market/peer features |
| Targets | Forward 10-candle direction (`-1/0/1`) with ATR-based dead-zone, forward return/high/low/volatility |
| Data | **MetaTrader 5 as the single source** (`provider: mt5` for forex/crypto/commodities); Binance adapter retained as an optional alternative |
| API | FastAPI (`/predict`, `/backtest`, `/train`, `/model-status`, `/models`, `/assets`, `/timeframes`, `/health`) |
| UI | React + TypeScript (Vite) served from `/` by the API |
| CLI | `scripts/train.py`, `predict.py`, `backtest.py`, `download_data.py`, `build_features.py`, `monitor.py` |

### No-look-ahead guarantees

* `timestamp` = candle **open** time (tz-aware UTC); every candle carries
  `available_at = timestamp + timeframe` (close time).
* Higher-timeframe context is joined with `merge_asof(left_on="timestamp",
  right_on="available_at", direction="backward")` → only **closed** HTF candles.
* Targets are built with a forward roll (`shift(-1)` + reversed rolling), never a
  backward-looking window.
* Target columns are prefixed `target_` and excluded from `feature_columns()`;
  `assert_no_target_leakage()` runs at build time, and `tests/test_no_lookahead.py`
  verifies prefix-invariance and closed-candle invariants.
* Walk-forward validation windows now drive per-horizon XGBoost early stopping;
  the median best iteration across folds is used for the final fit. Models must
  pass existing OOS accuracy/sample gates and beat both a fold-majority direction
  baseline and a zero-return price baseline before activation. Training reports
  include per-horizon and aggregate baseline-relative scores.

---

## 2. Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt     # includes MetaTrader5 (Windows only)

cd frontend
npm install          # Node 18+
npm run build        # produces frontend/dist, served at /
cd ..
```

Data source: **MetaTrader 5 terminal** (installed + logged in; the API attaches
to the running terminal). Credentials, if needed, live in `.env`
(`MT5_TERMINAL_PATH/LOGIN/PASSWORD/SERVER`) — see `.env.example`.

Optional Docker:

```powershell
docker compose up --build     # API on http://localhost:8000
```

---

## 3. Quickstart (real commands)

```powershell
# 0) MetaTrader 5 must be installed, running and logged in (Windows only).
#    Optional env (.env): MT5_TERMINAL_PATH, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

# 1) data — pulled from the MT5 terminal (all markets)
python -m scripts.download_data --market forex --symbol EURUSD --timeframe 1h --bars 8760
python -m scripts.download_data --market commodities --symbol XAUUSD --timeframe 1h --bars 8760
python -m scripts.download_data --market crypto --symbol BTCUSDT --timeframe 1h --bars 1000

# 2) train (walk-forward, embargoed, acceptance-gated)
python -m scripts.train --market forex --symbol EURUSD --timeframe 1h

# 3) predict the next 10 candles
python -m scripts.predict --market crypto --symbol BTCUSDT --timeframe 1m

# 4) backtest the stored out-of-sample predictions (with and without costs)
python -m scripts.backtest --market crypto --symbol BTCUSDT --timeframe 1m

# 5) drift monitoring since the training window
python -m scripts.monitor --market crypto --symbol BTCUSDT --timeframe 1m

# 6) API + UI
python main.py                       # http://127.0.0.1:8000
# dev UI with hot reload:  cd frontend; npm run dev   (proxies to :8000)

# tests
python -m pytest tests/ -q
```

**Live predictions:** each prediction fetches the selected symbol and timeframe
from MT5 before building the forecast. If the terminal cannot provide fresh data,
the prediction fails with an actionable error instead of returning a successful
forecast from stale local candles. Historical/training reads may still serve local
candles and refresh them when older than `MT5_STALE_CANDLES` (default 3 candle
durations); set `MT5_REFRESH=0` to disable that background refresh. MT5 downloads
are incremental (append + dedupe), never a full overwrite.
The UI displays market, forecast, and training timestamps in India Standard Time
(IST, `Asia/Kolkata`); API and stored candle timestamps remain UTC.

**1-minute scalp signals:** BUY/SELL is emitted only for an active (acceptance-
passed) model, fresh non-stored candle data, a trend regime, sufficient horizon-1
probability/confidence and expected return over configured round-trip costs, plus
agreement from at least two of the first three forecast candles. Otherwise the
signal is WAIT and the UI lists the blocking reasons. The displayed entry, stop,
and target are advisory levels using the 1-minute ATR settings; verify current
spread and execution price yourself. This is not an execution service and does
not guarantee profitable or accurate trades. Candidate models must pass training
acceptance before they can emit a 1-minute scalp entry.

When an MT5 date-range request returns no bars, the provider retries with the
terminal's latest-bars API in batches of at most 10,000 bars (up to
`MT5_MAX_BARS`, default 100,000 total). Training can only use history the broker
supplies; if the requested window is still empty or too short, check Market Watch
visibility and the terminal's **Max bars in chart** setting before retrying.

Symbols are broker-dependent: symbols your broker does not offer fail with an
actionable error listing close candidates — no substitute symbol is ever
invented. Drop a CSV/Parquet export at `data/raw/{market}/{SYMBOL}/{timeframe}.parquet`
and use `provider: csv` if you have no MT5 terminal.

---

## 4. Results measured on real data (2026-10-06)

Everything below was produced by the commands in §3 on **data pulled from the
MetaTrader 5 terminal** (MetaQuotes-Demo account, real EURUSD/XAUUSD candles) or
earlier from Binance public klines (BTCUSDT, before MT5 became the source).
Nothing here is simulated or copied from a paper.

| Check | Result |
|---|---|
| MT5 pull (forex) | OK — EURUSD 1h: **6,210 candles** (2025-10-06 → 2026-10-06), weekend gaps flagged as warnings only |
| MT5 pull (commodities) | OK — XAUUSD 1h: **5,781 candles** |
| MT5 pull (crypto) | **Not offered by this broker** — `BTCUSDT` fails with an actionable error listing the broker's `BTC*` symbols (never a fabricated substitute) |
| Train EURUSD 1h | OK — **full walk-forward, 5 folds** (180/30/30, step 30), 100 models, 509 s |
| Acceptance gate (EURUSD) | **FAILED**: mean OOS directional accuracy **0.380** < 0.50 → model saved as **`candidate`**, not `active` |
| Predict EURUSD 1h | OK — price 1.12668, `signal=WAIT`, `confidence=0.444`, model status `candidate` |
| Backtest EURUSD 1h (OOS May→Oct 2026) | 250 trades, gross **+0.29 %**, after costs **−2.19 %** → `profitable_after_costs: false` |
| Monitor EURUSD 1h | `insufficient_recent_data` (training used the full stored window) — correct, no fabricated drift numbers |
| Train BTCUSDT 1m (earlier, Binance data) | 90,000 candles, reduced single split (62 d < 180/30/30), OOS accuracy **0.383** < 0.50 → `candidate` |
| Backtest BTCUSDT 1m | 15 trades, gross +0.15 %, after costs −0.97 % → `profitable_after_costs: false` |
| API + UI | `/health`, `/` (React), `/predict`, `/model-status` all 200 on a live server |
| Tests | **94 passed** (incl. 12 MT5 provider tests against a fake terminal) |

**Interpretation (honest):** the pipeline works end to end and refuses to lie.
The baseline model does **not** beat the acceptance gate on 1-minute BTCUSDT and its
edge does not survive round-trip costs at that timeframe — so it stays a `candidate`.
Typical levers: longer/larger history, 5m–1h timeframes (fewer, larger moves per unit
cost), richer features (funding/OI/on-chain extras), and horizon-specific thresholds
(`config/risk.yaml`). Metrics from synthetic unit-test data carry no market claim.

---

## 5. Configuration

| File | Purpose |
|---|---|
| `config/markets.yaml` | enabled symbols, costs per market/timeframe, 1m signal overrides |
| `config/models.yaml` | forecast horizon, walk-forward windows, XGBoost params, calibration, acceptance gates, monitoring thresholds |
| `config/timeframes.yaml` | timeframes + higher-timeframe context per timeframe |
| `config/features.yaml` | feature toggles and versions |
| `config/risk.yaml` | signal thresholds, R:R, position sizing caps |
| `config/assets.yaml` | default UI asset list, enabled timeframes |

Key rules: probability columns are always ordered `[P(DOWN), P(FLAT), P(UP)]`;
targets are stored as `-1/0/1` and shifted to `0/1/2` only inside the trainer;
a model is promoted to `active` only if `check_acceptance()` passes.

Optional probability calibration (`config/models.yaml → calibration.enabled`) fits
Platt/isotonic calibrators on the **first half** of OOS predictions and scores them
on the **second half**; calibrators are stored with the package and applied at
predict time.

---

## 6. Extending

* **New broker/data source** — implement `download()` in `app/data/` and register
  it in `DataProviderFactory` (`app/data/base.py`). `app/data/mt5_provider.py`
  shows the full pattern: connect → resolve symbol → fetch rates → normalize →
  validate → store incrementally. Binance (`app/data/crypto/binance.py`) remains
  available as `provider: binance`. Credentials come from environment variables
  only.
* **New feature** — add it in `app/features/`, keep the `target_` prefix reserved,
  add a no-look-ahead test, bump `feature_version`.
* **New asset** — add symbol/costs to `config/markets.yaml`.

### Self-learning loop (design)

1. `scripts.monitor` compares recent data to the training window (PSI per feature,
   target-share PSI) and reports `drift_detected` when thresholds are crossed.
2. On drift (or a schedule), re-run `scripts.train` — walk-forward evaluation and
   acceptance gates decide promotion; the registry keeps the previous model.
3. `scripts.backtest` on the new OOS window compares the candidate against the
   deployed one; backtest results are stored and the API exposes both.
4. Nothing is promoted automatically in this version: drift and gate results are
   surfaced for an operator to trigger retraining.

---

## 7. API sketch

```http
GET  /health                                  # status + library versions
GET  /assets | /timeframes                    # what the UI can select
GET  /model-status?market=&symbol=&timeframe= # trained? status? window?
POST /predict   {market,symbol,timeframe,equity?}
POST /backtest  {market,symbol,timeframe,horizon,min_probability,min_confidence}
POST /train     {market,symbol,timeframe,train_days?,val_days?,test_days?,step_days?}
GET  /models                                  # registry listing
POST /models/{id}/activate | /deactivate
```

Auth: set `API_KEY` in the environment to require `X-API-Key` on mutating endpoints;
`RATE_LIMIT_PER_MINUTE` (default 120) rate-limits per client IP. Errors are JSON:
`{"error": "ModelNotFoundError", "detail": "...run scripts.train ..."}`.

---

## 8. Project layout

```
app/
  core/ config, logging, exceptions
  data/ timeframes, normalizer, validator, storage, mt5_provider (default source), providers (binance/file)
  features/ technical, common, multi_timeframe, forex, crypto, commodities, cross_market, builder
  targets/ ten_candle.py (ATR dead-zone direction + forward windows)
  training/ walk_forward, trainer, evaluation, acceptance
  models/ xgboost_model, registry, router, calibration
  prediction/ predictor, forecast, confidence
  signals/ engine, risk, regime
  backtesting/ engine, costs, metrics
  monitoring/ drift.py (PSI feature/target drift)
  api/ routes + schemas + deps
  database/ sqlite persistence of predictions/backtests
scripts/ download_data, build_features, train, predict, backtest, monitor
tests/ 82 tests incl. mandatory no-look-ahead suite
frontend/ React + TypeScript (Vite)
config/ yaml configuration
```

---

## 9. Disclaimer

Outputs are statistical estimates, not financial advice. Probabilities are model
outputs that can and do drift; backtest results are costs- and threshold-sensitive.
There is no trade execution anywhere in this codebase. Disable outbound network
access at the infrastructure level if your environment requires it.
