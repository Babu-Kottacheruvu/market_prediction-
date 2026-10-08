import { useEffect, useState } from "react";
import AssetSelector from "./components/AssetSelector";
import BacktestView from "./views/BacktestView";
import PredictView from "./views/PredictView";
import TrainingView from "./views/TrainingView";
import type { Market } from "./types";
import { api } from "./api";

type Tab = "predict" | "backtest" | "training";

export default function App() {
  const [tab, setTab] = useState<Tab>("predict");
  const [market, setMarket] = useState<Market>("crypto");
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [timeframe, setTimeframe] = useState("1m");
  const [health, setHealth] = useState<string>("…");

  useEffect(() => {
    api
      .health()
      .then((h) => setHealth(`API ok · xgboost ${h.versions.xgboost ?? "?"}`))
      .catch(() => setHealth("API offline — start with: python main.py"));
  }, []);

  return (
    <>
      <header className="topbar">
        <h1>MARKET PREDICTION PLATFORM</h1>
        <span className="sub">{health}</span>
      </header>

      <AssetSelector
        market={market}
        symbol={symbol}
        timeframe={timeframe}
        onChange={(next) => {
          if (next.market) {
            setMarket(next.market);
            setSymbol(next.symbol ?? defaultsFor(next.market).symbol);
          }
          if (next.symbol) setSymbol(next.symbol);
          if (next.timeframe) setTimeframe(next.timeframe);
        }}
      />

      <nav className="tabs">
        {(["predict", "backtest", "training"] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </nav>

      {tab === "predict" && (
        <PredictView
          market={market}
          symbol={symbol}
          timeframe={timeframe}
          onOpenTraining={() => setTab("training")}
        />
      )}
      {tab === "backtest" && (
        <BacktestView market={market} symbol={symbol} timeframe={timeframe} />
      )}
      {tab === "training" && (
        <TrainingView market={market} symbol={symbol} timeframe={timeframe} />
      )}

      <div className="footer">
        Predictions are statistical estimates from an XGBoost model with walk-forward validation.
        They are not financial advice and no orders are ever executed. Disable direct network
        access at the infrastructure level if your environment requires it.
      </div>
    </>
  );
}

function defaultsFor(market: Market): { symbol: string } {
  if (market === "crypto") return { symbol: "BTCUSDT" };
  if (market === "forex") return { symbol: "EURUSD" };
  return { symbol: "XAUUSD" };
}
