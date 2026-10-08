import { useState } from "react";
import { api } from "../api";
import type { Market } from "../types";

interface Props {
  market: Market;
  symbol: string;
  timeframe: string;
}

export default function BacktestView({ market, symbol, timeframe }: Props) {
  const [horizon, setHorizon] = useState(1);
  const [minProb, setMinProb] = useState(0.65);
  const [minConf, setMinConf] = useState(0.55);
  const [result, setResult] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.backtest(market, symbol, timeframe, horizon, minProb, minConf);
      setResult(res);
    } catch (e) {
      setError((e as Error).message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="panel">
        <div className="controls">
          <label className="field">
            Horizon (candles)
            <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
              {Array.from({ length: 10 }, (_, i) => i + 1).map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Min probability
            <input
              value={minProb}
              onChange={(e) => setMinProb(Number(e.target.value))}
              inputMode="decimal"
            />
          </label>
          <label className="field">
            Min confidence
            <input
              value={minConf}
              onChange={(e) => setMinConf(Number(e.target.value))}
              inputMode="decimal"
            />
          </label>
          <button className="primary" onClick={() => void run()} disabled={loading}>
            {loading ? "Running…" : "Run backtest"}
          </button>
        </div>
        <div className="muted" style={{ marginTop: 8 }}>
          Runs on stored out-of-sample predictions only — never on training rows.
        </div>
        {error && (
          <div className="error" style={{ marginTop: 10 }}>
            {error}
          </div>
        )}
      </div>
      {result && <pre className="json">{JSON.stringify(result, null, 2)}</pre>}
    </>
  );
}
