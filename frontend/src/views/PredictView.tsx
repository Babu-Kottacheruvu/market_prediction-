import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { Market, Prediction } from "../types";
import ForecastTable from "../components/ForecastTable";
import SignalCard from "../components/SignalCard";

interface Props {
  market: Market;
  symbol: string;
  timeframe: string;
  onOpenTraining: () => void;
}

export default function PredictView({ market, symbol, timeframe, onOpenTraining }: Props) {
  const [pred, setPred] = useState<Prediction | null>(null);
  const [equity, setEquity] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelAvailable, setModelAvailable] = useState<boolean | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const eq = equity ? Number(equity) : undefined;
      const result = await api.predict(market, symbol, timeframe, eq);
      setPred(result);
    } catch (e) {
      setError((e as Error).message);
      setPred(null);
    } finally {
      setLoading(false);
    }
  }, [market, symbol, timeframe, equity]);

  // auto-predict when the selection changes and a model exists
  useEffect(() => {
    setPred(null);
    setError(null);
    setModelAvailable(null);
    api
      .modelStatus(market, symbol, timeframe)
      .then((st) => {
        setModelAvailable(st.trained);
        if (st.trained) void run();
      })
      .catch((e: Error) => setError(e.message));
  }, [market, symbol, timeframe, run]);

  const needsTraining = modelAvailable === false || error?.includes("No model directory") === true;

  return (
    <>
      <div className="panel">
        <div className="controls">
          <label className="field">
            Equity (optional, for position sizing)
            <input
              value={equity}
              onChange={(e) => setEquity(e.target.value)}
              placeholder="10000"
              inputMode="decimal"
            />
          </label>
          <button className="primary" onClick={() => void run()} disabled={loading}>
            {loading ? "Predicting…" : "Predict next 10 candles"}
          </button>
          <span className="muted">Prediction only — no orders are ever placed.</span>
        </div>
        {error && (
          <div className="error" style={{ marginTop: 10 }}>
            {error}
          </div>
        )}
        {needsTraining && (
          <div className="muted" style={{ marginTop: 10 }}>
            No model is trained for {symbol} on {timeframe}. Train this exact combination before
            requesting a forecast; training may take several minutes and does not guarantee
            predictive accuracy.{" "}
            <button className="primary" onClick={onOpenTraining}>
              Train {symbol} {timeframe}
            </button>
          </div>
        )}
      </div>

      {pred && <SignalCard pred={pred} />}
      {pred && <ForecastTable rows={pred.forecast} />}
      {pred && (
        <div className="panel">
          <div className="muted">
            Overall agreement {(pred.agreement * 100).toFixed(1)}% · signal probability{" "}
            {(pred.signal_detail.probability * 100).toFixed(1)}% · probabilities are model outputs,
            not guarantees.
          </div>
        </div>
      )}
    </>
  );
}
