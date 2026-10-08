import type { Prediction } from "../types";
import { formatIndiaTime } from "../utils/time";

function badgeClass(signal: Prediction["signal"]): string {
  if (signal === "BUY") return "badge buy";
  if (signal === "SELL") return "badge sell";
  return "badge wait";
}

function fmtPrice(v: number): string {
  if (v >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (v >= 1) return v.toFixed(4);
  return v.toFixed(6);
}

export default function SignalCard({ pred }: { pred: Prediction }) {
  const biasClass =
    pred.overall_bias === "BULLISH" ? "up" : pred.overall_bias === "BEARISH" ? "down" : "flat";
  return (
    <>
      <div className="grid">
        <div className="stat">
          <div className="label">Signal</div>
          <div className="value">
            <span className={badgeClass(pred.signal)}>{pred.signal}</span>
          </div>
        </div>
        <div className="stat">
          <div className="label">Bias</div>
          <div className={`value ${biasClass}`}>{pred.overall_bias}</div>
        </div>
        <div className="stat">
          <div className="label">Confidence</div>
          <div className="value">{(pred.confidence * 100).toFixed(1)}%</div>
        </div>
        <div className="stat">
          <div className="label">Regime</div>
          <div className="value" style={{ fontSize: 16 }}>
            {pred.regime}
          </div>
        </div>
        <div className="stat">
          <div className="label">Price</div>
          <div className="value">{fmtPrice(pred.current_price)}</div>
        </div>
        <div className="stat">
          <div className="label">ATR ({pred.timeframe})</div>
          <div className="value">{fmtPrice(pred.atr)}</div>
        </div>
      </div>
      <div className="panel" style={{ marginTop: 12 }}>
        <div className="muted">
          Model <span className="mono">{pred.model.model_id}</span> · status{" "}
          <span className="mono">{pred.model.status}</span> · {pred.model.n_features} features ·
          source <span className="mono">{pred.data_source}</span> · data as of{" "}
          <span className="mono">{formatIndiaTime(pred.data_timestamp)}</span> · {pred.latency_ms} ms
        </div>
        {pred.risk && (
          <div className="muted" style={{ marginTop: 6 }}>
            Risk: entry <span className="mono">{String(pred.risk.entry ?? "-")}</span> · stop{" "}
            <span className="mono">{String(pred.risk.stop_loss ?? "-")}</span> · target{" "}
            <span className="mono">{String(pred.risk.take_profit ?? "-")}</span> · R:R{" "}
            <span className="mono">{String(pred.risk.risk_reward ?? "-")}</span> · size{" "}
            <span className="mono">{String(pred.risk.position_size_units ?? "-")}</span>
            {pred.risk.actionable === false && " · not actionable (WAIT)"}
          </div>
        )}
        {pred.signal_detail.scalp && (
          <div className="muted" style={{ marginTop: 6 }}>
            1m scalp setup · near-term agreement{" "}
            {pred.signal_detail.directional_agreement == null
              ? "-"
              : `${(pred.signal_detail.directional_agreement * 100).toFixed(0)}%`}
            {pred.signal_detail.reasons?.length ? (
              <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
                {pred.signal_detail.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            ) : (
              " · entry levels are advisory; verify spread and execution price before trading"
            )}
          </div>
        )}
      </div>
    </>
  );
}
