import { useEffect, useState } from "react";
import { api } from "../api";
import type { Market, ModelStatus, TrainingProgress, TrainingReport } from "../types";
import { formatIndiaTime } from "../utils/time";

interface Props {
  market: Market;
  symbol: string;
  timeframe: string;
}

export default function TrainingView({ market, symbol, timeframe }: Props) {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [report, setReport] = useState<TrainingReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<TrainingProgress | null>(null);

  useEffect(() => {
    setReport(null);
    setError(null);
    api
      .modelStatus(market, symbol, timeframe)
      .then(setStatus)
      .catch((e: Error) => setError(e.message));
  }, [market, symbol, timeframe, report]);

  const train = async () => {
    setLoading(true);
    setError(null);
    setProgress({ percentage: 0, phase: "queued", message: "Starting training job" });
    try {
      const res = await api.train(market, symbol, timeframe, setProgress);
      setReport(res);
      setStatus(await api.modelStatus(market, symbol, timeframe));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const agg = report?.out_of_sample_metrics.aggregate ?? {};
  const accuracyLift = agg.mean_accuracy_lift_vs_baseline;
  const priceMaeImprovement = agg.mean_mae_improvement_vs_baseline;

  return (
    <>
      <div className="panel">
        <div className="controls">
          <button className="primary" onClick={() => void train()} disabled={loading}>
            {loading ? "Training… (walk-forward)" : "Train model"}
          </button>
          <span className="muted">
            Walk-forward XGBoost training can take several minutes. Acceptance gates decide whether
            the model becomes active.
          </span>
        </div>
        {error && (
          <div className="error" style={{ marginTop: 10 }}>
            {error}
          </div>
        )}
        {loading && progress && (
          <div className="training-progress" aria-live="polite">
            <div className="training-progress-heading">
              <div>
                <span className="training-pulse" aria-hidden="true" />
                <span className="training-phase">{progress.phase.replace(/_/g, " ")}</span>
              </div>
              <span className="training-percent">{Math.round(progress.percentage)}%</span>
            </div>
            <div
              className="training-progress-track"
              role="progressbar"
              aria-label="Model training completion"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(progress.percentage)}
            >
              <div
                className="training-progress-fill"
                style={{ width: `${Math.min(100, Math.max(0, progress.percentage))}%` }}
              />
            </div>
            <div className="training-progress-caption">
              <span>{progress.message}</span>
              {progress.completed != null && progress.total != null && (
                <span>
                  {progress.completed} / {progress.total}
                </span>
              )}
            </div>
            <div className="muted">Training completion is not a measure of prediction accuracy.</div>
          </div>
        )}
      </div>

      <div className="panel">
        <div className="label muted">Current model status</div>
        {!status?.trained ? (
          <div className="muted" style={{ marginTop: 6 }}>
            No trained model for {symbol} {timeframe} yet.
          </div>
        ) : (
          <div className="muted" style={{ marginTop: 6 }}>
            <span className="mono">{status.model_id}</span> · {status.status} ·{" "}
            {status.n_rows?.toLocaleString()} rows · {status.n_features} features · trained{" "}
            {status.training_start ? formatIndiaTime(status.training_start) : "n/a"} →{" "}
            {status.training_end ? formatIndiaTime(status.training_end) : "n/a"}
          </div>
        )}
      </div>

      {report && (
        <>
          <div className="grid">
            <div className="stat">
              <div className="label">Status</div>
              <div className="value" style={{ fontSize: 18 }}>
                {report.status}
              </div>
            </div>
            <div className="stat">
              <div className="label">OOS accuracy (mean)</div>
              <div className="value">
                {agg.mean_accuracy != null ? Number(agg.mean_accuracy).toFixed(4) : "n/a"}
              </div>
            </div>
            <div className="stat">
              <div className="label">Brier (mean)</div>
              <div className="value">
                {agg.mean_brier != null ? Number(agg.mean_brier).toFixed(4) : "n/a"}
              </div>
            </div>
            <div className="stat">
              <div className="label">Accuracy lift vs baseline</div>
              <div className="value">
                {accuracyLift != null
                  ? `${Number(accuracyLift) >= 0 ? "+" : ""}${(Number(accuracyLift) * 100).toFixed(2)} pp`
                  : "n/a"}
              </div>
            </div>
            <div className="stat">
              <div className="label">Price MAE improvement</div>
              <div className="value">
                {priceMaeImprovement != null
                  ? `${Number(priceMaeImprovement) >= 0 ? "+" : ""}${(Number(priceMaeImprovement) * 10000).toFixed(2)} bps`
                  : "n/a"}
              </div>
            </div>
            <div className="stat">
              <div className="label">Acceptance</div>
              <div className="value" style={{ fontSize: 18, color: report.acceptance.passed ? "var(--up)" : "var(--down)" }}>
                {report.acceptance.passed ? "PASSED" : "FAILED"}
              </div>
            </div>
          </div>
          <div className="panel" style={{ marginTop: 12 }}>
            <div className="muted" style={{ marginBottom: 10 }}>
              Out-of-sample accuracy lift compares with a train-fold majority-direction baseline;
              price MAE improvement compares with predicting no price change. Positive values are
              better. These are historical test metrics, not guarantees.
            </div>
            <table className="forecast">
              <thead>
                <tr>
                  <th>Fold</th>
                  <th>Train rows</th>
                  <th>Test rows</th>
                  <th>H1 accuracy</th>
                </tr>
              </thead>
              <tbody>
                {report.folds.map((f) => (
                  <tr key={f.fold}>
                    <td>{f.fold}</td>
                    <td>{f.train_rows.toLocaleString()}</td>
                    <td>{f.test_rows.toLocaleString()}</td>
                    <td>{f.h1_accuracy != null ? f.h1_accuracy.toFixed(4) : "n/a"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {report.acceptance.reasons.length > 0 && (
              <div className="error" style={{ marginTop: 8 }}>
                {report.acceptance.reasons.join("; ")}
              </div>
            )}
          </div>
        </>
      )}
    </>
  );
}
