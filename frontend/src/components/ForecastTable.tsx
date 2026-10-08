import type { CSSProperties } from "react";
import type { ForecastRow } from "../types";
import { formatIndiaTime } from "../utils/time";

function pct(v: number): string {
  return (v * 100).toFixed(1) + "%";
}

function fmt(v: number, digits = 4): string {
  return v.toFixed(digits);
}

export default function ForecastTable({ rows }: { rows: ForecastRow[] }) {
  return (
    <div className="panel">
      <div className="label muted" style={{ marginBottom: 8 }}>
        Next 10 candles — direction probabilities, expected range and confidence
      </div>
      <table className="forecast">
        <thead>
          <tr>
            <th>H</th>
            <th>Candle time (IST)</th>
            <th>Direction</th>
            <th>Prob</th>
            <th>P(UP)</th>
            <th>P(FLAT)</th>
            <th>P(DOWN)</th>
            <th>Est price</th>
            <th>Exp ret</th>
            <th>High</th>
            <th>Low</th>
            <th>Conf</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.horizon}>
              <td>{r.horizon}</td>
              <td>{formatIndiaTime(r.candle_time)}</td>
              <td
                style={{
                  color:
                    r.direction === "UP"
                      ? "var(--up)"
                      : r.direction === "DOWN"
                        ? "var(--down)"
                        : "var(--flat)",
                }}
              >
                {r.direction}
              </td>
              <td colSpan={1}>
                <div
                  className="prob-bar"
                  style={
                    {
                      "--down-pct": `${r.probability_down * 100}%`,
                      "--flat-pct": `${(r.probability_down + r.probability_flat) * 100}%`,
                    } as CSSProperties
                  }
                />
              </td>
              <td>{pct(r.probability_up)}</td>
              <td>{pct(r.probability_flat)}</td>
              <td>{pct(r.probability_down)}</td>
              <td>{fmt(r.expected_price)}</td>
              <td>{(r.expected_return * 100).toFixed(3)}%</td>
              <td>{fmt(r.expected_high)}</td>
              <td>{fmt(r.expected_low)}</td>
              <td>{pct(r.confidence)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
