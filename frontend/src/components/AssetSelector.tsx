import { useEffect, useState } from "react";
import { api } from "../api";
import type { Market } from "../types";

interface Props {
  market: Market;
  symbol: string;
  timeframe: string;
  onChange: (next: { market?: Market; symbol?: string; timeframe?: string }) => void;
}

export default function AssetSelector({ market, symbol, timeframe, onChange }: Props) {
  const [assets, setAssets] = useState<Record<string, string[]>>({});
  const [timeframes, setTimeframes] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .assets()
      .then((res) => {
        if (alive) setAssets(res.assets);
      })
      .catch((e: Error) => {
        if (alive) setError(e.message);
      });
    fetch("/timeframes")
      .then((r) => r.json())
      .then((data: { enabled?: string[] }) => {
        if (alive && Array.isArray(data.enabled)) setTimeframes(data.enabled);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  const symbols = assets[market] ?? [];
  const tfList = timeframes.length ? timeframes : ["1m", "5m", "15m", "1h"];

  return (
    <div className="panel">
      <div className="controls">
        <label className="field">
          Market
          <select
            value={market}
            onChange={(e) => onChange({ market: e.target.value as Market, symbol: undefined })}
          >
            <option value="crypto">Crypto</option>
            <option value="forex">Forex</option>
            <option value="commodities">Commodities</option>
          </select>
        </label>
        <label className="field">
          Symbol
          <select value={symbol} onChange={(e) => onChange({ symbol: e.target.value })}>
            {symbols.length === 0 && <option value={symbol}>{symbol}</option>}
            {symbols.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Timeframe
          <select value={timeframe} onChange={(e) => onChange({ timeframe: e.target.value })}>
            {tfList.map((tf) => (
              <option key={tf} value={tf}>
                {tf}
              </option>
            ))}
          </select>
        </label>
        {error && <span className="error">{error}</span>}
      </div>
    </div>
  );
}
