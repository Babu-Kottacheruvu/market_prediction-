import type {
  BacktestResult,
  Market,
  ModelStatus,
  Prediction,
  TrainingProgress,
  TrainingReport,
} from "./types";

const BASE: string = (import.meta as unknown as { env: Record<string, string> }).env
  ?.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init?.headers as Record<string, string> | undefined),
  };
  const apiKey = localStorage.getItem("gmp_api_key");
  if (apiKey) headers["X-API-KEY"] = apiKey;

  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { detail: text };
  }
  if (!res.ok) {
    const detail =
      (body as { detail?: string })?.detail ??
      (body as { error?: string })?.error ??
      `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return body as T;
}

export const api = {
  health: () => request<{ status: string; versions: Record<string, string> }>("/health"),
  assets: () =>
    request<{ assets: Record<string, string[]>; stored_data: unknown[]; mvp: string[] }>(
      "/assets",
    ),
  modelStatus: (market: Market, symbol: string, timeframe: string) =>
    request<ModelStatus>(
      `/model-status?market=${market}&symbol=${encodeURIComponent(symbol)}&timeframe=${timeframe}`,
    ),
  predict: (market: Market, symbol: string, timeframe: string, equity?: number) =>
    request<Prediction>("/predict", {
      method: "POST",
      body: JSON.stringify({ market, symbol, timeframe, equity: equity || null }),
    }),
  backtest: (
    market: Market,
    symbol: string,
    timeframe: string,
    horizon: number,
    minProbability: number,
    minConfidence: number,
  ) =>
    request<BacktestResult>("/backtest", {
      method: "POST",
      body: JSON.stringify({
        market,
        symbol,
        timeframe,
        horizon,
        min_probability: minProbability,
        min_confidence: minConfidence,
      }),
    }),
  train: async (
    market: Market,
    symbol: string,
    timeframe: string,
    onProgress: (progress: TrainingProgress) => void,
  ): Promise<TrainingReport> => {
    const started = await request<{ training_id: string }>("/train", {
      method: "POST",
      body: JSON.stringify({ market, symbol, timeframe }),
    });
    while (true) {
      await new Promise((resolve) => window.setTimeout(resolve, 750));
      const job = await request<{
        status: "queued" | "running" | "succeeded" | "failed";
        progress: TrainingProgress;
        report: TrainingReport | null;
        error: string | null;
      }>(`/train/${encodeURIComponent(started.training_id)}`);
      onProgress(job.progress);
      if (job.status === "failed") throw new Error(job.error ?? "Training failed");
      if (job.status === "succeeded" && job.report) return job.report;
    }
  },
};
