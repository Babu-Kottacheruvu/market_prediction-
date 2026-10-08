export type Market = "crypto" | "forex" | "commodities";

export interface ForecastRow {
  horizon: number;
  candle_time: string;
  direction: "UP" | "DOWN" | "FLAT";
  probability: number;
  probability_up: number;
  probability_flat: number;
  probability_down: number;
  expected_return: number;
  expected_price: number;
  expected_high: number;
  expected_low: number;
  expected_volatility: number;
  confidence: number;
}

export interface Prediction {
  market: string;
  symbol: string;
  timeframe: string;
  timestamp: string;
  data_timestamp: string;
  data_source: string;
  current_price: number;
  atr: number;
  overall_bias: "BULLISH" | "BEARISH" | "NEUTRAL";
  regime: string;
  confidence: number;
  agreement: number;
  signal: "BUY" | "SELL" | "WAIT";
  signal_detail: {
    probability: number;
    min_probability?: number;
    expected_return?: number;
    reasons?: string[];
    scalp?: boolean;
    directional_agreement?: number | null;
  };
  risk: Record<string, number | string | boolean> | null;
  forecast: ForecastRow[];
  model: {
    model_id: string;
    status: string;
    model_version: string;
    feature_version: string;
    trained_at: string;
    n_features: number;
  };
  latency_ms: number;
}

export interface ModelStatus {
  trained: boolean;
  market?: string;
  symbol?: string;
  timeframe?: string;
  model_id?: string;
  status?: string;
  training_start?: string;
  training_end?: string;
  n_rows?: number;
  n_features?: number;
}

export interface BacktestResult {
  [key: string]: unknown;
}

export interface TrainingReport {
  model_id: string;
  status: string;
  n_rows: number;
  n_features: number;
  folds: Array<{
    fold: number;
    train_rows: number;
    test_rows: number;
    h1_accuracy: number | null;
  }>;
  out_of_sample_metrics: {
    aggregate: Record<string, number | null>;
    [key: string]: unknown;
  };
  acceptance: { passed: boolean; reasons: string[] };
  duration_seconds: number;
}

export interface TrainingProgress {
  percentage: number;
  phase: string;
  message: string;
  completed?: number;
  total?: number;
}
