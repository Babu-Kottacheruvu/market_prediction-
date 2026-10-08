"""Forex specific features: sessions and session statistics.

Sessions are expressed in UTC:
    Asian   00:00-07:00
    London  07:00-16:00
    New York 16:00-21:00
    Overlap is encoded inside London (12:00-16:00 is London/NY overlap start)
    Off     21:00-00:00 (low liquidity) and weekends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_CODES = {0: "off", 1: "asian", 2: "london", 3: "london_ny_overlap", 4: "new_york"}


def _session_code(hour: int) -> int:
    if 0 <= hour < 7:
        return 1
    if 7 <= hour < 12:
        return 2
    if 12 <= hour < 16:
        return 3
    if 16 <= hour < 21:
        return 4
    return 0


def build_forex_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    hour = df["timestamp"].dt.hour
    session = hour.map(_session_code).astype(float)
    out["session"] = session
    out["session_asian"] = (session == 1).astype(float)
    out["session_london"] = (session == 2).astype(float)
    out["session_overlap"] = (session == 3).astype(float)
    out["session_new_york"] = (session == 4).astype(float)
    out["session_off"] = (session == 0).astype(float)

    # session id = date + session code (each contiguous block gets its own id)
    day = df["timestamp"].dt.floor("D")
    out["session_id"] = (day.astype("int64") // 10**9).astype("int64") * 10 + session.astype("int64")

    ret = df["close"].pct_change().fillna(0.0)
    grp = out.groupby("session_id", sort=False)
    open_first = df["open"].groupby(out["session_id"], sort=False).transform("first")
    out["session_return"] = df["close"] / open_first - 1.0

    cum_ret2 = (ret**2).groupby(out["session_id"], sort=False).cumsum()
    count = ret.groupby(out["session_id"], sort=False).cumcount() + 1
    out["session_volatility"] = np.sqrt(cum_ret2 / count)

    out["session_candle_index"] = count.astype(float)

    # causal intra-session range position
    day_high = df["high"].groupby(out["session_id"], sort=False).cummax()
    day_low = df["low"].groupby(out["session_id"], sort=False).cummin()
    span = (day_high - day_low).replace(0.0, np.nan)
    out["session_range_position"] = (df["close"] - day_low) / span
    return out.drop(columns=["session_id"])
