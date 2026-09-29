"""Create technical-analysis features and supervised learning targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "gold_clean.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "gold_features.csv"


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Luôn ép OHLCV sang số trước khi tính chỉ báo.
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Returns và lag features.
    df["Return_1d"] = close.pct_change()
    for lag in [1, 2, 3, 5, 7, 14, 30]:
        df[f"Close_Lag_{lag}"] = close.shift(lag)
        df[f"Return_Lag_{lag}"] = df["Return_1d"].shift(lag)

    # Moving averages.
    for window in [7, 20, 50, 100, 200]:
        df[f"MA_{window}"] = close.rolling(window).mean()
        df[f"MA_Ratio_{window}"] = close / df[f"MA_{window}"]

    # Exponential moving averages.
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # RSI.
    df["RSI_14"] = calculate_rsi(close, 14)

    # MACD.
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # True Range / ATR.
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # Rolling volatility.
    df["Volatility_10"] = df["Return_1d"].rolling(10).std()
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()
    df["Volatility_60"] = df["Return_1d"].rolling(60).std()

    # Price range / momentum.
    df["HL_Range"] = (high - low) / close
    df["Momentum_10"] = close / close.shift(10) - 1
    df["Momentum_20"] = close / close.shift(20) - 1

    # Supervised targets: next-day and 7-day closing prices.
    df["Target_Close_1d"] = close.shift(-1)
    df["Target_Close_7d"] = close.shift(-7)
    df["Target_Return_1d"] = close.shift(-1) / close - 1

    # Remove infinities; keep NaN for training scripts to drop after they know the target.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {INPUT_FILE}. Hãy chạy cleaner.py trước."
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
    features = add_features(df)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Ép các cột feature về numeric trước khi ghi CSV.
    protected = {"Date"}
    for col in features.columns:
        if col not in protected:
            features[col] = pd.to_numeric(features[col], errors="coerce")

    features.to_csv(OUTPUT_FILE, index=False)

    print(f"Đã tạo {features.shape[1]} cột từ {len(features):,} dòng.")
    print(f"Đã lưu: {OUTPUT_FILE}")
    print(features.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
