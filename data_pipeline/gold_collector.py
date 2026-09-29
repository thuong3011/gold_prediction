"""Download historical Gold Futures data (GC=F) from Yahoo Finance.

Usage examples:
    python data_pipeline/gold_collector.py --period max
    python data_pipeline/gold_collector.py --start 2015-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "gold_gc_f.csv"
TICKER = "GC=F"

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten Yahoo Finance MultiIndex columns when necessary."""
    if isinstance(df.columns, pd.MultiIndex):
        # Example: ('Close', 'GC=F') -> 'Close'
        df.columns = [str(col[0]) for col in df.columns]
    else:
        df.columns = [str(col) for col in df.columns]
    return df


def normalize_download(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the downloaded DataFrame into a stable CSV schema."""
    if df.empty:
        raise RuntimeError("Yahoo Finance không trả về dữ liệu.")

    df = flatten_columns(df.copy()).reset_index()

    # Yahoo may return Datetime as the index name instead of Date.
    if "Datetime" in df.columns and "Date" not in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Thiếu cột dữ liệu bắt buộc: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None)

    for col in REQUIRED_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Date", "Open", "High", "Low", "Close"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def download_history(
    period: str = "max",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download GC=F daily historical data."""
    if start and end:
        raw = yf.download(
            TICKER,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=True,
            progress=False,
            group_by="column",
            threads=False,
        )
    else:
        raw = yf.download(
            TICKER,
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=True,
            progress=False,
            group_by="column",
            threads=False,
        )

    return normalize_download(raw)


def merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge newly downloaded rows with existing CSV, avoiding duplicates."""
    if RAW_FILE.exists():
        try:
            old_df = pd.read_csv(RAW_FILE, parse_dates=["Date"])
            combined = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            combined = new_df.copy()
    else:
        combined = new_df.copy()

    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    for col in REQUIRED_COLUMNS[1:]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined.dropna(subset=["Date", "Open", "High", "Low", "Close"], inplace=True)
    combined.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    combined.sort_values("Date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined[REQUIRED_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải dữ liệu Gold Futures GC=F từ Yahoo Finance")
    parser.add_argument("--period", default="max", help="Yahoo period, ví dụ max, 10y, 5y, 1y")
    parser.add_argument("--start", default=None, help="Ngày bắt đầu YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Ngày kết thúc YYYY-MM-DD")
    args = parser.parse_args()

    if bool(args.start) != bool(args.end):
        parser.error("Phải truyền cả --start và --end hoặc không truyền cả hai.")

    print(f"[1/2] Đang tải {TICKER} từ Yahoo Finance...")
    data = download_history(args.period, args.start, args.end)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    merged = merge_with_existing(data)
    merged.to_csv(RAW_FILE, index=False)

    print(f"[2/2] Đã lưu {len(merged):,} dòng vào: {RAW_FILE}")
    print("Khoảng thời gian:", merged["Date"].min().date(), "→", merged["Date"].max().date())
    print(merged.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
