"""Clean raw GC=F data before feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "gold_gc_f.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_FILE = PROCESSED_DIR / "gold_clean.csv"

NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Chuẩn hóa tên cột và loại bỏ khoảng trắng thừa.
    df.columns = [str(c).strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc: {col}")
        # Ép toàn bộ OHLCV về numeric để tránh dtype=object từ CSV/Yahoo.
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Loại bỏ ngày không hợp lệ và giá <= 0.
    df.dropna(subset=["Date", "Open", "High", "Low", "Close"], inplace=True)
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]

    # OHLC phải thỏa mãn các ràng buộc cơ bản.
    df = df[df["High"] >= df[["Open", "Close", "Low"]].max(axis=1)]
    df = df[df["Low"] <= df[["Open", "Close", "High"]].min(axis=1)]

    df["Volume"] = df["Volume"].fillna(0).clip(lower=0)
    df.sort_values("Date", inplace=True)
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {RAW_FILE}. Hãy chạy gold_collector.py trước."
        )

    df = pd.read_csv(RAW_FILE)
    cleaned = clean_data(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_FILE, index=False)

    print(f"Đã làm sạch: {len(cleaned):,} dòng")
    print(f"Đã lưu: {CLEAN_FILE}")
    print(cleaned.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
