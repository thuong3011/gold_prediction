"""Inference utilities for the trained XGBoost model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "gold_features.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "xgboost_gold.json"
META_FILE = PROJECT_ROOT / "models" / "xgboost_metadata.json"


def predict_next_day() -> dict:
    if not DATA_FILE.exists():
        raise FileNotFoundError("Chưa có gold_features.csv.")
    if not MODEL_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("Chưa có model XGBoost. Hãy chạy train_xgboost.py.")

    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
    features = metadata["feature_columns"]

    # Phòng trường hợp CSV lưu/đọc các cột số thành object.
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu feature mà model yêu cầu: {missing}")

    for col in features + ["Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    model = XGBRegressor()
    model.load_model(MODEL_FILE)

    usable = df.dropna(subset=features + ["Close"]).sort_values("Date")
    if usable.empty:
        raise ValueError("Không còn dòng dữ liệu hợp lệ sau khi chuyển feature sang numeric.")

    latest = usable.iloc[-1]
    X = pd.DataFrame([latest[features].to_dict()], columns=features).astype("float64")
    prediction = float(model.predict(X)[0])
    current = float(latest["Close"])
    change = prediction - current
    change_pct = (change / current * 100) if current else 0.0

    result = {
        "ticker": "GC=F",
        "date": str(pd.Timestamp(latest["Date"]).date()),
        "current_close": current,
        "predicted_next_close": prediction,
        "expected_change": change,
        "expected_change_percent": change_pct,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = predict_next_day()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
