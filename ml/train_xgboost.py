"""Train an XGBoost model to predict next-day Gold Futures close."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "gold_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "xgboost_gold.json"
META_FILE = MODEL_DIR / "xgboost_metadata.json"
TEST_FILE = PROJECT_ROOT / "reports" / "xgboost_test_predictions.csv"

EXCLUDED_COLUMNS = {
    "Date",
    "Target_Close_1d",
    "Target_Close_7d",
    "Target_Return_1d",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]


def train() -> dict:
    if not DATA_FILE.exists():
        raise FileNotFoundError("Chưa có gold_features.csv. Hãy chạy pipeline dữ liệu trước.")

    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])

    # CSV có thể chứa dữ liệu dạng object (đặc biệt khi được tạo từ MultiIndex
    # của yfinance). Ép toàn bộ cột trừ Date về numeric trước khi chọn feature.
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    feature_columns = get_feature_columns(df)
    if not feature_columns:
        raise RuntimeError("Không tìm thấy feature dạng số để huấn luyện XGBoost.")

    trainable = df.dropna(subset=feature_columns + ["Target_Close_1d"]).copy()
    if len(trainable) < 300:
        raise RuntimeError("Cần ít nhất khoảng 300 dòng dữ liệu sạch để huấn luyện.")

    split_idx = int(len(trainable) * 0.80)
    train_df = trainable.iloc[:split_idx]
    test_df = trainable.iloc[split_idx:]

    X_train = train_df[feature_columns].astype("float64")
    y_train = pd.to_numeric(train_df["Target_Close_1d"], errors="coerce").astype("float64")
    X_test = test_df[feature_columns].astype("float64")
    y_test = pd.to_numeric(test_df["Target_Close_1d"], errors="coerce").astype("float64")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    predictions = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    mape = float(np.mean(np.abs((y_test - predictions) / y_test)) * 100)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_FILE)

    test_output = test_df[["Date", "Close", "Target_Close_1d"]].copy()
    test_output["Prediction"] = predictions
    test_output.to_csv(TEST_FILE, index=False)

    metadata = {
        "model": "XGBRegressor",
        "target": "Target_Close_1d",
        "ticker": "GC=F",
        "feature_columns": feature_columns,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_end": str(train_df["Date"].max().date()),
        "test_start": str(test_df["Date"].min().date()),
        "metrics": {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape, "R2": r2},
    }
    META_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Save a sklearn-compatible copy as well for environments that prefer joblib.
    joblib.dump(model, MODEL_DIR / "xgboost_gold.joblib")

    print("=== XGBoost training hoàn tất ===")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Model: {MODEL_FILE}")
    print(f"Test predictions: {TEST_FILE}")
    return metadata


if __name__ == "__main__":
    train()
