"""Train an LSTM model using the latest 60 trading days of technical features."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Truy xuất keras qua tensorflow giúp Pylance nhận diện đúng module
keras = tf.keras
layers = tf.keras.layers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "gold_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "lstm_gold.keras"
SCALER_FILE = MODEL_DIR / "lstm_scaler.joblib"
META_FILE = MODEL_DIR / "lstm_metadata.json"

LOOKBACK = 60

FEATURES = [
    "Close",
    "Volume",
    "MA_7",
    "MA_20",
    "MA_50",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "ATR_14",
    "Volatility_20",
    "Momentum_10",
    "Momentum_20",
]
TARGET = "Close"


def make_sequences(values: np.ndarray, target_index: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i, target_index])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def train() -> dict:
    if not DATA_FILE.exists():
        raise FileNotFoundError("Chưa có gold_features.csv. Hãy chạy pipeline dữ liệu trước.")

    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    data = df[["Date", *FEATURES]].dropna().reset_index(drop=True)

    if len(data) <= LOOKBACK + 100:
        raise RuntimeError("Cần nhiều dữ liệu hơn để huấn luyện LSTM.")

    split_row = int(len(data) * 0.80)
    scaler = MinMaxScaler()
    scaler.fit(data.iloc[:split_row][FEATURES])
    scaled = scaler.transform(data[FEATURES]).astype(np.float32)

    X, y = make_sequences(scaled, target_index=0, lookback=LOOKBACK)
    target_dates = data["Date"].iloc[LOOKBACK:].reset_index(drop=True)

    # Sequence target thuộc train nếu ngày target nằm trước train/test cutoff.
    split_date = data["Date"].iloc[split_row]
    train_mask = target_dates < split_date
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = keras.Sequential(
        [
            keras.Input(shape=(LOOKBACK, len(FEATURES))),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Inverse-transform only the Close column.
    pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
    close_min = scaler.data_min_[0]
    close_max = scaler.data_max_[0]
    pred = pred_scaled * (close_max - close_min) + close_min
    actual = y_test * (close_max - close_min) + close_min

    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    metadata = {
        "model": "LSTM",
        "ticker": "GC=F",
        "target": TARGET,
        "lookback": LOOKBACK,
        "features": FEATURES,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_start": str(target_dates[test_mask].min().date()),
        "metrics": {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape},
    }
    META_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("=== LSTM training hoàn tất ===")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


if __name__ == "__main__":
    train()