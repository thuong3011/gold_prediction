"""FastAPI backend for the Gold Prediction dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "gold_features.csv"
META_FILE = PROJECT_ROOT / "models" / "xgboost_metadata.json"
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "Dashboard"

# Cho phép import ml.predict khi chạy: uvicorn backend.main:app --reload
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.predict import predict_next_day  # noqa: E402

app = FastAPI(
    title="Gold Prediction API",
    version="1.0.0",
    description="API phân tích và dự báo Gold Futures (GC=F).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "project": "gold_prediction"}


@app.get("/api/history")
def history(limit: int = Query(365, ge=1, le=5000)) -> dict:
    if not DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="Chưa có dữ liệu. Hãy chạy pipeline.")
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"]).sort_values("Date").tail(limit)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "date": str(row["Date"].date()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
                "ma20": None if pd.isna(row.get("MA_20")) else float(row["MA_20"]),
                "ma50": None if pd.isna(row.get("MA_50")) else float(row["MA_50"]),
            }
        )
    return {"ticker": "GC=F", "count": len(rows), "data": rows}


@app.get("/api/technical")
def technical() -> dict:
    if not DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="Chưa có gold_features.csv.")
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"]).sort_values("Date")
    row = df.iloc[-1]
    return {
        "date": str(row["Date"].date()),
        "close": float(row["Close"]),
        "ma7": float(row["MA_7"]),
        "ma20": float(row["MA_20"]),
        "ma50": float(row["MA_50"]),
        "ma200": None if pd.isna(row["MA_200"]) else float(row["MA_200"]),
        "rsi14": float(row["RSI_14"]),
        "macd": float(row["MACD"]),
        "macd_signal": float(row["MACD_Signal"]),
        "volatility20": float(row["Volatility_20"]),
    }


@app.get("/api/forecast")
def forecast() -> dict:
    try:
        return predict_next_day()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Không thể dự báo: {exc}") from exc


@app.get("/api/model")
def model_info() -> dict:
    if not META_FILE.exists():
        raise HTTPException(status_code=404, detail="Chưa có metadata model.")
    return json.loads(META_FILE.read_text(encoding="utf-8"))


# Serve the small dashboard directly from FastAPI.
if FRONTEND_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=FRONTEND_DIR, html=True), name="dashboard")


@app.get("/", include_in_schema=False)
def root():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return {"message": "Gold Prediction API", "docs": "/docs"}
    return FileResponse(index)
