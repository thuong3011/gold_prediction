# Gold Prediction

Hệ thống AI thu thập, phân tích và dự báo Gold Futures `GC=F` từ Yahoo Finance.

## Kiến trúc

```text
Yahoo Finance
     |
     v
 data_pipeline/gold_collector.py
     |
     v
 data/raw/gold_gc_f.csv
     |
     v
 data_pipeline/cleaner.py
     |
     v
 data/processed/gold_clean.csv
     |
     v
 data_pipeline/feature_engineering.py
     |
     v
 data/processed/gold_features.csv
     |
     +----------------------+
     |                      |
     v                      v
ml/train_xgboost.py    ml/train_lstm.py
     |                      |
     +----------+-----------+
                v
             models/
                |
                v
       backend/main.py
                |
                v
    frontend/Dashboard
```

## 1. Tạo môi trường ảo trên Windows

Khuyến nghị Python 3.12 hoặc 3.13.

```powershell
cd D:\gold_prediction
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn `Activate.ps1`, có thể chạy Python trực tiếp trong venv hoặc dùng CMD:

```cmd
.venv\Scripts\activate.bat
```

## 2. Tải dữ liệu vàng

```powershell
python data_pipeline/gold_collector.py --period max
```

Hoặc theo khoảng ngày:

```powershell
python data_pipeline/gold_collector.py --start 2015-01-01 --end 2026-01-01
```

File đầu ra:

```text
data/raw/gold_gc_f.csv
```

## 3. Làm sạch và tạo feature

```powershell
python data_pipeline/cleaner.py
python data_pipeline/feature_engineering.py
```

File đầu ra:

```text
data/processed/gold_clean.csv
data/processed/gold_features.csv
```

Feature gồm MA, EMA, RSI, MACD, ATR, volatility, momentum và lag features. Target chính là `Target_Close_1d`.

## 4. Huấn luyện XGBoost

```powershell
python ml/train_xgboost.py
```

Artifacts:

```text
models/xgboost_gold.json
models/xgboost_gold.joblib
models/xgboost_metadata.json
reports/xgboost_test_predictions.csv
```

## 5. Huấn luyện LSTM

```powershell
python ml/train_lstm.py
```

Artifacts:

```text
models/lstm_gold.keras
models/lstm_scaler.joblib
models/lstm_metadata.json
```

## 6. Dự báo ngày kế tiếp bằng XGBoost

```powershell
python ml/predict.py
```

Ví dụ kết quả:

```json
{
  "ticker": "GC=F",
  "date": "2026-09-28",
  "current_close": 3800.0,
  "predicted_next_close": 3815.2,
  "expected_change": 15.2,
  "expected_change_percent": 0.4
}
```

Các số trên chỉ là ví dụ định dạng, không phải giá thực tế.

## 7. Chạy FastAPI + Dashboard

Từ thư mục gốc:

```powershell
uvicorn backend.main:app --reload
```

Mở:

```text
http://127.0.0.1:8000/
```

Swagger API:

```text
http://127.0.0.1:8000/docs
```

Dashboard API chính:

```text
GET /api/health
GET /api/history?limit=365
GET /api/technical
GET /api/forecast
GET /api/model
```

## 8. Chạy toàn bộ pipeline MVP

```powershell
python run_pipeline.py
```

Sau đó chạy API:

```powershell
uvicorn backend.main:app --reload
```

## Lưu ý về dữ liệu

`GC=F` là Gold Futures, nên hệ thống hiện tại dự báo giá vàng futures theo dữ liệu Yahoo Finance, chưa phải trực tiếp giá vàng miếng SJC/DOJI/PNJ tại Việt Nam.

Mô hình nên được đánh giá theo dữ liệu chuỗi thời gian và không nên xem kết quả dự báo là khuyến nghị mua/bán.
