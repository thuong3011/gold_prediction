# Gold Prediction
<img width="603" height="619" alt="image" src="https://github.com/user-attachments/assets/4fd6cbbf-491b-4d95-81eb-25959c624703" />

Hệ thống **dự đoán giá vàng bằng Machine Learning và Deep Learning**, sử dụng dữ liệu lịch sử giá vàng từ **Yahoo Finance – Gold Futures (`GC=F`)**.

Dự án thực hiện toàn bộ quy trình từ thu thập dữ liệu, làm sạch, tạo đặc trưng kỹ thuật, huấn luyện mô hình đến dự đoán và hiển thị kết quả trên Dashboard.

---

## Mục tiêu dự án

Dự án được xây dựng nhằm:

* Thu thập dữ liệu giá vàng lịch sử.
* Làm sạch và chuẩn hóa dữ liệu.
* Phân tích xu hướng giá vàng.
* Tạo các đặc trưng kỹ thuật phục vụ Machine Learning.
* Dự đoán giá vàng trong tương lai.
* So sánh mô hình **XGBoost** và **LSTM**.
* Cung cấp API thông qua **FastAPI**.
* Hiển thị dữ liệu và kết quả dự đoán trên Dashboard.

> **Lưu ý:** Kết quả dự đoán chỉ phục vụ mục đích nghiên cứu và học tập.

---

# Nguồn dữ liệu

Nguồn dữ liệu chính của dự án là:

**Yahoo Finance – Gold Futures (`GC=F`)**

https://finance.yahoo.com/quote/GC%3DF/history/

Mã `GC=F` đại diện cho hợp đồng tương lai vàng (Gold Futures).

Dữ liệu gốc bao gồm:

| Trường | Ý nghĩa              |
| ------ | -------------------- |
| Date   | Ngày giao dịch       |
| Open   | Giá mở cửa           |
| High   | Giá cao nhất         |
| Low    | Giá thấp nhất        |
| Close  | Giá đóng cửa         |
| Volume | Khối lượng giao dịch |

Dữ liệu được thu thập thông qua thư viện:

```text
yfinance
```

Dữ liệu gốc được lưu tại:

```text
data/raw/gold_gc_f.csv
```

---

# Kiến trúc hệ thống

```text
                    Yahoo Finance
                         │
                         │ GC=F
                         ▼
                ┌─────────────────┐
                │ Gold Collector  │
                │   yfinance      │
                └────────┬────────┘
                         │
                         ▼
                 gold_gc_f.csv
                         │
                         ▼
                ┌─────────────────┐
                │     Cleaner     │
                │ Làm sạch dữ liệu│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │Feature Engineering│
                │ MA / RSI / MACD │
                │ ATR / Volatility│
                │ Lag / Momentum  │
                └────────┬────────┘
                         │
                         ▼
                gold_features.csv
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       ┌─────────────┐       ┌─────────────┐
       │   XGBoost   │       │     LSTM    │
       └──────┬──────┘       └──────┬──────┘
              │                     │
              └──────────┬──────────┘
                         ▼
                    Prediction
                         │
                         ▼
                  ┌─────────────┐
                  │   FastAPI   │
                  └──────┬──────┘
                         │
                         ▼
                    Dashboard


# Công nghệ sử dụng

## Backend

* Python
* FastAPI
* Uvicorn

## Data Processing

* Pandas
* NumPy
* yfinance

## Machine Learning

* Scikit-learn
* XGBoost

## Deep Learning

* TensorFlow
* Keras
* LSTM

## Technical Analysis

Các chỉ báo kỹ thuật được sử dụng:

* Moving Average (MA)
* Exponential Moving Average (EMA)
* Relative Strength Index (RSI)
* Moving Average Convergence Divergence (MACD)
* Average True Range (ATR)
* Volatility
* Momentum
* Lag features

## Frontend

* HTML
* CSS
* JavaScript
* Chart.js

---

# Các đặc trưng được tạo

Sau khi xử lý dữ liệu, hệ thống tạo ra nhiều đặc trưng phục vụ mô hình.

### Lag Features

```text
Close_Lag_1
Close_Lag_2
Close_Lag_3
Close_Lag_5
Close_Lag_7
Close_Lag_14
Close_Lag_30
```

Các đặc trưng này biểu diễn giá vàng của những ngày trước.

### Moving Average

```text
MA_7
MA_20
MA_50
MA_100
MA_200
```

### EMA

```text
EMA_12
EMA_26
```

### RSI

```text
RSI_14
```

### MACD

```text
MACD
MACD_Signal
MACD_Hist
```

### Volatility

```text
Volatility_10
Volatility_20
Volatility_60
```

### Momentum

```text
Momentum_10
Momentum_20
```

### ATR

```text
ATR_14
```

---

# Huấn luyện mô hình LSTM

Chạy:

```powershell
python ml/train_lstm.py
```

LSTM được sử dụng để xử lý dữ liệu chuỗi thời gian và học các mẫu biến động của giá vàng theo thời gian.

Mô hình được lưu trong:

```text
models/
```

---

# Chạy dự đoán

Sau khi huấn luyện mô hình:

```powershell
python ml/predict.py
```

Hệ thống sẽ sử dụng dữ liệu mới nhất để tạo dự báo.

---
# Quy trình dự đoán

Hệ thống thực hiện quy trình:

```text
Dữ liệu lịch sử
      ↓
Làm sạch
      ↓
Technical Indicators
      ↓
Feature Engineering
      ↓
Train/Test theo thời gian
      ↓
XGBoost / LSTM
      ↓
Đánh giá mô hình
      ↓
Dự đoán giá vàng
      ↓
FastAPI
      ↓
Dashboard
```

Dự án sử dụng **Time Series Split** thay vì chia dữ liệu ngẫu nhiên nhằm hạn chế việc sử dụng thông tin tương lai trong quá trình huấn luyện.

---

# Đánh giá mô hình

Các chỉ số đánh giá có thể sử dụng:

### MAE

Mean Absolute Error:

```text
MAE = mean(|y_true - y_pred|)
```

MAE càng nhỏ thì sai số dự đoán trung bình càng thấp.

### RMSE

Root Mean Squared Error:

```text
RMSE = sqrt(mean((y_true - y_pred)^2))
```

RMSE phạt mạnh hơn đối với các sai số lớn.

### MAPE

Mean Absolute Percentage Error:

```text
MAPE = mean(|(y_true - y_pred) / y_true|) × 100
```

MAPE biểu diễn sai số dưới dạng phần trăm.

### R²

Được sử dụng để đánh giá mức độ giải thích biến động của dữ liệu bởi mô hình.

---

# Các mô hình

## XGBoost

XGBoost phù hợp với dữ liệu dạng bảng sau Feature Engineering.

Đầu vào có thể bao gồm:

```text
Open
High
Low
Close
Volume
MA
EMA
RSI
MACD
ATR
Volatility
Momentum
Lag Features
```

## LSTM

LSTM được sử dụng để học các quan hệ theo chuỗi thời gian.

Ví dụ:

```text
Ngày 1 ─┐
Ngày 2 ─┤
Ngày 3 ─┤
Ngày 4 ─┤──→ LSTM ──→ Giá dự đoán
Ngày 5 ─┘
```

---

# Lưu ý

Mô hình Machine Learning không đảm bảo dự đoán chính xác giá vàng trong tương lai.

Giá vàng chịu ảnh hưởng bởi nhiều yếu tố như:

* Chính sách tiền tệ.
* Lãi suất.
* Lạm phát.
* Tỷ giá USD.
* Tình hình kinh tế.
* Thị trường tài chính.
* Các sự kiện kinh tế và địa chính trị.

Vì vậy, kết quả của hệ thống chỉ nên được sử dụng cho **mục đích nghiên cứu, học tập và phân tích dữ liệu**.




