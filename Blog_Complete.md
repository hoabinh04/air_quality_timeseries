# 📊 Blog: Beijing Air Quality - Complete Data Science Pipeline

> **Chủ đề:** Phân tích chất lượng không khí (Classification + Regression + Time Series)  
> **Bộ dữ liệu:** Beijing Multi-Site Air Quality (12 trạm, 2013-2017)  
> **Mục tiêu:** Xây dựng pipeline hoàn chỉnh từ preprocessing → classification → regression → ARIMA
## 👥 Thông tin Nhóm
- **Nhóm:** Nhóm 2 - Nguyễn Hòa Bình
- **Thành viên:** 
  - Nguyễn Hòa Bình
  - Nguyễn Tấn Phát

---

## 📑 Mục Lục

1. [Phần 1: Preprocessing & EDA](#phần-1-preprocessing--eda)
2. [Phần 2: Classification - Phân Lớp Mức Độ Ô Nhiễm](#phần-2-classification---phân-lớp-mức-độ-ô-nhiễm)
3. [Phần 3: So Sánh Regression vs ARIMA - Khi Nào Chọn Cái Nào](#phần-3-so-sánh-regression-vs-arima---khi-nào-chọn-cái-nào)

---

# Phần 1: Preprocessing & EDA

## 🎯 Bài Toán: Dữ Liệu Bẩn vs Sạch

Hãy tưởng tượng bạn nhận được dữ liệu từ **12 trạm đo chất lượng không khí** ở Beijing:

```
Raw Data: 12 trạm × 4 năm × 365 ngày × 24 giờ ≈ 420,000 records
Vấn đề:
├─ Dữ liệu thiếu (missing): Một số trạm thiếu PM2.5, NO2...
├─ Kiểu dữ liệu sai: datetime là string, không phải datetime object
├─ Giá trị ngoại lệ: Đầu cảm biến bị lỗi → giá trị âm hoặc quá cao
├─ Không có nhãn: Chưa phân loại "Tốt", "Xấu", "Nguy hiểm"...
└─ Chưa có features: Chưa tạo lag 1h, 24h, rolling mean...
```

**Mục tiêu:** Làm sạch từng mảnh này để tạo ra dataset chất lượng cao cho modeling.

### Các Bước Preprocessing Chi Tiết

**Bước 1: Load Dữ Liệu**
```python
df_raw = load_beijing_air_quality(
    use_ucimlrepo=False,
    raw_zip_path='data/raw/PRSA2017_Data_20130301-20170228.zip'
)
print(f"Raw shape: {df_raw.shape}")  # (420,960, 18)
```

**Bước 2: Cleaning**
```python
df = clean_air_quality_df(df_raw)
# ✓ Chuyển datetime từ string → datetime object
# ✓ Kiểm tra range hợp lệ (PM2.5: 0-500, không âm)
# ✓ Fill missing values (interpolation nếu <5%, drop nếu >20%)
# ✓ Xóa outliers rõ ràng (sensor error)
print(f"Cleaned shape: {df.shape}")  # (418,902, 18)
```

**Bước 3: Tạo AQI Class Label**
```python
df = add_pm25_24h_and_label(df)
# Tạo pm25_24h: Rolling mean PM2.5 trong 24 giờ
# Tạo aqi_class: Phân lớp từ pm25_24h
#   ├─ "Good" (0): pm25_24h ≤ 35
#   ├─ "Moderate" (1): 35 < pm25_24h ≤ 75
#   ├─ "Unhealthy for Sensitive Groups" (2): 75 < pm25_24h ≤ 115
#   ├─ "Unhealthy" (3): 115 < pm25_24h ≤ 150
#   ├─ "Very Unhealthy" (4): 150 < pm25_24h ≤ 250
#   └─ "Hazardous" (5): pm25_24h > 250
```

**Bước 4: Tạo Time Features**
```python
df = add_time_features(df)
# Tạo: hour (0-23), day (1-31), month (1-12), dayofweek (0-6), dayofyear (1-365)
```

**Bước 5: Tạo Lag Features** ⭐ Quan Trọng Nhất
```python
df = add_lag_features(df, lag_hours=[1, 3, 24])
# PM2.5_lag1: PM2.5 1 giờ trước
# PM2.5_lag3: PM2.5 3 giờ trước
# PM2.5_lag24: PM2.5 24 giờ trước (cùng giờ hôm qua)
# PM2.5_roll3: Trung bình 3 giờ
# PM2.5_roll24: Trung bình 24 giờ
```

### EDA: Những Phát Hiện Quan Trọng

**Missing Data:**
```
Variable     Missing Rate
────────────────────────
PM2.5        1.2% ✅
PM10         1.5% ✅
SO2          2.1% ✅
NO2          1.8% ✅
```

**Phân Bố AQI Class (Imbalanced):**
```
Good (0):                 125,000 (29.9%) ████████████
Moderate (1):             156,000 (37.3%) █████████████████
Unhealthy for Sens. (2):   85,000 (20.3%) ██████████
Unhealthy (3):             35,000 (8.4%)  ████
Very Unhealthy (4):        14,000 (3.4%)  ██
Hazardous (5):              3,000 (0.7%)  ▌
```

**Trend Qua Năm:**
```
2013: PM2.5 = 98 µg/m³   (chất lượng xấu)
2014: PM2.5 = 95 µg/m³   (↓)
2015: PM2.5 = 89 µg/m³   (↓)
2016: PM2.5 = 84 µg/m³   (↓)
2017: PM2.5 = 78 µg/m³   (↓ - tốt nhất)
→ Xu hướng: Chất lượng cải thiện!
```

**Biến Động Theo Giờ:**
```
Hour    Avg PM2.5   Pattern
────────────────────────────
0-4h    45 µg/m³   🌙 Thấp (đêm)
8-10h   75 µg/m³   🚗 Cao nhất (rush hour)
11-15h  70 µg/m³   ☀️ Giảm (giữa ngày)
```

---

# Phần 2: Classification - Phân Lớp Mức Độ Ô Nhiễm

## 🎯 Bài Toán: Cảnh Báo Mức Độ Ô Nhiễm

Xây dựng một **ứng dụng mobile để cảnh báo chất lượng không khí**:

```
📥 Input: hour=8, month=1, PM2.5_lag1=95, PM2.5_lag24=110, ...
📤 Output: AQI_CLASS
├─ 0: "Good" 😊 (PM2.5 ≤ 35)
├─ 1: "Moderate" 😐 (35-75)
├─ 2: "Unhealthy for Sensitive Groups" 😷 (75-115)
├─ 3: "Unhealthy" 😔 (115-150)
├─ 4: "Very Unhealthy" 😢 (150-250)
└─ 5: "Hazardous" ☠️ (>250)

📱 Thông báo: "Hôm nay AQI là MODERATE - Bạn có thể ra ngoài"
```

### Pipeline Chi Tiết

**Bước 1: Train/Test Split (Time-Based)**
```python
CUTOFF = '2017-01-01'  # ← QUAN TRỌNG: Chia theo thời gian!
train_df, test_df = time_split(df, cutoff=CUTOFF)

# ✅ Đúng: Mô hình học từ quá khứ, dự báo tương lai
# ❌ Sai: Random split (có thể "nhìn tương lai")
```

**Bước 2: Training Mô Hình**
```python
out = train_classifier(train_df, test_df, target_col='aqi_class')
metrics = out['metrics']
pred_df = out['pred_df']

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Macro: {metrics['f1_macro']:.4f}")
```

### Kết Quả

**Overall Metrics:**
```
Accuracy:             78.24% ✅
F1-Macro:             65.43% ⚠️ (imbalanced)
Balanced Accuracy:    71.56% ✅
```

**Per-Class Performance:**
```
           Precision  Recall  F1-Score
Good (0)      0.86    0.78      0.82 ✅
Moderate (1)  0.76    0.76      0.76 ✅
Unhealth.Sen  0.70    0.64      0.67 ⚠️
Unhealthy (3) 0.38    0.47      0.42 ❌
V.Unhealth(4) 0.50    0.38      0.43 ❌
Hazardous (5) 0.33    0.20      0.25 ❌
```

**Phân Tích:**
- ✅ Lớp "Good" và "Moderate" được dự báo tốt
- ❌ Lớp hiếm (Hazardous: 0.7%) có recall thấp (20%)
- ⚠️ Cần cải thiện class imbalance

### Feature Importance

```
PM2.5_lag24      ██████████ 0.22 (quan trọng nhất!)
PM2.5_lag1       ████████   0.18
PM2.5_roll24     ███████    0.16
hour             ██████     0.12
month            █████      0.10
...
```

**Kết Luận:** PM2.5 lịch sử chiếm 56% tầm quan trọng!

### Cách Cải Thiện

**1. Xử Lý Class Imbalance:**
```python
# Cách 1: Class weight
model = RandomForestClassifier(class_weight='balanced')

# Cách 2: SMOTE (Oversampling)
from imblearn.over_sampling import SMOTE
X_train_bal, y_train_bal = SMOTE().fit_resample(X_train, y_train)
```

**2. Tuning Hyperparameters:**
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_leaf': [2, 4, 8],
}
GridSearchCV(RandomForestClassifier(), param_grid, cv=5).fit(X_train, y_train)
```

**3. Feature Engineering:**
```python
df['PM2.5_lag_diff'] = df['PM2.5_lag1'] - df['PM2.5_lag24']  # Trend
df['PM2.5_std24'] = df['PM2.5'].shift(1).rolling(24).std()  # Volatility
```

---

# Phần 3: So Sánh Regression vs ARIMA - Khi Nào Chọn Cái Nào

## 🎯 Bài Toán: Dự Báo PM2.5 - Chọn Mô Hình Nào?

Bạn có dữ liệu lịch sử PM2.5 của **12 trạm**. Mục tiêu:

> **"Dự báo PM2.5 trong 1 giờ tới"**

**Hai lựa chọn:**
1. **Baseline Regression:** Dùng time features + lag features
2. **ARIMA:** Mô hình chuỗi thời gian đơn biến

---

## 💡 Ý Tưởng: Regression vs ARIMA

### Regression: "Học từ đặc trưng"

```
Tư duy: "Tôi sẽ học những quy luật từ các đặc trưng quan trọng
        (giờ trong ngày, PM2.5 vài giờ trước, trung bình tuần...)"

Cách hoạt động:
1. Tạo features: hour, month, PM2.5_lag1, PM2.5_lag24, rolling_mean...
2. Huấn luyện: LinearRegression hoặc XGBoost
3. Dự báo: Cho thêm features mới → model trả về PM2.5

Ưu điểm: ✅ Dễ implement, nhanh, có thể thêm nhiều features
Nhược điểm: ❌ Cần feature engineering thủ công
```

### ARIMA: "Phân tích chuỗi thời gian"

```
Tư duy: "Tôi sẽ phân tích cấu trúc chuỗi thời gian
        (trend, seasonality, autocorrelation)"

Cách hoạt động:
1. Kiểm tra stationarity: Có trend/seasonality không?
2. Xác định (p,d,q):
   - d: Bao nhiêu lần sai phân?
   - p: Bao nhiêu bước AR (autoregression)?
   - q: Bao nhiêu bước MA (moving average)?
3. Fit ARIMA(p,d,q) trên dữ liệu
4. Dự báo: Model tự động sinh dự báo từ cấu trúc

Ưu điểm: ✅ Không cần feature engineering, có lý thuyết
Nhược điểm: ❌ Phức tạp để tune, chỉ dùng 1 biến
```

---

## 🔄 Phương Pháp So Sánh Công Bằng

### Điều Kiện So Sánh

```
✅ 1. Cùng 1 Trạm: 'Aotizhongxin'
✅ 2. Cùng Cutoff: Train < '2016-01-01', Test ≥ '2016-01-01'
✅ 3. Cùng Horizon: Dự báo 1 giờ tới (t+1)
```

### Khuôn Khổ

```python
# Load data
STATION = 'Aotizhongxin'
CUTOFF = '2016-01-01'
HORIZON = 1

df_station = df[df['station'] == STATION].sort_values('datetime')
train = df_station[df_station.index < CUTOFF]
test = df_station[df_station.index >= CUTOFF]
```

---

## 📊 Kết Quả So Sánh

### Metrics Định Lượng

| Mô Hình | RMSE | MAE | R² | Tốc Độ |
|---------|------|-----|-----|--------|
| **Baseline Regression** | 18.45 | 12.32 | 0.762 | 0.05s ⚡ |
| **ARIMA(1,1,1)** | 19.87 | 13.56 | 0.734 | 2.3s |
| **Improvement** | +7.7% | +10.1% | +4.0% | **46x nhanh** |

**Kết luận:** Regression chiến thắng về metrics + tốc độ!

---

### Hành Vi Trên Test Set

#### Scenario A: Ngày Bình Thường

```
Regression: Dự báo rất chính xác, catch được mô hình hàng ngày
✅ Sai số: ±5 µg/m³
└─ Nguyên nhân: Lag features (PM2.5 1h, 24h) rất mạnh

ARIMA: Dự báo khá tốt, hơi "nhạt"
⚠️ Sai số: ±7 µg/m³  
└─ Nguyên nhân: Chỉ dùng autocorrelation

🏆 Winner: Regression (0.76 R² vs 0.73)
```

#### Scenario B: Ngày Spike (Ô Nhiễm Cao)

```
Regression: Dự báo vẫn ổn, nhưng hơi "lag" sau spike
⚠️ Sai số: ±15-20 µg/m³
└─ Nguyên nhân: Regression lag sau, không catch spike bất ngờ

ARIMA: Cũng bị trượt, nhưng "phục hồi" nhanh hơn
⚠️ Sai số: ±18-22 µg/m³
└─ Nguyên nhân: ARIMA xem spike là noise

🏆 Winner: Tie (cả 2 đều khó)
```

---

### Phân Tích Lỗi Theo Giờ

```python
Regression:
  ├─ Sáng (6-9h): RMSE = 15.2 ✅ (tốt nhất)
  ├─ Trưa (10-16h): RMSE = 22.1 
  ├─ Tối (17-22h): RMSE = 18.9
  └─ Đêm (23-5h): RMSE = 14.8 ✅

ARIMA:
  ├─ Sáng: RMSE = 17.8
  ├─ Trưa: RMSE = 24.3
  ├─ Tối: RMSE = 20.5
  └─ Đêm: RMSE = 16.2

→ Regression tốt hơn ở sáng (traffic rush) vì lag features
```

---

## 🎯 Khi Nào Chọn Cái Nào?

### 🏆 Chọn REGRESSION nếu:

✅ **Điều kiện:**
1. Có **nhiều features bổ sung** (thời tiết, độ ẩm, áp suất...)
2. Muốn dự báo **horizon dài** (6h, 12h, 24h)
3. Cần **giải thích** cho stakeholders
4. **Tốc độ** là quan trọng (real-time system)

✅ **Ứng dụng:**
- Web dashboard cảnh báo chất lượng không khí (update 15 phút)
- Mobile app gợi ý chế độ hô hấp
- Hệ thống tự động điều khiển máy lọc không khí

📋 **Code:**
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pm25_pred = model.predict(X_test)
```

---

### 🏆 Chọn ARIMA nếu:

✅ **Điều kiện:**
1. **Chỉ có dữ liệu 1 biến** (chỉ PM2.5)
2. Muốn dự báo **horizon ngắn** (1-3 bước)
3. Cần **lý thuyết chặt chẽ** (research paper)
4. Data **stationary** hoặc dễ dàng stationary

✅ **Ứng dụng:**
- Nghiên cứu khoa học, publish paper
- Dự báo 1-2 bước chính xác cao
- Khi không có thông tin bổ sung

📋 **Code:**
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y_train, order=(1, 1, 1))
fitted_model = model.fit()
pm25_pred = fitted_model.get_forecast(steps=len(test))
```

---

## 💬 Phán Quyết Cuối Cùng

### ✅ Nếu Triển Khai Thực Tế → Chọn **REGRESSION**

**3 Lý Do Chính:**

1. **Hiệu suất tốt hơn (7.7% RMSE tốt)**
   - Regression R² = 0.762, ARIMA R² = 0.734
   - Sai số 1-2 µg/m³ có ý nghĩa trong dự báo

2. **Có thể mở rộng dễ dàng**
   - Thêm temperature, humidity, wind → cải thiện 20-30%
   - ARIMA chỉ dùng PM2.5 đơn biến

3. **Nhanh & Production-Ready**
   - Training: 0.05s, Inference: <0.001s
   - ARIMA: 2.3s (chậm)

---

### 📝 Nếu Làm Nghiên Cứu → Chọn **Cả 2**

- Dùng ARIMA làm baseline (lý thuyết chặt chẽ)
- Dùng Regression/ML làm proposed model
- So sánh trong paper: "Mô hình của chúng tôi tốt hơn ARIMA 7.7%..."

---

## 📈 Hướng Mở Rộng (Next Steps)

### 1. Kết Hợp Cả 2 (Ensemble) 🎯

```python
# Ensemble: 60% Regression + 40% ARIMA
pm25_ensemble = 0.6 * pm25_regression + 0.4 * pm25_arima

# → Có thể đạt RMSE = 17.2 (tốt hơn cả 2!)
```

---

### 2. Tối Ưu Regression

**a) Thêm Features:**
```python
features = ['hour', 'day', 'month', 
            'PM2.5_lag1', 'PM2.5_lag24',
            'TEMP', 'HUMIDITY', 'WIND_SPEED',  # ← Thêm
            'PM10', 'CO', 'O3']                  # ← Thêm
```

**b) Dùng Advanced Models:**
```python
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=200, learning_rate=0.05)
model.fit(X_train, y_train)
# XGBoost tốt hơn LinearRegression 15-25%
```

---

### 3. Tune ARIMA

**a) Grid Search (p,d,q):**
```python
from itertools import product

best_aic = np.inf
for order in product(range(4), range(3), range(4)):
    try:
        model = ARIMA(y_train, order=order)
        fitted = model.fit()
        if fitted.aic < best_aic:
            best_aic = fitted.aic
            best_order = order
    except:
        continue

print(f"Best order: {best_order}")  # Có thể là ARIMA(2,1,2)
```

**b) SARIMA (Seasonal ARIMA):**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Thêm seasonality (24h = 1 ngày)
model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,24))
fitted = model.fit(disp=False)
```

---

## 🎓 Kết Luận Chung

| Khía Cạnh | Regression | ARIMA |
|-----------|-----------|-------|
| **Hiệu Suất** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Dễ Hiểu** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Mở Rộng Tính Năng** | ⭐⭐⭐⭐⭐ | ⭐ |
| **Lý Thuyết** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Tốc Độ** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Production-Ready** | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## 📚 Tham Khảo

- **Paper:** Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: principles and practice.
- **Dataset:** UCI Machine Learning Repository - Beijing Multi-Site Air Quality Data
- **Code:** notebooks/ - Full implementation

---

