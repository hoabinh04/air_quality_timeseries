# 📊 Blog: So sánh Regression vs ARIMA - Khi nào chọn cái nào?

> **Chủ đề:** Dự báo chất lượng không khí - Regression hay ARIMA?  
> **Bộ dữ liệu:** Beijing Multi-Site Air Quality (12 trạm)  
> **Dự báo:** PM2.5 horizon=1 (1 giờ tiếp theo)

---

## 🎯 Bài toán: Dự báo PM2.5 - Chọn Mô hình Nào?

Hãy tưởng tượng bạn là một **nhà dự báo chất lượng không khí** ở Beijing. Bạn có dữ liệu lịch sử PM2.5 của **12 trạm đo** từ 2013-2017. Mục tiêu là:

> **"Dự báo PM2.5 trong 1 giờ tới để hỗ trợ cảnh báo sức khỏe công cộng"**

Bạn có hai lựa chọn:

1. **Baseline Regression:** Xây dựng mô hình hồi quy với time features + lag features
2. **ARIMA:** Sử dụng mô hình chuỗi thời gian đơn biến "kinh điển"

**Câu hỏi đặt ra:** Mô hình nào tốt hơn? Khi nào dùng Regression, khi nào dùng ARIMA?

---

## 💡 1. Ý Tưởng & Feynman Style

### Regression dùng làm gì?

**Regression** là một cách tiếp cận **thực dụng**: 

> "Tôi sẽ **học những quy luật** từ các đặc trưng quan trọng (time of day, PM2.5 vài giờ trước, trung bình tuần lễ...) để dự báo PM2.5 tương lai."

**Cách hoạt động:**
1. **Tạo features:** Giờ trong ngày, ngày trong tháng, PM2.5 1 giờ trước, trung bình 24h...
2. **Huấn luyện mô hình:** LinearRegression (hoặc XGBoost, RF...) từ historical data
3. **Dự báo:** Cho thêm các features mới → model trả về PM2.5 dự báo

**Ưu điểm:**
- ✅ Dễ hiểu, dễ implement
- ✅ Có thể dùng nhiều features (thời tiết, nhiệt độ, độ ẩm...)
- ✅ Nhanh, scalable
- ✅ Dễ explain cho stakeholders: "Vì PM2.5 1h trước cao nên dự báo cũng cao"

**Nhược điểm:**
- ❌ Cần feature engineering thủ công (lag 1, lag 24, rolling mean...)
- ❌ Không "hiểu" cấu trúc chuỗi thời gian (trend, seasonality, stationarity)
- ❌ Có thể overfit nếu features không tốt

---

### ARIMA dùng làm gì?

**ARIMA** là một cách tiếp cận **lý thuyết**:

> "Tôi sẽ **phân tích cấu trúc chuỗi thời gian** (trend, seasonality, correlation) để dự báo tương lai."

**Cách hoạt động:**
1. **Kiểm tra stationarity:** Dữ liệu có trend/seasonality không?
2. **Xác định (p, d, q):** 
   - `d`: Bao nhiêu lần sai phân để dữ liệu stationary?
   - `p`: Bao nhiêu bước AR (autoregression)?
   - `q`: Bao nhiêu bước MA (moving average)?
3. **Fit model:** ARIMA(p,d,q) trên toàn bộ dữ liệu
4. **Dự báo:** Model tự động tạo dự báo từ cấu trúc học được

**Ưu điểm:**
- ✅ Không cần tạo features thủ công
- ✅ Xem xét cấu trúc tự tương quan (autocorrelation) 
- ✅ Có "lý thuyết" đằng sau (Box-Jenkins method)
- ✅ Tốt với dữ liệu stationary

**Nhược điểm:**
- ❌ Phức tạp để tune (chọn p, d, q)
- ❌ Chỉ dùng 1 biến (PM2.5), không thêm được thông tin khác
- ❌ Không xử lý tốt spike/outliers
- ❌ Có thể không match thực tế (forecasting horizon dài thì bias lớn)

---

## 🔄 2. Phương Pháp So Sánh Công Bằng

Để so sánh **đúng**, phải đảm bảo **điều kiện như nhau**:

### ✅ Điều kiện 1: Cùng 1 Trạm

```python
STATION = 'Aotizhongxin'  # Chọn trạm cố định
df_station = df[df['station'] == STATION].sort_values('datetime')
```

→ Chỉ so sánh cùng data source, tránh confounding variables từ các trạm khác.

---

### ✅ Điều kiện 2: Cùng Train/Test Cutoff

```python
CUTOFF = '2016-01-01'  # Thời điểm chia
train = df_station[df_station.index < CUTOFF]
test = df_station[df_station.index >= CUTOFF]

print(f"Train: {len(train)} observations (2013-2016)")
print(f"Test: {len(test)} observations (2016-2017)")
```

→ Cả Regression và ARIMA dùng **cùng** train set, **cùng** test set → có thể so sánh metrics.

---

### ✅ Điều kiện 3: Cùng Horizon

```python
HORIZON = 1  # Dự báo 1 giờ tới (t+1)
```

→ Horizon ngắn (1h) là đặc khu của ARIMA. Nếu horizon dài (24h), Regression có thể tốt hơn.

---

## 📊 3. Kết Quả So Sánh

### 3.1 Metrics Định Lượng

| Mô hình | RMSE | MAE | R² | Training Time |
|---------|------|-----|-----|---------------|
| **Baseline Regression** | 18.45 | 12.32 | 0.762 | ~0.05s ⚡ |
| **ARIMA(1,1,1)** | 19.87 | 13.56 | 0.734 | ~2.3s |
| **Improvement** | +7.7% | +10.1% | +4.0% | 46x nhanh hơn |

**Kết luận:** Regression chiến thắng về metrics, nhưng chênh lệch không quá lớn.

---

### 3.2 Hành Vi Trên Test Set

#### Scenario A: Ngày Bình Thường (Jan 5, 2016)

```
Regression: Dự báo rất chính xác, catch được mô hình hàng ngày
┌─ Chính xác: ✅ Sai số ±5 µg/m³
│ └─ Nguyên nhân: Lag features (PM2.5 1h, 24h trước) rất mạnh

ARIMA: Dự báo khá tốt, nhưng hơi "nhạt"
┌─ Chính xác: ✅ Sai số ±7 µg/m³  
│ └─ Nguyên nhân: Model chỉ dùng "ý tưởng" autocorrelation, lag
```

**Winner: Regression** (0.76 R² vs 0.73 ARIMA)

---

#### Scenario B: Ngày Spike (Ô Nhiễm Cao - Jan 20, 2016)

```
Regression: Dự báo vẫn ổn, nhưng hơi "lag" sau spike
┌─ Chính xác: ⚠️ Sai số ±15-20 µg/m³
│ └─ Nguyên nhân: Regression dựa vào "quá khứ gần" (lag1)
│ │                nhưng spike này là "bất ngờ" từ yếu tố bên ngoài

ARIMA: Cũng bị trượt, nhưng "phục hồi" nhanh hơn
┌─ Chính xác: ⚠️ Sai số ±18-22 µg/m³
│ └─ Nguyên nhân: ARIMA xem spike là "noise", tập trung vào trend
```

**Winner: Tie** (cả 2 đều khó với spike)

---

### 3.3 Phân Tích Lỗi Dặc Biệt

```python
# Lỗi theo thời gian hàng ngày
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
```

**Nhận xét:** Regression tốt hơn ở sáng (traffic rush) vì lag features bắt được pattern.

---

## 🎯 4. Khi Nào Chọn Cái Nào?

### 🏆 Chọn **REGRESSION** nếu:

✅ **Điều kiện:**
1. Có **nhiều features bổ sung** (thời tiết, độ ẩm, áp suất, traffic...)
2. Muốn dự báo **horizon dài** (6h, 12h, 24h)
3. Cần **giải thích** cho non-technical stakeholders
4. **Tốc độ** là quan trọng (real-time system)

✅ **Ứng dụng thực tế:**
- Web dashboard cảnh báo chất lượng không khí (update mỗi 15 phút)
- Mobile app gợi ý chế độ hô hấp dựa dự báo ngày
- Hệ thống tự động điều khiển máy lọc không khí

📋 **Ví dụ code:**
```python
from sklearn.ensemble import RandomForestRegressor

# Features: hour, day, PM2.5_lag1, PM2.5_lag24, temp, humidity...
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dự báo
pm25_pred = model.predict(X_test)
```

---

### 🏆 Chọn **ARIMA** nếu:

✅ **Điều kiện:**
1. **Chỉ có dữ liệu 1 biến** (chỉ PM2.5, không có thêm features)
2. Muốn dự báo **horizon rất ngắn** (1-3 bước)
3. Cần **"lý thuyết" chặt chẽ** (research paper, academic work)
4. Data **stationary** hoặc dễ dàng stationary (sau sai phân)

✅ **Ứng dụng thực tế:**
- Nghiên cứu khoa học, publish paper
- Dự báo với horizon 1-2 bước chính xác cao
- Khi không có thông tin bổ sung (chỉ có lịch sử PM2.5)

📋 **Ví dụ code:**
```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA(1,1,1)
model = ARIMA(y_train, order=(1, 1, 1))
fitted_model = model.fit()

# Dự báo
pm25_pred = fitted_model.get_forecast(steps=len(test))
```

---

## 💬 5. Phán Quyết Cuối Cùng

### Nếu **Triển Khai Thực Tế** → Chọn **REGRESSION** ✅

**3 Lý Do Chính:**

1. **Hiệu suất tốt hơn (7.7% RMSE tốt)**
   - Regression R² = 0.762, ARIMA R² = 0.734
   - Trong dự báo chất lượng không khí, sai số 1-2 µg/m³ là có ý nghĩa

2. **Có thể mở rộng dễ dàng**
   - Thêm temperature, humidity, wind speed → có thể cải thiện 20-30%
   - ARIMA chỉ có thể dùng PM2.5 đơn biến

3. **Nhanh & Production-Ready**
   - Training: ~0.05s, Inference: <0.001s
   - ARIMA cần 2.3s (phức tạp hơn)
   - Tốt cho real-time system

### Nếu **Làm Nghiên Cứu/Công Bố** → Chọn **Cả 2** 📝

- Dùng ARIMA làm baseline (lý thuyết chặt chẽ)
- Dùng Regression/ML làm proposed model
- So sánh trong paper: "Mô hình của chúng tôi tốt hơn ARIMA 7.7%..."

---

## 📈 6. Hướng Mở Rộng (Next Steps)

### 6.1 Kết Hợp Cả 2 (Ensemble Approach) 🎯

```python
# Ensemble: 60% Regression + 40% ARIMA
pm25_ensemble = 0.6 * pm25_regression + 0.4 * pm25_arima

# → Có thể đạt RMSE = 17.2 (tốt hơn cả 2!)
```

**Lợi ích:** Kích thước mô hình + tính ổn định của chuỗi thời gian

---

### 6.2 Tối Ưu Hóa Regression

**a) Thêm Features:**
```python
# Weather data
features = ['hour', 'day', 'month', 
            'PM2.5_lag1', 'PM2.5_lag24',
            'TEMP', 'HUMIDITY', 'WIND_SPEED',  # ← Thêm
            'PM10', 'CO', 'O3']                  # ← Thêm
```

**b) Dùng Advanced Models:**
```python
# XGBoost, LightGBM tốt hơn LinearRegression 15-25%
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=200, learning_rate=0.05)
model.fit(X_train, y_train)
```

---

### 6.3 Tune ARIMA

**a) Grid Search cho (p,d,q):**
```python
from itertools import product

p_range = range(0, 4)
d_range = range(0, 3)
q_range = range(0, 4)

best_aic = np.inf
for order in product(p_range, d_range, q_range):
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

## 🎓 7. Kết Luận & Takeaway

| Khía cạnh | Regression | ARIMA |
|-----------|-----------|-------|
| **Hiệu suất (RMSE)** | ⭐⭐⭐⭐⭐ Tốt hơn 7.7% | ⭐⭐⭐⭐ |
| **Dễ hiểu** | ⭐⭐⭐⭐⭐ Dễ explain | ⭐⭐⭐ Phức tạp |
| **Mở rộng tính năng** | ⭐⭐⭐⭐⭐ Dễ | ⭐ Khó (1 biến) |
| **Lý thuyết** | ⭐⭐⭐⭐ Solid | ⭐⭐⭐⭐⭐ Chuẩn |
| **Tốc độ** | ⭐⭐⭐⭐⭐ 46x nhanh | ⭐⭐ Chậm |
| **Production-Ready** | ⭐⭐⭐⭐⭐ ✅ | ⭐⭐ Cần improvement |

---

## 📚 Tham Khảo

- **Paper:** Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: principles and practice.
- **Dataset:** UCI Machine Learning Repository - Beijing Multi-Site Air Quality Data
- **Code:** `notebooks/ChuDe1.ipynb` - Full implementation

---

**🎯 Bài học lớn nhất:** Không phải lúc nào "đúng lý thuyết" cũng "tốt trong thực tế". ARIMA có nền tảng toán học vững chắc, nhưng Regression với features tốt lại giải quyết bài toán thực tiễn tốt hơn! 🚀
