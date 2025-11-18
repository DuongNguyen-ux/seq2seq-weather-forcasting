# ============================================
#  WEATHER FORECASTING 72 → 24 (4 FEATURES)
#  FULL AUTO — TRAIN + EVALUATE + PLOT
# ============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers


# ============================================
# 1) LOAD DATASET
# ============================================
DATA_FILE = "jena_climate_2009_2016.csv"
MODEL_FILE = "lstm_jena_model_small.keras"

print(" Đang load dữ liệu...")
df = pd.read_csv(DATA_FILE)

# Giữ đúng 4 feature tối ưu
cols = ["T (degC)", "p (mbar)", "wv (m/s)", "max. wv (m/s)"]
data = df[cols].values


# ============================================
# 2) SCALE DATA
# ============================================
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


# ============================================
# 3) CREATE SEQUENCE 72 → 24
# ============================================
def create_sequences(dataset, input_len=72, target_len=24):
    X, y = [], []
    for i in range(len(dataset) - input_len - target_len):
        X.append(dataset[i:i + input_len])
        y.append(dataset[i + input_len:i + input_len + target_len, 0])
    return np.array(X), np.array(y)


print(" Đang tạo sequence (72 → 24)...")
X, y = create_sequences(data_scaled)

# train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("✔ SHAPE:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ============================================
# 4) BUILD MODEL (if not exist)
# ============================================
def build_model():
    model = keras.Sequential([
        layers.LSTM(64, input_shape=(72, 4)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(24)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ============================================
# 5) LOAD OR TRAIN MODEL
# ============================================
if os.path.exists(MODEL_FILE):
    print(" Đang load model có sẵn...")
    model = keras.models.load_model(MODEL_FILE)
else:
    print(" Không thấy model. Đang train model mới...")
    model = build_model()
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=64
    )
    model.save(MODEL_FILE)
    print(" Model đã được lưu:", MODEL_FILE)

model.summary()


# ============================================
# 6) PREDICT
# ============================================
print(" Đang dự đoán...")
y_pred = model.predict(X_test)


# ============================================
# 7) RESCALE TO REAL VALUES
# ============================================
temp_min = scaler.data_min_[0]
temp_max = scaler.data_max_[0]

y_pred_rescaled = y_pred * (temp_max - temp_min) + temp_min
y_test_rescaled = y_test * (temp_max - temp_min) + temp_min


# ============================================
# 8) METRICS
# ============================================
mse = mean_squared_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten())
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten())

print("\n==============================")
print(" KẾT QUẢ ĐÁNH GIÁ")
print("==============================")
print("MSE  :", mse)
print("RMSE :", rmse)
print("MAE  :", mae)


# ============================================
# 9) PLOT FIRST 24H FORECAST
# ============================================
plt.figure(figsize=(10, 4))
plt.plot(y_test_rescaled[0], label="Thực tế")
plt.plot(y_pred_rescaled[0], label="Dự đoán")
plt.title("Dự báo 24 giờ đầu tiên")
plt.legend()
plt.grid()
plt.show()


# ============================================
# 10) PLOT ERROR METRICS
# ============================================
plt.figure(figsize=(6, 4))
errors = [mse, rmse, mae]
names = ["MSE", "RMSE", "MAE"]

plt.bar(names, errors)
plt.title("So sánh các chỉ số lỗi")
plt.ylabel("Giá trị")
plt.grid(axis="y")
plt.show()
# =========================================================
# 11) SO SÁNH GIỮA CÁC MÔ HÌNH
# =========================================================

# --- Nhập kết quả các mô hình ---
# Bạn chỉnh lại giá trị bên dưới theo mô hình của bạn

ket_qua = {
    "LSTM":   {"MSE": 0.85, "RMSE": 0.92, "MAE": 0.71},
    "GRU":    {"MSE": 0.80, "RMSE": 0.89, "MAE": 0.68},
    "CNN-LSTM": {"MSE": 0.75, "RMSE": 0.86, "MAE": 0.64},
    "Seq2Seq":  {"MSE": 0.72, "RMSE": 0.84, "MAE": 0.61},
}

# --- Tách dữ liệu ---
ten_mo_hinh = list(ket_qua.keys())
mse_list  = [ket_qua[m]["MSE"] for m in ten_mo_hinh]
rmse_list = [ket_qua[m]["RMSE"] for m in ten_mo_hinh]
mae_list  = [ket_qua[m]["MAE"] for m in ten_mo_hinh]


# =========================================================
# 11.1) VẼ BIỂU ĐỒ MSE
# =========================================================
plt.figure(figsize=(8, 4))
plt.bar(ten_mo_hinh, mse_list)
plt.title("So sánh MSE giữa các mô hình")
plt.ylabel("MSE (càng thấp càng tốt)")
plt.grid(axis="y")
plt.show()


# =========================================================
# 11.2) VẼ BIỂU ĐỒ RMSE
# =========================================================
plt.figure(figsize=(8, 4))
plt.bar(ten_mo_hinh, rmse_list)
plt.title("So sánh RMSE giữa các mô hình")
plt.ylabel("RMSE (càng thấp càng tốt)")
plt.grid(axis="y")
plt.show()


# =========================================================
# 11.3) VẼ BIỂU ĐỒ MAE
# =========================================================
plt.figure(figsize=(8, 4))
plt.bar(ten_mo_hinh, mae_list)
plt.title("So sánh MAE giữa các mô hình")
plt.ylabel("MAE (càng thấp càng tốt)")
plt.grid(axis="y")
plt.show()