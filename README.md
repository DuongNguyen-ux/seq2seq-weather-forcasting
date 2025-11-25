# seq2seq-weather-forcasting
Sequence-to-Sequence Weather Forecasting 
Goal: Predict temperature sequences for upcoming days. 
Dataset: Jena Climate Dataset (https://www.kaggle.com/datasets/mnassrib/jena-climate) 
Model: Encoder–Decoder LSTM 
Task Type: Regression (multi-step)
Extension: Add attention mechanism; compare against Transformer. 
| Cột                 | Ý nghĩa                                    | Đơn vị           |
| ------------------- | ------------------------------------------ | ---------------- |
|   Date Time         | Thời gian đo                               | yyyy-mm-dd HH:MM |
|   p (mbar)          | Áp suất khí quyển                          | millibar         |
|   T (degC)          | Nhiệt độ không khí                         | °C               |
|   Tpot (K)          | Nhiệt độ tiềm năng (Potential Temperature) | Kelvin           |
|   Tdew (degC)       | Nhiệt độ điểm sương (Dew Point)            | °C               |
|   rh (%)            | Độ ẩm tương đối                            | %                |
|   VPmax (mbar)      | Áp suất hơi bão hòa                        | millibar         |
|   VPact (mbar)      | Áp suất hơi thực tế                        | millibar         |
|   VPdef (mbar)      | Độ thiếu hụt áp suất hơi (VPmax – VPact)   | millibar         |
|   sh (g/kg)         | Độ ẩm tuyệt đối (Specific humidity)        | g/kg             |
|   H2OC (mmol/mol)   | Hàm lượng hơi nước                         | mmol/mol         |
|   rho (g/m**3)      | Mật độ không khí                           | g/m³             |
|   wv (m/s)          | Tốc độ gió trung bình                      | m/s              |
|   max. wv (m/s)     | Tốc độ gió lớn nhất đo được                | m/s              |
|   wd (deg)          | Hướng gió                                  | độ (0–360°)      |
