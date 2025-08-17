import joblib
import pandas as pd
import numpy as np

# 加载模型
model = joblib.load("../models/cpu_anomaly_detector_with_unix_time.pkl")

# 模拟新数据点（Unix 时间戳）
unix_timestamp = 1712345678  # 替换为实际的 Unix 时间戳
cpu_usage = 85.0

# 转换 Unix 时间戳为 datetime
timestamp = pd.to_datetime(unix_timestamp, unit='s')

# 提取特征（必须和训练一致）
hour = timestamp.hour
weekday = timestamp.weekday()
is_weekend = 1 if weekday >= 5 else 0
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

# 组合特征
features = np.array([[cpu_usage, hour, weekday, is_weekend, hour_sin, hour_cos]])

# 预测
pred = model.predict(features)[0]
prob = model.predict_proba(features)[0]

print(f"时间: {timestamp}, CPU: {cpu_usage}%")
print(f"预测: {'🚨 异常' if pred == 1 else '✅ 正常'}")
print(f"异常概率: {prob[1]:.2f}")