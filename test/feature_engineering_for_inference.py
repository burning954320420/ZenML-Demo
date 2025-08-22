# inference_with_zenml_model.py
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List
from datetime import datetime
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="DataFrame.fillna with 'method' is deprecated")

# ======================
# 特征工程类（与 ZenML pipeline 一致）
# ======================
class CPUFeatureEngineer:
    """CPU 特征工程处理器，用于异常检测（与训练 pipeline 保持一致）"""

    def __init__(self, window_sizes: List[int] = None):
        self.window_sizes = window_sizes or [5, 15, 30, 60]  # 分钟
        self.feature_names = []

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建基础特征"""
        df = df.copy()
        cpu_col = "cpu_utilization"

        # 归一化
        df["cpu_normalized"] = (df[cpu_col] - df[cpu_col].min()) / (df[cpu_col].max() - df[cpu_col].min())

        # 排名百分位
        df["cpu_rank"] = df[cpu_col].rank(pct=True)

        # 全局偏离度
        mean_cpu = df[cpu_col].mean()
        std_cpu = df[cpu_col].std()
        df["cpu_deviation_from_mean"] = (df[cpu_col] - mean_cpu) / (std_cpu + 1e-8)

        # 按小时、按日基线偏离
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["date"] = df["timestamp"].dt.date

        hourly_mean = df.groupby("hour")[cpu_col].transform("mean")
        daily_mean = df.groupby("date")[cpu_col].transform("mean")
        df["cpu_deviation_hourly"] = df[cpu_col] - hourly_mean
        df["cpu_deviation_daily"] = df[cpu_col] - daily_mean

        # 阈值标记
        df["is_high_usage"] = (df[cpu_col] > 80).astype(int)
        df["is_very_high_usage"] = (df[cpu_col] > 90).astype(int)
        df["is_low_usage"] = (df[cpu_col] < 10).astype(int)

        self.feature_names.extend([
            "cpu_normalized", "cpu_rank", "cpu_deviation_from_mean",
            "cpu_deviation_hourly", "cpu_deviation_daily",
            "is_high_usage", "is_very_high_usage", "is_low_usage"
        ])
        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间序列特征"""
        df = df.copy()
        cpu_col = "cpu_utilization"

        for window in self.window_sizes:
            win = f"{window}min"
            # 将 window 转换为基于 60 秒的整数（如 5min = 5*60 = 300 行）
            rolled = df[cpu_col].rolling(window=window * 60, min_periods=1)

            df[f"{win}_mean"] = rolled.mean()
            df[f"{win}_std"] = rolled.std().fillna(0)
            df[f"{win}_max"] = rolled.max()
            df[f"{win}_min"] = rolled.min()
            df[f"{win}_range"] = df[f"{win}_max"] - df[f"{win}_min"]
            df[f"{win}_change"] = df[cpu_col] - df[f"{win}_mean"]
            df[f"{win}_pct_change"] = df[cpu_col] / (df[f"{win}_mean"] + 1e-8) - 1

            # 趋势：线性回归斜率近似
            try:
                df[f"{win}_trend"] = rolled.apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0,
                    raw=True
                )
            except:
                df[f"{win}_trend"] = 0

            self.feature_names.extend([
                f"{win}_mean", f"{win}_std", f"{win}_max", f"{win}_min",
                f"{win}_range", f"{win}_change", f"{win}_pct_change", f"{win}_trend"
            ])

        # 时间上下文
        df["hour"] = df["timestamp"].dt.hour
        df["weekday"] = df["timestamp"].dt.dayofweek
        df["minute"] = df["timestamp"].dt.minute
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)
        df["is_business_hour"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
        df["is_peak_hour"] = ((df["hour"] >= 10) & (df["hour"] <= 16)).astype(int)

        self.feature_names.extend(["hour", "weekday", "minute", "is_weekend", "is_business_hour", "is_peak_hour"])
        return df

    def create_algorithm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建基于算法的异常检测特征"""
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor

        df = df.copy()
        cpu_col = "cpu_utilization"

        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        df["iso_forest_anomaly"] = iso_forest.fit_predict(df[[cpu_col]])
        df["iso_forest_anomaly"] = (df["iso_forest_anomaly"] == -1).astype(int)

        # LOF
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        df["lof_anomaly"] = lof.fit_predict(df[[cpu_col]])
        df["lof_anomaly"] = (df["lof_anomaly"] == -1).astype(int)

        # Z-score
        df["z_score"] = (df[cpu_col] - df[cpu_col].mean()) / (df[cpu_col].std() + 1e-8)
        df["z_score_anomaly"] = (df["z_score"].abs() > 3).astype(int)

        # IQR
        Q1 = df[cpu_col].quantile(0.25)
        Q3 = df[cpu_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df["iqr_anomaly"] = ((df[cpu_col] < lower) | (df[cpu_col] > upper)).astype(int)

        # 算法共识
        df["algorithm_consensus"] = df[["iso_forest_anomaly", "lof_anomaly", "z_score_anomaly", "iqr_anomaly"]].sum(axis=1)
        df["algorithm_agreement"] = (df["algorithm_consensus"] >= 2).astype(int)

        self.feature_names.extend([
            "iso_forest_anomaly", "lof_anomaly", "z_score", "z_score_anomaly",
            "iqr_anomaly", "algorithm_consensus", "algorithm_agreement"
        ])
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交互与复合特征"""
        df = df.copy()

        # 算法与统计交互
        df["iso_x_deviation"] = df["iso_forest_anomaly"] * df["cpu_deviation_from_mean"]
        df["lof_x_zscore"] = df["lof_anomaly"] * df["z_score"]

        # 持续性特征
        for n in [3, 5, 10]:
            df[f"high_usage_streak_{n}"] = (
                df["is_high_usage"].rolling(n, min_periods=1).sum()
            )
            df[f"anomaly_streak_{n}"] = (
                df["algorithm_agreement"].rolling(n, min_periods=1).sum()
            )

        # 波动性
        for window in [15, 30, 60]:
            win = f"{window}min"
            df[f"{win}_cv"] = df[f"{win}_std"] / (df[f"{win}_mean"] + 1e-8)  # 变异系数

        self.feature_names.extend([
            "iso_x_deviation", "lof_x_zscore",
            "high_usage_streak_3", "high_usage_streak_5", "high_usage_streak_10",
            "anomaly_streak_3", "anomaly_streak_5", "anomaly_streak_10",
            "15min_cv", "30min_cv", "60min_cv"
        ])
        return df

    def create_all_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """主流程：依次构建所有特征"""
        start_time = 0  # 不记录日志

        # 确保 timestamp 是 datetime 类型
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')  # ✅ 关键：Unix 时间戳转 datetime
        df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"📊 开始特征工程，原始数据形状: {df.shape}")

        df = self.create_basic_features(df)
        df = self.create_temporal_features(df)
        df = self.create_algorithm_features(df)
        df = self.create_interaction_features(df)

        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        total_time = 0
        print(f"✅ 特征工程完成，最终数据形状: {df.shape}")
        print(f"🧩 生成特征数量: {len(self.feature_names)}")

        return df, self.feature_names


# ======================
# 主推理函数
# ======================
def predict_anomalies(
    test_csv_path: str,
    model_dir: str,
    output_csv_path: str
):
    """
    推理函数：加载模型，预测异常，输出结果

    Args:
        test_csv_path: 测试集路径（CSV，含 timestamp, cpu_utilization）
        model_dir: 模型目录（含 final_model.pkl, scaler, feature_names）
        output_csv_path: 输出路径
    """
    print(f"🚀 开始推理任务...")
    print(f"📁 测试集: {test_csv_path}")
    print(f"📁 模型目录: {model_dir}")
    print(f"📁 输出路径: {output_csv_path}")

    # 1. 加载模型组件
    model_path = os.path.join(model_dir, "cpu_anomaly_detector_with_unix_time.pkl")
    scaler_path = os.path.join(model_dir, "cpu_anomaly_detector_with_unix_time_scaler.pkl")
    feature_names_path = os.path.join(model_dir, "cpu_anomaly_detector_with_unix_time_feature_names.pkl")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_names_path]):
        raise FileNotFoundError("缺少模型文件，请检查模型目录")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)

    print(f"✅ 模型组件加载完成")
    print(f"   特征数量: {len(feature_names)}")

    # 2. 读取测试数据（只取前两列）
    df = pd.read_csv(
        test_csv_path,
        usecols=[0, 1],
        header=0,
        names=['timestamp', 'cpu_utilization']
    )

    # 类型转换
    df['timestamp'] = df['timestamp'].astype(int)
    df['cpu_utilization'] = df['cpu_utilization'].astype(float)

    # ✅ 保留 2 位小数（四舍五入）
    df['cpu_utilization'] = df['cpu_utilization'].round(2)

    print(f"📊 测试数据形状: {df.shape}")
    print(f"ℹ️  已保留 CPU 使用率 2 位小数")

    # 3. 特征工程
    engineer = CPUFeatureEngineer(window_sizes=[5, 15, 30, 60])
    feature_df, _ = engineer.create_all_features(df)

    # 4. 提取模型所需特征
    X = feature_df[feature_names].values

    # 5. 标准化
    X_scaled = scaler.transform(X)

    # 6. 预测
    y_pred = model.predict(X_scaled)  # 0: 正常, 1: 异常

    # 7. 构造输出
    output_df = df[['timestamp', 'cpu_utilization']].copy()
    output_df['is_anomaly'] = y_pred.astype(int)

    # ✅ 再次确保 CPU 保留 2 位小数
    output_df['cpu_utilization'] = output_df['cpu_utilization'].round(2)

    # 8. 保存结果
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False, float_format='%.1f')
    print(f"✅ 推理完成！结果已保存至: {output_csv_path}")
    print(f"📊 总数据点: {len(y_pred)}, 异常点: {y_pred.sum()} ({y_pred.mean():.2%})")

    return output_df


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # ✅ 配置路径
    TEST_CSV = "../data/cpu_data_timestamp.csv"           # 输入测试集
    MODEL_DIR = "../models/"                    # 模型文件夹
    OUTPUT_CSV = "../output/anomalies.csv"      # 输出结果

    # 运行推理
    result = predict_anomalies(TEST_CSV, MODEL_DIR, OUTPUT_CSV)