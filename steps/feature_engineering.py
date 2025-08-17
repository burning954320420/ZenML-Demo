# steps/feature_engineering.py

import pandas as pd
import numpy as np
from zenml import step
from zenml.logger import get_logger
from typing import Tuple, List
import time
from typing import Optional, List, Tuple

logger = get_logger(__name__)

class CPUFeatureEngineer:
    """CPU 特征工程处理器，用于异常检测"""

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
            rolled = df[cpu_col].rolling(window, min_periods=1)

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
        start_time = time.time()

        # 确保 timestamp 是 datetime 类型
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"📊 开始特征工程，原始数据形状: {df.shape}")

        df = self.create_basic_features(df)
        df = self.create_temporal_features(df)
        df = self.create_algorithm_features(df)
        df = self.create_interaction_features(df)

        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        total_time = time.time() - start_time
        logger.info(f"✅ 特征工程完成，耗时: {total_time:.2f} 秒")
        logger.info(f"📊 最终数据形状: {df.shape}")
        logger.info(f"🧩 生成特征数量: {len(self.feature_names)}")

        return df, self.feature_names


@step
def feature_engineering_step(
    raw_df: pd.DataFrame,
    window_sizes: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    ZenML Step：执行 CPU 特征工程

    Args:
        raw_df: 输入的原始数据，包含 timestamp, cpu_utilization, is_anomaly
        window_sizes: 滑动窗口大小（分钟），默认 [5, 15, 30, 60]

    Returns:
        Tuple[增强后的 DataFrame, 特征名称列表]
    """
    logger.info("🔧 开始执行特征工程 Step...")

    if raw_df.empty:
        raise ValueError("输入数据为空")

    required_columns = {"timestamp", "cpu_utilization", "is_anomaly"}
    missing = required_columns - set(raw_df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    # 初始化特征工程师
    engineer = CPUFeatureEngineer(window_sizes=window_sizes)

    # 执行特征工程
    try:
        processed_df, feature_names = engineer.create_all_features(raw_df)
        logger.info(f"✅ 特征工程成功，生成 {len(feature_names)} 个特征")
        return processed_df, feature_names
    except Exception as e:
        logger.error(f"❌ 特征工程失败: {e}")
        raise