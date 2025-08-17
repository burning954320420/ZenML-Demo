# steps/data_validator.py
import pandas as pd
from zenml import step
from zenml.logger import get_logger
from typing import Dict, Any

logger = get_logger(__name__)

@step
def data_validation_step(df: pd.DataFrame) -> Dict[str, float]:
    """
    ZenML Step: 验证 CPU 数据质量（自动处理时间索引）

    支持输入：
      - 索引为 DatetimeIndex ✅
      - 包含 'datetime' 列（自动转为索引）
      - 包含 'timestamp' 列（自动转为 datetime 并设为索引）

    Args:
        df: 输入 DataFrame

    Returns:
        验证指标字典
    """
    logger.info("🔍 开始数据质量验证...")

    # ========================
    # 1. 自动处理时间索引（关键修复）
    # ========================
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("⚠️ 索引不是 DatetimeIndex，尝试从列构建时间索引...")

        # 情况1: 存在 'datetime' 列（字符串或时间类型）
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').sort_index()
            logger.info("✅ 使用 'datetime' 列构建时间索引")

        # 情况2: 存在 'timestamp' 列（Unix 时间戳）
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            # 判断是秒还是毫秒
            is_millisecond = (df['timestamp'].max() > 1e10)
            unit = 'ms' if is_millisecond else 's'
            df['datetime'] = pd.to_datetime(df['timestamp'], unit=unit)
            df = df.set_index('datetime').sort_index()
            logger.info(f"✅ 使用 'timestamp' 列（unit={unit}）构建时间索引")

        else:
            raise ValueError(
                "❌ 无法构建时间索引：DataFrame 必须包含 'datetime' 或 'timestamp' 列，"
                "或其索引必须是 DatetimeIndex"
            )
    else:
        logger.info("✅ 输入索引已是 DatetimeIndex，跳过转换")

    # ========================
    # 2. 检查必需列
    # ========================
    cpu_col = 'cpu_utilization'
    required_columns = [cpu_col, 'is_anomaly']
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"❌ 缺少必要列: {missing_cols}")

    print("🔍 Performing data quality validation...")
    print("=" * 50)

    # ========================
    # 3. 基础质量检查
    # ========================
    print("📊 Basic quality metrics:")

    # 完整性
    completeness = (1 - df.isnull().sum().sum() / len(df)) * 100
    print(f"   Data completeness: {completeness:.2f}%")

    # 时间连续性（单调递增）
    is_continuous = df.index.is_monotonic_increasing
    print(f"   Time continuity: {'✅ Good' if is_continuous else '❌ Issues'}")

    # 值有效性（CPU 应在 0~100）
    invalid_values = ((df[cpu_col] < 0) | (df[cpu_col] > 100)).sum()
    validity_ratio = (len(df) - invalid_values) / len(df) * 100
    validity_status = '✅' if invalid_values == 0 else '❌'
    print(f"   Value validity: {len(df) - invalid_values}/{len(df)} ({validity_status})")

    # 异常比例
    anomaly_ratio = df['is_anomaly'].mean()
    anomaly_status = '✅' if 0.05 <= anomaly_ratio <= 0.15 else '⚠️'
    print(f"   Anomaly ratio: {anomaly_ratio:.2%} ({anomaly_status})")

    # ========================
    # 4. 模式合理性检查
    # ========================
    print(f"\n📈 Pattern validation:")

    # 工作日 vs 周末
    weekday_mean = df[df.index.dayofweek < 5][cpu_col].mean()
    weekend_mean = df[df.index.dayofweek >= 5][cpu_col].mean()
    workday_difference = weekday_mean - weekend_mean
    workday_status = '✅' if workday_difference > 5 else '⚠️'
    print(f"   Weekday difference: {workday_difference:.1f}% ({workday_status})")

    # 日间波动（小时级均值最大最小差）
    hourly_mean = df.groupby(df.index.hour)[cpu_col].mean()
    daily_variation = hourly_mean.max() - hourly_mean.min()
    daily_status = '✅' if daily_variation > 20 else '⚠️'
    print(f"   Daily variation: {daily_variation:.1f}% ({daily_status})")

    # 异常分布（防聚集）
    anomaly_points = df[df['is_anomaly'] == 1]
    if len(anomaly_points) > 0:
        time_gaps = anomaly_points.index.to_series().diff().dt.total_seconds() / 60  # 分钟
        avg_gap = time_gaps.mean()
        gap_status = '✅' if avg_gap > 60 else '⚠️'
        print(f"   Anomaly distribution: avg gap {avg_gap:.0f} min ({gap_status})")
    else:
        avg_gap = float('nan')
        print("   Anomaly distribution: no anomalies found")

    # ========================
    # 5. 统计摘要
    # ========================
    print(f"\n📋 Data statistics summary:")
    stats = df[cpu_col].describe()
    for stat_name, value in stats.items():
        print(f"   {stat_name}: {value:.2f}%")

    # ========================
    # 6. 返回结构化结果
    # ========================
    results = {
        'completeness': float(completeness),
        'validity': float(validity_ratio),
        'anomaly_ratio': float(anomaly_ratio * 100),
        'workday_difference': float(workday_difference),
        'daily_variation': float(daily_variation),
        'avg_anomaly_gap_min': avg_gap if not pd.isna(avg_gap) else -1,
        'n_anomalies': int(anomaly_points.shape[0]),
        'n_total': int(df.shape[0]),
        'time_continuity': 1 if is_continuous else 0
    }

    logger.info("✅ 数据质量验证完成")
    return results