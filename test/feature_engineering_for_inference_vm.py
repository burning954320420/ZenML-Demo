# inference_with_victoriametrics.py
# 从Victoriametrics获取CPU使用率数据并进行异常检测
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime, timedelta
import warnings
import requests
import json
from urllib.parse import urlencode

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="DataFrame.fillna with 'method' is deprecated")

# 忽略SSL验证警告（仅用于开发环境）
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
# 数据获取与写入函数
# ======================
def fetch_cpu_data_from_victoriametrics(
    vm_url: str,
    query: str,
    start_time: str,
    end_time: str,
    step: str = "30",
    token: str = "xxxxxxx",
    namespace: str = "dip-Prod",
    host_name: str = "PDIPMCNINFVM005"
) -> pd.DataFrame:
    """
    从 Victoriametrics 获取 CPU 使用率数据
    
    Args:
        vm_url: Victoriametrics API URL
        query: 查询语句，如果为None则使用默认查询
        start_time: 开始时间，ISO 格式 (YYYY-MM-DDTHH:MM:SSZ)
        end_time: 结束时间，ISO 格式 (YYYY-MM-DDTHH:MM:SSZ)
        step: 步长，单位秒
        token: 认证令牌
        namespace: 命名空间
        host_name: 主机名
        
    Returns:
        包含 timestamp 和 cpu_utilization 的 DataFrame
    """
    if query is None:
        # 默认查询
        query = f'(1 - avg by (project, namespace, host_name) (rate(system_cpu_time_seconds_total{{state="idle",namespace="{namespace}",host_name="{host_name}"}}[1m]))) * 100'
    
    # 构建请求参数
    params = {
        'query': query,
        'start': start_time,
        'end': end_time,
        'step': step
    }
    
    # 构建请求头
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    print(f"🔍 正在从 Victoriametrics 获取数据...")
    print(f"   时间范围: {start_time} 至 {end_time}")
    print(f"   步长: {step}秒")
    
    try:
        # 发送请求
        response = requests.post(
            vm_url,
            data=params,
            headers=headers,
            verify=False  # 忽略SSL证书验证
        )
        
        # 检查响应状态
        response.raise_for_status()
        
        # 解析响应
        data = response.json()
        
        if data['status'] != 'success':
            raise ValueError(f"API 返回错误: {data.get('error', '未知错误')}")
        
        # 提取结果
        result = data['data']['result']
        
        if not result:
            raise ValueError("查询结果为空")
        
        # 提取第一个结果的值
        values = result[0]['values']
        
        # 转换为 DataFrame
        df = pd.DataFrame(values, columns=['timestamp', 'cpu_utilization'])
        
        # 转换类型
        df['timestamp'] = df['timestamp'].astype(int)
        df['cpu_utilization'] = df['cpu_utilization'].astype(float).round(2)
        
        print(f"✅ 数据获取成功，共 {len(df)} 条记录")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {str(e)}")
        raise
    except (ValueError, KeyError) as e:
        print(f"❌ 数据解析失败: {str(e)}")
        raise
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")
        raise

def write_to_victoriametrics(
    vm_write_url: str,
    timestamp: int,
    cpu_value: float,
    status: int
) -> bool:
    """
    将CPU使用率和状态写入Victoriametrics
    
    Args:
        vm_write_url: Victoriametrics写入API URL
        timestamp: 时间戳（Unix时间戳，秒）
        cpu_value: CPU使用率
        status: 状态（0表示正常，1表示异常）
        
    Returns:
        写入是否成功
    """
    # 构建Prometheus格式的数据
    # 确保时间戳是毫秒级（Victoriametrics要求）
    timestamp_ms = timestamp * 1000
    
    # 构建指标数据，确保标签格式正确
    metric_data = f'cpu_usage{{status="{status}"}} {cpu_value} {timestamp_ms}\n'
    
    print(f"📤 正在写入数据到Victoriametrics...")
    print(f"   URL: {vm_write_url}")
    print(f"   数据: {metric_data.strip()}")
    
    try:
        # 发送请求
        response = requests.post(
            vm_write_url,
            data=metric_data,
            headers={'Content-Type': 'application/x-protobuf;proto=prometheus.WriteRequest'},
            verify=False  # 忽略SSL证书验证
        )
        
        # 检查响应状态
        response.raise_for_status()
        
        print(f"✅ 数据写入成功！")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 写入失败: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {str(e)}")
        return False

# ======================
# 主推理函数
# ======================
def predict_anomalies(
    model_dir: str,
    output_csv_path: str = None,
    threshold: float = 0.5,
    vm_url: str = "https://1.1.1.1:30070/select/0/prometheus/api/v1/query_range",
    vm_token: str = "xxxxxxx",
    vm_namespace: str = "xxx-xxx",
    vm_host_name: str = "xxxxxxx",
    hours_to_fetch: float = 1,
    vm_write_url: str = "http://2.2.2.2:8428/api/v1/write"
):
    """
    推理函数：加载模型，预测异常，输出结果

    Args:
        model_dir: 模型目录（含 final_model.pkl, scaler, feature_names）
        output_csv_path: 输出路径，如果为None则不保存结果
        threshold: 自定义阈值（默认0.5），用于判定异常的概率阈值
        vm_url: Victoriametrics API URL
        vm_token: Victoriametrics认证令牌
        vm_namespace: 命名空间
        vm_host_name: 主机名
        hours_to_fetch: 获取最近多少小时的数据
        vm_write_url: Victoriametrics写入API URL
    """
    print(f"🚀 开始推理任务...")
    print(f"📁 模型目录: {model_dir}")
    if output_csv_path:
        print(f"📁 输出路径: {output_csv_path}")
    
    print(f"📡 数据源: Victoriametrics")

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

    # 2. 获取数据
    # 从Victoriametrics获取数据
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours_to_fetch)
    
    # 格式化时间为ISO格式
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # 获取数据
    df = fetch_cpu_data_from_victoriametrics(
        vm_url=vm_url,
        query=None,  # 使用默认查询
        start_time=start_time_str,
        end_time=end_time_str,
        step="30",  # 30秒步长
        token=vm_token,
        namespace=vm_namespace,
        host_name=vm_host_name
    )

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
    if hasattr(model, 'predict_proba'):
        # 如果模型支持概率预测
        y_proba = model.predict_proba(X_scaled)[:, 1]  # 获取异常类的概率
        y_pred = (y_proba >= threshold).astype(int)  # 应用自定义阈值
        print(f"ℹ️  使用自定义阈值 {threshold} 进行异常判定")
        print(f"ℹ️  预测概率范围: min={y_proba.min():.2f}, max={y_proba.max():.2f}")
        # 验证是否有概率低于阈值却被标记为异常的情况
        invalid_mask = (y_proba < threshold) & (y_pred == 1)
        if invalid_mask.any():
            print(f"⚠️  发现 {invalid_mask.sum()} 条数据概率低于阈值但被标记为异常！")
    else:
        # 不支持概率预测的模型
        y_pred = model.predict(X_scaled)  # 0: 正常, 1: 异常
        print(f"⚠️  模型不支持概率预测，使用默认预测结果")

    # 7. 构造输出（转换为 Unix 时间戳）
    output_df = df[['timestamp', 'cpu_utilization']].copy()
    # 将 timestamp 转换为 Unix 时间戳（整数秒）
    if not pd.api.types.is_numeric_dtype(output_df['timestamp']):
        output_df['timestamp'] = pd.to_datetime(output_df['timestamp']).astype('int64') // 10**9
    output_df['is_anomaly'] = y_pred.astype(int)

    # ✅ 再次确保 CPU 保留 2 位小数
    output_df['cpu_utilization'] = output_df['cpu_utilization'].round(2)

    # 8. 保存结果
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        output_df.to_csv(output_csv_path, index=False, float_format='%.2f')
        print(f"✅ 推理完成！结果已保存至: {output_csv_path}")
    else:
        print(f"✅ 推理完成！")
    
    # 输出结果统计
    print(f"📊 总数据点: {len(y_pred)}, 异常点: {y_pred.sum()} ({y_pred.mean():.2%})")
    
    # 输出最新数据点的预测结果
    latest_point = output_df.iloc[-1]
    latest_time = pd.to_datetime(latest_point['timestamp'], unit='s')
    latest_cpu = latest_point['cpu_utilization']
    latest_anomaly = "异常" if latest_point['is_anomaly'] == 1 else "正常"
    
    print(f"\n📌 最新数据点 ({latest_time}):")
    print(f"   CPU 使用率: {latest_cpu:.2f}%")
    print(f"   状态: {latest_anomaly}")
    
    # 如果是异常，计算异常持续时间
    if latest_point['is_anomaly'] == 1:
        # 查找最近连续异常的起始点
        anomaly_start_idx = len(output_df) - 1
        while anomaly_start_idx > 0 and output_df.iloc[anomaly_start_idx-1]['is_anomaly'] == 1:
            anomaly_start_idx -= 1
        
        anomaly_start_time = pd.to_datetime(output_df.iloc[anomaly_start_idx]['timestamp'], unit='s')
        duration_seconds = (latest_time - anomaly_start_time).total_seconds()
        duration_minutes = duration_seconds / 60
        
        print(f"   异常持续时间: {duration_minutes:.1f}分钟")
    
    # 将最新数据点写入Victoriametrics
    if vm_write_url:
        print("\n📤 正在将最新数据点写入Victoriametrics...")
        # 获取最新数据点的时间戳、CPU使用率和状态
        latest_timestamp = int(latest_point['timestamp'])
        latest_cpu_value = float(latest_point['cpu_utilization'])
        latest_status = int(latest_point['is_anomaly'])
        
        # 写入Victoriametrics
        write_success = write_to_victoriametrics(
            vm_write_url=vm_write_url,
            timestamp=latest_timestamp,
            cpu_value=latest_cpu_value,
            status=latest_status
        )
        
        if write_success:
            print(f"✅ 最新数据点已成功写入Victoriametrics")
        else:
            print(f"❌ 最新数据点写入Victoriametrics失败")
    
    return output_df

if __name__ == "__main__":
    # ✅ 配置路径和参数
    MODEL_DIR = "../models/"                    # 模型文件夹
    OUTPUT_CSV = "None"      # 输出结果，设为None则不保存
    THRESHOLD = 0.75                            # 自定义阈值
    
    # Victoriametrics配置
    VM_URL = "https://1.1.1.1:30070/select/0/prometheus/api/v1/query_range"
    VM_TOKEN = "xxxxxxx"  # 实际使用时请替换为有效的token
    VM_NAMESPACE = "xxx-Prod"
    VM_HOST_NAME = "xxx"
    HOURS_TO_FETCH = 1  # 获取最近1小时的数据
    
    # Victoriametrics写入配置
    VM_WRITE_URL = "http://2.2.2.2:8428/api/v1/import/prometheus"
    
    # 添加定时循环
    import time
    while True:
        try:
            print(f"\n⏰ 开始新一轮推理任务 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            # 运行推理
            result = predict_anomalies(
                model_dir=MODEL_DIR,
                output_csv_path=OUTPUT_CSV,
                threshold=THRESHOLD,
                vm_url=VM_URL,
                vm_token=VM_TOKEN,
                vm_namespace=VM_NAMESPACE,
                vm_host_name=VM_HOST_NAME,
                hours_to_fetch=HOURS_TO_FETCH,
                vm_write_url=VM_WRITE_URL
            )
            print(f"⏳ 等待30秒后继续...")
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n🛑 用户中断，停止执行")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")
            print(f"⏳ 等待30秒后重试...")
            time.sleep(30)