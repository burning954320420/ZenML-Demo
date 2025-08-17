import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from zenml import pipeline, step
from zenml.logger import get_logger

# 创建 logger
logger = get_logger(__name__)

# ------------------ 步骤 1: 加载数据 ------------------
@step
def load_data(data_path: str) -> pd.DataFrame:
    """加载带 Unix 时间戳的 CPU 数据"""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"✅ 成功加载数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"❌ 加载数据失败: {e}")
        raise

# ------------------ 步骤 2: 特征工程 - 提取时间特征 ------------------
@step
def feature_engineering(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """解析 Unix 时间戳，提取时间特征，并组合 cpu_usage 进行训练"""
    required_columns = ['timestamp', 'cpu_usage', 'is_normal']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV 文件必须包含列: {required_columns}")

    # 解析 Unix 时间戳
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')  # 假设时间戳为秒级

    # 提取时间特征
    df['hour'] = df['datetime'].dt.hour                    # 小时 (0-23)
    df['weekday'] = df['datetime'].dt.weekday              # 星期几 (0=周一, 6=周日)
    df['is_weekend'] = (df['datetime'].dt.weekday >= 5).astype(int)  # 是否周末
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)    # 周期性编码：sin
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)    # 周期性编码：cos

    # 特征列
    feature_columns = [
        'cpu_usage',
        'hour',
        'weekday',
        'is_weekend',
        'hour_sin',
        'hour_cos'
    ]

    X = df[feature_columns].values
    y = df['is_normal'].values

    logger.info(f"📊 特征矩阵形状: {X.shape}")
    logger.info(f"🏷️  标签向量形状: {y.shape}")
    logger.info(f"🧩 使用的特征: {feature_columns}")

    return X, y

# ------------------ 步骤 3: 训练随机森林模型 ------------------
@step
def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """使用包含时间特征的随机森林模型"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"📈 测试集准确率: {acc:.4f}")

    logger.info("📋 分类报告:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["正常 (0)", "异常 (1)"]))

    return model

# ------------------ 步骤 4: 保存模型 ------------------
@step
def save_model(model: RandomForestClassifier, save_path: str = "cpu_anomaly_detector_with_unix_time.pkl"):
    """保存模型"""
    import joblib
    joblib.dump(model, save_path)
    model_path = os.path.abspath(save_path)
    logger.info(f"💾 模型已保存至: {model_path}")
    return model_path

# ------------------ 定义 ZenML Pipeline ------------------
@pipeline
def cpu_training_pipeline_with_unix_time(data_path: str, model_save_path: str):
    """包含时间特征的 CPU 异常检测管道（Unix 时间戳版本）"""
    df = load_data(data_path)
    X, y = feature_engineering(df)
    model = train_random_forest(X, y)
    save_model(model, model_save_path)

# ------------------ 主函数 ------------------
if __name__ == "__main__":
    DATA_PATH = "../data/cpu_data_timestamp.csv"
    MODEL_SAVE_PATH = "../models/cpu_anomaly_detector_with_unix_time.pkl"

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {os.path.abspath(DATA_PATH)}")

    # 运行管道
    cpu_training_pipeline_with_unix_time(
        data_path=DATA_PATH,
        model_save_path=MODEL_SAVE_PATH
    )

    print(f"\n🎉 训练完成！模型已保存至: {os.path.abspath(MODEL_SAVE_PATH)}")