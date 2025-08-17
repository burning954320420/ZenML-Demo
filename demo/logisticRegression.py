import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from zenml import pipeline, step
from zenml.logger import get_logger  # 注意：这是 zenml 自带的 logger

# 获取 logger（必须传入名字）
logger = get_logger(__name__)

# ------------------ 步骤 1: 加载数据 ------------------
@step
def load_data(data_path: str) -> pd.DataFrame:
    """加载 CPU 数据 CSV 文件"""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"✅ 成功加载数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"❌ 加载数据失败: {e}")
        raise

# ------------------ 步骤 2: 预处理数据 ------------------
@step
def preprocess_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """提取 cpu_usage 作为特征，is_normal 作为标签"""
    required_columns = ['cpu_usage', 'is_normal']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV 文件必须包含列: {required_columns}")

    X = df[['cpu_usage']].values           # 特征：CPU 使用率
    y = df['is_normal'].values             # 标签：0=正常，1=异常

    logger.info(f"📊 特征矩阵形状: {X.shape}")
    logger.info(f"🏷️  标签向量形状: {y.shape}")

    return X, y

# ------------------ 步骤 3: 训练模型 ------------------
@step
def train_model(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """使用逻辑回归训练异常检测模型"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 在测试集上评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"📈 测试集准确率: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["正常 (0)", "异常 (1)"]))

    return model

# ------------------ 步骤 4: 保存模型 ------------------
@step
def save_model(model: LogisticRegression, save_path: str = "cpu_anomaly_model.pkl"):
    """将训练好的模型保存到本地文件"""
    import joblib
    joblib.dump(model, save_path)
    model_path = os.path.abspath(save_path)
    logger.info(f"💾 模型已保存至: {model_path}")
    return model_path

# ------------------ 定义 ZenML Pipeline ------------------
@pipeline
def cpu_training_pipeline(data_path: str, model_save_path: str):
    """CPU 异常检测训练流程"""
    df = load_data(data_path)
    X, y = preprocess_data(df)
    model = train_model(X, y)
    save_model(model, model_save_path)

# ------------------ 主函数 ------------------
if __name__ == "__main__":
    DATA_PATH = "data/cpu_data.csv"              # 数据文件路径
    MODEL_SAVE_PATH = "models/cpu_anomaly_detector.pkl" # 模型保存路径

    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {os.path.abspath(DATA_PATH)}")

    # 运行 ZenML 管道
    cpu_training_pipeline(
        data_path=DATA_PATH,
        model_save_path=MODEL_SAVE_PATH
    )

    print(f"\n🎉 训练完成！模型已保存至: {os.path.abspath(MODEL_SAVE_PATH)}")