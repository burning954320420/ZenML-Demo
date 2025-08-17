import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from zenml import pipeline, step
from zenml.logger import get_logger

# 创建 logger（必须传参）
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

    X = df[['cpu_usage']].values           # 特征：CPU 使用率（二维数组）
    y = df['is_normal'].values             # 标签：0=正常，1=异常

    logger.info(f"📊 特征矩阵形状: {X.shape}")
    logger.info(f"🏷️  标签向量形状: {y.shape}")

    return X, y

# ------------------ 步骤 3: 训练随机森林模型 ------------------
@step
def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """使用随机森林训练异常检测模型"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 使用随机森林分类器
    model = RandomForestClassifier(
        n_estimators=100,      # 100 棵树
        max_depth=5,           # 控制过拟合
        random_state=42,
        n_jobs=-1              # 使用所有 CPU 核心
    )
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"📈 测试集准确率: {acc:.4f}")

    logger.info("📋 分类报告:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["正常 (0)", "异常 (1)"]))

    logger.info("📊 混淆矩阵:")
    logger.info("\n" + str(confusion_matrix(y_test, y_pred)))

    return model

# ------------------ 步骤 4: 保存模型 ------------------
@step
def save_model(model: RandomForestClassifier, save_path: str = "cpu_anomaly_detector.pkl"):
    """将训练好的模型保存到本地"""
    import joblib
    joblib.dump(model, save_path)
    model_path = os.path.abspath(save_path)
    logger.info(f"💾 模型已保存至: {model_path}")
    return model_path

# ------------------ 定义 ZenML Pipeline ------------------
@pipeline
def cpu_training_pipeline(data_path: str, model_save_path: str):
    """基于随机森林的 CPU 异常检测训练流程"""
    df = load_data(data_path)
    X, y = preprocess_data(df)
    model = train_random_forest(X, y)
    save_model(model, model_save_path)

# ------------------ 主函数 ------------------
if __name__ == "__main__":
    DATA_PATH = "data/cpu_data.csv"              # 数据文件路径
    MODEL_SAVE_PATH = "cpu_anomaly_detector.pkl" # 模型保存路径

    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {os.path.abspath(DATA_PATH)}")

    # 运行 ZenML 管道
    cpu_training_pipeline(
        data_path=DATA_PATH,
        model_save_path=MODEL_SAVE_PATH
    )

    print(f"\n🎉 训练完成！随机森林模型已保存至: {os.path.abspath(MODEL_SAVE_PATH)}")