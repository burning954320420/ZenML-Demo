# steps/model_predictor.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from zenml import step
from zenml.logger import get_logger
from typing import Dict, List, Tuple, Any

logger = get_logger(__name__)

class CPUAnomalyPredictor:
    """CPU异常检测预测器"""
    
    def __init__(self, model_path: str):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径（其他相关文件会基于此路径推导）
        """
        self.model_path = Path(model_path)
        self.model_dir = self.model_path.parent
        self.model_name = self.model_path.stem
        
        # 推导其他文件路径
        self.scaler_path = self.model_dir / f"{self.model_name}_scaler.pkl"
        self.feature_names_path = self.model_dir / f"{self.model_name}_feature_names.pkl"
        
        # 加载组件
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_components()
        
        logger.info(f"✅ 预测器初始化完成，使用模型: {self.model_path}")
    
    def _load_components(self):
        """加载模型、标准化器和特征名称"""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ 模型加载成功: {self.model_path}")
            
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"✅ 标准化器加载成功: {self.scaler_path}")
            
            self.feature_names = joblib.load(self.feature_names_path)
            logger.info(f"✅ 特征名称加载成功: {self.feature_names_path}")
            logger.info(f"   特征数量: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"❌ 加载模型组件失败: {e}")
            raise
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        对输入数据进行预测
        
        Args:
            df: 包含特征的DataFrame
            
        Returns:
            预测结果和预测概率
        """
        # 检查特征
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            logger.warning(f"⚠️ 缺少特征: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            for feature in missing_features:
                df[feature] = 0  # 填充缺失特征
        
        # 提取特征
        X = df[self.feature_names].copy()
        
        # 处理无穷值和缺失值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 预测
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]  # 异常的概率
        
        return y_pred, y_proba

@step
def predict_anomalies_step(
    df: pd.DataFrame,
    model_path: str,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    ZenML Step: 使用训练好的模型预测CPU异常
    
    Args:
        df: 输入数据
        model_path: 模型路径
        threshold: 异常概率阈值
        
    Returns:
        包含预测结果的字典
    """
    logger.info(f"🔍 开始异常检测预测...")
    
    # 初始化预测器
    predictor = CPUAnomalyPredictor(model_path)
    
    # 执行预测
    y_pred, y_proba = predictor.predict(df)
    
    # 应用自定义阈值（如果不是默认值）
    if threshold != 0.5:
        logger.info(f"🎚️ 使用自定义阈值: {threshold}")
        y_pred_custom = (y_proba >= threshold).astype(int)
    else:
        y_pred_custom = y_pred
    
    # 统计结果
    anomaly_count = y_pred_custom.sum()
    total_count = len(y_pred_custom)
    anomaly_ratio = anomaly_count / total_count if total_count > 0 else 0
    
    logger.info(f"✅ 预测完成:")
    logger.info(f"   总样本数: {total_count}")
    logger.info(f"   检测到的异常数: {anomaly_count}")
    logger.info(f"   异常比例: {anomaly_ratio:.2%}")
    
    # 返回结果
    return {
        "predictions": y_pred_custom,
        "probabilities": y_proba,
        "anomaly_count": int(anomaly_count),
        "total_count": total_count,
        "anomaly_ratio": float(anomaly_ratio)
    }