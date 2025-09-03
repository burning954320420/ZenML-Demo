# pipelines/inference_pipeline.py
from zenml import pipeline
from steps.data_loader import load_data_step
from steps.feature_engineering import feature_engineering_step
from steps.model_predictor import predict_anomalies_step

@pipeline(enable_cache=False)
def cpu_inference_pipeline(data_path: str, model_path: str, threshold: float = 0.5):
    """CPU 异常检测推理管道"""
    
    # 1. 数据加载
    raw_df = load_data_step(data_path)
    
    # 2. 特征工程（与训练时相同的特征处理）
    features_df, feature_list = feature_engineering_step(raw_df)
    
    # 3. 模型预测
    prediction_results = predict_anomalies_step(
        df=features_df,
        model_path=model_path,
        threshold=threshold
    )
    
    return prediction_results