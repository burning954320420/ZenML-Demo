# pipelines/training_pipeline.py
from zenml import pipeline
from steps.data_loader import load_data_step
from steps.data_validation import data_validation_step
from steps.feature_engineering import feature_engineering_step
from steps.model_trainer import train_model_step
from steps.model_saver import save_model_step


@pipeline(enable_cache=False)
def cpu_training_pipeline(data_path: str, model_save_path: str):
    """CPU 异常检测训练管道"""

    # 1. 数据加载
    raw_df = load_data_step(data_path)

    # 2. 数据质量验证（关键质量门禁）
    validation_results = data_validation_step(raw_df)

    # 3. 特征工程
    features_df, feature_list = feature_engineering_step(raw_df)

    # 4. 模型训练
    model, scaler, feature_names, training_history = train_model_step(features_df, feature_list)

    # 5. 模型保存
    save_model_step(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        training_history=training_history,
        model_path=model_save_path
    )