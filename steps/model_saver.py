# steps/model_saver.py
import joblib
import json
import os
from pathlib import Path
from zenml.steps import step

@step
def save_model_step(
    model,
    scaler,
    feature_names,
    training_history,
    model_path: str  # 例如: "models/v1/model.pkl"
):
    """
    保存模型、标准化器和训练历史为独立文件

    Args:
        model: 训练好的模型
        scaler: 特征标准化器
        feature_names: 特征名称列表
        training_history: 训练历史记录
        model_path: 模型主文件路径（用于推导其他路径）

    Returns:
        dict: 包含所有保存路径的字典
    """
    print(f"💾 开始保存模型组件...")

    # 确保目录存在
    model_dir = os.path.dirname(model_path)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # 推导各个文件的路径
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # ✅ 独立保存每个组件
    model_file = os.path.join(model_dir, f"{model_name}.pkl")           # model.pkl
    scaler_file = os.path.join(model_dir, f"{model_name}_scaler.pkl")   # model_scaler.pkl
    features_file = os.path.join(model_dir, f"{model_name}_feature_names.pkl")  # model_feature_names.pkl
    history_file = os.path.join(model_dir, f"{model_name}_history.json")

    # 1. 保存纯模型
    joblib.dump(model, model_file)
    print(f"   ✅ 模型已保存: {model_file}")

    # 2. 保存 scaler
    joblib.dump(scaler, scaler_file)
    print(f"   ✅ Scaler 已保存: {scaler_file}")

    # 3. 保存 feature_names
    joblib.dump(feature_names, features_file)
    print(f"   ✅ 特征名称已保存: {features_file}")

    # 4. 保存训练历史（JSON）
    with open(history_file, 'w') as f:
        history_copy = (training_history.copy() if training_history else {})
        if 'feature_importance' in history_copy and hasattr(history_copy['feature_importance'], 'to_dict'):
            history_copy['feature_importance'] = history_copy['feature_importance'].to_dict('records')
        json.dump(history_copy, f, indent=2, default=str)
    print(f"   ✅ 训练历史已保存: {history_file}")

    return {
        "model_path": model_file,
        "scaler_path": scaler_file,
        "feature_names_path": features_file,
        "history_path": history_file
    }