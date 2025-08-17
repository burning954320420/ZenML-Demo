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
    model_path: str
):
    """
    保存模型、标准化器和训练历史

    Args:
        model: 训练好的模型
        scaler: 特征标准化器
        feature_names: 特征名称列表
        training_history: 训练历史记录
        model_path: 模型保存路径

    Returns:
        dict: 包含保存路径的字典
    """
    print(f"💾 保存模型...")

    # 确保目录存在
    model_dir = os.path.dirname(model_path)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # 构建历史文件路径
    history_path = os.path.join(
        model_dir,
        f"{os.path.splitext(os.path.basename(model_path))[0]}_history.json"
    )

    # 保存模型和标准化器
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }, model_path)

    # 保存训练历史
    with open(history_path, 'w') as f:
        # 将不能序列化的对象转换为可序列化格式
        history_copy = training_history.copy() if training_history else {}

        # 处理DataFrame类型的特征重要性
        if 'feature_importance' in history_copy and hasattr(history_copy['feature_importance'], 'to_dict'):
            history_copy['feature_importance'] = history_copy['feature_importance'].to_dict('records')

        json.dump(history_copy, f, indent=2)

    print(f"   ✅ 模型保存至: {model_path}")
    print(f"   ✅ 历史保存至: {history_path}")

    return {
        "model_path": model_path,
        "history_path": history_path
    }
