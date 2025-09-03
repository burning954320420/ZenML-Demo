# run.py
import traceback
import sys
from pathlib import Path
from configs.config import DATA_PATH, MODEL_SAVE_PATH, setup_directories
from pipelines.training_pipeline import cpu_training_pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)

def run_training(data_path=None, model_path=None):
    """
    运行CPU异常检测训练管道
    
    Args:
        data_path: 数据文件路径，默认使用配置中的路径
        model_path: 模型保存路径，默认使用配置中的路径
        
    Returns:
        训练结果信息字典
    """
    try:
        # 创建必要目录
        setup_directories()
        
        # 设置默认值
        data_path = data_path or DATA_PATH
        model_path = model_path or MODEL_SAVE_PATH
        
        # 路径转字符串（ZenML 要求）
        data_path = str(data_path)
        model_path = str(model_path)
        
        # 检查数据文件
        if not Path(data_path).exists():
            raise FileNotFoundError(f"数据文件未找到: {Path(data_path).resolve()}")
        
        logger.info(f"📄 数据路径: {data_path}")
        logger.info(f"🤖 模型保存路径: {model_path}")
        
        # 运行 pipeline
        pipeline_instance = cpu_training_pipeline(
            data_path=data_path,
            model_save_path=model_path
        )
        
        logger.info(f"✅ 训练完成！模型已保存至: {Path(model_path).resolve()}")
        
        # 返回训练结果信息
        return {
            "pipeline_instance": pipeline_instance,
            "model_path": model_path
        }
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        raise

if __name__ == "__main__":
    try:
        run_training()
        print(f"\n🎉 训练完成！模型已保存至: {Path(MODEL_SAVE_PATH).resolve()}")
    except Exception as e:
        print(f"发生错误: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        sys.exit(1)