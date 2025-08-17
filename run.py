# main.py
import traceback
import sys
from pathlib import Path
from configs.config import DATA_PATH, MODEL_SAVE_PATH, setup_directories
from pipelines.training_pipeline import cpu_training_pipeline

if __name__ == "__main__":
    try:
        # 创建必要目录
        setup_directories()

        # 路径转字符串（ZenML 要求）
        data_path = str(DATA_PATH)
        model_save_path = str(MODEL_SAVE_PATH)

        # 检查数据文件
        if not Path(data_path).exists():
            raise FileNotFoundError(f"数据文件未找到: {Path(data_path).resolve()}")

        print(f"数据路径: {data_path}")
        print(f"模型保存路径: {model_save_path}")

        # 运行 pipeline
        pipeline_instance = cpu_training_pipeline(
            data_path=data_path,
            model_save_path=model_save_path
        )

        print(f"\n🎉 训练完成！模型已保存至: {Path(model_save_path).resolve()}")
    except Exception as e:
        print(f"发生错误: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        sys.exit(1)