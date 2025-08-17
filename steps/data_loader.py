# steps/data_loader.py
import pandas as pd
from zenml import step
from zenml.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)

@step
def load_data_step(data_path: str) -> pd.DataFrame:
    """加载数据并打印调试信息"""
    data_path = Path(data_path)  # 确保是 Path 对象

    # 🔥 打印：正在读取哪个文件
    print(f"\n📄 正在读取数据文件: {data_path.resolve()}")
    if not data_path.exists():
        raise FileNotFoundError(f"文件不存在: {data_path.resolve()}")

    try:
        # 使用 utf-8-sig 处理 BOM
        df = pd.read_csv(data_path, encoding='utf-8-sig')

        # 🔥 打印：CSV 第一行（列名）
        print(f"📋 CSV 文件列名: {df.columns.tolist()}")
        print(f"🔍 列名原始表示 (repr): {[repr(col) for col in df.columns]}")

        # 清理列名：去除引号、空格
        df.columns = df.columns.str.replace(r"[\'\"\s]+$|^[\'\"\s]+", "", regex=True)
        print(f"✅ 清理后列名: {df.columns.tolist()}")

        required_columns = ['timestamp', 'cpu_utilization', 'is_anomaly']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"缺少列: {missing}，当前列: {df.columns.tolist()}")

        logger.info(f"✅ 成功加载数据，共 {len(df)} 条记录")
        return df

    except Exception as e:
        logger.error(f"❌ 加载数据失败: {e}")
        raise