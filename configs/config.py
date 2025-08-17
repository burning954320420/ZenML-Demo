# configs/config.py
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
# 数据路径
DATA_PATH = ROOT_DIR / "data" / "cpu_data_timestamp.csv"
# 模型保存路径
MODEL_SAVE_PATH = ROOT_DIR / "models" / "cpu_anomaly_detector_with_unix_time.pkl"

# 确保目录存在
def setup_directories():
    (ROOT_DIR / "models").mkdir(exist_ok=True)