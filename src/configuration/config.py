from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent

# 源数据目录
RAW_DATA_DIR = ROOT_DIR / 'data'/'raw'

# 预处理之后数据目录
PROCESSED_DATA_DIR = ROOT_DIR /'data'/'processed'

# 日志
LOGS_DIR = ROOT_DIR / 'logs'

# 保存训练好的模型
SAVE_MODELS_DIR = ROOT_DIR / 'models'

# 预训练模型
PRETRAINED_DIR = ROOT_DIR /'pretrained'