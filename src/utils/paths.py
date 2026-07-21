import os
from pathlib import Path

# Thư mục gốc của project (HealthSense-MachineLearning)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Thư mục chứa dữ liệu
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"

# Thư mục chứa model đã huấn luyện
MODELS_DIR = ROOT_DIR / "models"

# Tạo các thư mục nếu chưa tồn tại
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
