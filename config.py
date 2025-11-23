from pathlib import Path

# === Đường dẫn chính ===
ROOT_PATH = Path("/mnt/hdd2tb/Uni-Temporal-Flood-Detection-Sentinel-1_Frontiers22")

# === Đường dẫn Sen1Floods11 dataset ===
SEN1FLOODS_PATH = ROOT_PATH / "Sen1Flood11/v1.1"
CATALOG_PATH = SEN1FLOODS_PATH / "catalog"

# === Đường dẫn dữ liệu ===
HAND_LABELED_SOURCE = CATALOG_PATH / "sen1floods11_hand_labeled_source"
HAND_LABELED_LABEL = CATALOG_PATH / "sen1floods11_hand_labeled_label"
WEAK_LABELED_SOURCE = CATALOG_PATH / "sen1floods11_weak_labeled_source"
WEAK_LABELED_LABEL = CATALOG_PATH / "sen1floods11_weak_labeled_label"

IMG_PATH = SEN1FLOODS_PATH / "data/flood_events/HandLabeled/S1Hand"  
LABEL_PATH = SEN1FLOODS_PATH / "data/flood_events/HandLabeled/LabelHand"  # Thư mục chứa labels
DEM_PATH = ROOT_PATH / "DEM_Patches"  # Thư mục chứa DEM data
JRC_PATH = SEN1FLOODS_PATH / "data/flood_events/HandLabeled/JRCWaterHand"  # Thư mục chứa JRC data
DATA_PATH = SEN1FLOODS_PATH / "data/flood_events"  # Backup data path

WEIGHT_PATH = ROOT_PATH / "weights"
WEIGHT_FILE = "standard_checkpoint.h5"  # Sửa tên biến cho consistent

lr = 0.0001
IMG_HEIGHT = 512
IMG_WIDTH = 512
val_batchSize = 1  # Giảm để tránh OOM error
train_batchSize = 1  # Giảm để tránh OOM error
CLASSES = ['flood']

OUT_FOLDER = ROOT_PATH / "outputs"

WEIGHT_PATH.mkdir(parents=True, exist_ok=True)
OUT_FOLDER.mkdir(parents=True, exist_ok=True)
