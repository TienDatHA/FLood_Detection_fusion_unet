import os
from pathlib import Path

def get_project_root() -> Path:
    """Get project root directory, allowing override via environment variable."""
    # Allow override via environment variable
    if "PROJECT_ROOT" in os.environ:
        return Path(os.environ["PROJECT_ROOT"]).resolve()
    
    # Default: use current script's directory as project root
    return Path(__file__).parent.resolve()

def get_data_root() -> Path:
    """Get data root directory, with fallback options."""
    # Priority 1: Environment variable
    if "DATA_ROOT" in os.environ:
        return Path(os.environ["DATA_ROOT"]).resolve()
    
    # Priority 2: Relative to project root
    project_root = get_project_root()
    relative_data = project_root / "data"
    if relative_data.exists():
        return relative_data
    
    # Priority 3: Original hardcoded path (as fallback)
    original_path = Path("/mnt/hdd2tb/Uni-Temporal-Flood-Detection-Sentinel-1_Frontiers22")
    if original_path.exists():
        return original_path
    
    # Priority 4: Create relative data directory
    relative_data.mkdir(parents=True, exist_ok=True)
    return relative_data

# === Main paths ===
PROJECT_ROOT = get_project_root()
ROOT_PATH = get_data_root()

# === Sen1Floods11 dataset paths ===
SEN1FLOODS_PATH = ROOT_PATH / "Sen1Flood11/v1.1"
CATALOG_PATH = SEN1FLOODS_PATH / "catalog"

# === Data paths ===
HAND_LABELED_SOURCE = CATALOG_PATH / "sen1floods11_hand_labeled_source"
HAND_LABELED_LABEL = CATALOG_PATH / "sen1floods11_hand_labeled_label"
WEAK_LABELED_SOURCE = CATALOG_PATH / "sen1floods11_weak_labeled_source"
WEAK_LABELED_LABEL = CATALOG_PATH / "sen1floods11_weak_labeled_label"

IMG_PATH = SEN1FLOODS_PATH / "data/flood_events/HandLabeled/S1Hand"  
LABEL_PATH = SEN1FLOODS_PATH / "data/flood_events/HandLabeled/LabelHand"
DEM_PATH = ROOT_PATH / "DEM_Patches"
JRC_PATH = SEN1FLOODS_PATH / "data/flood_events/HandLabeled/JRCWaterHand"
DATA_PATH = SEN1FLOODS_PATH / "data/flood_events"

WEIGHT_PATH = PROJECT_ROOT / "weights"
WEIGHT_FILE = "standard_checkpoint.h5"

# === Training parameters ===
LR = float(os.getenv("LEARNING_RATE", "0.0001"))
IMG_HEIGHT = int(os.getenv("IMG_HEIGHT", "512"))
IMG_WIDTH = int(os.getenv("IMG_WIDTH", "512"))
VAL_BATCH_SIZE = int(os.getenv("VAL_BATCH_SIZE", "1"))
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "1"))
CLASSES = ['flood']

# === Output paths ===
OUT_FOLDER = PROJECT_ROOT / "outputs"

def ensure_dirs():
    """Create necessary directories. Call this before training/inference."""
    WEIGHT_PATH.mkdir(parents=True, exist_ok=True)
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"📁 Directories ensured:")
    print(f"   Weights: {WEIGHT_PATH}")
    print(f"   Outputs: {OUT_FOLDER}")

# Legacy variable names for backward compatibility
lr = LR
val_batchSize = VAL_BATCH_SIZE  
train_batchSize = TRAIN_BATCH_SIZE
