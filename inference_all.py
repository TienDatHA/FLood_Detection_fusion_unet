import tensorflow as tf
import numpy as np
import rasterio
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from skimage.morphology import opening, closing, remove_small_objects, remove_small_holes, disk
import seaborn as sns
from rasterio.transform import Affine
from config import *  # expects ROOT_PATH, WEIGHT_PATH, IMG_HEIGHT, IMG_WIDTH
from model import Resnet50_UNet
from utils import scale_img

# =========================
# CẤU HÌNH NGƯỠNG RIÊNG BIỆT
# =========================
INPUT_THRESHOLD = 0.6  # Ngưỡng cho đầu vào: JRC occurrence, binary conversion
POST_PROCESS_THRESHOLD = 0.3  # Ngưỡng cho hậu xử lý: morphology, filtering

# =========================
# CẤU HÌNH ĐƯỜNG DẪN 
# =========================
INFER_DATA_PATH   = ROOT_PATH / "Data"
BENCH_BASE = ROOT_PATH / "Bench_Mark"
INFER_S1_PATH     = BENCH_BASE / "Sentinel1_Tiles"
INFER_DEM_PATH    = BENCH_BASE / "DEM_Tiles_4326"
INFER_JRC_PATH    = BENCH_BASE / "JRC_Tiles_Cut"   
INFER_LABEL_PATH  = BENCH_BASE / "Labels_Tiles_filtered"

# Đường dẫn output - sửa để lưu vào benchmarks
INFER_OUTPUT_PATH   = ROOT_PATH / "inference_results" / "benchmarks" / "BacGiang_20240910"
PRED_OUTPUT_PATH    = INFER_OUTPUT_PATH / "predictions"
METRICS_OUTPUT_PATH = INFER_OUTPUT_PATH / "metrics"
RAW_OUTPUT_PATH     = INFER_OUTPUT_PATH / "raw_predictions"
MORPH_OUTPUT_PATH   = INFER_OUTPUT_PATH / "morph_predictions" 
REFINED_OUTPUT_PATH = INFER_OUTPUT_PATH / "refined_predictions"
VIS_OUTPUT_PATH     = INFER_OUTPUT_PATH / "visualizations"

# Tạo tất cả thư mục output
(INFER_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
(PRED_OUTPUT_PATH).mkdir(exist_ok=True)
(METRICS_OUTPUT_PATH).mkdir(exist_ok=True)
(RAW_OUTPUT_PATH).mkdir(exist_ok=True)
(MORPH_OUTPUT_PATH).mkdir(exist_ok=True)
(REFINED_OUTPUT_PATH).mkdir(exist_ok=True)
(VIS_OUTPUT_PATH).mkdir(exist_ok=True)


import os

def _env_path(name, default_path):
    v = os.environ.get(name)
    return Path(v) if v else default_path

# Replace module-level paths with environment-provided ones if present
INFER_DATA_PATH = _env_path('INFER_DATA_PATH', INFER_DATA_PATH)
INFER_S1_PATH = _env_path('INFER_S1_PATH', INFER_S1_PATH)
INFER_DEM_PATH = _env_path('INFER_DEM_PATH', INFER_DEM_PATH)
INFER_JRC_PATH = _env_path('INFER_JRC_PATH', INFER_JRC_PATH)
INFER_LABEL_PATH = _env_path('INFER_LABEL_PATH', INFER_LABEL_PATH)

# Also override outputs if provided
INFER_OUTPUT_PATH = _env_path('INFER_OUTPUT_PATH', INFER_OUTPUT_PATH)
PRED_OUTPUT_PATH = _env_path('PRED_OUTPUT_PATH', PRED_OUTPUT_PATH)
METRICS_OUTPUT_PATH = _env_path('METRICS_OUTPUT_PATH', METRICS_OUTPUT_PATH)
RAW_OUTPUT_PATH = _env_path('RAW_OUTPUT_PATH', RAW_OUTPUT_PATH)
MORPH_OUTPUT_PATH = _env_path('MORPH_OUTPUT_PATH', MORPH_OUTPUT_PATH)
REFINED_OUTPUT_PATH = _env_path('REFINED_OUTPUT_PATH', REFINED_OUTPUT_PATH)
VIS_OUTPUT_PATH = _env_path('VIS_OUTPUT_PATH', VIS_OUTPUT_PATH)

# Ensure overridden output dirs exist
INFER_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
PRED_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
METRICS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
RAW_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MORPH_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
REFINED_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
VIS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# If INFER_JRC_PATH points to a region folder (e.g. Bench_Mark/JRC_Tiles_Cut/BacGiang_20240910),
# and S1/DEM/LABEL env vars were not set explicitly, derive their paths from Bench_Mark layout.
try:
    # If INFER_JRC_PATH is a directory containing *_occurrence_jrc.tif files, assume it's per-region
    if INFER_JRC_PATH.is_dir():
        sample = next(INFER_JRC_PATH.glob('*_occurrence_jrc.tif'), None)
        if sample is not None:
            # region name is the prefix before the first underscore in filename or folder name
            region_name = INFER_JRC_PATH.name if INFER_JRC_PATH.name and '_' in INFER_JRC_PATH.name else sample.name.split('_')[0]
            bench = BENCH_BASE
            # Only set defaults if not explicitly overridden via env
            if os.environ.get('INFER_S1_PATH') is None:
                INFER_S1_PATH = bench / 'Sentinel1_Tiles' / region_name
            if os.environ.get('INFER_DEM_PATH') is None:
                INFER_DEM_PATH = bench / 'DEM_Tiles_4326' / region_name
            if os.environ.get('INFER_LABEL_PATH') is None:
                INFER_LABEL_PATH = bench / 'Labels_Tiles_filtered' / region_name
except Exception:
    pass

# =========================
# HÀM HẬU XỬ LÝ
# =========================
def smart_noise_filter(binary_mask: np.ndarray, 
                      min_area: int = 10,
                      max_noise_area: int = 50, 
                      min_river_aspect_ratio: float = 2.5) -> np.ndarray:
    """
    Lọc thông minh: loại bỏ đốm nhỏ (noise) nhưng giữ sông mỏng
    
    Args:
        binary_mask: Mask nhị phân đầu vào
        min_area: Diện tích tối thiểu (pixels)
        max_noise_area: Diện tích tối đa coi là noise (pixels)
        min_river_aspect_ratio: Tỷ lệ dài/rộng tối thiểu để coi là sông
    """
    from skimage.measure import label, regionprops
    
    # Label các connected components
    labeled = label(binary_mask)
    result = np.zeros_like(binary_mask, dtype=bool)
    
    for region in regionprops(labeled):
        area = region.area
        
        # Loại bỏ objects quá nhỏ
        if area < min_area:
            continue
            
        # Giữ lại objects lớn (không phải noise)
        if area > max_noise_area:
            result[labeled == region.label] = True
            continue
            
        # Đối với objects vừa (min_area <= area <= max_noise_area)
        # Kiểm tra aspect ratio để phân biệt noise vs sông
        if region.major_axis_length > 0 and region.minor_axis_length > 0:
            aspect_ratio = region.major_axis_length / region.minor_axis_length
            
            # Nếu có aspect ratio cao -> có thể là sông mỏng
            if aspect_ratio >= min_river_aspect_ratio:
                result[labeled == region.label] = True
            # Nếu aspect ratio thấp -> có thể là noise, loại bỏ
            # (không làm gì, để False)
    
    return result


def post_process_mask_advanced(pred_mask: np.ndarray, 
                               jrc_mask: np.ndarray, 
                               threshold: float = POST_PROCESS_THRESHOLD,
                               morphology_level: str = "medium",  # "light", "medium", "strong", "river_optimized"
                               allow_growth: bool = False,
                               preserve_thin_objects: bool = True,  # NEW: Bảo vệ đối tượng mỏng
                               return_intermediate: bool = False) -> np.ndarray:
    """Hậu xử lý mask với thứ tự MỚI: inference → lấp đầy & xóa nhiễu → trừ JRC → làm mịn cạnh.
    Các bước:
      1) Binarize theo threshold.
      2) Lấp đầy hố nhỏ và xóa nhiễu (remove_small_holes + smart_noise_filter).
      3) Loại nước tĩnh theo JRC (ngưỡng 30%).
      4) Làm mịn cạnh bằng morphological operations (closing + opening).
      5) Nếu allow_growth=False → đảm bảo mask ⊆ mask gốc.
    
    Args:
        return_intermediate: Nếu True, trả về tuple (final_result, after_fill_denoise)
    """
    mask_bin = (pred_mask > threshold).astype(bool)
    refined = mask_bin.copy()

    # === BƯỚC 2: LẤP ĐẦY & XÓA NHIỄU TRƯỚC ===
    if morphology_level == "light":
        hole_sz = 30;  k = 1  # Rất nhẹ cho sông mỏng
        smart_filter_params = {'min_area': 5, 'max_noise_area': 25, 'min_river_aspect_ratio': 2.0}
    elif morphology_level == "strong":
        hole_sz = 400; k = 3
        smart_filter_params = {'min_area': 20, 'max_noise_area': 100, 'min_river_aspect_ratio': 3.0}
    elif morphology_level == "river_optimized":  # Tối ưu cho sông
        hole_sz = 100; k = 2  # Cân bằng tốt
        smart_filter_params = {'min_area': 10, 'max_noise_area': 50, 'min_river_aspect_ratio': 2.5}
    elif morphology_level == "balanced":  # NEW: Nhẹ nhàng, bảo vệ vùng lớn
        hole_sz = 150; k = 2  # Tăng để giảm nhiễu và lấp đầy đẹp hơn
        smart_filter_params = {'min_area': 8, 'max_noise_area': 40, 'min_river_aspect_ratio': 2.2}
    else:  # medium
        hole_sz = 250; k = 2
        smart_filter_params = {'min_area': 15, 'max_noise_area': 75, 'min_river_aspect_ratio': 2.5}

    print(f"    🔧 Step 1 - Fill holes & denoise ({morphology_level}): hole_size={hole_sz}, smart_filter")
    
    # 1) LẤP ĐẦY HỐ NHỎ
    refined = remove_small_holes(refined, area_threshold=hole_sz)
    
    # 2) XÓA NHIỄU BẰNG SMART FILTER
    refined = smart_noise_filter(refined, **smart_filter_params)
    
    # Lưu kết quả sau fill & denoise để visualization
    after_fill_denoise = refined.copy().astype(np.float32)

    # === BƯỚC 3: TRỪ JRC SAU KHI ĐÃ LẤP ĐẦY & XÓA NHIỄU ===
    if jrc_mask is not None:
        permanent = (jrc_mask == 1.0)  # JRC occurrence > 30%
        jrc_removed_pixels = np.sum(refined & permanent)
        total_flood_pixels = np.sum(refined)
        refined[permanent] = False
        print(f"    🌊 Step 2 - JRC filtering: removed {jrc_removed_pixels}/{total_flood_pixels} flood pixels (permanent water)")

    # === BƯỚC 4: LÀM MỊN CẠNH SAU CÙNG ===
    print(f"    ✨ Step 3 - Edge smoothing: kernel={k}")
    
    if k > 0:
        # 1) Closing để kết nối các phần gần nhau
        refined = closing(refined, disk(k))
        
        # 2) Opening để làm mịn cạnh (nếu cần)
        if morphology_level not in ["light", "balanced"]:  # Không opening với light và balanced level
            if preserve_thin_objects:
                # Lưu lại đối tượng mỏng trước khi opening
                original_objects = refined.copy()
                refined = opening(refined, disk(max(1, k-1)))  # Kernel nhỏ hơn
                
                # Tìm đối tượng bị mất do opening
                lost_objects = original_objects & (~refined)
                
                # Chỉ khôi phục những đối tượng mỏng (có aspect ratio cao)
                from skimage.measure import label, regionprops
                labeled = label(lost_objects)
                for region in regionprops(labeled):
                    # Tính aspect ratio (chiều dài / chiều rộng)
                    if region.major_axis_length > 0 and region.minor_axis_length > 0:
                        aspect_ratio = region.major_axis_length / region.minor_axis_length
                        if aspect_ratio > 3:  # Đối tượng mỏng (ratio > 3)
                            refined[labeled == region.label] = True
            else:
                refined = opening(refined, disk(max(1, k-1)))  # Kernel nhỏ hơn

    # === BƯỚC 5: NO-GROWTH CONSTRAINT ===
    if not allow_growth:
        refined = refined & mask_bin

    final_result = refined.astype(np.float32)
    
    if return_intermediate:
        return final_result, after_fill_denoise
    else:
        return final_result


# =========================
# LOAD & PREPROCESS PATCH
# =========================
def load_and_preprocess_patch(s1_filename: str):
    try:
        s1_path = INFER_S1_PATH / s1_filename
        with rasterio.open(s1_path) as src:
            s1_data = src.read().astype(np.float32)
            s1_crs = src.crs
            s1_transform = src.transform
        if s1_data.shape[0] == 2:
            vv_img, vh_img = s1_data[0], s1_data[1]
        elif s1_data.shape[0] == 1:
            vv_img = s1_data[0]
            vh_img = s1_data[0].copy()
        else:
            vv_img, vh_img = s1_data[0], s1_data[1]

        orig_h, orig_w = vv_img.shape
        target_size = (IMG_HEIGHT, IMG_WIDTH)
        if (orig_h, orig_w) != target_size:
            vv_img = cv.resize(vv_img, target_size, interpolation=cv.INTER_AREA)
            vh_img = cv.resize(vh_img, target_size, interpolation=cv.INTER_AREA)
        scale_x = orig_w / target_size[1]
        scale_y = orig_h / target_size[0]
        new_transform = Affine(s1_transform.a * scale_x, s1_transform.b, s1_transform.c,
                               s1_transform.d, s1_transform.e * scale_y, s1_transform.f)

        x_arr = np.stack([vv_img, vh_img], axis=-1)
        valid_mask = np.isfinite(vv_img) & np.isfinite(vh_img)

        base = s1_filename.replace('.tif', '')
        # Convert format from "BinhDinh_20171021_patch_025_002" to "BinhDinh_20171021_025-002"
        if '_patch_' in base:
            parts = base.split('_patch_')
            if len(parts) == 2 and '_' in parts[1]:
                coord_parts = parts[1].split('_', 1)
                base = f"{parts[0]}_{coord_parts[0]}-{coord_parts[1]}"
        dem_filename = f"{base}_dem.tif"
        dem_path = INFER_DEM_PATH / dem_filename
        try:
            with rasterio.open(dem_path) as dsrc:
                dem_img = dsrc.read(1).astype(np.float32)
            if dem_img.shape != target_size:
                dem_img = cv.resize(dem_img, target_size, interpolation=cv.INTER_AREA)
        except Exception:
            print(f"Warning: DEM not found for {dem_filename}, using zeros")
            dem_img = np.zeros(target_size, dtype=np.float32)

        jrc_filename = f"{base}_occurrence_jrc.tif"
        jrc_path = INFER_JRC_PATH / jrc_filename
        jrc_img = None
        try:
            with rasterio.open(jrc_path) as jsrc:
                j = jsrc.read(1).astype(np.float32)
            if j.shape != target_size:
                j = cv.resize(j, target_size, interpolation=cv.INTER_NEAREST)
            occurrence_threshold = INPUT_THRESHOLD * 100  # Convert to percentage (e.g., 0.8 -> 80.0)
            jrc_img = (j > occurrence_threshold).astype(np.float32)
            pw = int(np.sum(jrc_img == 1.0)); tot = jrc_img.size
            print(f"    JRC occurrence (>{occurrence_threshold}%): {pw}/{tot} ({pw/tot:.3f}) permanent water")
        except Exception:
            print(f"Warning: JRC occurrence not found for {jrc_filename}, trying extent...")
            try:
                fb_name = f"{base}_extent_jrc.tif"
                with rasterio.open(INFER_JRC_PATH / fb_name) as esrc:
                    e = esrc.read(1).astype(np.float32)
                if e.shape != target_size:
                    e = cv.resize(e, target_size, interpolation=cv.INTER_NEAREST)
                extent_threshold = INPUT_THRESHOLD
                jrc_img = (e > extent_threshold).astype(np.float32)
                pw = int(np.sum(jrc_img == 1.0)); tot = jrc_img.size
                print(f"    JRC extent (>{extent_threshold}): {pw}/{tot} ({pw/tot:.3f}) permanent water")
            except Exception:
                print("Warning: No suitable JRC data found, using zeros")
                jrc_img = np.zeros(target_size, dtype=np.float32)

        x_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        x_img[..., :2] = x_arr
        x_img[..., 2]  = jrc_img

        x_inf = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        x_inf[..., :2] = x_arr
        x_inf[..., 2]  = dem_img

        x_img = scale_img(x_img)
        x_inf = scale_img(x_inf)

        return x_img, x_inf, jrc_img, s1_crs, new_transform, valid_mask
    except Exception as e:
        print(f"Error loading {s1_filename}: {e}")
        return None, None, None, None, None, None


# =========================
# GROUND TRUTH
# =========================
def load_ground_truth(filename: str):
    try:
        base = filename.replace('.tif', '')
        # Convert format from "BinhDinh_20171021_patch_025_002" to "BinhDinh_20171021_025-002"
        if '_patch_' in base:
            parts = base.split('_patch_')
            if len(parts) == 2 and '_' in parts[1]:
                coord_parts = parts[1].split('_', 1)
                base = f"{parts[0]}_{coord_parts[0]}-{coord_parts[1]}"
        label_filename = f"{base}_label.tif"
        # Try per-region folder first (INFER_LABEL_PATH/region/filename), then fallback to flat folder
        label_path_candidates = [INFER_LABEL_PATH / label_filename,
                                 INFER_LABEL_PATH / Path(INFER_JRC_PATH).name / label_filename]
        label_path = None
        for p in label_path_candidates:
            if p.exists():
                label_path = p
                break
        if label_path is None:
            raise FileNotFoundError(f"Label not found in candidates: {label_path_candidates}")
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.float32)
        if label.shape != (IMG_HEIGHT, IMG_WIDTH):
            label = cv.resize(label, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_NEAREST)
        return (label > 0).astype(np.float32)
    except Exception:
        print(f"Warning: Label not found for {label_filename}")
        return None


# =========================
# METRICS & LƯU KẾT QUẢ
# =========================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = INPUT_THRESHOLD):
    y_pred_bin = (y_pred > threshold).astype(int)
    y_true_bin = (y_true > threshold).astype(int)
    yt = y_true_bin.flatten(); yp = y_pred_bin.flatten()

    acc = accuracy_score(yt, yp)
    pre = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    f1  = f1_score(yt, yp, zero_division=0)
    inter = np.logical_and(y_true_bin, y_pred_bin).sum()
    uni   = np.logical_or(y_true_bin, y_pred_bin).sum()
    iou = inter / uni if uni > 0 else 0.0
    try:
        ssim_score = ssim(y_true, y_pred, data_range=1.0)
    except Exception:
        ssim_score = 0.0
    return {
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'f1_score': f1,
        'iou': iou,
        'ssim': ssim_score,
        'confusion_matrix': confusion_matrix(yt, yp)
    }


def save_prediction(pred_mask: np.ndarray, filename: str, output_dir: Path,
                    crs, transform, bin_threshold: float = INPUT_THRESHOLD):
    out = output_dir / filename
    pred_bin = (pred_mask > bin_threshold).astype(np.uint8)
    with rasterio.open(out, 'w', driver='GTiff', height=pred_bin.shape[0], width=pred_bin.shape[1],
                       count=1, dtype=rasterio.uint8, crs=crs, transform=transform) as dst:
        dst.write(pred_bin, 1)


def save_diff(pred_mask: np.ndarray, refined_mask: np.ndarray, filename: str,
              output_dir: Path, crs, transform, bin_threshold: float = INPUT_THRESHOLD):
    output_path = output_dir / filename
    pred_bin = (pred_mask > bin_threshold).astype(np.int8)
    ref_bin  = (refined_mask > bin_threshold).astype(np.int8)
    diff = pred_bin - ref_bin  # {-1,0,1}
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=diff.shape[0],
        width=diff.shape[1],
        count=1,
        dtype=rasterio.int8,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(diff, 1)


def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Flood', 'Flood'], yticklabels=['No Flood', 'Flood'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_inference_patch_detailed(s1_img: np.ndarray, jrc_img: np.ndarray,
                                       pred_original: np.ndarray, pred_fill_denoise: np.ndarray,
                                       refined_mask: np.ndarray, label: np.ndarray, save_path: Path):
    """
    Hiển thị đầy đủ 6 ảnh: Sentinel-1, dự đoán gốc, sau fill & denoise, JRC, kết quả cuối cùng, label
    """
    plt.figure(figsize=(18, 12))
    
    # Chuẩn bị Sentinel-1 RGB
    if s1_img.ndim == 3 and s1_img.shape[-1] == 2:
        vv, vh = s1_img[..., 0], s1_img[..., 1]
    elif s1_img.ndim == 3 and s1_img.shape[0] == 2:
        vv, vh = s1_img[0], s1_img[1]
    else:
        vv = s1_img[..., 0] if s1_img.ndim == 3 else s1_img
        vh = s1_img[..., 1] if (s1_img.ndim == 3 and s1_img.shape[-1] >= 2) else s1_img[..., 0]
    rgb = np.stack([vv, vh, vv - vh], axis=-1)
    for i in range(3):
        ch = rgb[..., i]
        p2, p98 = np.percentile(ch, (2, 98))
        rgb[..., i] = np.clip((ch - p2) / (p98 - p2), 0, 1) if p98 > p2 else np.zeros_like(ch)

    # Hiển thị 6 ảnh theo layout 2x3
    plt.subplot(2, 3, 1)
    plt.imshow(rgb)
    plt.title('1. Sentinel-1 RGB\n(VV, VH, VV-VH)', fontsize=12, pad=10)
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(pred_original, cmap='Reds', vmin=0, vmax=1)
    plt.title('2. Dự đoán gốc\n(Model output)', fontsize=12, pad=10)
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(pred_fill_denoise, cmap='Oranges', vmin=0, vmax=1)
    plt.title('3. Fill holes & Denoise\n(Lấp đầy + Xóa nhiễu)', fontsize=12, pad=10)
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(jrc_img, cmap='Blues', vmin=0, vmax=1)
    plt.title('4. JRC Permanent Water\n(Occurrence > 30%)', fontsize=12, pad=10)
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(refined_mask, cmap='Purples', vmin=0, vmax=1)
    plt.title('5. Kết quả cuối cùng\n(- JRC + Edge smooth)', fontsize=12, pad=10)
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(label, cmap='Greens', vmin=0, vmax=1)
    plt.title('6. Ground Truth\n(Label)', fontsize=12, pad=10)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_inference_patch(s1_img: np.ndarray, jrc_img: np.ndarray,
                               refined_mask: np.ndarray, label: np.ndarray, save_path: Path):
    plt.figure(figsize=(12, 8))
    if s1_img.ndim == 3 and s1_img.shape[-1] == 2:
        vv, vh = s1_img[..., 0], s1_img[..., 1]
    elif s1_img.ndim == 3 and s1_img.shape[0] == 2:
        vv, vh = s1_img[0], s1_img[1]
    else:
        vv = s1_img[..., 0] if s1_img.ndim == 3 else s1_img
        vh = s1_img[..., 1] if (s1_img.ndim == 3 and s1_img.shape[-1] >= 2) else s1_img[..., 0]
    rgb = np.stack([vv, vh, vv - vh], axis=-1)
    for i in range(3):
        ch = rgb[..., i]
        p2, p98 = np.percentile(ch, (2, 98))
        rgb[..., i] = np.clip((ch - p2) / (p98 - p2), 0, 1) if p98 > p2 else np.zeros_like(ch)

    plt.subplot(2, 2, 1); plt.imshow(rgb); plt.title('Sentinel-1 RGB'); plt.axis('off')
    plt.subplot(2, 2, 2); plt.imshow(jrc_img, cmap='Blues', vmin=0, vmax=1); plt.title('JRC Permanent Water'); plt.axis('off')
    plt.subplot(2, 2, 3); plt.imshow(refined_mask, cmap='Reds', vmin=0, vmax=1); plt.title(f'Prediction (Refined, thres={POST_PROCESS_THRESHOLD})'); plt.axis('off')
    plt.subplot(2, 2, 4); plt.imshow(label, cmap='Greens', vmin=0, vmax=1); plt.title('Ground Truth'); plt.axis('off')
    plt.tight_layout(); plt.savefig(save_path, dpi=200); plt.close()


# =========================
# LOAD MODEL
# =========================
def load_model():
    in_img = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='img_input')
    in_inf = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='inf_input')
    model = Resnet50_UNet(n_classes=1, in_img=in_img, in_inf=in_inf)

    weight_path = WEIGHT_PATH / "Fusion_unet_checkpoint.h5"
    if not weight_path.exists():
        weight_path = ROOT_PATH / "Fusion_unet_checkpoint.h5"
    if weight_path.exists():
        print(f"Loading weights: {weight_path}")
        model.load_weights(str(weight_path))
    else:
        raise FileNotFoundError(f"Không tìm thấy weights file: {weight_path}")
    return model


# =========================
# MAIN
# =========================
def main():
    print("Đang load model…")
    model = load_model()

    s1_files = list(INFER_S1_PATH.glob("*.tif"))
    print(f"Tìm thấy {len(s1_files)} files để inference")

    all_metrics_original = []
    all_metrics_refined  = []
    batch_pred_original  = []
    batch_pred_refined   = []
    batch_gt             = []

    print("\nBắt đầu inference…")
    for i, s1_file in enumerate(tqdm(s1_files)):
        fname = s1_file.name
        x_img, x_inf, jrc_img, crs, transform, valid_mask = load_and_preprocess_patch(fname)
        if x_img is None:
            continue

        gt = load_ground_truth(fname)

        pred = model.predict([np.expand_dims(x_img, 0), np.expand_dims(x_inf, 0)], verbose=0)[0]
        pred = np.squeeze(pred)

        print(f"Processing {fname}…")
        refined, after_fill_denoise = post_process_mask_advanced(
            pred, jrc_img, 
            threshold=POST_PROCESS_THRESHOLD, 
            morphology_level="balanced",  # Đổi sang level cân bằng, nhẹ nhàng hơn
            preserve_thin_objects=True,  # Bảo vệ sông mỏng
            return_intermediate=True
        )

        # ====== ÁP MẶT NẠ HỢP LỆ ======
        pred_masked         = np.where(valid_mask, pred, 0.0)
        refined_masked      = np.where(valid_mask, refined, 0.0)
        fill_denoise_masked = np.where(valid_mask, after_fill_denoise, 0.0)  # NEW: sau fill & denoise

        # ====== LƯU CÁC LOẠI ẢNH VÀO THỬMỤC TƯƠNG ỨNG ======
        # 1) Dự đoán CHƯA morphology, CHƯA trừ JRC (raw) - lưu vào RAW_OUTPUT_PATH
        save_prediction(pred_masked, f"raw_{fname}", RAW_OUTPUT_PATH, crs, transform)

        # 2) Dự đoán ĐÃ fill & denoise, CHƯA trừ JRC (fill-denoise) - lưu vào MORPH_OUTPUT_PATH
        save_prediction(fill_denoise_masked, f"fill_denoise_{fname}", MORPH_OUTPUT_PATH, crs, transform)

        # 3) Dự đoán KẾT QUẢ CUỐI (ĐÃ fill & denoise & ĐÃ trừ JRC & ĐÃ smooth) - lưu vào REFINED_OUTPUT_PATH
        save_prediction(refined_masked, f"refined_{fname}", REFINED_OUTPUT_PATH, crs, transform)

        # 4) JRC permanent water reference - lưu vào PRED_OUTPUT_PATH
        save_prediction(jrc_img, f"jrc_perm_{fname}", PRED_OUTPUT_PATH, crs, transform)

        # 5) Ảnh hiệu giữa raw và final - lưu vào PRED_OUTPUT_PATH
        save_diff(pred_masked, refined_masked, f"pred_diff_{fname}", PRED_OUTPUT_PATH, crs, transform)

        # 6) Lưu JRC đã qua ngưỡng vào thư mục riêng để merge sau này
        jrc_threshold_dir = INFER_OUTPUT_PATH / "jrc_threshold"
        jrc_threshold_dir.mkdir(exist_ok=True)
        save_prediction(jrc_img, f"jrc_thresh_{fname}", jrc_threshold_dir, crs, transform)

        # ====== TÍNH METRICS (nếu có GT) ======
        if gt is not None:
            permanent = (jrc_img == 1.0)
            gt_ref = gt.copy()
            gt_ref[permanent] = 0.0
            valid_pixels = valid_mask & (~permanent)
            if np.any(valid_pixels):
                y_true = gt_ref[valid_pixels].astype(np.float32)
                m_ori  = calculate_metrics(y_true, pred_masked[valid_pixels].astype(np.float32))
                m_ref  = calculate_metrics(y_true, refined_masked[valid_pixels].astype(np.float32))
            else:
                m_ori = m_ref = {'accuracy':0,'precision':0,'recall':0,'f1_score':0,'iou':0,'ssim':0,'confusion_matrix':np.array([[0,0],[0,0]])}
            m_ori['filename'] = fname
            m_ref['filename'] = fname
            all_metrics_original.append(m_ori)
            all_metrics_refined.append(m_ref)
            batch_pred_original.append(pred_masked)
            batch_pred_refined.append(refined_masked)
            batch_gt.append(gt_ref)

            # Visualization chi tiết với 6 ảnh
            VIS_OUTPUT_PATH.mkdir(exist_ok=True)
            with rasterio.open(INFER_S1_PATH / fname) as src:
                s1 = src.read().astype(np.float32)
            if s1.shape[0] == 2:
                s1_img = np.stack([cv.resize(s1[0], (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA),
                                   cv.resize(s1[1], (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA)], axis=-1)
            elif s1.shape[0] == 1:
                a = cv.resize(s1[0], (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA)
                s1_img = np.stack([a, a], axis=-1)
            else:
                s1_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 2), dtype=np.float32)
            
            # Tạo visualization chi tiết với 6 ảnh
            pred_binary = (pred > INPUT_THRESHOLD).astype(np.float32)  # Dự đoán gốc nhị phân
            visualize_inference_patch_detailed(s1_img, jrc_img, pred_binary, after_fill_denoise, 
                                             refined, gt_ref, VIS_OUTPUT_PATH / f"{fname.replace('.tif','.png')}")

        if (i + 1) % 10 == 0:
            print(f"Đã xử lý {i + 1}/{len(s1_files)} files")

    # ===== Sau vòng lặp: tổng hợp metrics & lưu báo cáo =====
    if batch_pred_original:
        print("" + "="*60)  # section divider
        print("TÍNH TOÁN METRICS TỔNG THỂ")
        print("="*60)
        all_gt   = np.concatenate([gt.flatten() for gt in batch_gt])
        all_ori  = np.concatenate([p.flatten() for p in batch_pred_original])
        all_ref  = np.concatenate([p.flatten() for p in batch_pred_refined])
        m_ori = calculate_metrics(all_gt, all_ori)
        m_ref = calculate_metrics(all_gt, all_ref)
        print(f"📊 ORIGINAL: acc={m_ori['accuracy']:.4f} pre={m_ori['precision']:.4f} rec={m_ori['recall']:.4f} f1={m_ori['f1_score']:.4f} iou={m_ori['iou']:.4f} ssim={m_ori['ssim']:.4f}")
        print(f"📊 REFINED : acc={m_ref['accuracy']:.4f} pre={m_ref['precision']:.4f} rec={m_ref['recall']:.4f} f1={m_ref['f1_score']:.4f} iou={m_ref['iou']:.4f} ssim={m_ref['ssim']:.4f}")
        imp = {k: float(m_ref[k] - m_ori[k]) for k in ['accuracy','precision','recall','f1_score','iou']}
        print(f"📈 IMPROVEMENT (REFINED - ORIGINAL): {imp}")

        metrics_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_files_processed': len(batch_gt),
            'jrc_processing_info': {
                'occurrence_threshold_percent': 30.0,
                'extent_threshold': INPUT_THRESHOLD,
                'note': 'Pixels with JRC=1 are set to 0 in flood predictions and GT before scoring.'
            },
            'overall_metrics_original': {k: float(m_ori[k]) for k in ['accuracy','precision','recall','f1_score','iou','ssim']},
            'overall_metrics_refined' : {k: float(m_ref[k]) for k in ['accuracy','precision','recall','f1_score','iou','ssim']},
            'improvement_metrics': {k+"_improvement": float(imp[k]) for k in imp},
            'per_file_metrics_original': [
                {k: (float(v) if k in ['accuracy','precision','recall','f1_score','iou','ssim'] else v) for k, v in m.items() if k != 'confusion_matrix'}
                for m in all_metrics_original
            ],
            'per_file_metrics_refined': [
                {k: (float(v) if k in ['accuracy','precision','recall','f1_score','iou','ssim'] else v) for k, v in m.items() if k != 'confusion_matrix'}
                for m in all_metrics_refined
            ]
        }
        with open(METRICS_OUTPUT_PATH / "inference_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

        plot_confusion_matrix(m_ori['confusion_matrix'], METRICS_OUTPUT_PATH / "confusion_matrix_original.png")
        plot_confusion_matrix(m_ref['confusion_matrix'], METRICS_OUTPUT_PATH / "confusion_matrix_refined.png")
        print("📁 Lưu tại:")  # output locations
        print(f"   • Raw Predictions: {RAW_OUTPUT_PATH}")
        print(f"   • Fill & Denoise Predictions: {MORPH_OUTPUT_PATH}")
        print(f"   • Refined Predictions: {REFINED_OUTPUT_PATH}")
        print(f"   • JRC Threshold: {INFER_OUTPUT_PATH / 'jrc_threshold'}")
        print(f"   • Other Predictions: {PRED_OUTPUT_PATH}")
        print(f"   • Metrics: {METRICS_OUTPUT_PATH / 'inference_metrics.json'}")
        print(f"   • Confusion Matrix: {METRICS_OUTPUT_PATH}")
        print(f"   • Visualizations: {VIS_OUTPUT_PATH}")
        
        # === MERGE PREDICTIONS ===
        print("\n" + "="*60)
        print("MERGING PREDICTIONS TO FULL IMAGE")
        print("="*60)
        
        # Lấy region name từ output path hoặc từ file đầu tiên
        if INFER_OUTPUT_PATH.name and '_' in INFER_OUTPUT_PATH.name:
            # Lấy từ output path như benchmarks/BacGiang_20240910
            region_name = INFER_OUTPUT_PATH.name
        else:
            # Fallback, tìm từ file đầu tiên hoặc dùng "merged"
            region_name = "merged"
            if s1_files:
                first_file = s1_files[0].name
                if '_' in first_file:
                    # Extract từ filename như BacGiang_20240910_001-001.tif
                    parts = first_file.split('_')
                    if len(parts) >= 2:
                        region_name = f"{parts[0]}_{parts[1]}"
        
        # Merge refined predictions
        refined_output = INFER_OUTPUT_PATH / f"{region_name}_merged_refined.tif"
        
        print(f"🔗 Merging refined predictions from: {REFINED_OUTPUT_PATH}")
        print(f"📁 Output: {refined_output}")
        
        success = merge_predictions(REFINED_OUTPUT_PATH, refined_output, pattern="refined_*.tif")
        if success:
            print(f"✅ Merged refined predictions saved to: {refined_output}")
        else:
            print("❌ Failed to merge refined predictions")
            
        # Merge original (RAW) predictions
        original_output = INFER_OUTPUT_PATH / f"{region_name}_merged_raw.tif"
        print(f"\n🔗 Merging original (raw) predictions from: {RAW_OUTPUT_PATH}")
        print(f"📁 Output: {original_output}")
        
        success_orig = merge_predictions(RAW_OUTPUT_PATH, original_output, pattern="raw_*.tif")
        if success_orig:
            print(f"✅ Merged raw predictions saved to: {original_output}")
        else:
            print("❌ Failed to merge raw predictions")

        # Merge fill & denoise predictions (thay cho morphological)
        fill_denoise_output = INFER_OUTPUT_PATH / f"{region_name}_merged_fill_denoise.tif"
        print(f"\n🔗 Merging fill & denoise predictions from: {MORPH_OUTPUT_PATH}")
        print(f"📁 Output: {fill_denoise_output}")
        
        success_fill = merge_predictions(MORPH_OUTPUT_PATH, fill_denoise_output, pattern="fill_denoise_*.tif")
        if success_fill:
            print(f"✅ Merged fill & denoise predictions saved to: {fill_denoise_output}")
        else:
            print("❌ Failed to merge fill & denoise predictions")

        # Merge JRC threshold predictions
        jrc_threshold_dir = INFER_OUTPUT_PATH / "jrc_threshold"
        jrc_output = INFER_OUTPUT_PATH / f"{region_name}_merged_jrc_threshold.tif"
        print(f"\n🔗 Merging JRC threshold predictions from: {jrc_threshold_dir}")
        print(f"📁 Output: {jrc_output}")
        
        success_jrc = merge_predictions(jrc_threshold_dir, jrc_output, pattern="jrc_thresh_*.tif")
        if success_jrc:
            print(f"✅ Merged JRC threshold predictions saved to: {jrc_output}")
        else:
            print("❌ Failed to merge JRC threshold predictions")
    else:
        print("⚠️ Không có ground truth để đánh giá. Predictions đã lưu ở:")
        print(f"   • Raw: {RAW_OUTPUT_PATH}")
        print(f"   • Fill & Denoise: {MORPH_OUTPUT_PATH}") 
        print(f"   • Refined: {REFINED_OUTPUT_PATH}")
        print(f"   • JRC Threshold: {INFER_OUTPUT_PATH / 'jrc_threshold'}")

    print("✅ INFERENCE HOÀN THÀNH!")  # done banner


def merge_predictions(input_dir, output_path, pattern="*.tif", exclude_pattern=None):
    """
    Merge các prediction tiles thành một ảnh lớn
    """
    import glob
    from rasterio.merge import merge as rio_merge
    
    try:
        # Tìm files theo pattern
        search_pattern = str(input_dir / pattern)
        tif_files = glob.glob(search_pattern)
        
        # Loại bỏ files không mong muốn
        if exclude_pattern:
            tif_files = [f for f in tif_files if exclude_pattern not in os.path.basename(f)]
        
        if not tif_files:
            print(f"❌ Không tìm thấy file nào với pattern: {pattern}")
            return False
            
        print(f"📊 Tìm thấy {len(tif_files)} files để merge")
        
        # Mở và merge
        srcs = []
        for file_path in sorted(tif_files):
            try:
                src = rasterio.open(file_path)
                srcs.append(src)
                print(f"  ✓ {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  ❌ Không thể mở {os.path.basename(file_path)}: {e}")
        
        if not srcs:
            print("❌ Không thể mở file nào!")
            return False
        
        print(f"🔗 Merging {len(srcs)} files...")
        mosaic, transform = rio_merge(srcs, method='first')
        
        # Chuẩn bị metadata
        out_meta = srcs[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "compress": "lzw"
        })
        
        # Lưu kết quả
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        # Đóng files
        for src in srcs:
            src.close()
            
        print(f"✅ Merged successfully: {mosaic.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi merge: {e}")
        return False

if __name__ == "__main__":
    main()
