#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Change Detection cho flood giữa 2 ngày, xuất đầy đủ 8 loại ảnh + 1 ảnh tổng hợp overlay PNG.
- An toàn hình học: mọi phép so sánh/overlay đều quy về lưới của AFTER morph mosaic.
- GeoTIFF: before_raw, after_raw, raw_intersection, after_raw_minus_intersection,
           before_morph, after_morph, change_morph (AFTER−BEFORE),
           after_morph_minus_raw_intersection (MỚI).
- PNG: 'tong_hop_overlay.png' (gộp 6 ô overlay/preview, tiêu đề tiếng Việt).

BỔ SUNG:
- Nếu ngày sau (AFTER) không tìm thấy DEM/JRC cho patch nào đó, sẽ thử fallback
  sang DEM/JRC của ngày trước (BEFORE). Nếu cũng không có thì dùng zeros.
"""

import os
import sys
import argparse
from pathlib import Path
import shutil

import numpy as np
import cv2 as cv
import tensorflow as tf
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from skimage.morphology import opening, closing, remove_small_objects, remove_small_holes, disk
from matplotlib.patches import Patch

from config import *
from model import Resnet50_UNet
from utils import scale_img
from merge import merge_directory as merge_predictions

# ================== CẤU HÌNH TOÀN CỤC ==================
REGION_BEFORE = None
REGION_AFTER = None
SHARED_REGION = None
BENCH_BASE = ROOT_PATH / "Bench_Mark"

INFER_S1_PATH_BEFORE = None
INFER_S1_PATH_AFTER = None
INFER_DEM_PATH = None
INFER_JRC_PATH = None
INFER_OUTPUT_PATH = None
PRED_OUTPUT_PATH = None
VIS_OUTPUT_PATH = None

RAW_OUTPUTS_DIR = None
MORPH_OUTPUTS_DIR = None

# Đường dẫn fallback cho AFTER: dùng DEM/JRC của BEFORE nếu cần
ALT_DEM_PATH_BEFORE = None
ALT_JRC_PATH_BEFORE = None


# ================== TIỆN ÍCH TF ==================
def _configure_tf_memory_growth():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


# ================== MODEL ==================
def load_model():
    """Load model từ checkpoint"""
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
    print("✅ Model loaded successfully")
    return model


# ================== HẬU XỬ LÝ ==================
def post_process_mask_advanced(mask: np.ndarray,
                               jrc_img: np.ndarray,
                               threshold: float = 0.3,
                               morphology_level: str = "medium",
                               allow_growth: bool = False) -> np.ndarray:
    """Morphology trước, KHÔNG trừ JRC; giữ no-growth nếu cần."""
    mask_bin = (mask > threshold).astype(bool)
    refined = mask_bin.copy()

    if morphology_level == "light":
        min_sz = 80;  hole_sz = 150;  k = 2
    elif morphology_level == "strong":
        min_sz = 250; hole_sz = 400;  k = 3
    else:  # medium
        min_sz = 150; hole_sz = 250;  k = 2

    refined = remove_small_objects(refined, min_size=min_sz)
    refined = remove_small_holes(refined, area_threshold=hole_sz)
    if k > 0:
        refined = closing(refined, disk(k))
        refined = opening(refined, disk(k))
    refined = remove_small_objects(refined, min_size=min_sz)

    if not allow_growth:
        refined = refined & mask_bin

    return refined.astype(np.float32)


# ================== I/O RASTER ==================
def _gtiff_profile_like(arr_uint8, crs, transform):
    return dict(
        driver="GTiff",
        height=arr_uint8.shape[0],
        width=arr_uint8.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform,
        compress="LZW",
        tiled=True,
        blockxsize=min(512, arr_uint8.shape[1]),
        blockysize=min(512, arr_uint8.shape[0]),
    )


def save_prediction(pred_mask: np.ndarray, filename: str, output_dir: Path,
                    crs, transform, bin_threshold: float = 0.5):
    """Lưu mask nhị phân 0/1"""
    out = output_dir / filename
    pred_bin = (pred_mask > bin_threshold).astype(np.uint8)
    profile = _gtiff_profile_like(pred_bin, crs, transform)
    with rasterio.open(out, 'w', **profile) as dst:
        dst.write(pred_bin, 1)


def _try_read_raster(path: Path) -> np.ndarray:
    """Thử đọc một raster đơn băng, trả np.ndarray (float32) hoặc raise Exception."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return arr


def load_and_preprocess_patch(
    s1_filename: str,
    s1_path: Path,
    dem_path: Path,
    jrc_path: Path,
    dem_path_alt: Path = None,
    jrc_path_alt: Path = None
):
    """
    Load + chuẩn hoá patch S1/DEM/JRC về kích thước IMG_* và build 2 input tensor.
    Fallback: nếu không tìm thấy DEM/JRC ở dem_path/jrc_path, thử dem_path_alt/jrc_path_alt.
    """
    try:
        s1_file_path = s1_path / s1_filename
        with rasterio.open(s1_file_path) as src:
            s1_data = src.read().astype(np.float32)
            s1_crs = src.crs
            s1_transform = src.transform

        if s1_data.shape[0] == 2:
            vv_img, vh_img = s1_data[0], s1_data[1]
        elif s1_data.shape[0] == 1:
            vv_img = s1_data[0]; vh_img = s1_data[0].copy()
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
        # Convert "Region_YYYYMMDD_patch_r_c" -> "Region_YYYYMMDD_r-c"
        if '_patch_' in base:
            parts = base.split('_patch_')
            if len(parts) == 2 and '_' in parts[1]:
                coord_parts = parts[1].split('_', 1)
                base = f"{parts[0]}_{coord_parts[0]}-{coord_parts[1]}"

        # ===== DEM (có fallback) =====
        dem_filename = f"{base}_dem.tif"
        dem_img = None
        # primary
        try:
            dem_img = _try_read_raster(dem_path / dem_filename)
        except Exception:
            # fallback alt (thường là DEM ngày trước)
            if dem_path_alt is not None:
                try:
                    dem_img = _try_read_raster(dem_path_alt / dem_filename)
                    print(f"[Fallback DEM] Dùng DEM từ: {dem_path_alt / dem_filename}")
                except Exception:
                    pass
        if dem_img is None:
            print(f"Warning: DEM not found in both primary/alt for {dem_filename}, using zeros")
            dem_img = np.zeros(target_size, dtype=np.float32)
        if dem_img.shape != target_size:
            dem_img = cv.resize(dem_img, target_size, interpolation=cv.INTER_AREA)

        # ===== JRC (có fallback) =====
        jrc_img = None
        # thử occurrence trước
        jrc_occ_filename = f"{base}_occurrence_jrc.tif"
        def _read_jrc_occ(pth: Path):
            with rasterio.open(pth) as jsrc:
                j = jsrc.read(1).astype(np.float32)
            return j

        def _read_jrc_extent(pth: Path):
            with rasterio.open(pth) as esrc:
                e = esrc.read(1).astype(np.float32)
            return e

        def _post_jrc(j, is_occ=True):
            if j.shape != target_size:
                j = cv.resize(j, target_size, interpolation=cv.INTER_NEAREST)
            if is_occ:
                return (j > 30.0).astype(np.float32)
            else:
                return (j > 0.5).astype(np.float32)

        # primary occurrence
        try:
            j = _read_jrc_occ(jrc_path / jrc_occ_filename)
            jrc_img = _post_jrc(j, is_occ=True)
        except Exception:
            # alt occurrence
            if jrc_path_alt is not None:
                try:
                    j = _read_jrc_occ(jrc_path_alt / jrc_occ_filename)
                    jrc_img = _post_jrc(j, is_occ=True)
                    print(f"[Fallback JRC] Dùng OCC từ: {jrc_path_alt / jrc_occ_filename}")
                except Exception:
                    pass

        # nếu chưa có -> thử extent
        if jrc_img is None:
            jrc_ext_filename = f"{base}_extent_jrc.tif"
            # primary extent
            try:
                e = _read_jrc_extent(jrc_path / jrc_ext_filename)
                jrc_img = _post_jrc(e, is_occ=False)
            except Exception:
                # alt extent
                if jrc_path_alt is not None:
                    try:
                        e = _read_jrc_extent(jrc_path_alt / jrc_ext_filename)
                        jrc_img = _post_jrc(e, is_occ=False)
                        print(f"[Fallback JRC] Dùng EXTENT từ: {jrc_path_alt / jrc_ext_filename}")
                    except Exception:
                        pass

        if jrc_img is None:
            print(f"Warning: JRC not found in both primary/alt for base={base}, using zeros")
            jrc_img = np.zeros(target_size, dtype=np.float32)

        # 3 kênh: VV, VH, JRC (x_img) và VV, VH, DEM (x_inf)
        x_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        x_img[..., :2] = x_arr
        x_img[..., 2]   = jrc_img

        x_inf = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        x_inf[..., :2] = x_arr
        x_inf[..., 2]   = dem_img

        x_img = scale_img(x_img)
        x_inf = scale_img(x_inf)

        return x_img, x_inf, jrc_img, s1_crs, new_transform, valid_mask
    except Exception as e:
        print(f"Error loading {s1_filename}: {e}")
        return None, None, None, None, None, None


# ================== WARP / ALIGN ==================
def reproject_to_like(src_arr, src_transform, src_crs, like_transform, like_crs, like_shape, resampling=Resampling.nearest):
    """Đưa src_arr về cùng grid (crs/transform/shape) của 'like'."""
    dst = np.zeros(like_shape, dtype=src_arr.dtype)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=like_transform,
        dst_crs=like_crs,
        resampling=resampling,
        num_threads=2
    )
    return dst


# ================== TẠO CÁC ẢNH THEO YÊU CẦU ==================
def write_and_align_boolean_ops(before_tif, after_tif, out_path_intersection_raw,
                                out_path_change_morph, out_path_after_raw_minus_inter_raw,
                                out_path_before_raw, out_path_after_raw,
                                out_path_before_morph, out_path_after_morph,
                                out_path_after_morph_minus_raw_inter):
    """
    Nhận 4 raster mosaic và tạo các ảnh:
      1) intersection_raw
      2) before_raw (đã có)
      3) after_raw (đã có)
      4) before_morph (đã có)
      5) after_morph (đã có)
      6) change_morph_after_minus_before (AFTER_morph==1 & BEFORE_morph==0)
      7) after_raw_minus_intersection_raw (AFTER_raw==1 & INTERSECTION_raw==0)
      8) after_morph_minus_raw_intersection (AFTER_morph==1 & INTERSECTION_raw==0)  <-- MỚI
    Lưới chuẩn: AFTER morph mosaic.
    """
    # ---- Lưới chuẩn = AFTER morph ----
    with rasterio.open(out_path_after_morph) as src_like:
        like_crs = src_like.crs
        like_transform = src_like.transform
        like_shape = src_like.read(1).shape

    # ===== RAW (before/after) về lưới chuẩn =====
    with rasterio.open(out_path_before_raw) as src_bf_raw, \
         rasterio.open(out_path_after_raw) as src_af_raw:

        bf_raw = (src_bf_raw.read(1) > 0).astype(np.uint8)
        if (src_bf_raw.transform != like_transform) or (src_bf_raw.crs != like_crs) or (bf_raw.shape != like_shape):
            bf_raw = reproject_to_like(bf_raw, src_bf_raw.transform, src_bf_raw.crs, like_transform, like_crs, like_shape)

        af_raw = (src_af_raw.read(1) > 0).astype(np.uint8)
        if (src_af_raw.transform != like_transform) or (src_af_raw.crs != like_crs) or (af_raw.shape != like_shape):
            af_raw = reproject_to_like(af_raw, src_af_raw.transform, src_af_raw.crs, like_transform, like_crs, like_shape)

    # (1) Giao (RAW)
    inter_raw = ((bf_raw == 1) & (af_raw == 1)).astype(np.uint8)
    with rasterio.open(out_path_intersection_raw, "w", **_gtiff_profile_like(inter_raw, like_crs, like_transform)) as dst:
        dst.write(inter_raw, 1)

    # ===== MORPH (before/after) về lưới chuẩn =====
    with rasterio.open(out_path_before_morph) as src_bf_m, \
         rasterio.open(out_path_after_morph) as src_af_m:

        bf_m = (src_bf_m.read(1) > 0).astype(np.uint8)
        if (src_bf_m.transform != like_transform) or (src_bf_m.crs != like_crs) or (bf_m.shape != like_shape):
            bf_m = reproject_to_like(bf_m, src_bf_m.transform, src_bf_m.crs, like_transform, like_crs, like_shape)

        af_m = (src_af_m.read(1) > 0).astype(np.uint8)
        if (src_af_m.transform != like_transform) or (src_af_m.crs != like_crs) or (af_m.shape != like_shape):
            af_m = reproject_to_like(af_m, src_af_m.transform, src_af_m.crs, like_transform, like_crs, like_shape)

    # (6) Thay đổi sau morphological: AFTER − BEFORE
    change_morph = ((af_m == 1) & (bf_m == 0)).astype(np.uint8)
    with rasterio.open(out_path_change_morph, "w", **_gtiff_profile_like(change_morph, like_crs, like_transform)) as dst:
        dst.write(change_morph, 1)

    # (7) AFTER (RAW) − INTERSECTION (RAW)
    after_raw_minus_inter = ((af_raw == 1) & (inter_raw == 0)).astype(np.uint8)
    with rasterio.open(out_path_after_raw_minus_inter_raw, "w", **_gtiff_profile_like(after_raw_minus_inter, like_crs, like_transform)) as dst:
        dst.write(after_raw_minus_inter, 1)

    # (8) AFTER (MORPH) − INTERSECTION (RAW)   <-- MỚI theo yêu cầu
    after_morph_minus_raw_inter = ((af_m == 1) & (inter_raw == 0)).astype(np.uint8)
    with rasterio.open(out_path_after_morph_minus_raw_inter, "w", **_gtiff_profile_like(after_morph_minus_raw_inter, like_crs, like_transform)) as dst:
        dst.write(after_morph_minus_raw_inter, 1)

    # Trả về để vẽ PNG
    return dict(
        like_crs=like_crs, like_transform=like_transform, like_shape=like_shape,
        bf_raw=bf_raw, af_raw=af_raw, inter_raw=inter_raw,
        bf_morph=bf_m, af_morph=af_m, change_morph=change_morph,
        after_raw_minus_inter=after_raw_minus_inter,
        after_morph_minus_raw_inter=after_morph_minus_raw_inter,
    )


# ================== VIZ: 1 ẢNH PNG GỘP 6 Ô (tiếng Việt) ==================
def _rgba_from_mask(mask, rgba, shape=None):
    """Tạo layer RGBA từ mask nhị phân 0/1"""
    if shape is None:
        h, w = mask.shape
    else:
        h, w = shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    out[mask == 1] = rgba
    return out

def save_combined_overlay_png(data, out_png, region_before, region_after):
    """
    Vẽ 6 ô trong 1 ảnh:
      (1) Raw BEFORE vs Raw AFTER (chồng lớp)
      (2) Giao (RAW)
      (3) AFTER (RAW) − Giao (RAW)
      (4) Morph BEFORE vs Morph AFTER (chồng lớp)
      (5) Thay đổi sau morphology (AFTER − BEFORE)
      (6) AFTER (Morph) − Giao (RAW)
    """
    h, w = data['like_shape']
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.ravel()

    # (1) Raw BEFORE vs Raw AFTER
    img = np.zeros((h, w, 4), dtype=np.float32)
    img += _rgba_from_mask(data['af_raw'], (0, 0, 1, 0.6), (h, w))  # xanh
    img += _rgba_from_mask(data['bf_raw'], (1, 0, 0, 0.6), (h, w))  # đỏ
    axes[0].imshow(img, origin='upper', aspect='equal')
    axes[0].set_title(f"Chồng ảnh thô: Sau (xanh) vs Trước (đỏ)\nAFTER: {region_after} | BEFORE: {region_before}", fontsize=12)

    # (2) Giao (RAW)
    axes[1].imshow(_rgba_from_mask(data['inter_raw'], (0, 1, 0, 0.85), (h, w)), origin='upper', aspect='equal')
    axes[1].set_title("Giao hai ngày (Raw Intersection)", fontsize=12)

    # (3) AFTER (RAW) − Giao (RAW)
    axes[2].imshow(_rgba_from_mask(data['after_raw_minus_inter'], (1, 0.5, 0, 0.9), (h, w)), origin='upper', aspect='equal')
    axes[2].set_title("Phần riêng của ngày sau (Raw − Giao)", fontsize=12)

    # (4) Morph BEFORE vs Morph AFTER
    img4 = np.zeros((h, w, 4), dtype=np.float32)
    img4 += _rgba_from_mask(data['af_morph'], (0, 0, 1, 0.6), (h, w))  # xanh
    img4 += _rgba_from_mask(data['bf_morph'], (1, 0, 0, 0.6), (h, w))  # đỏ
    axes[3].imshow(img4, origin='upper', aspect='equal')
    axes[3].set_title(f"Chồng ảnh đã xử lý: Sau (xanh) vs Trước (đỏ)\nAFTER: {region_after} | BEFORE: {region_before}", fontsize=12)

    # (5) Thay đổi sau morphology (AFTER − BEFORE)
    axes[4].imshow(_rgba_from_mask(data['change_morph'], (1, 1, 0, 0.95), (h, w)), origin='upper', aspect='equal')
    axes[4].set_title("Ngập mới sau xử lý (AFTER − BEFORE)", fontsize=12)

    # (6) AFTER (Morph) − Giao (RAW)
    axes[5].imshow(_rgba_from_mask(data['after_morph_minus_raw_inter'], (0.5, 0, 1, 0.9), (h, w)), origin='upper', aspect='equal')
    axes[5].set_title("Ngày sau (đã xử lý) trừ phần giao (RAW)", fontsize=12)

    for ax in axes:
        ax.set_xlabel("Pixel X"); ax.set_ylabel("Pixel Y")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🖼  Đã lưu ảnh tổng hợp: {out_png}")


# ================== INFERENCE CHÍNH ==================
def main():
    parser = argparse.ArgumentParser(description="Change Detection Inference between two dates")
    parser.add_argument("--before", required=True, help="Region trước (e.g., BinhDinh_20171021)")
    parser.add_argument("--after", required=True, help="Region sau (e.g., BinhDinh_20171110)")
    args = parser.parse_args()

    global REGION_BEFORE, REGION_AFTER, SHARED_REGION
    global INFER_S1_PATH_BEFORE, INFER_S1_PATH_AFTER, INFER_DEM_PATH, INFER_JRC_PATH
    global INFER_OUTPUT_PATH, PRED_OUTPUT_PATH, VIS_OUTPUT_PATH
    global RAW_OUTPUTS_DIR, MORPH_OUTPUTS_DIR
    global ALT_DEM_PATH_BEFORE, ALT_JRC_PATH_BEFORE

    REGION_BEFORE = args.before
    REGION_AFTER = args.after
    SHARED_REGION = args.after  # dùng DEM/JRC từ ngày sau cho cả 2 ngày (như trước đây)

    # Input paths
    INFER_S1_PATH_BEFORE = ROOT_PATH / "Sen1_Before" / REGION_BEFORE
    INFER_S1_PATH_AFTER  = BENCH_BASE / "Sentinel1_Tiles" / REGION_AFTER

    # DEM/JRC primary theo AFTER
    INFER_DEM_PATH = BENCH_BASE / "DEM_Tiles_4326" / SHARED_REGION
    INFER_JRC_PATH = BENCH_BASE / "JRC_Tiles_Cut" / SHARED_REGION

    # DEM/JRC fallback theo BEFORE (nếu thiếu cho AFTER)
    ALT_DEM_PATH_BEFORE = BENCH_BASE / "DEM_Tiles_4326" / REGION_BEFORE
    ALT_JRC_PATH_BEFORE = BENCH_BASE / "JRC_Tiles_Cut" / REGION_BEFORE

    # Output root
    region_name = REGION_AFTER.split('_')[0]
    after_date = REGION_AFTER.split('_')[1]
    INFER_OUTPUT_PATH = ROOT_PATH / "inference_results" / "change_detection" / f"{region_name}_change_detection_{after_date}"
    PRED_OUTPUT_PATH  = INFER_OUTPUT_PATH / "predictions"
    VIS_OUTPUT_PATH   = INFER_OUTPUT_PATH / "visualizations"
    RAW_OUTPUTS_DIR   = PRED_OUTPUT_PATH / "raw_patches"
    MORPH_OUTPUTS_DIR = PRED_OUTPUT_PATH / "morph_patches"

    # Create dirs
    for p in [INFER_OUTPUT_PATH, PRED_OUTPUT_PATH, VIS_OUTPUT_PATH, RAW_OUTPUTS_DIR, MORPH_OUTPUTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Change Detection Inference")
    print(f"   BEFORE: {REGION_BEFORE}")
    print(f"   AFTER : {REGION_AFTER}")
    print(f"   DEM/JRC dùng region chính: {SHARED_REGION} (fallback: {REGION_BEFORE})")
    print(f"   Output root: {INFER_OUTPUT_PATH}")

    # Kiểm tra input dirs
    for path, desc in [
        (INFER_S1_PATH_BEFORE, f"S1 BEFORE {REGION_BEFORE}"),
        (INFER_S1_PATH_AFTER,  f"S1 AFTER  {REGION_AFTER}"),
        (INFER_DEM_PATH,       f"DEM (primary) {SHARED_REGION}"),
        (INFER_JRC_PATH,       f"JRC (primary) {SHARED_REGION}"),
    ]:
        if not path.exists():
            print(f"⚠️ Cảnh báo: Không tìm thấy {desc}: {path} (sẽ dùng fallback/zeros nếu thiếu file)")

    # TF config + load model
    _configure_tf_memory_growth()
    print("📥 Loading model...")
    model = load_model()

    # Liệt kê patch
    s1_files_before = sorted([p for p in INFER_S1_PATH_BEFORE.glob("*.tif")])
    s1_files_after  = sorted([p for p in INFER_S1_PATH_AFTER.glob("*.tif")])
    if not s1_files_before:
        print(f"❌ Không tìm thấy file S1 trong {INFER_S1_PATH_BEFORE}"); return
    if not s1_files_after:
        print(f"❌ Không tìm thấy file S1 trong {INFER_S1_PATH_AFTER}"); return

    print(f"📊 BEFORE patches: {len(s1_files_before)}")
    print(f"📊 AFTER  patches: {len(s1_files_after)}")

    # ---------- Inference BEFORE ----------
    print(f"🔄 Processing BEFORE: {REGION_BEFORE}")
    for i, s1_file in enumerate(s1_files_before):
        try:
            print(f"   [{i+1}/{len(s1_files_before)}] {s1_file.name}")
            x_img, x_inf, jrc_img, crs, transform, valid_mask = load_and_preprocess_patch(
                s1_file.name,
                s1_path=INFER_S1_PATH_BEFORE,
                dem_path=INFER_DEM_PATH,
                jrc_path=INFER_JRC_PATH,
                dem_path_alt=None,           # BEFORE giữ nguyên như trước
                jrc_path_alt=None
            )
            if x_img is None: continue

            pred = model.predict([np.expand_dims(x_img, 0), np.expand_dims(x_inf, 0)], verbose=0)[0]
            pred = np.squeeze(pred)

            # ---- RAW (trước morphology) ----
            pred_raw = np.where(valid_mask, pred, 0.0)
            save_prediction(pred_raw, f"before_raw_{s1_file.name}", RAW_OUTPUTS_DIR, crs, transform, bin_threshold=0.5)

            # ---- MORPH ----
            refined = post_process_mask_advanced(pred, jrc_img, threshold=0.3, morphology_level="medium")
            refined_masked = np.where(valid_mask, refined, 0.0)
            save_prediction(refined_masked, f"before_morph_{s1_file.name}", MORPH_OUTPUTS_DIR, crs, transform, bin_threshold=0.5)

        except Exception as e:
            print(f"⚠️  Lỗi processing {s1_file.name}: {e}")
            continue

    # ---------- Inference AFTER ----------
    print(f"🔄 Processing AFTER: {REGION_AFTER}")
    for i, s1_file in enumerate(s1_files_after):
        try:
            print(f"   [{i+1}/{len(s1_files_after)}] {s1_file.name}")
            x_img, x_inf, jrc_img, crs, transform, valid_mask = load_and_preprocess_patch(
                s1_file.name,
                s1_path=INFER_S1_PATH_AFTER,
                dem_path=INFER_DEM_PATH,
                jrc_path=INFER_JRC_PATH,
                dem_path_alt=ALT_DEM_PATH_BEFORE,    # Fallback sang BEFORE
                jrc_path_alt=ALT_JRC_PATH_BEFORE
            )
            if x_img is None: continue

            pred = model.predict([np.expand_dims(x_img, 0), np.expand_dims(x_inf, 0)], verbose=0)[0]
            pred = np.squeeze(pred)

            # ---- RAW (trước morphology) ----
            pred_raw = np.where(valid_mask, pred, 0.0)
            save_prediction(pred_raw, f"after_raw_{s1_file.name}", RAW_OUTPUTS_DIR, crs, transform, bin_threshold=0.5)

            # ---- MORPH ----
            refined = post_process_mask_advanced(pred, jrc_img, threshold=0.3, morphology_level="medium")
            refined_masked = np.where(valid_mask, refined, 0.0)
            save_prediction(refined_masked, f"after_morph_{s1_file.name}", MORPH_OUTPUTS_DIR, crs, transform, bin_threshold=0.5)

        except Exception as e:
            print(f"⚠️  Lỗi processing {s1_file.name}: {e}")
            continue

    # ---------- Merge các nhóm patch ----------
    print("🔧 Merging predictions (RAW & MORPH) ...")
    before_temp_raw  = RAW_OUTPUTS_DIR / "before_temp"
    after_temp_raw   = RAW_OUTPUTS_DIR / "after_temp"
    before_temp_morph = MORPH_OUTPUTS_DIR / "before_temp"
    after_temp_morph  = MORPH_OUTPUTS_DIR / "after_temp"
    for p in [before_temp_raw, after_temp_raw, before_temp_morph, after_temp_morph]:
        p.mkdir(exist_ok=True)

    # gom file theo tiền tố
    for f in RAW_OUTPUTS_DIR.glob("before_raw_*.tif"): shutil.copy2(f, before_temp_raw / f.name)
    for f in RAW_OUTPUTS_DIR.glob("after_raw_*.tif"):  shutil.copy2(f, after_temp_raw / f.name)
    for f in MORPH_OUTPUTS_DIR.glob("before_morph_*.tif"): shutil.copy2(f, before_temp_morph / f.name)
    for f in MORPH_OUTPUTS_DIR.glob("after_morph_*.tif"):  shutil.copy2(f, after_temp_morph / f.name)

    # Đường dẫn các raster hợp nhất (mosaic)
    before_raw_mosaic   = INFER_OUTPUT_PATH / f"{REGION_BEFORE}_before_raw.tif"
    after_raw_mosaic    = INFER_OUTPUT_PATH / f"{REGION_AFTER}_after_raw.tif"
    before_morph_mosaic = INFER_OUTPUT_PATH / f"{REGION_BEFORE}_before_morph.tif"
    after_morph_mosaic  = INFER_OUTPUT_PATH / f"{REGION_AFTER}_after_morph.tif"

    # Merge
    merge_predictions(str(before_temp_raw),   str(before_raw_mosaic),   method='first', recursive=False, use_safe_merge=True)
    merge_predictions(str(after_temp_raw),    str(after_raw_mosaic),    method='first', recursive=False, use_safe_merge=True)
    merge_predictions(str(before_temp_morph), str(before_morph_mosaic), method='first', recursive=False, use_safe_merge=True)
    merge_predictions(str(after_temp_morph),  str(after_morph_mosaic),  method='first', recursive=False, use_safe_merge=True)

    # Dọn tạm
    for p in [before_temp_raw, after_temp_raw, before_temp_morph, after_temp_morph]:
        shutil.rmtree(p, ignore_errors=True)

    # ---------- Tạo các ảnh theo yêu cầu 1/6/7/8 ----------
    intersection_raw_tif        = INFER_OUTPUT_PATH / "raw_intersection.tif"                     # (1)
    change_morph_tif            = INFER_OUTPUT_PATH / "change_morph_after_minus_before.tif"      # (6)
    after_raw_minus_inter_tif   = INFER_OUTPUT_PATH / "after_raw_minus_raw_intersection.tif"     # (7)
    after_morph_minus_raw_inter_tif = INFER_OUTPUT_PATH / "after_morph_minus_raw_intersection.tif"  # (8) MỚI

    data = write_and_align_boolean_ops(
        before_tif=before_morph_mosaic,
        after_tif=after_morph_mosaic,
        out_path_intersection_raw=intersection_raw_tif,
        out_path_change_morph=change_morph_tif,
        out_path_after_raw_minus_inter_raw=after_raw_minus_inter_tif,
        out_path_before_raw=before_raw_mosaic,         # (2)
        out_path_after_raw=after_raw_mosaic,           # (3)
        out_path_before_morph=before_morph_mosaic,     # (4)
        out_path_after_morph=after_morph_mosaic,       # (5)
        out_path_after_morph_minus_raw_inter=after_morph_minus_raw_inter_tif  # (8)
    )

    # ---------- PNG TỔNG HỢP (1 ảnh) ----------
    combined_png = VIS_OUTPUT_PATH / "tong_hop_overlay.png"
    save_combined_overlay_png(
        data=data,
        out_png=combined_png,
        region_before=REGION_BEFORE,
        region_after=REGION_AFTER
    )

    # ---------- In ra đường dẫn ----------
    print("📊 Outputs (GeoTIFF):")
    print(f"  (1) RAW Intersection                        : {intersection_raw_tif}")
    print(f"  (2) BEFORE Raw                              : {before_raw_mosaic}")
    print(f"  (3) AFTER  Raw                              : {after_raw_mosaic}")
    print(f"  (4) BEFORE Morph                            : {before_morph_mosaic}")
    print(f"  (5) AFTER  Morph                            : {after_morph_mosaic}")
    print(f"  (6) Change Morph (AFTER−BEFORE)             : {change_morph_tif}")
    print(f"  (7) AFTER Raw − RAW Intersection            : {after_raw_minus_inter_tif}")
    print(f"  (8) AFTER Morph − RAW Intersection (MỚI)    : {after_morph_minus_raw_inter_tif}")
    print("🖼  Ảnh tổng hợp PNG:", combined_png)


if __name__ == "__main__":
    main()
