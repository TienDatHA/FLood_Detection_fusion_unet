#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Đánh giá ảnh dự đoán (sau morphology hoặc sau khi trừ giao) so với label.
Xuất ra các chỉ số: Accuracy, IoU, Precision, Recall.
Ngoài ra, nếu chạy chế độ AFTER_MORPH_MINUS_INTER thì lưu thêm GeoTIFF ảnh trừ (_resultfinal.tif).
"""

from pathlib import Path
import json, csv
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import os
import sys

# Add current directory to Python path to import config
sys.path.insert(0, str(Path(__file__).parent))
from config import get_data_root, get_project_root

# =============== CONFIGURATION (EDIT HERE) =================
# Get configurable data root
DATA_ROOT = get_data_root()
PROJECT_ROOT = get_project_root()

# Example configuration - modify these paths as needed
REGION_NAME = os.getenv("REGION_NAME", "BinhDinh_20171110")
DETECTION_DATE = os.getenv("DETECTION_DATE", "20171110") 

# Paths relative to DATA_ROOT
AFTER_MORPH_PATH = DATA_ROOT / f"inference_results/change_detection/{REGION_NAME}_change_detection_{DETECTION_DATE}/{REGION_NAME}_{DETECTION_DATE}_after_raw.tif"
LABEL_PATH = DATA_ROOT / f"Bench_Mark/Label_4326/{REGION_NAME}.tif"
RAW_INTERSECTION_PATH = DATA_ROOT / f"inference_results/change_detection/{REGION_NAME}_change_detection_{DETECTION_DATE}/raw_intersection.tif"  # or None

# Output directory relative to PROJECT_ROOT
OUTDIR = PROJECT_ROOT / "eval_results" / REGION_NAME

# Thresholds
THRESH_PRED = float(os.getenv("THRESH_PRED", "0.5"))
THRESH_LABEL = float(os.getenv("THRESH_LABEL", "0.5"))

# Print configuration for verification
print(f"📍 Configuration:")
print(f"   DATA_ROOT: {DATA_ROOT}")
print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
print(f"   Region: {REGION_NAME}")
print(f"   Output dir: {OUTDIR}")
# =============================================================


def _safe_div(num, den):
    return float(num) / float(den) if den != 0 else 0.0


def _reproject_to_like(src_arr, src_transform, src_crs,
                       like_shape, like_transform, like_crs,
                       resampling=Resampling.nearest):
    dst = np.zeros(like_shape, dtype=src_arr.dtype)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=like_transform,
        dst_crs=like_crs,
        resampling=resampling,
    )
    return dst


def _load_band_one(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        tfm = src.transform
        crs = src.crs
        nod = src.nodatavals[0]
    return arr, tfm, crs, nod


def _save_geotiff_mask_uint8(path_out: Path, mask_uint8, like_transform, like_crs):
    profile = {
        "driver": "GTiff",
        "height": mask_uint8.shape[0],
        "width":  mask_uint8.shape[1],
        "count": 1,
        "dtype": rasterio.uint8,
        "crs": like_crs,
        "transform": like_transform,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": min(512, mask_uint8.shape[1]),
        "blockysize": min(512, mask_uint8.shape[0]),
    }
    with rasterio.open(path_out, "w", **profile) as dst:
        dst.write(mask_uint8.astype(np.uint8), 1)
    print(f"🗺️  Đã lưu GeoTIFF: {path_out}")


def main():
    # 1) Đọc AFTER_MORPH làm chuẩn
    with rasterio.open(AFTER_MORPH_PATH) as src_pred:
        pred = src_pred.read(1).astype(np.float32)
        like_transform = src_pred.transform
        like_crs = src_pred.crs
        like_shape = pred.shape
        pred_nodata = src_pred.nodatavals[0]

    # 2) Nếu có RAW_INTERSECTION
    inter_like = None
    if RAW_INTERSECTION_PATH is not None:
        inter_raw, inter_tfm, inter_crs, inter_nodata = _load_band_one(RAW_INTERSECTION_PATH)
        inter_like = _reproject_to_like(
            inter_raw, inter_tfm, inter_crs,
            like_shape, like_transform, like_crs,
            resampling=Resampling.nearest
        ).astype(np.float32)

    # 3) Label -> reproject về lưới AFTER_MORPH
    lab, lab_tfm, lab_crs, lab_nodata = _load_band_one(LABEL_PATH)
    lab_like = _reproject_to_like(
        lab, lab_tfm, lab_crs,
        like_shape, like_transform, like_crs,
        resampling=Resampling.nearest
    ).astype(np.float32)

    # 4) Mask hợp lệ
    pred_valid = np.isfinite(pred) & ((pred_nodata is None) | (pred != pred_nodata))
    lab_valid  = np.isfinite(lab_like) & ((lab_nodata is None) | (lab_like != lab_nodata))
    valid = pred_valid & lab_valid
    if inter_like is not None:
        valid &= np.isfinite(inter_like)

    # 5) Nhị phân hóa
    pred_bin = (pred > THRESH_PRED).astype(np.uint8)
    lab_bin  = (lab_like > THRESH_LABEL).astype(np.uint8)

    if inter_like is not None:
        inter_bin = (inter_like > 0.5).astype(np.uint8)
        eval_mask = ((pred_bin == 1) & (inter_bin == 0)).astype(np.uint8)
        mode = "AFTER_MORPH_MINUS_INTER"

        # Lưu ảnh kết quả trừ
        OUTDIR.mkdir(parents=True, exist_ok=True)
        result_tif = OUTDIR / f"{mode}_resultfinal_raw.tif"
        _save_geotiff_mask_uint8(result_tif, eval_mask, like_transform, like_crs)
    else:
        eval_mask = pred_bin
        mode = "AFTER_MORPH"

    # 6) Áp mask hợp lệ
    p = eval_mask[valid]
    g = lab_bin[valid]

    # 7) TP/FP/FN/TN
    tp = int(np.sum((p == 1) & (g == 1)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))
    tn = int(np.sum((p == 0) & (g == 0)))

    # 8) Tính chỉ số
    precision = _safe_div(tp, tp + fp)
    recall    = _safe_div(tp, tp + fn)
    iou       = _safe_div(tp, tp + fp + fn)
    acc       = _safe_div(tp + tn, tp + fp + fn + tn)

    results = {
        "mode": mode,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": acc,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }

    # 9) Lưu kết quả
    with open(OUTDIR / f"{mode}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(OUTDIR / f"{mode}_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(results.keys()); w.writerow(results.values())

    print("📊 Kết quả:")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
