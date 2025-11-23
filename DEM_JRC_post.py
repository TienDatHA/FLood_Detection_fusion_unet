#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FINAL FLOOD MAP PIPELINE
Bước A: STATIC = (DEM <= min(DEM) + 10 m) AND (JRC >= threshold)
Bước B: FINAL  = MORPHO_MASK AND (NOT STATIC)

I/O:
- DEM_PATH: GeoTIFF elevation (m)
- JRC_PATH: GeoTIFF (occurrence/recurrence), 0–100 (mặc định) hoặc 0–1
- MORPHO_TIF: GeoTIFF nhị phân sau morphology (1 = nước, 0 = không)
- OUT_STATIC_TIF: GeoTIFF nước tĩnh (uint8)
- OUT_FINAL_TIF : GeoTIFF ngập sau khi loại nước tĩnh (uint8)
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# Add configuration import
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import get_data_root, get_project_root, ensure_dirs
import os

# ================== CONFIGURATION (EDIT HERE) ==================
DATA_ROOT = get_data_root()
PROJECT_ROOT = get_project_root()

# Example configuration - modify these as needed
REGION_NAME = os.getenv("REGION_NAME", "BacGiang_20240910")
MORPH_THRESHOLD = os.getenv("MORPH_THRESHOLD", "0.6")

# Input paths relative to DATA_ROOT
DEM_PATH = DATA_ROOT / f"Bench_Mark/JRC + DEM/{REGION_NAME}_DEM.tif"
JRC_PATH = DATA_ROOT / f"Bench_Mark/JRC + DEM/{REGION_NAME}_occurrence.tif"
MORPHO_TIF = DATA_ROOT / f"morphology_results/{REGION_NAME}_{MORPH_THRESHOLD}/{REGION_NAME}_merged_morph.tif"

# Output paths relative to PROJECT_ROOT  
OUTPUT_DIR = PROJECT_ROOT / "final_flood_results" / REGION_NAME
OUT_STATIC_TIF = OUTPUT_DIR / "output_static_water_mask.tif"
OUT_FINAL_TIF = OUTPUT_DIR / "final_flood_mask.tif"

# Ensure output directory exists
ensure_dirs([OUTPUT_DIR])

# Parameters
DELTA_M = float(os.getenv("DELTA_M", "5.0"))           # DEM threshold: min(DEM) + 5 m
JRC_THRESHOLD = int(os.getenv("JRC_THRESHOLD", "99"))   # % occurrence/recurrence for static water
JRC_BAND_INDEX = 1                                      # 1-based
JRC_IS_PERCENT = True                                   # True if JRC 0–100; False if JRC 0–1

print(f"📍 Final Flood Map Configuration:")
print(f"   DATA_ROOT: {DATA_ROOT}")
print(f"   Region: {REGION_NAME}")
print(f"   DEM: {DEM_PATH}")
print(f"   JRC: {JRC_PATH}")
print(f"   Morphology: {MORPHO_TIF}")
print(f"   Output dir: {OUTPUT_DIR}")

# =============== HÀM PHỤ DÙNG CHUNG ===============

def read_band_as_float(path: Path):
    """Đọc band 1 -> float64, trả về (arr, valid_mask, profile)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        prof = src.profile
        nodata = src.nodata
        if nodata is not None:
            valid = ~np.isclose(arr, nodata)
        else:
            valid = ~np.isnan(arr)
    return arr, valid, prof

def reproject_to_like(src_path: Path, like_profile: dict, band_index: int = 1, resampling=Resampling.nearest):
    """Warp band (band_index) của src_path về grid like_profile; trả (arr_like, valid_like)."""
    with rasterio.open(src_path) as src:
        if band_index < 1 or band_index > src.count:
            raise ValueError(f"JRC_BAND_INDEX={band_index} nhưng file có {src.count} band.")
        src_arr = src.read(band_index).astype(np.float64)
        dst_h, dst_w = like_profile["height"], like_profile["width"]
        dst_arr = np.zeros((dst_h, dst_w), dtype=np.float64)

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=like_profile["transform"], dst_crs=like_profile["crs"],
            resampling=resampling
        )

        # Warp mask hợp lệ (dựa trên nodata nếu có; nếu không dùng NaN)
        nodata = src.nodata
        if nodata is not None:
            src_valid = ~np.isclose(src_arr, nodata)
            dst_valid = np.zeros((dst_h, dst_w), dtype=np.uint8)
            reproject(
                source=src_valid.astype(np.uint8),
                destination=dst_valid,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=like_profile["transform"], dst_crs=like_profile["crs"],
                resampling=Resampling.nearest
            )
            valid_like = dst_valid.astype(bool)
        else:
            valid_like = ~np.isnan(dst_arr)

    return dst_arr, valid_like

def save_mask_uint8(mask_bool: np.ndarray, profile_like: dict, out_path: Path, set_nodata_zero=True):
    """Lưu bool mask -> GeoTIFF uint8 (1/0)."""
    out_prof = profile_like.copy()
    out_prof.update({
        "dtype": "uint8", "count": 1,
        "compress": "deflate", "predictor": 2, "zlevel": 6
    })
    if set_nodata_zero:
        out_prof["nodata"] = 0
    with rasterio.open(out_path, "w", **out_prof) as dst:
        dst.write(mask_bool.astype(np.uint8), 1)

# =============== BƯỚC A: TẠO NƯỚC TĨNH ===============

def build_static_water_mask(dem_path: Path, jrc_path: Path,
                            delta_m: float, jrc_thr: float,
                            jrc_band_index: int, jrc_is_percent: bool,
                            out_static_tif: Path):
    """STATIC = (DEM <= min(DEM)+delta_m) AND (JRC >= jrc_thr)"""

    # 1) DEM
    dem, dem_valid, dem_prof = read_band_as_float(dem_path)
    if not np.any(dem_valid):
        raise ValueError("DEM không có pixel hợp lệ.")
    min_elev = float(np.min(dem[dem_valid]))
    dem_low_mask = (dem <= (min_elev + delta_m)) & dem_valid

    # 2) JRC về lưới DEM
    jrc_arr, jrc_valid = reproject_to_like(jrc_path, dem_prof, band_index=jrc_band_index, resampling=Resampling.nearest)
    if not jrc_is_percent:
        jrc_arr *= 100.0
    jrc_static_mask = (jrc_arr == 99) & jrc_valid  # Chỉ loại bỏ JRC = 99% (permanent 100%)

    # 3) STATIC = chỉ dùng JRC (bỏ điều kiện DEM)
    static_mask = jrc_static_mask  # Thay vì: dem_low_mask & jrc_static_mask

    # 4) Lưu
    out_static_tif.parent.mkdir(parents=True, exist_ok=True)
    save_mask_uint8(static_mask, dem_prof, out_static_tif, set_nodata_zero=True)

    # 5) Log
    print("[STATIC] Min elevation (m):", f"{min_elev:.3f}")
    print("[STATIC] DEM threshold (m):", f"{min_elev + delta_m:.3f} (Δ={delta_m} m)")
    print("[STATIC] Pixels DEM low   :", int(np.sum(dem_low_mask)))
    print("[STATIC] Pixels JRC static:", int(np.sum(jrc_static_mask)))
    print("[STATIC] Pixels STATIC (JRC only):", int(np.sum(static_mask)))
    print("[STATIC] Saved:", out_static_tif)

    return static_mask, dem_prof

# =============== BƯỚC B: TRỪ NƯỚC TĨNH KHỎI MORPHO ===============

def subtract_static_from_morpho(morpho_tif: Path, static_tif: Path, out_final_tif: Path):
    """FINAL = MORPHO AND (NOT STATIC) (warp STATIC về lưới MORPHO nếu cần)."""
    # Lưới chuẩn: MORPHO
    morpho_arr, morpho_valid, morpho_prof = read_band_as_float(morpho_tif)
    morpho_mask = (morpho_arr > 0) & morpho_valid
    
    print(f"[DEBUG] MORPHO shape: {morpho_arr.shape}")
    print(f"[DEBUG] MORPHO profile: {morpho_prof['width']}x{morpho_prof['height']}")

    # STATIC về lưới MORPHO
    static_like, static_valid = reproject_to_like(static_tif, morpho_prof, band_index=1, resampling=Resampling.nearest)
    static_mask = (static_like > 0) & static_valid
    
    print(f"[DEBUG] STATIC shape after reproject: {static_like.shape}")
    print(f"[DEBUG] STATIC valid pixels: {np.sum(static_valid)}")
    print(f"[DEBUG] STATIC mask pixels: {np.sum(static_mask)}")

    # FINAL - Logic đúng: chỉ cần morpho valid, không cần static valid ở khắp nơi
    final_mask = morpho_mask & (~static_mask) & morpho_valid
    
    # Debug chi tiết  
    morpho_in_valid = morpho_mask & morpho_valid
    static_in_morpho = static_mask & morpho_valid
    not_static_in_morpho = (~static_mask) & morpho_valid
    
    print(f"[DEBUG] MORPHO in valid area: {np.sum(morpho_in_valid)}")
    print(f"[DEBUG] STATIC in morpho area: {np.sum(static_in_morpho)}")
    print(f"[DEBUG] NOT STATIC in morpho area: {np.sum(not_static_in_morpho)}")
    print(f"[DEBUG] FINAL = MORPHO AND NOT STATIC: {np.sum(final_mask)}")
    
    overlap_valid = morpho_valid & static_valid  # Chỉ để debug
    
    print(f"[DEBUG] Overlap valid pixels: {np.sum(overlap_valid)}")
    print(f"[DEBUG] MORPHO pixels in overlap: {np.sum(morpho_mask & overlap_valid)}")
    print(f"[DEBUG] STATIC pixels in overlap: {np.sum(static_mask & overlap_valid)}")

    # Lưu
    out_final_tif.parent.mkdir(parents=True, exist_ok=True)
    save_mask_uint8(final_mask, morpho_prof, out_final_tif, set_nodata_zero=True)

    # Log
    print("[FINAL] Pixels MORPHO:", int(np.sum(morpho_mask)))
    print("[FINAL] Pixels STATIC:", int(np.sum(static_mask)))
    print("[FINAL] Pixels FINAL :", int(np.sum(final_mask)))
    print("[FINAL] Saved:", out_final_tif)

# ===================== MAIN =====================

def main():
    # A) tạo nước tĩnh từ DEM + JRC (min + 10 m, ngưỡng JRC)
    _static_mask, _dem_prof = build_static_water_mask(
        DEM_PATH, JRC_PATH,
        DELTA_M, JRC_THRESHOLD,
        JRC_BAND_INDEX, JRC_IS_PERCENT,
        OUT_STATIC_TIF
    )

    # B) trừ nước tĩnh khỏi mask sau morphology
    subtract_static_from_morpho(MORPHO_TIF, OUT_STATIC_TIF, OUT_FINAL_TIF)

if __name__ == "__main__":
    main()
