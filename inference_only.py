#!/usr/bin/env python3
"""
Script change detection cho flood - so sánh 2 ngày
Sử dụng để phát hiện thay đổi lũ lụt giữa 2 thời điểm
"""

import tensorflow as tf
import numpy as np
import rasterio
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import argparse
from config import *
from model import Resnet50_UNet
from utils import scale_img
from merge import merge_tiles_to_geotiff as merge_predictions
from merge import merge_tiles_to_geotiff  # Import merge function

# Cấu hình sẽ được set từ command line arguments
REGION_BEFORE = None
REGION_AFTER = None
SHARED_REGION = None  # Region cho JRC và DEM chung
BENCH_BASE = ROOT_PATH / "Bench_Mark"

# Đường dẫn sẽ được set trong main
INFER_S1_PATH_BEFORE = None
INFER_S1_PATH_AFTER = None
INFER_DEM_PATH = None
INFER_JRC_PATH = None
INFER_OUTPUT_PATH = None
PRED_OUTPUT_PATH = None
VIS_OUTPUT_PATH = None

def load_model():
    """Load model từ checkpoint"""
    # Giữ nguyên 3 channels như model gốc
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

def load_and_preprocess_patch(s1_filename, s1_path, dem_path, jrc_path):
    """Load và preprocess một patch cho inference"""
    
    # Đọc S1 data và lấy profile
    with rasterio.open(s1_filename) as src:
        s1_data = src.read([1, 2])  # VV, VH
        s1_data = np.transpose(s1_data, (1, 2, 0))
        profile = src.profile
        crs = src.crs
        transform = src.transform
    
    # Extract coordinates từ filename
    base_name = Path(s1_filename).stem
    try:
        coords = base_name.split('_')[-4:]  # Lấy 4 số cuối: minx_miny_maxx_maxy
        minx, miny, maxx, maxy = map(int, coords)
    except:
        print(f"⚠️  Không thể extract coordinates từ {base_name}")
        return None, None, None, None
    
    # Tạo filename pattern cho DEM và JRC
    dem_filename = f"DEM_{minx}_{miny}_{maxx}_{maxy}.tif"
    jrc_filename = f"JRC_{minx}_{miny}_{maxx}_{maxy}.tif"
    
    # Construct full paths
    dem_full_path = dem_path / dem_filename
    jrc_full_path = jrc_path / jrc_filename
    
    # Load DEM
    if dem_full_path.exists():
        with rasterio.open(dem_full_path) as src:
            dem_data = src.read(1)
    else:
        print(f"⚠️  Không tìm thấy DEM file: {dem_filename}")
        dem_data = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    
    # Load JRC
    if jrc_full_path.exists():
        with rasterio.open(jrc_full_path) as src:
            jrc_data = src.read(1)
    else:
        print(f"⚠️  Không tìm thấy JRC file: {jrc_filename}")
        jrc_data = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    
    # Scale S1 data
    s1_scaled = scale_img(s1_data)
    
    # Resize nếu cần
    if s1_scaled.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
        s1_scaled = cv.resize(s1_scaled, (IMG_WIDTH, IMG_HEIGHT))
    if dem_data.shape != (IMG_HEIGHT, IMG_WIDTH):
        dem_data = cv.resize(dem_data, (IMG_WIDTH, IMG_HEIGHT))
    if jrc_data.shape != (IMG_HEIGHT, IMG_WIDTH):
        jrc_data = cv.resize(jrc_data, (IMG_WIDTH, IMG_HEIGHT))
    
    # Prepare inputs cho model
    # img input: VV, VH, JRC
    img_input = np.dstack([s1_scaled[:, :, 0], s1_scaled[:, :, 1], jrc_data])
    
    # inf input: VV, VH, DEM
    inf_input = np.dstack([s1_scaled[:, :, 0], s1_scaled[:, :, 1], dem_data])
    
    # Profile cho save
    s1_profile = {
        'crs': crs,
        'transform': transform,
        'width': profile['width'],
        'height': profile['height']
    }
    
    return img_input, inf_input, coords, s1_profile

def post_process_mask_advanced(pred, threshold=0.3, morphology_level="medium", use_jrc_filtering=False, jrc_img=None):
    """Post-processing mask với morphological operations - tùy chọn có dùng JRC hay không"""
    from skimage.morphology import opening, closing, remove_small_objects, remove_small_holes, disk
    
    # Binary threshold
    binary_mask = (pred > threshold).astype(np.uint8)
    
    # Remove permanent water (JRC = 1) - chỉ khi use_jrc_filtering=True
    if use_jrc_filtering and jrc_img is not None:
        permanent_water = (jrc_img == 1.0)
        binary_mask[permanent_water] = 0
        print(f"    🌊 JRC filtering: removed permanent water pixels")
    else:
        print(f"    🚫 Bỏ qua JRC filtering - chỉ dùng morphological processing")
    
    # Morphological operations
    if morphology_level == "light":
        kernel = disk(1)
    elif morphology_level == "medium":
        kernel = disk(2)
    elif morphology_level == "heavy":
        kernel = disk(3)
    else:
        kernel = disk(2)
        
    # Opening để loại bỏ noise
    opened = opening(binary_mask, kernel)
    
    # Closing để fill holes
    closed = closing(opened, kernel)
    
    # Remove small objects
    cleaned = remove_small_objects(closed.astype(bool), min_size=50, connectivity=2)
    
    # Remove small holes
    filled = remove_small_holes(cleaned, area_threshold=50, connectivity=2)
    
    return filled.astype(np.float32)

def save_prediction(prediction, filename, output_dir, crs, transform, bin_threshold=0.5):
    """Lưu prediction"""
    try:
        output_path = output_dir / filename
        
        # Convert to binary
        binary_pred = (prediction > bin_threshold).astype(np.uint8)
        
        # Metadata
        meta = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'width': binary_pred.shape[1],
            'height': binary_pred.shape[0],
            'count': 1,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw'
        }
        
        # Lưu
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(binary_pred, 1)
            
    except Exception as e:
        print(f"❌ Lỗi lưu {filename}: {e}")

def merge_predictions(input_dir, output_path, pattern="*.tif", exclude_pattern=None):
    """Merge các prediction tiles thành một ảnh lớn"""
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

def create_visualization(s1_data, dem_data, pred_original, pred_refined, filename, save_path):
    """Tạo visualization với 4 ảnh: S1 VV, S1 VH, DEM, Predictions"""
    try:
        # Chuẩn bị dữ liệu
        vv = s1_data[..., 0]
        vh = s1_data[..., 1]
        
        # Normalize để hiển thị
        def normalize_for_display(img):
            img_norm = img.copy()
            # Clip extreme values
            p2, p98 = np.percentile(img_norm[np.isfinite(img_norm)], [2, 98])
            img_norm = np.clip(img_norm, p2, p98)
            # Scale to 0-1
            img_norm = (img_norm - p2) / (p98 - p2)
            return img_norm
        
        vv_norm = normalize_for_display(vv)
        vh_norm = normalize_for_display(vh)
        dem_norm = normalize_for_display(dem_data)
        
        # Tạo figure với 2x3 layout
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # S1 VV
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(vv_norm, cmap='gray')
        ax1.set_title('Sentinel-1 VV', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # S1 VH
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(vh_norm, cmap='gray')
        ax2.set_title('Sentinel-1 VH', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # DEM
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(dem_norm, cmap='terrain')
        ax3.set_title('DEM (Elevation)', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # Prediction Original
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(pred_original, cmap='Blues', vmin=0, vmax=1)
        ax4.set_title('Prediction (Original)', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        # Prediction Refined
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(pred_refined, cmap='Blues', vmin=0, vmax=1)
        ax5.set_title('Prediction (Refined)', fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        # Difference
        ax6 = fig.add_subplot(gs[1, 2])
        diff = pred_refined - pred_original
        im6 = ax6.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax6.set_title('Difference (Refined - Original)', fontsize=12, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        
        # Thêm title chung
        fig.suptitle(f'Flood Detection Results: {filename}', fontsize=16, fontweight='bold', y=0.95)
        
        # Lưu và đóng
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi tạo visualization cho {filename}: {e}")
        return False

def create_summary_visualization(processed_files, save_path):
    """Tạo summary visualization với một số samples"""
    try:
        # Chọn một số files để hiển thị (tối đa 9)
        sample_files = processed_files[:9]
        n_samples = len(sample_files)
        
        if n_samples == 0:
            return False
            
        # Tính layout
        rows = int(np.ceil(np.sqrt(n_samples)))
        cols = int(np.ceil(n_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_samples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (fname, pred_refined) in enumerate(sample_files):
            if i < len(axes):
                axes[i].imshow(pred_refined, cmap='Blues', vmin=0, vmax=1)
                axes[i].set_title(f'{fname}', fontsize=10)
                axes[i].axis('off')
        
        # Ẩn axes thừa
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Flood Detection Summary ({n_samples} samples)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi tạo summary visualization: {e}")
        return False

def main():
    """Main function for change detection"""
    parser = argparse.ArgumentParser(description="Change Detection Inference between two dates")
    parser.add_argument("--before", required=True, help="Region name cho ngày trước (e.g., BinhDinh_20171021)")
    parser.add_argument("--after", required=True, help="Region name cho ngày sau (e.g., BinhDinh_20171110)")
    parser.add_argument("--shared", required=True, help="Region name cho JRC và DEM chung (e.g., BinhDinh_20171021)")
    
    args = parser.parse_args()
    
    # Set global variables
    global REGION_BEFORE, REGION_AFTER, SHARED_REGION
    global INFER_S1_PATH_BEFORE, INFER_S1_PATH_AFTER, INFER_DEM_PATH, INFER_JRC_PATH
    global INFER_OUTPUT_PATH, PRED_OUTPUT_PATH, VIS_OUTPUT_PATH
    
    REGION_BEFORE = args.before
    REGION_AFTER = args.after
    SHARED_REGION = args.shared
    
    # Setup paths
    INFER_S1_PATH_BEFORE = BENCH_BASE / "Sentinel1_Tiles" / REGION_BEFORE
    INFER_S1_PATH_AFTER = BENCH_BASE / "Sentinel1_Tiles" / REGION_AFTER
    INFER_DEM_PATH = BENCH_BASE / "DEM_Tiles_4326" / SHARED_REGION
    INFER_JRC_PATH = BENCH_BASE / "JRC_Tiles_Cut" / SHARED_REGION
    
    # Output paths
    INFER_OUTPUT_PATH = ROOT_PATH / "inference_results" / "change_detection" / f"{REGION_BEFORE}_to_{REGION_AFTER}"
    PRED_OUTPUT_PATH = INFER_OUTPUT_PATH / "predictions"
    VIS_OUTPUT_PATH = INFER_OUTPUT_PATH / "visualizations"
    
    # Create output directories
    INFER_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    PRED_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    VIS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Change Detection Inference")
    print(f"   Ngày trước: {REGION_BEFORE}")
    print(f"   Ngày sau: {REGION_AFTER}")
    print(f"   Shared region (JRC/DEM): {SHARED_REGION}")
    print(f"   Output: {INFER_OUTPUT_PATH}")
    
    # Run change detection inference
    run_change_detection_inference()


def run_change_detection_inference():
    """Chạy change detection inference cho 2 ngày"""
    
    print(f"\n🚀 Bắt đầu change detection inference")
    print(f"   S1 trước: {INFER_S1_PATH_BEFORE}")
    print(f"   S1 sau: {INFER_S1_PATH_AFTER}")
    print(f"   DEM: {INFER_DEM_PATH}")
    print(f"   JRC: {INFER_JRC_PATH}")
    
    # Kiểm tra thư mục input
    paths_to_check = [
        (INFER_S1_PATH_BEFORE, f"S1 trước {REGION_BEFORE}"),
        (INFER_S1_PATH_AFTER, f"S1 sau {REGION_AFTER}"),
        (INFER_DEM_PATH, f"DEM {SHARED_REGION}"),
        (INFER_JRC_PATH, f"JRC {SHARED_REGION}")
    ]
    
    for path, desc in paths_to_check:
        if not path.exists():
            print(f"❌ Không tìm thấy thư mục {desc}: {path}")
            return
    
    # Load model
    print("\n📥 Loading model...")
    model = load_model()
    if model is None:
        return
    
    # Tìm các file S1 trước
    s1_files_before = list(INFER_S1_PATH_BEFORE.glob("*.tif"))
    if not s1_files_before:
        print(f"❌ Không tìm thấy file S1 trong {INFER_S1_PATH_BEFORE}")
        return
    
    # Tìm các file S1 sau
    s1_files_after = list(INFER_S1_PATH_AFTER.glob("*.tif"))
    if not s1_files_after:
        print(f"❌ Không tìm thấy file S1 trong {INFER_S1_PATH_AFTER}")
        return
    
    print(f"📊 Tìm thấy {len(s1_files_before)} files S1 ngày trước")
    print(f"📊 Tìm thấy {len(s1_files_after)} files S1 ngày sau")
    
    # Run inference cho ngày trước
    print(f"\n🔄 Processing ngày trước: {REGION_BEFORE}")
    predictions_before = {}
    
    for i, s1_file in enumerate(s1_files_before):
        try:
            print(f"   Processing {i+1}/{len(s1_files_before)}: {s1_file.name}")
            
            # Load và preprocess
            img_input, inf_input, coords, s1_profile = load_and_preprocess_patch(
                s1_file, INFER_S1_PATH_BEFORE, INFER_DEM_PATH, INFER_JRC_PATH
            )
            
            if img_input is None:
                continue
            
            # Predict
            prediction = model.predict([
                np.expand_dims(img_input, axis=0),
                np.expand_dims(inf_input, axis=0)
            ], verbose=0)
            
            binary_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
            
            # Post-process - need JRC image for filtering
            jrc_img = img_input[:, :, 2]  # JRC channel from img_input
            refined_mask = post_process_mask_advanced(
                binary_mask, 
                threshold=0.3, 
                morphology_level="medium",
                use_jrc_filtering=True, 
                jrc_img=jrc_img
            )
            
            # Save prediction - use correct signature
            pred_file = PRED_OUTPUT_PATH / f"before_{s1_file.name}"
            save_prediction(
                refined_mask, 
                f"before_{s1_file.name}",
                PRED_OUTPUT_PATH, 
                s1_profile['crs'], 
                s1_profile['transform']
            )
            
            # Store cho merge
            predictions_before[s1_file.stem] = refined_mask
            
        except Exception as e:
            print(f"⚠️  Lỗi processing {s1_file.name}: {e}")
            continue
    
    # Run inference cho ngày sau
    print(f"\n🔄 Processing ngày sau: {REGION_AFTER}")
    predictions_after = {}
    
    for i, s1_file in enumerate(s1_files_after):
        try:
            print(f"   Processing {i+1}/{len(s1_files_after)}: {s1_file.name}")
            
            # Load và preprocess
            img_input, inf_input, coords, s1_profile = load_and_preprocess_patch(
                s1_file, INFER_S1_PATH_AFTER, INFER_DEM_PATH, INFER_JRC_PATH
            )
            
            if img_input is None:
                continue
            
            # Predict
            prediction = model.predict([
                np.expand_dims(img_input, axis=0),
                np.expand_dims(inf_input, axis=0)
            ], verbose=0)
            
            binary_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
            
            # Post-process - need JRC image for filtering
            jrc_img = img_input[:, :, 2]  # JRC channel from img_input
            refined_mask = post_process_mask_advanced(
                binary_mask, 
                threshold=0.3, 
                morphology_level="medium",
                use_jrc_filtering=True, 
                jrc_img=jrc_img
            )
            
            # Save prediction - use correct signature
            pred_file = PRED_OUTPUT_PATH / f"after_{s1_file.name}"
            save_prediction(
                refined_mask, 
                f"after_{s1_file.name}",
                PRED_OUTPUT_PATH, 
                s1_profile['crs'], 
                s1_profile['transform']
            )
            
            # Store cho merge
            predictions_after[s1_file.stem] = refined_mask
            
        except Exception as e:
            print(f"⚠️  Lỗi processing {s1_file.name}: {e}")
            continue
    
    print(f"\n✅ Inference hoàn thành!")
    print(f"   Ngày trước: {len(predictions_before)} predictions")
    print(f"   Ngày sau: {len(predictions_after)} predictions")
    
    # Merge predictions và tạo các outputs
    print(f"\n🔧 Merging predictions...")
    
    try:
        # 1. Merged ảnh ngày trước (morphological processed)
        before_output = INFER_OUTPUT_PATH / f"{REGION_BEFORE}_flood_refined.tif"
        merge_predictions(
            list(PRED_OUTPUT_PATH.glob("before_*.tif")),
            before_output,
            f"Flood Detection - {REGION_BEFORE} (Morphological Processed)"
        )
        
        # 2. Merged ảnh ngày sau (morphological processed)  
        after_output = INFER_OUTPUT_PATH / f"{REGION_AFTER}_flood_refined.tif"
        merge_predictions(
            list(PRED_OUTPUT_PATH.glob("after_*.tif")),
            after_output,
            f"Flood Detection - {REGION_AFTER} (Morphological Processed)"
        )
        
        # 3. Change detection: sau trừ trước (with rules: 1-1=0, 0-1=0)
        change_output = INFER_OUTPUT_PATH / f"change_{REGION_AFTER}_minus_{REGION_BEFORE}.tif"
        create_change_detection(before_output, after_output, change_output)
        
        # 4. Visualization: overlay trước lên sau
        vis_output = VIS_OUTPUT_PATH / f"overlay_{REGION_BEFORE}_on_{REGION_AFTER}.png"
        create_overlay_visualization(before_output, after_output, vis_output)
        
        print(f"📊 Outputs saved:")
        print(f"   Before (refined): {before_output}")
        print(f"   After (refined): {after_output}")
        print(f"   Change detection: {change_output}")
        print(f"   Visualization: {vis_output}")
        
    except Exception as e:
        print(f"⚠️  Lỗi trong quá trình merge: {e}")
    

def create_change_detection(before_tif, after_tif, output_tif):
    """Tạo change detection: sau trừ trước với rules: 1-1=0, 0-1=0"""
    try:
        # Read before image
        with rasterio.open(before_tif) as src_before:
            before_data = src_before.read(1)
            profile = src_before.profile
        
        # Read after image  
        with rasterio.open(after_tif) as src_after:
            after_data = src_after.read(1)
        
        # Apply change detection rules: sau trừ trước
        # 1-1=0, 0-1=0, 1-0=1, 0-0=0
        change_data = np.where((after_data == 1) & (before_data == 0), 1, 0)
        
        # Save change detection result
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(change_data, 1)
            dst.descriptions = ["Change Detection: New Floods (After - Before)"]
        
        print(f"✅ Change detection saved: {output_tif}")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi tạo change detection: {e}")
        return False


def create_overlay_visualization(before_tif, after_tif, output_png):
    """Tạo visualization overlay: đè ảnh trước lên ảnh sau"""
    try:
        # Read images
        with rasterio.open(before_tif) as src_before:
            before_data = src_before.read(1)
        
        with rasterio.open(after_tif) as src_after:
            after_data = src_after.read(1)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Base layer: after (blue)
        after_colored = np.zeros((*after_data.shape, 4))
        after_colored[after_data == 1] = [0, 0, 1, 0.7]  # Blue with transparency
        
        # Overlay: before (red)
        before_colored = np.zeros((*before_data.shape, 4))
        before_colored[before_data == 1] = [1, 0, 0, 0.8]  # Red with transparency
        
        # Show after as base
        ax.imshow(after_colored, extent=[0, after_data.shape[1], 0, after_data.shape[0]])
        
        # Overlay before on top
        ax.imshow(before_colored, extent=[0, before_data.shape[1], 0, before_data.shape[0]])
        
        ax.set_title(f'Flood Overlay: {REGION_BEFORE} (Red) on {REGION_AFTER} (Blue)', fontsize=14)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, label=f'Before ({REGION_BEFORE})'),
            Patch(facecolor='blue', alpha=0.7, label=f'After ({REGION_AFTER})'),
            Patch(facecolor='purple', alpha=0.8, label='Overlap')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Overlay visualization saved: {output_png}")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi tạo overlay visualization: {e}")
        return False


if __name__ == "__main__":
    main()
    main()
