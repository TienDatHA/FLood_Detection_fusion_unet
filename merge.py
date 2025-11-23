import os
import glob
import math
import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from collections import Counter
import gc


def list_tifs(input_dir, recursive=True):
    pattern = "**/*.tif" if recursive else "*.tif"
    files = glob.glob(os.path.join(input_dir, pattern), recursive=recursive)
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files


def analyze_file_data_coverage(file_path, sample_count=5):
    """
    Phân tích coverage chi tiết của file bằng cách lấy nhiều sample
    """
    try:
        with rasterio.open(file_path) as src:
            width, height = src.width, src.height
            
            # Lấy multiple samples từ các vị trí khác nhau
            sample_positions = [
                (0, 0),  # Top-left
                (width//2, height//2),  # Center
                (width-100, height-100),  # Bottom-right
                (width//4, height//4),  # Quarter
                (3*width//4, 3*height//4),  # Three-quarter
            ]
            
            total_valid = 0
            total_sampled = 0
            
            for i, (x_start, y_start) in enumerate(sample_positions):
                if x_start >= width or y_start >= height:
                    continue
                    
                sample_size = min(100, width - x_start, height - y_start)
                if sample_size <= 0:
                    continue
                
                try:
                    window = rasterio.windows.Window(x_start, y_start, sample_size, sample_size)
                    sample = src.read(1, window=window)
                    
                    # Đếm valid pixels theo nhiều tiêu chí
                    if np.issubdtype(sample.dtype, np.floating):
                        # Float: check NaN và finite
                        valid_mask = np.isfinite(sample) & ~np.isnan(sample)
                        if src.nodata is not None:
                            valid_mask = valid_mask & (sample != src.nodata)
                    else:
                        # Integer: check nodata value
                        if src.nodata is not None:
                            valid_mask = sample != src.nodata
                        else:
                            valid_mask = np.ones_like(sample, dtype=bool)
                    
                    valid_count = np.sum(valid_mask)
                    total_valid += valid_count
                    total_sampled += sample.size
                    
                    print(f"     Sample {i+1} [{x_start}:{x_start+sample_size}, {y_start}:{y_start+sample_size}]: "
                          f"{valid_count}/{sample.size} valid ({100*valid_count/sample.size:.1f}%)")
                    
                    # Show actual values
                    if valid_count > 0:
                        valid_values = sample[valid_mask]
                        print(f"       Data range: {valid_values.min():.4f} to {valid_values.max():.4f}")
                    else:
                        unique_vals = np.unique(sample.flatten())[:5]  # First 5 unique values
                        print(f"       Unique values: {unique_vals}")
                        
                except Exception as e:
                    print(f"     Sample {i+1} error: {e}")
                    continue
            
            overall_coverage = (total_valid / total_sampled * 100) if total_sampled > 0 else 0
            return overall_coverage
            
    except Exception as e:
        print(f"   ❌ Cannot analyze coverage: {e}")
        return None


def print_file_info(file_path):
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        with rasterio.open(file_path) as src:
            print(f"📄 {os.path.basename(file_path)} | {src.width}x{src.height} | bands={src.count} | "
                  f"crs={src.crs} | res={src.res} | dtype={src.dtypes[0]} | nodata={src.nodata} | {file_size_mb:.1f}MB")
            print(f"   bounds: {src.bounds}")
            
            # **THÊM: Phân tích coverage chi tiết**
            if file_size_mb > 1000:  # File > 1GB
                print(f"   ⚠️ LARGE FILE: {file_size_mb:.1f}MB - Analyzing data coverage...")
                coverage = analyze_file_data_coverage(file_path)
                if coverage is not None:
                    print(f"   📊 Overall coverage: {coverage:.1f}%")
                    if coverage > 0:
                        print(f"   ✅ File contains valid data!")
                    else:
                        print(f"   ⚠️ File appears to be empty or all NoData")
            else:
                # Quick check for smaller files
                sample_size = min(100, src.width, src.height)
                sample = src.read(1, window=((0, sample_size), (0, sample_size)))
                
                if np.issubdtype(sample.dtype, np.floating):
                    valid_pixels = np.sum(np.isfinite(sample) & ~np.isnan(sample))
                    if src.nodata is not None:
                        valid_pixels = np.sum((sample != src.nodata) & np.isfinite(sample))
                else:
                    valid_pixels = np.sum(sample != src.nodata) if src.nodata is not None else sample.size
                
                coverage = (valid_pixels / sample.size) * 100
                print(f"   📊 Sample coverage: {coverage:.1f}% (quick check)")
                    
    except Exception as e:
        print(f"❌ Lỗi đọc file {file_path}: {e}")


def safe_merge_with_memory_management(srcs, method='first', nodata=None):
    """
    Merge files với memory management cho file lớn
    """
    print(f"🔄 Safe merge with memory management...")
    
    # Phân loại files theo kích thước
    large_files = []
    normal_files = []
    
    for i, src in enumerate(srcs):
        file_size_mb = os.path.getsize(src.name) / (1024 * 1024)
        if file_size_mb > 1000:  # > 1GB
            large_files.append((i, src, file_size_mb))
            print(f"  📦 Large file {i}: {os.path.basename(src.name)} ({file_size_mb:.1f}MB)")
        else:
            normal_files.append((i, src, file_size_mb))
    
    print(f"  📊 Normal files: {len(normal_files)}, Large files: {len(large_files)}")
    
    if len(large_files) == 0:
        # Không có file lớn → dùng merge bình thường
        print("  ✅ No large files, using standard merge...")
        return rio_merge(srcs, method=method.lower(), nodata=nodata)
    
    else:
        # Có file lớn → dùng VRT approach
        print("  🔄 Large files detected, using VRT approach...")
        
        try:
            # **KIỂM TRA**: Đảm bảo nodata được set đúng
            if nodata is None:
                # Tự động detect nodata từ sources
                nodata_candidates = [src.nodata for src in srcs if src.nodata is not None]
                if nodata_candidates:
                    nodata = nodata_candidates[0]
                    print(f"  🔧 Auto-detected nodata value: {nodata}")
                else:
                    # Set default nodata cho float
                    if srcs[0].dtypes[0] == 'float32':
                        nodata = -9999.0
                    else:
                        nodata = -9999
                    print(f"  🔧 Using default nodata value: {nodata}")
            
            # Tạo VRT cho từng file để tiết kiệm memory
            vrt_srcs = []
            for src in srcs:
                vrt = WarpedVRT(src, 
                               crs=srcs[0].crs,
                               nodata=nodata,  # Đảm bảo nodata được set
                               resampling=Resampling.nearest)
                vrt_srcs.append(vrt)
            
            # Merge VRTs thay vì files gốc
            mosaic, transform = rio_merge(vrt_srcs, method=method.lower(), nodata=nodata)
            
            # Cleanup VRTs
            for vrt in vrt_srcs:
                vrt.close()
            
            return mosaic, transform
            
        except Exception as e:
            print(f"  ❌ VRT approach failed: {e}")
            # Fallback: Thử merge từng file một cách tuần tự
            return sequential_merge(srcs, method, nodata)


def sequential_merge(srcs, method='first', nodata=None):
    """
    Merge files tuần tự để tránh memory overflow
    """
    print("  🔄 Sequential merge fallback...")
    
    if len(srcs) == 1:
        data = srcs[0].read()
        return data, srcs[0].transform
    
    # Bắt đầu với 2 files đầu tiên
    print(f"    Merging files 1-2...")
    mosaic, transform = rio_merge(srcs[:2], method=method.lower(), nodata=nodata)
    gc.collect()  # Free memory
    
    # Merge từng file tiếp theo
    for i in range(2, len(srcs)):
        print(f"    Adding file {i+1}...")
        
        # Tạo temporary raster từ mosaic hiện tại
        temp_profile = srcs[0].profile.copy()
        temp_profile.update({
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': transform,
            'dtype': mosaic.dtype,
            'nodata': nodata
        })
        
        # Ghi temporary file
        temp_path = f"/tmp/temp_mosaic_{i}.tif"
        with rasterio.open(temp_path, 'w', **temp_profile) as temp_dst:
            temp_dst.write(mosaic)
        
        # Merge temp file với file tiếp theo
        with rasterio.open(temp_path) as temp_src:
            mosaic, transform = rio_merge([temp_src, srcs[i]], method=method.lower(), nodata=nodata)
        
        # Cleanup
        os.remove(temp_path)
        gc.collect()
    
    return mosaic, transform


def analyze_output_data(output_path):
    """
    Phân tích dữ liệu trong file output
    """
    try:
        print(f"\n🔍 ANALYZING OUTPUT DATA:")
        with rasterio.open(output_path) as src:
            # Đọc toàn bộ band đầu tiên (hoặc sample lớn nếu file quá lớn)
            if src.width * src.height > 100_000_000:  # > 100M pixels
                # File rất lớn, chỉ sample
                sample_data = src.read(1, window=rasterio.windows.Window(0, 0, 
                                                                       min(1000, src.width), 
                                                                       min(1000, src.height)))
                print(f"  📊 Analyzing sample: {sample_data.shape}")
            else:
                sample_data = src.read(1)
                print(f"  📊 Analyzing full data: {sample_data.shape}")
            
            # Phân tích data
            total_pixels = sample_data.size
            
            if np.issubdtype(sample_data.dtype, np.floating):
                finite_mask = np.isfinite(sample_data)
                nan_count = np.sum(np.isnan(sample_data))
                inf_count = np.sum(np.isinf(sample_data))
                
                if src.nodata is not None:
                    nodata_count = np.sum(sample_data == src.nodata)
                    valid_mask = finite_mask & (sample_data != src.nodata)
                else:
                    nodata_count = 0
                    valid_mask = finite_mask
                
                valid_count = np.sum(valid_mask)
                
                print(f"  📈 Total pixels: {total_pixels:,}")
                print(f"  ✅ Valid pixels: {valid_count:,} ({100*valid_count/total_pixels:.2f}%)")
                print(f"  🚫 NoData pixels: {nodata_count:,}")
                print(f"  ⚠️ NaN pixels: {nan_count:,}")
                print(f"  ⚠️ Inf pixels: {inf_count:,}")
                
                if valid_count > 0:
                    valid_data = sample_data[valid_mask]
                    print(f"  📊 Data range: {valid_data.min():.6f} to {valid_data.max():.6f}")
                    print(f"  📊 Data mean: {valid_data.mean():.6f}")
                    print(f"  📊 Data std: {valid_data.std():.6f}")
                
            else:
                if src.nodata is not None:
                    valid_count = np.sum(sample_data != src.nodata)
                    nodata_count = np.sum(sample_data == src.nodata)
                else:
                    valid_count = total_pixels
                    nodata_count = 0
                
                print(f"  📈 Total pixels: {total_pixels:,}")
                print(f"  ✅ Valid pixels: {valid_count:,} ({100*valid_count/total_pixels:.2f}%)")
                print(f"  🚫 NoData pixels: {nodata_count:,}")
                
                if valid_count > 0:
                    valid_data = sample_data[sample_data != src.nodata] if src.nodata is not None else sample_data
                    print(f"  📊 Data range: {valid_data.min()} to {valid_data.max()}")
            
    except Exception as e:
        print(f"  ❌ Error analyzing output: {e}")


def merge_directory(input_dir, output_path, method='first', nodata=None, recursive=True, use_safe_merge=True):
    """
    Merge TẤT CẢ .tif trong thư mục thành 1 ảnh - ENHANCED VERSION cho file lớn
    method: 'first'|'last'|'min'|'max'|'sum'|'count'|'mean'
    nodata: giá trị NoData
    use_safe_merge: Sử dụng safe merge cho file lớn
    """
    tif_paths = list_tifs(input_dir, recursive=recursive)
    if not tif_paths:
        print("❌ Không tìm thấy file .tif nào trong thư mục.")
        return False

    print(f"📂 Tìm thấy {len(tif_paths)} file .tif - MERGE TẤT CẢ:")
    for i, p in enumerate(tif_paths, 1):
        print(f"  {i}. {os.path.basename(p)}")
    print("-" * 80)

    # Print detailed info
    total_size_mb = 0
    for p in tif_paths:
        print_file_info(p)
        total_size_mb += os.path.getsize(p) / (1024 * 1024)
    
    print(f"📊 Tổng dung lượng input: {total_size_mb:.1f} MB")
    print("-" * 80)

    # Mở tất cả files
    srcs = []
    failed_files = []
    for path in tif_paths:
        try:
            src = rasterio.open(path)
            srcs.append(src)
        except Exception as e:
            print(f"❌ Cannot open {os.path.basename(path)}: {e}")
            failed_files.append(path)
    
    if not srcs:
        print("❌ Không thể mở file nào!")
        return False
    
    if failed_files:
        print(f"⚠️ {len(failed_files)} file không thể mở, tiếp tục với {len(srcs)} files")
    
    try:        
        print(f"🎯 Target CRS: {srcs[0].crs}")
        print(f"📊 Merge {len(srcs)} ảnh thành 1 ảnh duy nhất bằng method '{method}'")
        
        if use_safe_merge and total_size_mb > 2000:  # > 2GB
            print(f"🔀 Using SAFE MERGE (total size: {total_size_mb:.1f}MB > 2GB)...")
            mosaic, transform = safe_merge_with_memory_management(srcs, method, nodata)
        else:
            print(f"🔀 Using STANDARD MERGE (total size: {total_size_mb:.1f}MB)...")
            mosaic, transform = rio_merge(srcs, method=method.lower(), nodata=nodata)
        
        print(f"✅ Merge completed! Mosaic shape: {mosaic.shape}")
        
        # Chuẩn bị metadata từ file đầu tiên
        out_meta = srcs[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "dtype": mosaic.dtype,
            "compress": "LZW",
            "tiled": True,
            "BIGTIFF": "YES" if total_size_mb > 1000 else "NO",
            "blockxsize": 512,
            "blockysize": 512,
        })
        
        # Set nodata nếu được chỉ định
        if nodata is not None:
            out_meta.update({"nodata": nodata})

        # Đảm bảo thư mục output tồn tại
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Ghi file
        print(f"💾 Writing to: {output_path}")
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        # **THÊM: Phân tích output data**
        analyze_output_data(output_path)

        # Clear memory
        del mosaic
        gc.collect()

        # Final report
        if os.path.exists(output_path):
            actual_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print("=" * 80)
            print("✅ MERGE HOÀN THÀNH!")
            print(f"📁 Output: {output_path}")
            print(f"📊 Đã merge: {len(srcs)} ảnh → 1 ảnh")
            print(f"📏 Kích thước: {out_meta['width']:,} x {out_meta['height']:,} x {out_meta['count']} bands")
            print(f"💾 Dung lượng file: {actual_size_mb:.1f} MB")
            print(f"🧭 CRS: {srcs[0].crs}")
            print(f"🚫 NoData: {out_meta.get('nodata', 'None')}")
            print("=" * 80)
            return True
        else:
            print("❌ Output file was not created!")
            return False

    except Exception as e:
        print(f"❌ Lỗi khi merge: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        for s in srcs:
            try: 
                s.close()
            except: 
                pass


if __name__ == "__main__":
    # Import config for flexible path resolution  
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from config import get_data_root, get_project_root, ensure_dirs
    
    # ====== CONFIGURATION ======
    DATA_ROOT = get_data_root()
    
    # Use environment variables for configuration
    REGION_NAME = os.getenv("REGION_NAME", "VietNamFlood_20221015")
    INPUT_DIR = str(DATA_ROOT / "GEE_EXPORTS/DEM/tiles")
    OUTPUT_FILE = str(DATA_ROOT / f"GEE_EXPORTS/DEM/{REGION_NAME}_DEM.tif")
    MERGE_METHOD = "first"   # 'first'|'last'|'min'|'max'|'sum'|'count'|'mean'
    NODATA_VALUE = None      # None for auto-detect, or set -9999
    
    # Ensure output directory exists
    ensure_dirs([Path(OUTPUT_FILE).parent])
    
    print(f"📍 Merge Configuration:")
    print(f"   Input dir: {INPUT_DIR}")
    print(f"   Output file: {OUTPUT_FILE}")

    print("🚀 ENHANCED MERGE - PHÂN TÍCH CHI TIẾT DATA")
    print("=" * 80)
    ok = merge_directory(INPUT_DIR, OUTPUT_FILE, method=MERGE_METHOD, nodata=NODATA_VALUE, 
                        recursive=True, use_safe_merge=True)
    print("🎉 THÀNH CÔNG!" if ok else "❌ THẤT BẠI!")