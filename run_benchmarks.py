#!/usr/bin/env python3
"""
Run inference_all across all regions in Bench_Mark/JRC_Tiles_Cut (or a specified subset).
This script temporarily patches `inference_all` module's INFER_* paths and METRICS/PRED output
locations per region, runs `main()`, then moves metrics/predictions into a region-specific folder.

Usage:
  python3 run_benchmarks.py --dry-run                 # list regions and commands
  python3 run_benchmarks.py --regions BacGiang_20240910 W...  # run specific
  python3 run_benchmarks.py --workers 1               # runs sequentially (default)

Notes:
 - Make sure conda env `flood` is active (has tensorflow).
 - Each region will create an output folder under inference_results/benchmarks/<region>.
 - This script imports `inference_all` directly and calls `main()`, so it will run in-process.
 - After running multiple regions, aggregated metrics will be computed automatically.
"""

import argparse
import importlib
import sys
from pathlib import Path
import subprocess
import os
import json
import numpy as np
from datetime import datetime
from multiprocessing import cpu_count

ROOT = Path(__file__).resolve().parent
BENCH_DIR = ROOT / "Bench_Mark" / "JRC_Tiles_Cut"

def _make_env_for_region(region_name, region_dir, out_dir):
    env = os.environ.copy()
    env['INFER_JRC_PATH'] = str(region_dir)
    env['INFER_OUTPUT_PATH'] = str(out_dir)
    env['PRED_OUTPUT_PATH'] = str(out_dir / 'preds')
    env['METRICS_OUTPUT_PATH'] = str(out_dir / 'metrics')
    # Also set S1/DEM/LABEL paths based on Bench_Mark layout
    bench = Path(__file__).resolve().parent / 'Bench_Mark'
    env['INFER_S1_PATH'] = str(bench / 'Sentinel1_Tiles' / region_name)
    env['INFER_DEM_PATH'] = str(bench / 'DEM_Tiles_4326' / region_name)
    env['INFER_LABEL_PATH'] = str(bench / 'Labels_Tiles_filtered' / region_name)
    return env

def run_region(region_name, dry_run=False):
    region_dir = BENCH_DIR / region_name
    if not region_dir.exists():
        print('Region dir not found:', region_name)
        return 1
    out_dir = ROOT / 'inference_results' / 'benchmarks' / region_name
    if dry_run:
        print('Would run:', region_name)
        print('  JRC input:', region_dir)
        print('  Output ->', out_dir)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'preds').mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics').mkdir(parents=True, exist_ok=True)

    env = _make_env_for_region(region_name, region_dir, out_dir)
    cmd = [sys.executable, str(ROOT / 'inference_all.py')]
    print('Spawning:', ' '.join(cmd), 'for region', region_name)
    # Run synchronously per region to avoid overloading GPU; caller can parallelize if desired
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def load_metrics_from_region(region_path: Path) -> dict:
    """Load metrics từ một khu vực"""
    metrics_file = region_path / "metrics" / "inference_metrics.json"
    if not metrics_file.exists():
        print(f"⚠️  Không tìm thấy metrics file: {metrics_file}")
        return None
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def aggregate_metrics_for_regions(regions: list, base_path: Path = None) -> dict:
    """Tính toán metrics tổng hợp từ nhiều khu vực"""
    if base_path is None:
        base_path = ROOT / "inference_results" / "benchmarks"
    
    all_metrics = []
    valid_regions = []
    total_files = 0
    
    print(f"\n{'='*80}")
    print("TÍNH TOÁN METRICS TỔNG HỢP TỪ NHIỀU KHU VỰC")
    print(f"{'='*80}")
    
    # Load metrics từ tất cả các khu vực
    for region in regions:
        region_path = base_path / region
        print(f"📂 Đang load metrics từ {region}...")
        
        metrics = load_metrics_from_region(region_path)
        if metrics is None:
            continue
            
        all_metrics.append(metrics)
        valid_regions.append(region)
        total_files += metrics.get('total_files_processed', 0)
        
        # In metrics của từng khu vực
        orig = metrics['overall_metrics_original']
        refined = metrics['overall_metrics_refined']
        print(f"   📊 {region}:")
        print(f"      Original : acc={orig['accuracy']:.4f} pre={orig['precision']:.4f} f1={orig['f1_score']:.4f} iou={orig['iou']:.4f}")
        print(f"      Refined  : acc={refined['accuracy']:.4f} pre={refined['precision']:.4f} f1={refined['f1_score']:.4f} iou={refined['iou']:.4f}")
    
    if not all_metrics:
        print("❌ Không tìm thấy metrics từ bất kỳ khu vực nào!")
        return None
    
    print(f"\n✅ Đã load thành công metrics từ {len(valid_regions)} khu vực ({total_files} files)")
    
    # Tính toán metrics tổng hợp
    print(f"\n{'='*60}")
    print("TÍNH TOÁN METRICS TỔNG HỢP")
    print(f"{'='*60}")
    
    # Weighted average theo số files của mỗi khu vực
    weighted_orig = {}
    weighted_refined = {}
    
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'iou', 'ssim']
    
    # Khởi tạo
    for key in metric_keys:
        weighted_orig[key] = 0.0
        weighted_refined[key] = 0.0
    
    # Tính weighted sum
    for metrics in all_metrics:
        num_files = metrics['total_files_processed']
        weight = num_files / total_files
        
        orig = metrics['overall_metrics_original']
        refined = metrics['overall_metrics_refined']
        
        for key in metric_keys:
            weighted_orig[key] += orig[key] * weight
            weighted_refined[key] += refined[key] * weight
    
    # Simple average
    simple_orig = {}
    simple_refined = {}
    
    for key in metric_keys:
        simple_orig[key] = np.mean([m['overall_metrics_original'][key] for m in all_metrics])
        simple_refined[key] = np.mean([m['overall_metrics_refined'][key] for m in all_metrics])
    
    # Tính improvement
    weighted_improvement = {
        f"{key}_improvement": weighted_refined[key] - weighted_orig[key] 
        for key in metric_keys
    }
    
    simple_improvement = {
        f"{key}_improvement": simple_refined[key] - simple_orig[key] 
        for key in metric_keys
    }
    
    # Tạo kết quả tổng hợp
    aggregated_results = {
        "timestamp": datetime.now().isoformat(),
        "regions_analyzed": valid_regions,
        "total_regions": len(valid_regions),
        "total_files_processed": total_files,
        "aggregation_method": "Both weighted (by files) and simple average",
        
        # Weighted average results
        "weighted_average": {
            "overall_metrics_original": weighted_orig,
            "overall_metrics_refined": weighted_refined,
            "improvement_metrics": weighted_improvement
        },
        
        # Simple average results  
        "simple_average": {
            "overall_metrics_original": simple_orig,
            "overall_metrics_refined": simple_refined,
            "improvement_metrics": simple_improvement
        },
        
        # Per-region summary
        "per_region_summary": []
    }
    
    # Thêm thông tin từng khu vực
    for i, region in enumerate(valid_regions):
        metrics = all_metrics[i]
        region_summary = {
            "region": region,
            "files_processed": metrics['total_files_processed'],
            "weight_in_analysis": metrics['total_files_processed'] / total_files,
            "overall_metrics_original": metrics['overall_metrics_original'],
            "overall_metrics_refined": metrics['overall_metrics_refined'],
            "improvement_metrics": metrics['improvement_metrics']
        }
        aggregated_results["per_region_summary"].append(region_summary)
    
    return aggregated_results


def print_aggregated_results(results: dict):
    """In kết quả tổng hợp một cách đẹp mắt"""
    print(f"\n{'='*80}")
    print("KẾT QUẢ METRICS TỔNG HỢP")
    print(f"{'='*80}")
    
    print(f"🔍 Số khu vực phân tích: {results['total_regions']}")
    print(f"📁 Tổng số files: {results['total_files_processed']}")
    print(f"📍 Các khu vực: {', '.join(results['regions_analyzed'])}")
    
    # Weighted Average
    print(f"\n{'='*50}")
    print("📊 WEIGHTED AVERAGE (theo số files)")
    print(f"{'='*50}")
    
    weighted = results['weighted_average']
    orig_w = weighted['overall_metrics_original']
    refined_w = weighted['overall_metrics_refined']
    improve_w = weighted['improvement_metrics']
    
    print(f"🔸 ORIGINAL : acc={orig_w['accuracy']:.4f} pre={orig_w['precision']:.4f} rec={orig_w['recall']:.4f} f1={orig_w['f1_score']:.4f} iou={orig_w['iou']:.4f} ssim={orig_w['ssim']:.4f}")
    print(f"🔹 REFINED  : acc={refined_w['accuracy']:.4f} pre={refined_w['precision']:.4f} rec={refined_w['recall']:.4f} f1={refined_w['f1_score']:.4f} iou={refined_w['iou']:.4f} ssim={refined_w['ssim']:.4f}")
    print(f"📈 IMPROVEMENT: acc={improve_w['accuracy_improvement']:+.4f} pre={improve_w['precision_improvement']:+.4f} rec={improve_w['recall_improvement']:+.4f} f1={improve_w['f1_score_improvement']:+.4f} iou={improve_w['iou_improvement']:+.4f} ssim={improve_w['ssim_improvement']:+.4f}")
    
    # Simple Average
    print(f"\n{'='*50}")
    print("📊 SIMPLE AVERAGE (trọng số đều)")
    print(f"{'='*50}")
    
    simple = results['simple_average']
    orig_s = simple['overall_metrics_original']
    refined_s = simple['overall_metrics_refined']
    improve_s = simple['improvement_metrics']
    
    print(f"🔸 ORIGINAL : acc={orig_s['accuracy']:.4f} pre={orig_s['precision']:.4f} rec={orig_s['recall']:.4f} f1={orig_s['f1_score']:.4f} iou={orig_s['iou']:.4f} ssim={orig_s['ssim']:.4f}")
    print(f"🔹 REFINED  : acc={refined_s['accuracy']:.4f} pre={refined_s['precision']:.4f} rec={refined_s['recall']:.4f} f1={refined_s['f1_score']:.4f} iou={refined_s['iou']:.4f} ssim={refined_s['ssim']:.4f}")
    print(f"📈 IMPROVEMENT: acc={improve_s['accuracy_improvement']:+.4f} pre={improve_s['precision_improvement']:+.4f} rec={improve_s['recall_improvement']:+.4f} f1={improve_s['f1_score_improvement']:+.4f} iou={improve_s['iou_improvement']:+.4f} ssim={improve_s['ssim_improvement']:+.4f}")


def save_aggregated_results(results: dict, output_path: Path):
    """Lưu kết quả tổng hợp ra file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Đã lưu kết quả tổng hợp vào: {output_path}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--regions', nargs='*', help='specific region folders to run')
    p.add_argument('--workers', type=int, default=1, help='number of parallel workers (default 1)')
    p.add_argument('--no-aggregate', action='store_true', help='skip aggregated metrics calculation')
    args = p.parse_args()

    regions = sorted([d.name for d in BENCH_DIR.iterdir() if d.is_dir()])
    if args.regions:
        regions = [r for r in regions if r in args.regions]

    if args.dry_run:
        print('Found regions:', len(regions))
        for r in regions:
            run_region(r, dry_run=True)
        print('Dry-run complete')
        sys.exit(0)

    failed_regions = []
    
    # Simple worker-based parallelism using subprocesses
    if args.workers <= 1:
        for r in regions:
            rc = run_region(r, dry_run=False)
            if rc != 0:
                print('Region', r, 'failed with exit code', rc)
                failed_regions.append(r)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        workers = min(args.workers, len(regions), max(1, cpu_count()//2))
        print('Running with', workers, 'workers')
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_region, r): r for r in regions}
            for fut in as_completed(futs):
                r = futs[fut]
                try:
                    rc = fut.result()
                except Exception as e:
                    print('Region', r, 'raised exception', e)
                    failed_regions.append(r)
                else:
                    if rc != 0:
                        print('Region', r, 'failed with exit code', rc)
                        failed_regions.append(r)
    
    # Tính toán metrics tổng hợp nếu có nhiều hơn 1 khu vực và không bị disable
    successful_regions = [r for r in regions if r not in failed_regions]
    
    if len(successful_regions) > 1 and not args.no_aggregate:
        try:
            print(f"\n{'='*80}")
            print(f"TÍNH TOÁN METRICS TỔNG HỢP CHO {len(successful_regions)} KHU VỰC")
            print(f"{'='*80}")
            
            # Tính toán metrics tổng hợp
            aggregated_results = aggregate_metrics_for_regions(successful_regions)
            
            if aggregated_results:
                # In kết quả
                print_aggregated_results(aggregated_results)
                
                # Lưu kết quả
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"aggregated_metrics_{len(successful_regions)}regions_{timestamp}.json"
                output_path = ROOT / "inference_results" / output_filename
                save_aggregated_results(aggregated_results, output_path)
                
                print(f"\n✅ HOÀN THÀNH TỔNG HỢP! Phân tích {len(successful_regions)} khu vực với {aggregated_results['total_files_processed']} files.")
            else:
                print("❌ Không thể tính toán metrics tổng hợp.")
        
        except Exception as e:
            print(f"❌ Lỗi khi tính toán metrics tổng hợp: {e}")
    
    elif len(successful_regions) <= 1:
        print(f"\n⚠️  Chỉ có {len(successful_regions)} khu vực thành công - không tính metrics tổng hợp.")
    
    else:
        print(f"\n📝 Bỏ qua tính toán metrics tổng hợp (--no-aggregate được chỉ định).")
    
    if failed_regions:
        print(f"\n❌ Các khu vực thất bại: {', '.join(failed_regions)}")
        sys.exit(1)
    else:
        print(f"\n🎉 Tất cả {len(successful_regions)} khu vực đã hoàn thành thành công!")
        sys.exit(0)
        
