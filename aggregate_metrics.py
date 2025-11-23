#!/usr/bin/env python3
"""
Script để tính toán metrics tổng hợp từ nhiều khu vực
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import argparse


def load_metrics_from_region(region_path: Path) -> Dict[str, Any]:
    """Load metrics từ một khu vực"""
    metrics_file = region_path / "metrics" / "inference_metrics.json"
    if not metrics_file.exists():
        print(f"⚠️  Không tìm thấy metrics file: {metrics_file}")
        return None
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def aggregate_metrics(regions: List[str], base_path: Path = None) -> Dict[str, Any]:
    """
    Tính toán metrics tổng hợp từ nhiều khu vực
    
    Args:
        regions: Danh sách tên các khu vực
        base_path: Đường dẫn gốc chứa các khu vực
    
    Returns:
        Dict chứa metrics tổng hợp
    """
    if base_path is None:
        base_path = Path("/mnt/hdd2tb/Uni-Temporal-Flood-Detection-Sentinel-1_Frontiers22/inference_results/benchmarks")
    
    all_metrics = []
    valid_regions = []
    total_files = 0
    
    print("=" * 80)
    print("TÍNH TOÁN METRICS TỔNG HỢP TỪ NHIỀU KHU VỰC")
    print("=" * 80)
    
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
        raise ValueError("Không tìm thấy metrics từ bất kỳ khu vực nào!")
    
    print(f"\n✅ Đã load thành công metrics từ {len(valid_regions)} khu vực ({total_files} files)")
    
    # Tính toán metrics tổng hợp
    print("\n" + "=" * 60)
    print("TÍNH TOÁN METRICS TỔNG HỢP")
    print("=" * 60)
    
    # Có 2 cách tính:
    # 1. Weighted average theo số files của mỗi khu vực
    # 2. Simple average (mỗi khu vực có trọng số như nhau)
    
    # === 1. WEIGHTED AVERAGE ===
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
    
    # === 2. SIMPLE AVERAGE ===
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


def print_aggregated_results(results: Dict[str, Any]):
    """In kết quả tổng hợp một cách đẹp mắt"""
    print("\n" + "=" * 80)
    print("KẾT QUẢ METRICS TỔNG HỢP")
    print("=" * 80)
    
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
    
    # Per-region breakdown
    print(f"\n{'='*50}")
    print("📋 CHI TIẾT TỪNG KHU VỰC")
    print(f"{'='*50}")
    
    for region_data in results['per_region_summary']:
        region = region_data['region']
        files = region_data['files_processed']
        weight = region_data['weight_in_analysis']
        orig = region_data['overall_metrics_original']
        refined = region_data['overall_metrics_refined']
        
        print(f"\n🏷️  {region} ({files} files, weight={weight:.3f})")
        print(f"   Original : acc={orig['accuracy']:.4f} pre={orig['precision']:.4f} f1={orig['f1_score']:.4f} iou={orig['iou']:.4f}")
        print(f"   Refined  : acc={refined['accuracy']:.4f} pre={refined['precision']:.4f} f1={refined['f1_score']:.4f} iou={refined['iou']:.4f}")


def save_aggregated_results(results: Dict[str, Any], output_path: Path):
    """Lưu kết quả tổng hợp ra file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Đã lưu kết quả tổng hợp vào: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tính toán metrics tổng hợp từ nhiều khu vực")
    parser.add_argument("--regions", nargs="+", required=True, 
                       help="Danh sách các khu vực cần phân tích")
    parser.add_argument("--base_path", type=str, 
                       default="/mnt/hdd2tb/Uni-Temporal-Flood-Detection-Sentinel-1_Frontiers22/inference_results/benchmarks",
                       help="Đường dẫn gốc chứa các khu vực")
    parser.add_argument("--output", type=str,
                       default="inference_results/aggregated_metrics.json",
                       help="File output để lưu kết quả")
    
    args = parser.parse_args()
    
    try:
        # Tính toán metrics tổng hợp
        results = aggregate_metrics(args.regions, Path(args.base_path))
        
        # In kết quả
        print_aggregated_results(results)
        
        # Lưu kết quả
        output_path = Path(args.output)
        save_aggregated_results(results, output_path)
        
        print(f"\n✅ HOÀN THÀNH! Phân tích {len(args.regions)} khu vực với {results['total_files_processed']} files.")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
