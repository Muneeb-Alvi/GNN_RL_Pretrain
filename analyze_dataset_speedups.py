#!/usr/bin/env python3
"""
Analyze speedup distributions in raw datasets (80k vs 1M) to understand
why 80k worked well and what we should expect from 1M dataset.
"""

import pickle
import numpy as np
import os
from pathlib import Path

def analyze_raw_speedup_distribution(dataset_path, dataset_name, max_programs=1000):
    """Analyze speedup distribution in a raw .pkl dataset"""
    
    print(f"\n🔍 Analyzing {dataset_name} dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return None
        
    try:
        print("📂 Loading dataset...")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"✅ Loaded {len(dataset)} programs")
        
        # Extract all speedup values
        speedups = []
        program_count = 0
        
        for func_name, func_data in dataset.items():
            if program_count >= max_programs:
                break
                
            program_exec_time = func_data["initial_execution_time"]
            
            for i, sched_json in enumerate(func_data["schedules_list"]):
                exec_times = sched_json.get("execution_times", [])
                if not exec_times or min(exec_times) <= 0:
                    continue
                    
                sched_exec_time = min(exec_times)
                speedup = program_exec_time / sched_exec_time
                speedups.append(speedup)
            
            program_count += 1
            if program_count % 100 == 0:
                print(f"   Processed {program_count} programs...")
        
        if not speedups:
            print("❌ No valid speedup values found")
            return None
            
        speedups = np.array(speedups)
        
        print(f"\n📊 {dataset_name} Speedup Distribution:")
        print(f"   Total samples: {len(speedups):,}")
        print(f"   Programs analyzed: {program_count}")
        print(f"   Min speedup: {speedups.min():.6f}")
        print(f"   Max speedup: {speedups.max():.6f}")
        print(f"   Mean speedup: {speedups.mean():.4f}")
        print(f"   Median speedup: {np.median(speedups):.4f}")
        print(f"   Std deviation: {speedups.std():.4f}")
        
        print(f"\n🎯 Percentiles:")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        for p in percentiles:
            print(f"   {p:4.1f}%: {np.percentile(speedups, p):8.4f}")
        
        print(f"\n⚡ Extreme Values:")
        extreme_ranges = [
            ("< 0.1x (major slowdowns)", speedups < 0.1),
            ("< 0.5x (slowdowns)", speedups < 0.5),
            ("0.5x - 2.0x (normal range)", (speedups >= 0.5) & (speedups <= 2.0)),
            ("> 2.0x (good speedups)", speedups > 2.0),
            ("> 5.0x (high speedups)", speedups > 5.0),
            ("> 10.0x (very high)", speedups > 10.0),
            ("> 20.0x (extreme)", speedups > 20.0),
            ("> 100.0x (outliers)", speedups > 100.0),
        ]
        
        for label, mask in extreme_ranges:
            count = np.sum(mask)
            pct = 100 * count / len(speedups)
            print(f"   {label:25s}: {count:6,} ({pct:5.2f}%)")
        
        # Calculate expected MSE if we predict median
        median_speedup = np.median(speedups)
        mse_if_predict_median = np.mean((speedups - median_speedup)**2)
        print(f"\n💡 Expected MSE if predicting median ({median_speedup:.2f}): {mse_if_predict_median:.2f}")
        
        # Calculate expected MSE if we predict 1.0
        mse_if_predict_1 = np.mean((speedups - 1.0)**2)
        print(f"   Expected MSE if predicting 1.0: {mse_if_predict_1:.2f}")
        
        return {
            'speedups': speedups,
            'count': len(speedups),
            'mean': speedups.mean(),
            'median': np.median(speedups),
            'std': speedups.std(),
            'min': speedups.min(),
            'max': speedups.max(),
            'mse_pred_median': mse_if_predict_median,
            'mse_pred_1': mse_if_predict_1
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")
        return None

def compare_datasets():
    """Compare speedup distributions between 80k and 1M datasets"""
    
    print("🔬 SPEEDUP DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Define dataset paths
    datasets = [
        # 80k dataset (the one that worked well)
        ("small_dataset/train_data_sample_500-programs_60k-schedules.pkl", "80k Dataset"),
        
        # 1M dataset (the problematic one)
        ("datasets/dataset_expr_dataset_batch550000-838143+batch101-1227605_train_part_1_of_22.pkl", "1M Dataset"),
        
        # Validation dataset
        ("datasets/LOOPer_dataset_val_250k.pkl", "250k Val Dataset"),
    ]
    
    results = {}
    
    for dataset_path, dataset_name in datasets:
        result = analyze_raw_speedup_distribution(dataset_path, dataset_name)
        if result:
            results[dataset_name] = result
    
    # Compare results
    if len(results) >= 2:
        print(f"\n🔍 COMPARISON SUMMARY:")
        print("=" * 50)
        
        for name, stats in results.items():
            print(f"\n📊 {name}:")
            print(f"   Samples: {stats['count']:,}")
            print(f"   Mean speedup: {stats['mean']:.4f}")
            print(f"   Median speedup: {stats['median']:.4f}")
            print(f"   Max speedup: {stats['max']:.2f}")
            print(f"   Expected MSE (pred=1.0): {stats['mse_pred_1']:.2f}")
        
        print(f"\n💡 INSIGHTS:")
        if '80k Dataset' in results and '1M Dataset' in results:
            r80k = results['80k Dataset']
            r1m = results['1M Dataset']
            
            print(f"   80k max speedup: {r80k['max']:.1f}x")
            print(f"   1M max speedup: {r1m['max']:.1f}x")
            print(f"   Ratio: {r1m['max']/r80k['max']:.1f}x higher in 1M dataset")
            print(f"   80k expected MSE: {r80k['mse_pred_1']:.2f}")
            print(f"   1M expected MSE: {r1m['mse_pred_1']:.2f}")
            
            if r1m['mse_pred_1'] > r80k['mse_pred_1'] * 2:
                print(f"   🚨 1M dataset has much higher variance - explains the high loss!")
            else:
                print(f"   ✅ Similar variance - issue might be elsewhere")

if __name__ == "__main__":
    compare_datasets()






