#!/usr/bin/env python3
"""
Debug script to examine actual speedup values in the processed dataset
to understand why loss values are still high even with clipping.
"""

import torch
import numpy as np
import pickle
import os
from pathlib import Path

def examine_speedup_distribution():
    """Examine the distribution of speedup values in processed data"""
    
    gnn_pickles_dir = "/scratch/maa9509/GNN_RL_Pretrain/gnn_pickles"
    
    print("🔍 Examining speedup values in processed dataset...")
    
    # Check both train and val
    for split in ["train", "val"]:
        split_dir = os.path.join(gnn_pickles_dir, split)
        if not os.path.exists(split_dir):
            print(f"❌ {split} directory not found: {split_dir}")
            continue
            
        print(f"\n📊 Analyzing {split.upper()} split:")
        
        speedups = []
        file_count = 0
        
        # Load all pickle files in the split
        for pkl_file in Path(split_dir).glob("*.pkl"):
            try:
                with open(pkl_file, 'rb') as f:
                    data_list = pickle.load(f)
                
                # Extract speedup values from Data objects
                for func_name, data_attr_pairs in data_list:
                    for data_obj, attrs in data_attr_pairs:
                        if hasattr(data_obj, 'y') and data_obj.y is not None:
                            speedup_val = data_obj.y.item()
                            speedups.append(speedup_val)
                
                file_count += 1
                if file_count >= 3:  # Sample first few files to avoid OOM
                    break
                    
            except Exception as e:
                print(f"❌ Error loading {pkl_file}: {e}")
                continue
        
        if speedups:
            speedups = np.array(speedups)
            print(f"   📈 Total samples examined: {len(speedups)}")
            print(f"   📊 Speedup statistics:")
            print(f"      Min: {speedups.min():.6f}")
            print(f"      Max: {speedups.max():.6f}")
            print(f"      Mean: {speedups.mean():.6f}")
            print(f"      Median: {np.median(speedups):.6f}")
            print(f"      Std: {speedups.std():.6f}")
            print(f"   🎯 Percentiles:")
            print(f"      1%: {np.percentile(speedups, 1):.6f}")
            print(f"      5%: {np.percentile(speedups, 5):.6f}")
            print(f"      95%: {np.percentile(speedups, 95):.6f}")
            print(f"      99%: {np.percentile(speedups, 99):.6f}")
            
            # Check if clipping is working
            clipped_count = np.sum(speedups == 0.01)
            print(f"   ✂️  Values clipped to 0.01: {clipped_count} ({100*clipped_count/len(speedups):.2f}%)")
            
            # Check for extreme values
            extreme_high = np.sum(speedups > 100)
            extreme_low = np.sum(speedups < 0.1)
            print(f"   ⚡ Values > 100x: {extreme_high} ({100*extreme_high/len(speedups):.2f}%)")
            print(f"   🐌 Values < 0.1x: {extreme_low} ({100*extreme_low/len(speedups):.2f}%)")
            
        else:
            print(f"   ❌ No speedup values found in {split} split")

def check_mse_calculation():
    """Check what MSE values we should expect"""
    print(f"\n🧮 Expected MSE values:")
    
    # If most speedups are around 1.0 and we predict 1.0
    print(f"   If actual=1.0, pred=1.0 → MSE = 0.0")
    print(f"   If actual=2.0, pred=1.0 → MSE = 1.0") 
    print(f"   If actual=10.0, pred=1.0 → MSE = 81.0")
    print(f"   If actual=100.0, pred=1.0 → MSE = 9801.0")
    
    print(f"\n💡 High loss (~95) suggests many targets are 10-15x speedup range")

if __name__ == "__main__":
    examine_speedup_distribution()
    check_mse_calculation()



