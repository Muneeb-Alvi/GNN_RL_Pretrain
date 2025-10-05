#!/usr/bin/env python3
"""
Simple test to check what features old_data_utils.py actually produces
by directly calling build_gnn_data_for_schedule on a few samples.
"""

import sys
import os
import pickle
import torch
import numpy as np

# Import the old data utils
sys.path.append('utils_gnn')
from old_data_utils import build_gnn_data_for_schedule

def test_old_features():
    """Test what features old_data_utils.py actually produces"""
    print("=" * 60)
    print("TESTING OLD_DATA_UTILS.PY FEATURE GENERATION")
    print("=" * 60)
    
    # Use the small dataset file like we always do
    dataset_file = "small_dataset/train_data_sample_500-programs_60k-schedules.pkl"
    if not os.path.exists(dataset_file):
        print(f"[ERROR] Dataset file not found: {dataset_file}")
        return
    print(f"Using dataset: {dataset_file}")
    
    # Load the dataset
    with open(dataset_file, 'rb') as f:
        programs_dict = pickle.load(f)
    
    print(f"Dataset contains {len(programs_dict)} programs")
    
    # Test on just 2 programs to keep it light
    function_names = list(programs_dict.keys())[:2]  # Just test 2 programs
    
    total_samples = 0
    all_non_zero_pcts = []
    
    for func_name in function_names:
        func_dict = programs_dict[func_name]
        print(f"\nTesting function: {func_name}")
        
        # Test first schedule only for this function
        schedules = func_dict.get("schedules_list", [])[:1]  # Just 1 schedule per function
        
        for i, sched_json in enumerate(schedules):
            try:
                print(f"  Schedule {i}:")
                
                # Call build_gnn_data_for_schedule from old_data_utils
                data_obj = build_gnn_data_for_schedule(
                    func_dict, 
                    sched_json, 
                    device="cpu"
                )
                
                x = data_obj.x
                print(f"    Shape: {x.shape}")
                print(f"    Min: {x.min().item():.6f}, Max: {x.max().item():.6f}")
                print(f"    Mean: {x.mean().item():.6f}")
                
                non_zero = (x != 0).sum().item()
                total = x.numel()
                non_zero_pct = 100 * non_zero / total
                print(f"    Non-zero: {non_zero}/{total} ({non_zero_pct:.1f}%)")
                
                all_non_zero_pcts.append(non_zero_pct)
                total_samples += 1
                
                # Show first node features
                if x.shape[0] > 0:
                    print(f"    First node: {x[0].tolist()}")
                    
                # Show feature breakdown by dimension
                print(f"    Feature breakdown:")
                for dim in range(min(x.shape[1], 8)):  # Show first 8 dims
                    col = x[:, dim]
                    non_zero_dim = (col != 0).sum().item()
                    if non_zero_dim > 0:
                        print(f"      Dim {dim}: {non_zero_dim}/{x.shape[0]} non-zero, range [{col.min().item():.3f}, {col.max().item():.3f}]")
                    else:
                        print(f"      Dim {dim}: ALL ZEROS")
                        
            except Exception as e:
                print(f"    [ERROR] Failed on schedule {i}: {e}")
    
    # Overall results
    if all_non_zero_pcts:
        avg_non_zero = np.mean(all_non_zero_pcts)
        min_non_zero = np.min(all_non_zero_pcts)
        max_non_zero = np.max(all_non_zero_pcts)
        
        print(f"\n{'='*60}")
        print(f"RESULTS FROM {total_samples} SAMPLES:")
        print(f"Non-zero percentage:")
        print(f"  Average: {avg_non_zero:.1f}%")
        print(f"  Range: {min_non_zero:.1f}% - {max_non_zero:.1f}%")
        
        # Verdict
        print(f"\n{'='*60}")
        print("VERDICT FOR OLD_DATA_UTILS.PY:")
        if avg_non_zero < 5:
            print(f"❌ Produces mostly ZERO features ({avg_non_zero:.1f}% non-zero)")
            print("   Your colleague's claim 'features are not empty' is INCORRECT")
        elif avg_non_zero < 15:
            print(f"⚠️  Produces sparse features ({avg_non_zero:.1f}% non-zero)")
            print("   Your colleague's claim 'not empty' is technically correct but misleading")
        else:
            print(f"✅ Produces populated features ({avg_non_zero:.1f}% non-zero)")
            print("   Your colleague's claim 'features are not empty' is CORRECT")
        print(f"{'='*60}")
    else:
        print("[ERROR] No valid samples processed")

if __name__ == "__main__":
    test_old_features()
