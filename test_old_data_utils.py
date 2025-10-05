#!/usr/bin/env python3
"""
Test script to examine what features the old_data_utils.py actually generates.
This will help verify if your colleague's version produces non-zero values.
"""

import sys
import torch
import pickle
import numpy as np
from pathlib import Path

# Import the old data utils
sys.path.append('utils_gnn')
from old_data_utils import build_gnn_data_for_schedule

def test_old_data_utils():
    """Test the old_data_utils.py feature generation"""
    print("=" * 60)
    print("TESTING OLD_DATA_UTILS.PY FEATURE GENERATION")
    print("=" * 60)
    
    # Try to find a dataset file to test with
    dataset_files = [
        "data_samples/train_data_sample_500-programs_60k-schedules.pkl",
        "data_samples/val_data_sample_500-programs_60k-schedules.pkl",
    ]
    
    dataset_file = None
    for f in dataset_files:
        if Path(f).exists():
            dataset_file = f
            break
    
    if dataset_file is None:
        print("[ERROR] No dataset file found.")
        return
    
    print(f"Using dataset file: {dataset_file}")
    
    try:
        # Load the dataset
        with open(dataset_file, 'rb') as f:
            programs_dict = pickle.load(f)
        
        print(f"Dataset contains {len(programs_dict)} programs")
        
        # Test with first few programs
        program_names = list(programs_dict.keys())[:5]  # Test 5 programs
        
        all_results = []
        
        for prog_name in program_names:
            prog_dict = programs_dict[prog_name]
            print(f"\n--- Testing program: {prog_name} ---")
            
            # Test first schedule
            if len(prog_dict['schedules_list']) > 0:
                first_schedule = prog_dict['schedules_list'][0]
                
                # Test with add_exec_time=True (17D)
                print("Testing old_data_utils with add_exec_time=True (17D):")
                data_obj_17d = build_gnn_data_for_schedule(
                    prog_dict,
                    first_schedule,
                    device="cpu",
                    add_exec_time=True
                )
                
                x_17d = data_obj_17d.x
                print(f"  Shape: {x_17d.shape}")
                print(f"  Min: {x_17d.min().item():.6f}, Max: {x_17d.max().item():.6f}")
                print(f"  Mean: {x_17d.mean().item():.6f}")
                non_zero_17d = (x_17d != 0).sum().item()
                total_17d = x_17d.numel()
                print(f"  Non-zero: {non_zero_17d}/{total_17d} ({100*non_zero_17d/total_17d:.1f}%)")
                
                # Show first few nodes
                for i in range(min(3, x_17d.shape[0])):
                    print(f"  Node {i}: {x_17d[i].tolist()}")
                
                # Test with add_exec_time=False (16D)
                print("\nTesting old_data_utils with add_exec_time=False (16D):")
                data_obj_16d = build_gnn_data_for_schedule(
                    prog_dict,
                    first_schedule,
                    device="cpu",
                    add_exec_time=False
                )
                
                x_16d = data_obj_16d.x
                print(f"  Shape: {x_16d.shape}")
                print(f"  Min: {x_16d.min().item():.6f}, Max: {x_16d.max().item():.6f}")
                print(f"  Mean: {x_16d.mean().item():.6f}")
                non_zero_16d = (x_16d != 0).sum().item()
                total_16d = x_16d.numel()
                print(f"  Non-zero: {non_zero_16d}/{total_16d} ({100*non_zero_16d/total_16d:.1f}%)")
                
                # Store results for summary
                all_results.append({
                    'program': prog_name,
                    'nodes_17d': x_17d.shape[0],
                    'features_17d': x_17d.shape[1],
                    'non_zero_pct_17d': 100*non_zero_17d/total_17d,
                    'nodes_16d': x_16d.shape[0],
                    'features_16d': x_16d.shape[1],
                    'non_zero_pct_16d': 100*non_zero_16d/total_16d,
                    'target': data_obj_16d.y.item()
                })
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY OF OLD_DATA_UTILS.PY RESULTS")
        print(f"{'='*60}")
        
        for result in all_results:
            print(f"Program: {result['program']}")
            print(f"  17D: {result['nodes_17d']} nodes × {result['features_17d']} features, {result['non_zero_pct_17d']:.1f}% non-zero")
            print(f"  16D: {result['nodes_16d']} nodes × {result['features_16d']} features, {result['non_zero_pct_16d']:.1f}% non-zero")
            print(f"  Target: {result['target']:.4f}")
        
        # Overall stats
        avg_non_zero_17d = np.mean([r['non_zero_pct_17d'] for r in all_results])
        avg_non_zero_16d = np.mean([r['non_zero_pct_16d'] for r in all_results])
        
        print(f"\nOverall averages:")
        print(f"  17D non-zero percentage: {avg_non_zero_17d:.1f}%")
        print(f"  16D non-zero percentage: {avg_non_zero_16d:.1f}%")
        
        # Verdict
        print(f"\n{'='*60}")
        print("VERDICT:")
        if avg_non_zero_16d < 5:
            print("❌ OLD_DATA_UTILS produces mostly ZERO features (< 5% non-zero)")
        elif avg_non_zero_16d < 15:
            print("⚠️  OLD_DATA_UTILS produces sparse features (5-15% non-zero)")
        else:
            print("✅ OLD_DATA_UTILS produces populated features (> 15% non-zero)")
        
        print(f"Your colleague's claim: Features are 'not empty'")
        print(f"Reality: {avg_non_zero_16d:.1f}% of values are non-zero")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"[ERROR] Failed to test old_data_utils: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing old_data_utils.py feature generation")
    print("This will show if your colleague's version actually produces non-zero features.")
    
    test_old_data_utils()
    
    print("\nTest complete!")
