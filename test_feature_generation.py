#!/usr/bin/env python3
"""
Test script to examine feature generation using old_data_utils.py
This will help verify what features are actually being generated.
"""

import sys
import torch
import pickle
import json
import numpy as np
from pathlib import Path

# Import the old data utils directly
sys.path.append('utils_gnn')
from old_data_utils import build_gnn_data_for_schedule

def test_feature_generation():
    """Test the feature generation process directly"""
    print("=" * 60)
    print("TESTING FEATURE GENERATION FROM RAW DATA")
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
        print("[ERROR] No dataset file found. Looking for any .pkl files...")
        pkl_files = list(Path(".").glob("**/*.pkl"))
        if pkl_files:
            print("Available .pkl files:")
            for f in pkl_files[:10]:  # Show first 10
                print(f"  {f}")
            # Try to use the first one
            dataset_file = str(pkl_files[0])
        else:
            print("[ERROR] No .pkl files found!")
            return
    
    print(f"Using dataset file: {dataset_file}")
    
    try:
        # Load the dataset
        with open(dataset_file, 'rb') as f:
            programs_dict = pickle.load(f)
        
        print(f"Dataset contains {len(programs_dict)} programs")
        
        # Get the first program
        first_program_name = list(programs_dict.keys())[0]
        first_program = programs_dict[first_program_name]
        
        print(f"\nExamining program: {first_program_name}")
        print(f"Number of schedules: {len(first_program['schedules_list'])}")
        print(f"Initial execution time: {first_program['initial_execution_time']}")
        
        # Get the first schedule
        first_schedule = first_program['schedules_list'][0]
        print(f"First schedule execution times: {first_schedule.get('execution_times', 'N/A')}")
        
        # Test feature generation with add_exec_time=False (16D)
        print("\n--- TESTING 16D FEATURES (add_exec_time=False) ---")
        data_obj_16d = build_gnn_data_for_schedule(
            first_program,
            first_schedule,
            device="cpu",
            add_exec_time=False
        )
        
        print(f"16D Features - Node tensor shape: {data_obj_16d.x.shape}")
        print(f"16D Features - Edge tensor shape: {data_obj_16d.edge_index.shape}")
        print(f"16D Features - Target: {data_obj_16d.y.item():.4f}")
        
        x_16d = data_obj_16d.x
        print(f"16D Features - Min: {x_16d.min().item():.6f}, Max: {x_16d.max().item():.6f}, Mean: {x_16d.mean().item():.6f}")
        print(f"16D Features - Non-zero: {(x_16d != 0).sum().item()}/{x_16d.numel()} ({100*(x_16d != 0).sum().item()/x_16d.numel():.1f}%)")
        
        print(f"\n16D Feature values for first 3 nodes:")
        for i in range(min(3, x_16d.shape[0])):
            print(f"Node {i}: {x_16d[i].tolist()}")
        
        # Test feature generation with add_exec_time=True (17D)  
        print("\n--- TESTING 17D FEATURES (add_exec_time=True) ---")
        data_obj_17d = build_gnn_data_for_schedule(
            first_program,
            first_schedule,
            device="cpu",
            add_exec_time=True
        )
        
        print(f"17D Features - Node tensor shape: {data_obj_17d.x.shape}")
        print(f"17D Features - Edge tensor shape: {data_obj_17d.edge_index.shape}")
        print(f"17D Features - Target: {data_obj_17d.y.item():.4f}")
        
        x_17d = data_obj_17d.x
        print(f"17D Features - Min: {x_17d.min().item():.6f}, Max: {x_17d.max().item():.6f}, Mean: {x_17d.mean().item():.6f}")
        print(f"17D Features - Non-zero: {(x_17d != 0).sum().item()}/{x_17d.numel()} ({100*(x_17d != 0).sum().item()/x_17d.numel():.1f}%)")
        
        print(f"\n17D Feature values for first 3 nodes:")
        for i in range(min(3, x_17d.shape[0])):
            print(f"Node {i}: {x_17d[i].tolist()}")
        
        # Compare the difference
        print(f"\n--- COMPARING 16D vs 17D ---")
        if x_16d.shape[0] == x_17d.shape[0]:
            print("Extra column in 17D (execution time feature):")
            extra_col = x_17d[:, -1]  # Last column
            print(f"Execution time feature - Min: {extra_col.min().item():.6f}, Max: {extra_col.max().item():.6f}")
            print(f"Execution time values: {extra_col[:5].tolist()}")  # First 5 values
        
        # Examine the program annotation structure
        print(f"\n--- PROGRAM ANNOTATION STRUCTURE ---")
        prog_annot = first_program['program_annotation']
        print(f"Program annotation keys: {list(prog_annot.keys())}")
        
        if 'iterators' in prog_annot:
            iterators = prog_annot['iterators']
            print(f"Iterators type: {type(iterators)}")
            if isinstance(iterators, dict):
                print(f"Iterator names: {list(iterators.keys())}")
            else:
                print(f"Iterators: {iterators}")
        
        if 'computations' in prog_annot:
            computations = prog_annot['computations']
            print(f"Number of computations: {len(computations)}")
            comp_names = list(computations.keys())
            print(f"Computation names: {comp_names[:5]}...")  # First 5
            
            # Examine first computation
            first_comp = computations[comp_names[0]]
            print(f"First computation keys: {list(first_comp.keys())}")
        
        # Examine schedule structure
        print(f"\n--- SCHEDULE STRUCTURE ---")
        print(f"Schedule keys: {list(first_schedule.keys())}")
        
        if 'tree_structure' in first_schedule:
            tree = first_schedule['tree_structure']
            print(f"Tree structure type: {type(tree)}")
            if isinstance(tree, dict) and 'roots' in tree:
                print(f"Number of roots: {len(tree['roots'])}")
        
    except Exception as e:
        print(f"[ERROR] Failed to test feature generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Feature Generation Test Script")
    print("This will test the feature generation process directly from raw data.")
    
    test_feature_generation()
    
    print("\n" + "=" * 60)
    print("FEATURE GENERATION TEST COMPLETE")
    print("=" * 60)
