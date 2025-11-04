#!/usr/bin/env python3
"""
Check the actual size of the dataset being used for training.
"""

import sys
import pickle
import os

sys.path.append('utils_gnn')
from data_utils import GNNDatasetParallel

def check_dataset_sizes():
    """Check the actual dataset sizes"""
    print("=" * 60)
    print("CHECKING ACTUAL DATASET SIZES")
    print("=" * 60)
    
    # Load train dataset
    print("Loading training dataset...")
    try:
        train_dataset = GNNDatasetParallel(
            pkl_output_folder="gnn_pickles/train",
            just_load_pickled=True
        )
        print(f"✅ Training dataset size: {len(train_dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading training dataset: {e}")
    
    # Load val dataset  
    print("Loading validation dataset...")
    try:
        val_dataset = GNNDatasetParallel(
            pkl_output_folder="gnn_pickles/val", 
            just_load_pickled=True
        )
        print(f"✅ Validation dataset size: {len(val_dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading validation dataset: {e}")
    
    # Check individual pkl files
    print(f"\n--- DETAILED BREAKDOWN ---")
    train_files = ["gnn_pickles/train/gnn_representation_part_0.pkl",
                   "gnn_pickles/train/gnn_representation_part_1.pkl", 
                   "gnn_pickles/train/gnn_representation_part_2.pkl",
                   "gnn_pickles/train/gnn_representation_part_3.pkl"]
    
    total_samples = 0
    for pkl_file in train_files:
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                local_list = pickle.load(f)
            
            file_samples = 0
            for fn, data_attr_pairs in local_list:
                file_samples += len(data_attr_pairs)
            
            print(f"  {os.path.basename(pkl_file)}: {file_samples} samples")
            total_samples += file_samples
    
    print(f"  Total from pkl files: {total_samples} samples")
    
    # Check original dataset
    print(f"\n--- ORIGINAL DATASET INFO ---")
    original_train = "datasets/dataset_expr_dataset_batch550000-838143+batch101-1227605_train_part_1_of_22.pkl"
    if os.path.exists(original_train):
        with open(original_train, 'rb') as f:
            programs_dict = pickle.load(f)
        
        total_schedules = 0
        for func_name, func_dict in programs_dict.items():
            total_schedules += len(func_dict.get("schedules_list", []))
        
        print(f"  Original dataset: {len(programs_dict)} programs, ~{total_schedules} schedules")
        print(f"  Expected samples after filtering: ~{total_schedules * 0.8:.0f} (rough estimate)")

if __name__ == "__main__":
    check_dataset_sizes()















