#!/usr/bin/env python3
"""
Check what's actually in .pkl files vs .pt files to understand the data pipeline.
"""

import pickle
import torch
import sys
import os

sys.path.append('utils_gnn')

def check_pkl_content():
    """Check what's in the raw .pkl files"""
    print("=" * 60)
    print("CHECKING .PKL FILE CONTENT (RAW DATA)")
    print("=" * 60)
    
    pkl_file = "small_dataset/train_data_sample_500-programs_60k-schedules.pkl"
    
    with open(pkl_file, 'rb') as f:
        programs_dict = pickle.load(f)
    
    print(f"Loaded {len(programs_dict)} programs from {pkl_file}")
    
    # Look at first program
    func_name = list(programs_dict.keys())[0]
    func_dict = programs_dict[func_name]
    
    print(f"\nFirst program: {func_name}")
    print(f"Keys in program dict: {list(func_dict.keys())}")
    
    # Look at first schedule
    if "schedules_list" in func_dict:
        first_schedule = func_dict["schedules_list"][0]
        print(f"\nFirst schedule keys: {list(first_schedule.keys())}")
        
        # Check if there are any feature vectors in the raw data
        if "program_annotation" in func_dict:
            prog_ann = func_dict["program_annotation"]
            print(f"Program annotation keys: {list(prog_ann.keys())}")
            
            if "computations_dict" in prog_ann:
                comp_dict = prog_ann["computations_dict"]
                first_comp = list(comp_dict.values())[0]
                print(f"First computation keys: {list(first_comp.keys())}")
                
                # Look for any existing feature vectors
                if "features" in first_comp:
                    print(f"❌ Found 'features' in computation - this would be unexpected!")
                else:
                    print(f"✅ No 'features' key found - raw data as expected")

def check_pt_content():
    """Check what's in the processed .pt files"""
    print("\n" + "=" * 60)
    print("CHECKING .PT FILE CONTENT (PROCESSED DATA)")
    print("=" * 60)
    
    # Find a .pt file
    pt_dir = "gnn_pickles/train"
    if not os.path.exists(pt_dir):
        print(f"❌ No {pt_dir} directory found")
        return
    
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
    if not pt_files:
        print(f"❌ No .pt files found in {pt_dir}")
        return
    
    pt_file = os.path.join(pt_dir, pt_files[0])
    print(f"Loading {pt_file}")
    
    try:
        data_obj = torch.load(pt_file, map_location='cpu')
        print(f"✅ Loaded .pt file successfully")
        print(f"Type: {type(data_obj)}")
        
        if hasattr(data_obj, 'x'):
            x = data_obj.x
            print(f"Feature matrix shape: {x.shape}")
            print(f"Feature matrix min/max: {x.min().item():.6f} / {x.max().item():.6f}")
            print(f"Non-zero elements: {(x != 0).sum().item()}/{x.numel()} ({100*(x != 0).sum().item()/x.numel():.1f}%)")
            print(f"✅ .pt file contains actual populated feature vectors!")
        else:
            print(f"❌ No 'x' attribute found in data object")
            
    except Exception as e:
        print(f"❌ Error loading .pt file: {e}")

def main():
    print("UNDERSTANDING THE DATA PIPELINE")
    print("Checking what's in .pkl vs .pt files...")
    
    check_pkl_content()
    check_pt_content()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(".pkl files = Raw program data (NO feature vectors)")
    print(".pt files = Processed graph data (WITH feature vectors)")
    print("Feature vectors are created by build_gnn_data_for_schedule()")
    print("=" * 60)

if __name__ == "__main__":
    main()
















