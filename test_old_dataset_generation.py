#!/usr/bin/env python3
"""
Generate a small dataset using old_data_utils.py to test what features it actually produces.
"""

import sys
import os
import shutil
from pathlib import Path

# Import the old data utils
sys.path.append('utils_gnn')
from old_data_utils import GNNDatasetParallel

def generate_test_dataset_with_old_utils():
    """Generate a small test dataset using old_data_utils.py"""
    print("=" * 60)
    print("GENERATING TEST DATASET WITH OLD_DATA_UTILS.PY")
    print("=" * 60)
    
    # Find an existing dataset file to use as input
    dataset_files = list(Path("datasets").glob("*.pkl"))
    if not dataset_files:
        print("[ERROR] No dataset files found in datasets/")
        return
    
    # Use the first available dataset file
    source_dataset = str(dataset_files[0])
    print(f"Using source dataset: {source_dataset}")
    
    # Create output folder for old_data_utils generated data
    output_folder = "test_old_gnn_pickles"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        print(f"Generating GNN data using old_data_utils.py...")
        print(f"Output folder: {output_folder}")
        
        # Generate dataset using old_data_utils
        gnn_dataset = GNNDatasetParallel(
            dataset_filename=source_dataset,
            pkl_output_folder=output_folder,
            nb_processes=2,  # Use fewer processes for testing
            device="cpu",
            just_load_pickled=False  # Generate new data
        )
        
        print(f"Generated {len(gnn_dataset)} data points")
        
        if len(gnn_dataset) > 0:
            # Examine first few data points
            print(f"\n--- EXAMINING GENERATED DATA ---")
            
            for i in range(min(5, len(gnn_dataset))):
                data_obj, attrs = gnn_dataset[i]
                x = data_obj.x
                
                print(f"\nData point {i}:")
                print(f"  Shape: {x.shape}")
                print(f"  Min: {x.min().item():.6f}, Max: {x.max().item():.6f}")
                print(f"  Mean: {x.mean().item():.6f}")
                non_zero = (x != 0).sum().item()
                total = x.numel()
                print(f"  Non-zero: {non_zero}/{total} ({100*non_zero/total:.1f}%)")
                print(f"  Target: {data_obj.y.item():.6f}")
                
                # Show first node features
                if x.shape[0] > 0:
                    print(f"  First node: {x[0].tolist()}")
                    
                # Show feature breakdown by dimension
                print(f"  Feature breakdown:")
                for dim in range(min(x.shape[1], 10)):  # Show first 10 dims
                    col = x[:, dim]
                    non_zero_dim = (col != 0).sum().item()
                    if non_zero_dim > 0:
                        print(f"    Dim {dim}: {non_zero_dim}/{x.shape[0]} non-zero, range [{col.min().item():.3f}, {col.max().item():.3f}]")
            
            # Overall statistics
            print(f"\n--- OVERALL STATISTICS ---")
            all_non_zero_pcts = []
            all_shapes = []
            
            for i in range(min(20, len(gnn_dataset))):
                data_obj, _ = gnn_dataset[i]
                x = data_obj.x
                non_zero_pct = (x != 0).sum().item() / x.numel() * 100
                all_non_zero_pcts.append(non_zero_pct)
                all_shapes.append(x.shape)
            
            import numpy as np
            avg_non_zero = np.mean(all_non_zero_pcts)
            min_non_zero = np.min(all_non_zero_pcts)
            max_non_zero = np.max(all_non_zero_pcts)
            
            print(f"Non-zero percentage across {len(all_non_zero_pcts)} samples:")
            print(f"  Average: {avg_non_zero:.1f}%")
            print(f"  Range: {min_non_zero:.1f}% - {max_non_zero:.1f}%")
            
            node_counts = [s[0] for s in all_shapes]
            feature_dims = [s[1] for s in all_shapes]
            print(f"Node counts: {min(node_counts)} - {max(node_counts)}")
            print(f"Feature dimensions: {min(feature_dims)} - {max(feature_dims)}")
            
            # Verdict
            print(f"\n{'='*60}")
            print("VERDICT FOR OLD_DATA_UTILS.PY:")
            if avg_non_zero < 5:
                print(f"❌ Produces mostly ZERO features ({avg_non_zero:.1f}% non-zero)")
                print("   Your colleague's claim 'not empty' appears to be INCORRECT")
            elif avg_non_zero < 15:
                print(f"⚠️  Produces sparse features ({avg_non_zero:.1f}% non-zero)")
                print("   Your colleague's claim 'not empty' is technically correct but features are very sparse")
            else:
                print(f"✅ Produces populated features ({avg_non_zero:.1f}% non-zero)")
                print("   Your colleague's claim 'not empty' is CORRECT")
            print(f"{'='*60}")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate test dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing old_data_utils.py by generating a fresh dataset")
    generate_test_dataset_with_old_utils()
    print("\nTest complete!")
