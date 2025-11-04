import os
import torch
from pathlib import Path

# --- Configuration ---
# Point this to the directory where your .pt files are saved.
PT_FILES_DIR = "/scratch/maa9509/GNN_RL_Pretrain/gnn_pickles/train" 

def inspect_data():
    """
    Loads a few .pt files from the specified directory and prints
    diagnostic information about their feature vectors.
    """
    print(f"--- Inspecting .pt files in: {PT_FILES_DIR} ---")
    
    try:
        # Find the first 3 .pt files in the directory
        pt_files = list(Path(PT_FILES_DIR).glob('*.pt'))[:3]
        if not pt_files:
            print(f"ERROR: No .pt files found in {PT_FILES_DIR}.")
            return
            
        # Loop through the files to inspect them
        for i, file_path in enumerate(pt_files):
            print(f"\n--- Loading Sample #{i+1}: {file_path.name} ---")
            
            data_obj = torch.load(file_path)
            
            # This is the crucial test:
            features = data_obj.x
            zero_percentage = ((features == 0).sum() / features.numel()) * 100
            
            print("Feature tensor shape:", features.shape)
            print("First 5 rows of the feature tensor:\n", features[:5])
            print(f"Percentage of zeros in this feature tensor: {zero_percentage:.2f}%")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_data()