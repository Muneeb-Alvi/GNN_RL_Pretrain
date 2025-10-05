#!/usr/bin/env python3
"""
Debug script to examine actual data tensor values in the GNN dataset.
This will help verify if features are actually populated or just zeros.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add the utils_gnn directory to path
sys.path.append('utils_gnn')

# Import the old data utils (the one your colleague is referring to)
from old_data_utils import GNNDatasetParallel, build_gnn_data_for_schedule
from torch_geometric.loader import DataLoader

def examine_single_datapoint():
    """Load a single .pt file and examine its contents"""
    print("=" * 60)
    print("EXAMINING SINGLE DATAPOINT FROM .PT FILES")
    print("=" * 60)
    
    # Look for existing .pt files
    pt_files = list(Path("gnn_pickles").glob("*.pt"))
    if not pt_files:
        print("[ERROR] No .pt files found in gnn_pickles/")
        return
    
    # Load first .pt file
    pt_file = pt_files[0]
    print(f"Loading: {pt_file}")
    
    try:
        data = torch.load(pt_file)
        print(f"Data object type: {type(data)}")
        print(f"Node features shape: {data.x.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
        print(f"Target shape: {data.y.shape}")
        print(f"Target value: {data.y.item():.4f}")
        
        print("\n--- NODE FEATURES ANALYSIS ---")
        x = data.x
        print(f"Min value: {x.min().item():.6f}")
        print(f"Max value: {x.max().item():.6f}")
        print(f"Mean value: {x.mean().item():.6f}")
        print(f"Std value: {x.std().item():.6f}")
        print(f"Non-zero elements: {(x != 0).sum().item()}/{x.numel()} ({100*(x != 0).sum().item()/x.numel():.1f}%)")
        
        print(f"\nFirst 5 nodes (all features):")
        for i in range(min(5, x.shape[0])):
            print(f"Node {i}: {x[i].tolist()}")
        
        print(f"\nFeature statistics per dimension:")
        for dim in range(x.shape[1]):
            col = x[:, dim]
            non_zero = (col != 0).sum().item()
            print(f"Dim {dim:2d}: min={col.min().item():8.4f}, max={col.max().item():8.4f}, mean={col.mean().item():8.4f}, non_zero={non_zero:3d}/{x.shape[0]}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load {pt_file}: {e}")

def examine_dataset_batch():
    """Load dataset and examine a batch"""
    print("\n" + "=" * 60)
    print("EXAMINING DATASET BATCH")
    print("=" * 60)
    
    try:
        # Load existing pickled data
        dataset = GNNDatasetParallel(
            pkl_output_folder="gnn_pickles",
            just_load_pickled=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) == 0:
            print("[ERROR] Dataset is empty!")
            return
        
        # Create a small dataloader
        # Note: dataset.__getitem__ returns (data_obj, attrs), but DataLoader expects just data_obj
        # So we need to create a wrapper
        class DataOnlyDataset:
            def __init__(self, dataset):
                self.dataset = dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                data_obj, _ = self.dataset[idx]
                return data_obj
        
        data_only_dataset = DataOnlyDataset(dataset)
        loader = DataLoader(data_only_dataset, batch_size=4, shuffle=False)
        
        # Get first batch
        batch = next(iter(loader))
        
        print(f"Batch type: {type(batch)}")
        print(f"Batch size: {batch.batch.max().item() + 1}")
        print(f"Total nodes in batch: {batch.x.shape[0]}")
        print(f"Node features shape: {batch.x.shape}")
        print(f"Edge index shape: {batch.edge_index.shape}")
        print(f"Targets: {batch.y}")
        
        print("\n--- BATCH FEATURES ANALYSIS ---")
        x = batch.x
        print(f"Min value: {x.min().item():.6f}")
        print(f"Max value: {x.max().item():.6f}")
        print(f"Mean value: {x.mean().item():.6f}")
        print(f"Std value: {x.std().item():.6f}")
        print(f"Non-zero elements: {(x != 0).sum().item()}/{x.numel()} ({100*(x != 0).sum().item()/x.numel():.1f}%)")
        
        print(f"\nFeature statistics per dimension (across all nodes in batch):")
        for dim in range(x.shape[1]):
            col = x[:, dim]
            non_zero = (col != 0).sum().item()
            print(f"Dim {dim:2d}: min={col.min().item():8.4f}, max={col.max().item():8.4f}, mean={col.mean().item():8.4f}, non_zero={non_zero:3d}/{x.shape[0]}")
        
    except Exception as e:
        print(f"[ERROR] Failed to examine dataset batch: {e}")
        import traceback
        traceback.print_exc()

def test_model_forward():
    """Test the model forward pass with debug prints enabled"""
    print("\n" + "=" * 60)
    print("TESTING MODEL FORWARD PASS")
    print("=" * 60)
    
    try:
        # Temporarily enable debug prints in the model
        import utils_gnn.modeling as modeling_module
        
        # Get the source code and enable debug prints
        import inspect
        import re
        
        # Load a small batch
        dataset = GNNDatasetParallel(
            pkl_output_folder="gnn_pickles",
            just_load_pickled=True
        )
        
        if len(dataset) == 0:
            print("[ERROR] Dataset is empty!")
            return
        
        # Get first data point
        data_obj, _ = dataset[0]
        
        print(f"Single data object:")
        print(f"  Node features shape: {data_obj.x.shape}")
        print(f"  Edge index shape: {data_obj.edge_index.shape}")
        print(f"  Target: {data_obj.y.item():.4f}")
        
        # Create model
        model = modeling_module.SimpleGraphSAGE(
            in_channels=data_obj.x.shape[1],
            hidden_channels=64,
            num_layers=2,
            dropout=0.1
        )
        
        print(f"\nModel input size: {data_obj.x.shape[1]}")
        
        # Temporarily modify the forward function to enable debug prints
        original_forward = model.forward
        
        def debug_forward(self, data):
            x, edge_index = data.x, data.edge_index
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
            print(f"[DEBUG] Input tensor shape: {x.shape}")
            print(f"[DEBUG] Input tensor stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
            print(f"[DEBUG] Non-zero elements: {(x != 0).sum().item()}/{x.numel()}")
            print(f"[DEBUG] First few values:\n{x[:3, :8]}")  # First 3 nodes, first 8 features
            
            return original_forward(data)
        
        # Bind the debug forward method
        import types
        model.forward = types.MethodType(debug_forward, model)
        
        # Run forward pass
        model.eval()
        with torch.no_grad():
            output = model(data_obj)
            print(f"\nModel output: {output.item():.4f}")
        
    except Exception as e:
        print(f"[ERROR] Failed to test model forward: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("GNN Data Tensor Debug Script")
    print("This will examine the actual values in your dataset tensors.")
    
    # Run all examinations
    examine_single_datapoint()
    examine_dataset_batch() 
    test_model_forward()
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
