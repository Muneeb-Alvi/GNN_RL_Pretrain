#!/usr/bin/env python3
"""
Simple debug script to examine existing .pt files in the dataset.
This avoids importing complex modules and just looks at the saved data.
"""

import torch
import numpy as np
from pathlib import Path
import sys

def examine_pt_files():
    """Load and examine existing .pt files"""
    print("=" * 60)
    print("EXAMINING EXISTING .PT FILES")
    print("=" * 60)
    
    # Look for .pt files in both train and val directories
    train_dir = Path("gnn_pickles/train")
    val_dir = Path("gnn_pickles/val")
    
    for data_dir in [train_dir, val_dir]:
        if not data_dir.exists():
            print(f"[WARNING] Directory {data_dir} does not exist")
            continue
            
        pt_files = list(data_dir.glob("*.pt"))
        print(f"\n{data_dir}: Found {len(pt_files)} .pt files")
        
        if len(pt_files) == 0:
            continue
            
        # Examine first few files
        for i, pt_file in enumerate(pt_files[:3]):  # Look at first 3 files
            print(f"\n--- Examining {pt_file.name} ---")
            
            try:
                data = torch.load(pt_file, map_location='cpu')
                
                print(f"Data type: {type(data)}")
                if hasattr(data, 'x'):
                    x = data.x
                    print(f"Node features shape: {x.shape}")
                    print(f"Feature stats - min: {x.min().item():.6f}, max: {x.max().item():.6f}, mean: {x.mean().item():.6f}")
                    print(f"Non-zero elements: {(x != 0).sum().item()}/{x.numel()} ({100*(x != 0).sum().item()/x.numel():.1f}%)")
                    
                    # Show first node's features
                    if x.shape[0] > 0:
                        print(f"First node features: {x[0].tolist()}")
                    
                    # Feature dimension analysis
                    print(f"Feature dimension breakdown:")
                    for dim in range(min(x.shape[1], 20)):  # Show first 20 dims
                        col = x[:, dim]
                        non_zero = (col != 0).sum().item()
                        if non_zero > 0:  # Only show dimensions with non-zero values
                            print(f"  Dim {dim:2d}: min={col.min().item():8.4f}, max={col.max().item():8.4f}, non_zero={non_zero:3d}/{x.shape[0]}")
                
                if hasattr(data, 'edge_index'):
                    print(f"Edge index shape: {data.edge_index.shape}")
                    
                if hasattr(data, 'y'):
                    print(f"Target value: {data.y.item():.6f}")
                    
            except Exception as e:
                print(f"[ERROR] Failed to load {pt_file}: {e}")
                
        # Quick statistics across multiple files
        if len(pt_files) > 10:
            print(f"\n--- STATISTICS ACROSS {min(20, len(pt_files))} FILES ---")
            all_shapes = []
            all_targets = []
            all_feature_stats = []
            
            for pt_file in pt_files[:20]:  # Sample 20 files
                try:
                    data = torch.load(pt_file, map_location='cpu')
                    if hasattr(data, 'x'):
                        all_shapes.append(data.x.shape)
                        all_feature_stats.append({
                            'min': data.x.min().item(),
                            'max': data.x.max().item(), 
                            'mean': data.x.mean().item(),
                            'non_zero_pct': (data.x != 0).sum().item() / data.x.numel() * 100
                        })
                    if hasattr(data, 'y'):
                        all_targets.append(data.y.item())
                except:
                    continue
            
            if all_shapes:
                print(f"Node count range: {min(s[0] for s in all_shapes)} - {max(s[0] for s in all_shapes)}")
                print(f"Feature dimensions: {all_shapes[0][1]} (consistent: {all(s[1] == all_shapes[0][1] for s in all_shapes)})")
                
            if all_feature_stats:
                mins = [s['min'] for s in all_feature_stats]
                maxs = [s['max'] for s in all_feature_stats]
                means = [s['mean'] for s in all_feature_stats]
                non_zeros = [s['non_zero_pct'] for s in all_feature_stats]
                
                print(f"Feature value ranges:")
                print(f"  Min values: {min(mins):.6f} to {max(mins):.6f}")
                print(f"  Max values: {min(maxs):.6f} to {max(maxs):.6f}")
                print(f"  Mean values: {min(means):.6f} to {max(means):.6f}")
                print(f"  Non-zero %: {min(non_zeros):.1f}% to {max(non_zeros):.1f}%")
                
            if all_targets:
                print(f"Target (speedup) range: {min(all_targets):.4f} to {max(all_targets):.4f}")
                print(f"Target mean: {np.mean(all_targets):.4f}, std: {np.std(all_targets):.4f}")

def examine_model_forward():
    """Test model forward with actual data"""
    print("\n" + "=" * 60)
    print("TESTING MODEL FORWARD WITH ACTUAL DATA")
    print("=" * 60)
    
    # Find a data file
    train_dir = Path("gnn_pickles/train")
    pt_files = list(train_dir.glob("*.pt"))
    
    if len(pt_files) == 0:
        print("[ERROR] No .pt files found")
        return
        
    # Load first file
    data = torch.load(pt_files[0], map_location='cpu')
    print(f"Loaded data from: {pt_files[0].name}")
    print(f"Input shape: {data.x.shape}")
    
    # Import and create model
    sys.path.append('utils_gnn')
    from modeling import SimpleGraphSAGE
    
    model = SimpleGraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=64,
        num_layers=2,
        dropout=0.1
    )
    
    print(f"Created model with input size: {data.x.shape[1]}")
    
    # Temporarily enable debug prints by modifying the forward pass
    original_forward = model.forward
    
    def debug_forward(data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        print(f"[DEBUG] Input tensor shape: {x.shape}")
        print(f"[DEBUG] Input tensor stats - min: {x.min().item():.4f}, max: {x.max().item():.4f}, mean: {x.mean().item():.4f}")
        print(f"[DEBUG] Non-zero elements: {(x != 0).sum().item()}/{x.numel()}")
        print(f"[DEBUG] First 3 nodes, first 8 features:")
        for i in range(min(3, x.shape[0])):
            print(f"  Node {i}: {x[i, :8].tolist()}")
        
        # Call original forward
        return original_forward(data)
    
    model.forward = debug_forward
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        try:
            output = model(data)
            print(f"\n[DEBUG] Model output: {output.item():.6f}")
            print(f"[DEBUG] Target value: {data.y.item():.6f}")
        except Exception as e:
            print(f"[ERROR] Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Simple Data Debug Script")
    print("Examining existing .pt files to understand the data tensors.")
    
    examine_pt_files()
    
    try:
        examine_model_forward()
    except Exception as e:
        print(f"[WARNING] Model forward test failed: {e}")
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
