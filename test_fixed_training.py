#!/usr/bin/env python3
"""
Test script to verify that the fixes work properly by training for a few epochs
on a small subset of the data.
"""

import os
import logging
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from utils_gnn.train_utils import train_gnn_model, mape_criterion
from utils_gnn.data_utils import GNNDatasetParallel  
from utils_gnn.modeling import SimpleGCN

def collate_with_attrs(batch):
    """Custom collate function for batching GNN data and accompanying attributes."""
    data_list, attr_list = zip(*batch)
    batched_data = Batch.from_data_list(data_list)
    return batched_data, list(attr_list)

def test_fixed_training():
    print("🧪 Testing fixed training code...")
    
    # Use a small subset for quick testing
    train_dataset_file = "/scratch/maa9509/GNN_RL_Pretrain/datasets/dataset_expr_dataset_batch550000-838143+batch101-1227605_train_part_1_of_22.pkl"
    val_dataset_file = "/scratch/maa9509/GNN_RL_Pretrain/datasets/LOOPer_dataset_val_250k.pkl"
    
    # Create test directories
    test_pkl_dir = "/scratch/maa9509/GNN_RL_Pretrain/test_gnn_pickles"
    os.makedirs(f"{test_pkl_dir}/train", exist_ok=True)
    os.makedirs(f"{test_pkl_dir}/val", exist_ok=True)
    
    print("📦 Loading datasets with fixed code...")
    
    # Test with a small subset first
    gnn_dataset_train = GNNDatasetParallel(
        dataset_filename=train_dataset_file,
        pkl_output_folder=f"{test_pkl_dir}/train",
        nb_processes=2,  # Fewer processes for testing
        device="cpu",
        just_load_pickled=False  # Force regeneration with fixed code
    )
    
    # Take only first 1000 samples for quick test
    if len(gnn_dataset_train) > 1000:
        gnn_dataset_train.data_list = gnn_dataset_train.data_list[:1000]
        gnn_dataset_train.attr_list = gnn_dataset_train.attr_list[:1000]
    
    print(f"✅ Loaded {len(gnn_dataset_train)} training samples")
    
    # Check a few speedup values
    print("🔍 Checking speedup values in processed data:")
    speedups = []
    for i in range(min(10, len(gnn_dataset_train))):
        data, attrs = gnn_dataset_train[i]
        speedup = data.y.item()
        speedups.append(speedup)
        print(f"   Sample {i}: speedup = {speedup:.4f}")
    
    print(f"📊 Speedup stats: min={min(speedups):.4f}, max={max(speedups):.4f}, avg={sum(speedups)/len(speedups):.4f}")
    
    # Create a small validation set from the same data
    val_size = min(200, len(gnn_dataset_train) // 5)
    gnn_dataset_val = GNNDatasetParallel(dataset_filename=None, just_load_pickled=True)
    gnn_dataset_val.data_list = gnn_dataset_train.data_list[-val_size:]
    gnn_dataset_val.attr_list = gnn_dataset_train.attr_list[-val_size:]
    gnn_dataset_train.data_list = gnn_dataset_train.data_list[:-val_size]
    gnn_dataset_train.attr_list = gnn_dataset_train.attr_list[:-val_size]
    
    print(f"✅ Split: {len(gnn_dataset_train)} train, {len(gnn_dataset_val)} val")
    
    # Create DataLoaders
    train_loader = DataLoader(
        gnn_dataset_train, 
        batch_size=32,  # Small batch size for testing
        shuffle=True,
        collate_fn=collate_with_attrs,
    )
    val_loader = DataLoader(
        gnn_dataset_val,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_with_attrs,
    )
    dataloaders = {"train": train_loader, "val": val_loader}
    
    # Create a simple model
    model = SimpleGCN(
        in_channels=17,  # From config
        hidden_channels=128,
        out_channels=1
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-2
    )
    
    # Mock config object
    class MockConfig:
        def __init__(self):
            self.experiment = type('obj', (object,), {'base_path': '/scratch/maa9509/GNN_RL_Pretrain'})()
            self.wandb = type('obj', (object,), {'use_wandb': False})()
    
    config = MockConfig()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    print("🚀 Starting test training (3 epochs)...")
    
    try:
        best_loss, best_model_state = train_gnn_model(
            config=config,
            model=model,
            criterion=mape_criterion,  # Use regular MAPE loss
            optimizer=optimizer,
            max_lr=0.001,
            dataloader_dict=dataloaders,
            num_epochs=3,  # Just a few epochs for testing
            logger=logger,
            log_every=1,
            train_device=device,
            validation_device=device,
        )
        
        print(f"✅ Test training completed! Best loss: {best_loss:.4f}")
        
        # Check if loss values are reasonable (should be < 100 for MAPE)
        if best_loss < 100:
            print("🎉 SUCCESS: Loss values are in reasonable range!")
            return True
        else:
            print(f"⚠️  WARNING: Loss still high ({best_loss:.4f}), may need further investigation")
            return False
            
    except Exception as e:
        print(f"❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_training()
    if success:
        print("\n🎯 Test passed! The fixes appear to be working.")
    else:
        print("\n💥 Test failed! More investigation needed.")
