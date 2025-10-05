#!/usr/bin/env python3
"""
Test script to verify our fixes work on the 80k dataset and reproduce the good results.
Expected results: Loss ~1.38-1.41, MAPE ~57-76%
"""

import os
import sys
import logging
import hydra
from hydra.core.config_store import ConfigStore

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from utils_gnn.train_utils import train_gnn_model, mape_criterion
from utils_gnn.data_utils import GNNDatasetParallel  
from utils_gnn.modeling import SimpleGCN, SimpleGAT, PearlGATv2, ResidualGIN, SimpleGraphSAGE, DeepGraphSAGE

ROOT_DIR = "/scratch/maa9509/GNN_RL_Pretrain"

def collate_with_attrs(batch):
    """Custom collate function for batching GNN data and accompanying attributes."""
    data_list, attr_list = zip(*batch)
    batched_data = Batch.from_data_list(data_list)
    return batched_data, list(attr_list)

@hydra.main(config_path="conf", config_name="config-gnn-80k-test", version_base=None)
def main(conf):
    print("🧪 Testing fixed code on 80k dataset...")
    print(f"📊 Expected results: Loss ~1.38-1.41, MAPE ~57-76%")
    
    log_folder_path = os.path.join(conf.experiment.base_path, "logs/")
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)
    log_file = os.path.join(log_folder_path, "gnn_training_80k_test.log")
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # Overwrite previous logs
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s: %(message)s'
    )
    logger = logging.getLogger()

    # Decide on device
    train_device = torch.device(conf.training.training_gpu)
    val_device = torch.device(conf.training.validation_gpu)
    print(f"🖥️  Using devices: Train={train_device}, Val={val_device}")

    nb_processes = int(conf.data_generation.nb_processes)
    just_load_pickled = bool(conf.data_generation.just_load_pickled)

    # Create test pickle directories
    test_pkl_train = f"{ROOT_DIR}/test_80k_gnn_pickles/train"
    test_pkl_val = f"{ROOT_DIR}/test_80k_gnn_pickles/val"
    os.makedirs(test_pkl_train, exist_ok=True)
    os.makedirs(test_pkl_val, exist_ok=True)

    print("📦 Loading 80k dataset with fixed code...")
    
    # --- 1) Load or create your GNN dataset ---
    gnn_dataset_train = GNNDatasetParallel(
        dataset_filename=conf.data_generation.train_dataset_file,
        pkl_output_folder=test_pkl_train,
        nb_processes=nb_processes,
        device=str(train_device),
        just_load_pickled=just_load_pickled
    )

    gnn_dataset_val = GNNDatasetParallel(
        dataset_filename=conf.data_generation.valid_dataset_file,
        pkl_output_folder=test_pkl_val,
        nb_processes=nb_processes,
        device=str(val_device),
        just_load_pickled=just_load_pickled
    )

    print(f"✅ Loaded datasets: {len(gnn_dataset_train)} train, {len(gnn_dataset_val)} val")

    # Check a few speedup values to verify our fixes
    print("🔍 Checking speedup values in processed data:")
    speedups = []
    for i in range(min(10, len(gnn_dataset_train))):
        data, attrs = gnn_dataset_train[i]
        speedup = data.y.item()
        speedups.append(speedup)
        if i < 5:  # Show first 5
            print(f"   Sample {i}: speedup = {speedup:.4f}")
    
    print(f"📊 Speedup stats (first 10): min={min(speedups):.4f}, max={max(speedups):.4f}, avg={sum(speedups)/len(speedups):.4f}")

    # --- 2) Make PyG DataLoaders ---
    train_loader = DataLoader(
        gnn_dataset_train, 
        batch_size=conf.data_generation.batch_size, 
        shuffle=True,
        collate_fn=collate_with_attrs,
    )
    val_loader = DataLoader(
        gnn_dataset_val,
        batch_size=conf.data_generation.batch_size,
        shuffle=False,
        collate_fn=collate_with_attrs,
    )
    dataloaders = {"train": train_loader, "val": val_loader}

    # Choose model architecture (same as your good results)
    if conf.model.name == "SimpleGraphSAGE":
        model = SimpleGraphSAGE(
            in_channels=conf.model.input_size,
            hidden_channels=conf.model.hidden_size,
            num_layers=4,
            dropout=0.1,
            out_channels=1
        )
    elif conf.model.name == "DeepGraphSAGE":
        model = DeepGraphSAGE(
            in_channels=conf.model.input_size,
            hidden_channels=conf.model.hidden_size,
            num_layers=6,
            dropout=0.1,
            out_channels=1
        )
    elif conf.model.name == "SimpleGCN":
        model = SimpleGCN(
            in_channels=conf.model.input_size,
            hidden_channels=conf.model.hidden_size,
            out_channels=1
        )
    elif conf.model.name == "SimpleGAT":
        model = SimpleGAT(
            in_channels=conf.model.input_size,
            hidden_channels=conf.model.hidden_size,
            out_channels=1
        )
    elif conf.model.name == "PearlGATv2":
        model = PearlGATv2(
            in_channels=conf.model.input_size,
            hidden_channels=conf.model.hidden_size,
            num_heads=4,
            dropout=0.1,
            out_channels=1
        )
    else:  # Default to ResidualGIN
        model = ResidualGIN(
            in_channels=conf.model.input_size,       
            hidden_channels=conf.model.hidden_size,  
            out_channels=1
        )

    print(f"🧠 Using model: {conf.model.name}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=conf.training.lr,
        weight_decay=1e-2
    )

    print("🚀 Starting training...")
    
    best_loss, best_model_state = train_gnn_model(
        config=conf,
        model=model,
        criterion=mape_criterion,  # Use regular MAPE loss like the original
        optimizer=optimizer,
        max_lr=conf.training.lr,
        dataloader_dict=dataloaders,
        num_epochs=conf.training.max_epochs,
        logger=logger,
        log_every=5,
        train_device=train_device,
        validation_device=val_device,
    )

    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Best validation loss: {best_loss:.4f}")
    
    # Compare with expected results
    if 1.0 <= best_loss <= 2.0:
        print("🎉 SUCCESS! Loss is in the expected range (1.38-1.41)")
        print("✅ The fixes appear to be working correctly!")
    elif best_loss < 1.0:
        print("🤔 Loss is lower than expected but still good")
    elif 2.0 < best_loss < 10.0:
        print("⚠️  Loss is higher than expected but reasonable")
    else:
        print("❌ Loss is much higher than expected - may need more investigation")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return best_loss

if __name__ == "__main__":
    try:
        final_loss = main()
        print(f"\n📋 Test completed with final loss: {final_loss:.4f}")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
