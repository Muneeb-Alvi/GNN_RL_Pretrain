import os
import io
import logging
import random
import gc
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import numpy as np

from utils_gnn.train_utils import train_gnn_model, mape_criterion, collate_with_attrs
from utils_gnn.data_utils import GNNDatasetParallel  
from utils_gnn.modeling import SimpleGCN, SimpleGAT, ResidualGIN, DeepResidualGAT, PearlGATv2, SimpleGraphSAGE

@hydra.main(config_path="conf", config_name="config-gnn")
def main(conf):
    log_folder_path = os.path.join(conf.experiment.base_path, "logs/")
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)
    log_file = os.path.join(log_folder_path, f"{conf.model.name}_{conf.training.lr}.log")
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s: %(message)s'
    )
    logger = logging.getLogger()

    # Setup wandb
    if conf.wandb.use_wandb:
        import wandb
        run_name = f"{conf.experiment.name}_{conf.model.name}_{conf.training.lr}"
        wandb.init(name=run_name, project=conf.wandb.project)
        wandb.config = dict(conf)
    
    # Decide on device
    train_device = torch.device(conf.training.training_gpu)
    val_device = torch.device(conf.training.validation_gpu)

    # --- 1) Load or create your GNN dataset ---
    gnn_dataset_train = GNNDatasetParallel(
        dataset_filename=conf.data_generation.train_dataset_file,
        pkl_output_folder="/scratch/maa9509/GNN_RL_Pretrain/gnn_pickles/train",
        nb_processes=4,
        device=conf.training.training_gpu,
        just_load_pickled=True
    )

    gnn_dataset_val = GNNDatasetParallel(
        dataset_filename=conf.data_generation.valid_dataset_file,
        pkl_output_folder="/scratch/maa9509/GNN_RL_Pretrain/gnn_pickles/val",
        nb_processes=4,
        device=conf.training.validation_gpu,
        just_load_pickled=True
    )


    # --- 2) Make PyG DataLoaders ---
    train_loader = DataLoader(
        gnn_dataset_train, 
        batch_size=conf.data_generation.batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_with_attrs
    )
    val_loader = DataLoader(
        gnn_dataset_val,
        batch_size=conf.data_generation.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_with_attrs
    )
    dataloaders = {"train": train_loader, "val": val_loader}
    if conf.model.name == "SimpleGAT":
        model = SimpleGAT(
            in_channels=conf.model.input_size,       # e.g. 16 or 17
            hidden_channels=conf.model.hidden_size,  # from config
            out_channels=1
        )
    elif conf.model.name == "SimpleGCN":
        model = SimpleGCN(
            in_channels=conf.model.input_size,       # e.g. 16 or 17
            hidden_channels=conf.model.hidden_size,  # from config
            out_channels=1
        )
        logger.info("SIMPLE GAT LOG")
    elif conf.model.name == "DeepResidualGAT":
        model = DeepResidualGAT(
            in_channels=conf.model.input_size,       # e.g. 16 or 17
            hidden_channels=conf.model.hidden_size,  # from config
            out_channels=1
        )
    elif conf.model.name == "ResidualGIN":
        model = ResidualGIN(
            in_channels=conf.model.input_size,       
            hidden_channels=conf.model.hidden_size,  
            out_channels=1
        )
    elif conf.model.name == "PearlGATv2":
        model = PearlGATv2(
            in_channels=conf.model.input_size,
            hidden_channels=64,
            num_heads=4,
            dropout=0.1,
            out_channels=1
        )
    elif conf.model.name == "SimpleGraphSAGE":
        model = SimpleGraphSAGE(
            in_channels=conf.model.input_size,
            hidden_channels=conf.model.hidden_size,
            out_channels=1
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=conf.training.lr,
        weight_decay=conf.training.get('weight_decay', 1e-2)
    )
    config_dump = OmegaConf.to_yaml(conf)
    logger.info("========== Experiment Configuration ==========\n" + config_dump)
    print("\n=== Dataset Diagnostics ===")
    sample_data = gnn_dataset_train.data_list[0]
    print(f"Sample graph: {sample_data.x.shape[0]} nodes, {sample_data.edge_index.shape[1]} edges")
    print(f"Node features shape: {sample_data.x.shape}")
    print(f"First 10 node features:\n{sample_data.x[:10]}")

    # Check statistics across first 100 graphs
    non_zero_counts = []
    for i in range(min(100, len(gnn_dataset_train.data_list))):
        data = gnn_dataset_train.data_list[i]
        non_zero_counts.append((data.x != 0).sum().item())
    print(f"\nNon-zero features per graph (first 100): mean={np.mean(non_zero_counts):.1f}, std={np.std(non_zero_counts):.1f}")
    best_loss, best_model_state = train_gnn_model(
        config=conf,
        model=model,
        criterion=mape_criterion,
        optimizer=optimizer,
        max_lr=conf.training.lr,
        dataloader_dict=dataloaders,
        num_epochs=conf.training.max_epochs,
        logger=logger,
        log_every=1,
        train_device=train_device,
        validation_device=val_device,
    )
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

if __name__ == "__main__":
    main()
