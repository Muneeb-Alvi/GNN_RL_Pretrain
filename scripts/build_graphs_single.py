import os
import sys
import argparse
import shutil
from pathlib import Path

# Ensure repository root is on sys.path for imports when run via SLURM
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_gnn.data_utils import GNNDatasetParallel

ROOT = "/scratch/maa9509/GNN_RL_Pretrain"
TRAIN_DIR = f"{ROOT}/gnn_pickles/train"
VAL_DIR = f"{ROOT}/gnn_pickles/val"


def process_single_shard(shard_path, index, nb_processes=16):
    """Process a single shard and move .pt files to final train directory."""
    temp_out = f"{ROOT}/gnn_pickles/train_tmp_{index}"
    print(f"Processing shard {index+1}: {shard_path}")
    
    # Ensure output directories exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    
    # Convert shard to .pt files
    GNNDatasetParallel(
        dataset_filename=shard_path,
        pkl_output_folder=temp_out,
        nb_processes=nb_processes,
        device="cpu",
        just_load_pickled=False,
    )
    
    # Move .pt files to final train directory
    pt_count = 0
    for pt in Path(temp_out).glob("*.pt"):
        shutil.move(str(pt), f"{TRAIN_DIR}/{pt.name}")
        pt_count += 1
    
    # Clean up temp directory
    shutil.rmtree(temp_out)
    print(f"Completed shard {index+1}: moved {pt_count} .pt files to {TRAIN_DIR}")


def process_val(nb_processes=16):
    """Process validation set (only run once)."""
    src = f"{ROOT}/datasets/LOOPer_dataset_val.pkl"
    print(f"Processing val set: {src}")
    os.makedirs(VAL_DIR, exist_ok=True)
    
    GNNDatasetParallel(
        dataset_filename=src,
        pkl_output_folder=VAL_DIR,
        nb_processes=nb_processes,
        device="cpu",
        just_load_pickled=False,
    )
    
    val_count = len(list(Path(VAL_DIR).glob("*.pt")))
    print(f"Completed val set: {val_count} .pt files in {VAL_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=False, help="Path to shard .pkl file")
    parser.add_argument("--index", type=int, required=False, help="Shard index (0-based)")
    parser.add_argument("--val-only", action="store_true", help="Process only validation set")
    parser.add_argument("--nb-processes", type=int, default=16, help="Parallel processes to use")
    
    args = parser.parse_args()
    
    if args.val_only:
        process_val(nb_processes=args.nb_processes)
        return

    if not args.shard or args.index is None:
        parser.error("--shard and --index are required unless --val-only is set")

    process_single_shard(args.shard, args.index, nb_processes=args.nb_processes)


if __name__ == "__main__":
    main()
