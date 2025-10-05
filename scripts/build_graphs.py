import os
import sys
import glob
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


def ensure_dirs():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)


def process_train_shards():
    shards = sorted(glob.glob(f"{ROOT}/datasets/dataset_expr_dataset_*_train_part_*_of_22.pkl"))
    print(f"Found {len(shards)} train shards")
    for i, shard in enumerate(shards):
        temp_out = f"{ROOT}/gnn_pickles/train_tmp_{i}"
        print(f"Processing shard {i+1}/{len(shards)}: {shard}")
        GNNDatasetParallel(
            dataset_filename=shard,
            pkl_output_folder=temp_out,
            nb_processes=4,
            device="cpu",
            just_load_pickled=False,
        )
        for pt in Path(temp_out).glob("*.pt"):
            shutil.move(str(pt), f"{TRAIN_DIR}/{pt.name}")
        shutil.rmtree(temp_out)


def process_val():
    src = f"{ROOT}/datasets/LOOPer_dataset_val.pkl"
    print(f"Processing val set: {src}")
    GNNDatasetParallel(
        dataset_filename=src,
        pkl_output_folder=VAL_DIR,
        nb_processes=4,
        device="cpu",
        just_load_pickled=False,
    )


def main():
    ensure_dirs()
    process_train_shards()
    process_val()
    n_train = len(list(Path(TRAIN_DIR).glob("*.pt")))
    n_val = len(list(Path(VAL_DIR).glob("*.pt")))
    print(f"Done. Train .pt files: {n_train}, Val .pt files: {n_val}")


if __name__ == "__main__":
    main()













