import os
import sys
import yaml

ROOT_DIR = "/scratch/maa9509/GNN_RL_Pretrain"
sys.path.insert(0, ROOT_DIR)

from utils_gnn.data_utils import GNNDatasetParallel


def main():
    conf_path = os.path.join(ROOT_DIR, "conf", "config-gnn.yaml")
    with open(conf_path, "r") as f:
        conf = yaml.safe_load(f)

    train_dataset = conf["data_generation"]["train_dataset_file"]
    val_dataset = conf["data_generation"]["valid_dataset_file"]
    nb_proc = int(conf["data_generation"].get("nb_processes", 4))

    out_train = os.path.join(ROOT_DIR, "gnn_pickles", "train")
    out_val = os.path.join(ROOT_DIR, "gnn_pickles", "val")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_val, exist_ok=True)

    print("[CPU BUILD] Generating TRAIN pickles →", out_train)
    _ = GNNDatasetParallel(
        dataset_filename=train_dataset,
        pkl_output_folder=out_train,
        nb_processes=nb_proc,
        device="cpu",
        just_load_pickled=False,
    )

    print("[CPU BUILD] Generating VAL pickles →", out_val)
    _ = GNNDatasetParallel(
        dataset_filename=val_dataset,
        pkl_output_folder=out_val,
        nb_processes=nb_proc,
        device="cpu",
        just_load_pickled=False,
    )

    print("[CPU BUILD] Done.")


if __name__ == "__main__":
    main()





