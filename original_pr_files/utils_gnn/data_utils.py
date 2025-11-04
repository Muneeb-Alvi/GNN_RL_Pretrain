import copy
import json
import pickle
import random
import re
import gc
import sys
import multiprocessing
import shutil
import resource
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import enum
import os, psutil
import sympy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Exceptions
class NbTranformationException(Exception):
    pass

class NbAccessException(Exception):
    pass

class LoopsDepthException(Exception):
    pass

# Global configuration
MAX_NUM_TRANSFORMATIONS = 4
MAX_TAGS = 16
MAX_DEPTH = 5
MAX_EXPR_LEN = 66

#############################################################
# GNN DATA BUILDING
#############################################################

def build_loop_feature(loop_name, program_annot, schedule_json):
    """
    Build an 8-dimensional feature vector for a loop node.
    Features: [parallelized, tiled, tile_factor, fused, unrolled, unroll_factor, shifted, shift_factor]
    """
    feat = torch.zeros(8, dtype=torch.float)
    
    # Initialize loop schedule dict with max values across all computations using this loop
    loop_schedule = {
        'parallelized': 0,
        'tiled': 0,
        'tile_factor': 0,
        'fused': 0,
        'unrolled': 0,
        'unroll_factor': 0,
        'shifted': 0,
        'shift_factor': 0
    }
    
    # Extract schedule information for this loop from all computations
    computations_dict = program_annot["computations"]
    for comp_name in sorted(computations_dict.keys()):
        if comp_name not in schedule_json:
            continue
            
        comp_dict = computations_dict[comp_name]
        comp_schedule = schedule_json[comp_name]
        
        # Check if this loop belongs to this computation's iterator list
        if loop_name not in comp_dict["iterators"]:
            continue
        
        # Get the position of this loop in the computation's nest
        loop_position = comp_dict["iterators"].index(loop_name)
        
        # Parallelization
        if comp_schedule.get("parallelized_dim") == loop_name:
            loop_schedule['parallelized'] = 1
        
        # Tiling - check both by name and by position
        if comp_schedule.get("tiling"):
            tiling_dims = comp_schedule["tiling"].get("tiling_dims", [])
            if loop_name in tiling_dims:
                loop_schedule['tiled'] = 1
                idx = tiling_dims.index(loop_name)
                tile_factor = int(comp_schedule["tiling"]["tiling_factors"][idx])
                # Take the max tile factor if multiple comps tile this loop differently
                loop_schedule['tile_factor'] = max(loop_schedule['tile_factor'], tile_factor)
        
        # Unrolling (applied to innermost loop)
        if comp_schedule.get("unrolling_factor") and len(comp_dict["iterators"]) > 0:
            innermost = comp_dict["iterators"][-1]
            if loop_name == innermost:
                loop_schedule['unrolled'] = 1
                unroll_factor = int(comp_schedule["unrolling_factor"])
                loop_schedule['unroll_factor'] = max(loop_schedule['unroll_factor'], unroll_factor)
        
        # Shifting - check if this loop is being shifted
        if comp_schedule.get('shiftings'):
            for shifting in comp_schedule['shiftings']:
                # Check both exact match and prefix match
                if loop_name == shifting[0] or loop_name.startswith(shifting[0]):
                    loop_schedule['shifted'] = 1
                    loop_schedule['shift_factor'] = shifting[1]
                    break
    
    # Check fusion across all computations
    if "fusions" in schedule_json and schedule_json["fusions"]:
        for fusion in schedule_json["fusions"]:
            if len(fusion) >= 3:
                comp1, comp2, level = fusion[0], fusion[1], fusion[2]
                if comp1 in computations_dict and comp2 in computations_dict:
                    # Check if this loop is at the fused level
                    comp1_iters = computations_dict[comp1]["iterators"]
                    comp2_iters = computations_dict[comp2]["iterators"]
                    
                    if level < len(comp1_iters) and comp1_iters[level] == loop_name:
                        loop_schedule['fused'] = 1
                    if level < len(comp2_iters) and comp2_iters[level] == loop_name:
                        loop_schedule['fused'] = 1
    
    # Pack into tensor
    feat[0] = loop_schedule['parallelized']
    feat[1] = loop_schedule['tiled']
    feat[2] = loop_schedule['tile_factor']
    feat[3] = loop_schedule['fused']
    feat[4] = loop_schedule['unrolled']
    feat[5] = loop_schedule['unroll_factor']
    feat[6] = loop_schedule['shifted']
    feat[7] = loop_schedule['shift_factor']
    
    return feat


def build_comp_feature(comp_name, program_annot, schedule_json):
    """
    Build a 16-dimensional feature vector for a computation node.
    Features: [is_reduction, num_transformations, parallelized, tiled, tile_factor, 
               unrolled, unroll_factor, shifted, shift_factor, fused, 
               num_iterators, num_accesses, write_buffer_id, ...(reserved)]
    """
    feat = torch.zeros(16, dtype=torch.float)
    
    comp_dict = program_annot["computations"][comp_name]
    comp_schedule = schedule_json.get(comp_name, {})
    
    # Basic properties
    feat[0] = 1.0 if comp_dict.get("comp_is_reduction", False) else 0.0
    
    # Number of transformations applied
    transformations_list = comp_schedule.get("transformations_list", [])
    feat[1] = len(transformations_list)
    
    # Parallelization
    feat[2] = 1.0 if comp_schedule.get("parallelized_dim") else 0.0
    
    # Tiling
    if comp_schedule.get("tiling"):
        feat[3] = 1.0
        # Average tile factor (if multiple dimensions tiled)
        tile_factors = comp_schedule["tiling"].get("tiling_factors", [])
        if tile_factors:
            feat[4] = np.mean([int(f) for f in tile_factors])
    
    # Unrolling
    if comp_schedule.get("unrolling_factor"):
        feat[5] = 1.0
        feat[6] = int(comp_schedule["unrolling_factor"])
    
    # Shifting
    if comp_schedule.get("shiftings"):
        feat[7] = 1.0
        # Average shift factor
        shift_factors = [s[1] for s in comp_schedule["shiftings"]]
        feat[8] = np.mean(shift_factors) if shift_factors else 0.0
    
    # Fusion
    feat[9] = 0.0
    if "fusions" in schedule_json and schedule_json["fusions"]:
        for fusion in schedule_json["fusions"]:
            if comp_name in fusion:
                feat[9] = 1.0
                break
    
    # Structural properties
    feat[10] = len(comp_dict.get("iterators", []))
    feat[11] = len(comp_dict.get("accesses", []))
    feat[12] = comp_dict.get("write_buffer_id", 0) + 1
    
    # Features 13-15 reserved for future use
    
    return feat


def build_gnn_data_for_schedule(
    function_dict,
    schedule_json,
    device="cpu",
    add_exec_time=False,
    debug=False
):
    """
    Build a PyG Data object for a single schedule.
    """
    program_annot = function_dict["program_annotation"]
    
    loop_node_features = []
    comp_node_features = []
    loop_name_to_id = {}
    comp_name_to_id = {}
    loop_loop_edges = []
    loop_comp_edges = []
    
    # Build Loop Nodes with actual features
    i_loop = 0
    for loop_name in program_annot["iterators"]:
        feat = build_loop_feature(loop_name, program_annot, schedule_json)
        if debug and feat.sum() > 0:
            print(f"Loop '{loop_name}' features: {feat}")
        loop_node_features.append(feat)
        loop_name_to_id[loop_name] = i_loop
        i_loop += 1
    
    # Build Computation Nodes with actual features
    computations_dict = program_annot["computations"]
    i_comp = 0
    for c_name in sorted(computations_dict.keys(),
                         key=lambda x: computations_dict[x]["absolute_order"]):
        feat = build_comp_feature(c_name, program_annot, schedule_json)
        comp_node_features.append(feat)
        comp_name_to_id[c_name] = i_comp
        i_comp += 1
    
    # Build loop->loop edges from the tree structure
    if "tree_structure" in schedule_json:
        if "roots" in schedule_json["tree_structure"]:
            for root in schedule_json["tree_structure"]["roots"]:
                build_loop_loop_edges(root, None, loop_name_to_id, loop_loop_edges)
    
    # Build loop->computation edges from the tree structure
    if "tree_structure" in schedule_json:
        if "roots" in schedule_json["tree_structure"]:
            for root in schedule_json["tree_structure"]["roots"]:
                build_loop_comp_edges(root, loop_name_to_id, comp_name_to_id, loop_comp_edges)
    
    # Merge node features into a single tensor
    loop_features = (torch.stack(loop_node_features, dim=0)
                     if loop_node_features else torch.zeros(0, 8))
    comp_features = (torch.stack(comp_node_features, dim=0)
                     if comp_node_features else torch.zeros(0, 16))
    
    max_dim = max((loop_features.shape[1] if loop_features.shape[0] > 0 else 0),
                  (comp_features.shape[1] if comp_features.shape[0] > 0 else 0))
    
    if loop_features.shape[1] < max_dim and loop_features.shape[0] > 0:
        pad_cols = max_dim - loop_features.shape[1]
        loop_features = torch.nn.functional.pad(loop_features, (0, pad_cols))
    
    if comp_features.shape[1] < max_dim and comp_features.shape[0] > 0:
        pad_cols = max_dim - comp_features.shape[1]
        comp_features = torch.nn.functional.pad(comp_features, (0, pad_cols))
    
    # Concatenate all node features
    x = torch.cat([loop_features, comp_features], dim=0)
    
    # Add initial execution time as an extra feature if requested
    if add_exec_time:
        initial_time_val = float(function_dict["initial_execution_time"])
        exec_time_feat = torch.full((x.shape[0], 1), fill_value=initial_time_val)
        x = torch.cat([x, exec_time_feat], dim=1)
    
    # Build edge index
    edge_index_list = []
    
    # Loop-loop edges (bidirectional)
    for (src, dst) in loop_loop_edges:
        edge_index_list.append((src, dst))
        edge_index_list.append((dst, src))
    
    # Loop-computation edges (bidirectional)
    offset = len(loop_node_features)
    for (l_id, c_id) in loop_comp_edges:
        edge_index_list.append((l_id, offset + c_id))
        edge_index_list.append((offset + c_id, l_id))
    
    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Calculate speedup
    orig_time = float(function_dict["initial_execution_time"])
    sched_times = schedule_json.get("execution_times", [])
    if sched_times:
        transformed_time = min(sched_times)
    else:
        transformed_time = 1e-9
    
    final_speedup = orig_time / transformed_time
    y = torch.tensor([final_speedup], dtype=torch.float)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )
    
    return data


def build_loop_loop_edges(node, parent_loop_name, loop_name_to_id, edge_list):
    """Recursively build parent-child edges between loops."""
    loop_name = node["loop_name"]
    if parent_loop_name is not None:
        edge_list.append((loop_name_to_id[parent_loop_name], loop_name_to_id[loop_name]))
    for child in node["child_list"]:
        build_loop_loop_edges(child, loop_name, loop_name_to_id, edge_list)


def build_loop_comp_edges(node, loop_name_to_id, comp_name_to_id, edge_list):
    """Recursively build edges between loops and their computations."""
    loop_name = node["loop_name"]
    for comp in node["computations_list"]:
        if comp in comp_name_to_id:
            edge_list.append((loop_name_to_id[loop_name], comp_name_to_id[comp]))
    for child in node["child_list"]:
        build_loop_comp_edges(child, loop_name_to_id, comp_name_to_id, edge_list)


#############################################################
# Parallel GNN Data Generation
#############################################################

def get_func_repr_task_gnn(input_q, output_q):
    process_id, programs_dict, pkl_output_folder, device = input_q.get()
    function_name_list = list(programs_dict.keys())
    local_list = []
    
    for function_name in tqdm(function_name_list):
        func_dict = programs_dict[function_name]
        if drop_program(func_dict, function_name):
            continue
        
        program_exec_time = func_dict["initial_execution_time"]
        data_and_attrs_list = []
        
        for i, sched_json in enumerate(func_dict["schedules_list"]):
            if drop_schedule(func_dict, i):
                continue
            
            sched_exec_time = np.min(sched_json["execution_times"])
            if sched_exec_time <= 0:
                continue
            
            speedup_val = program_exec_time / sched_exec_time
            speedup_val = speedup_clip(speedup_val)
            
            data_obj = build_gnn_data_for_schedule(
                func_dict,
                sched_json,
                device=device
            )
            
            datapoint_attrs = get_datapoint_attributes(function_name, func_dict, i, "<gnn_footprint>")
            data_and_attrs_list.append((data_obj, datapoint_attrs))
        
        if len(data_and_attrs_list) > 0:
            local_list.append((function_name, data_and_attrs_list))
    
    pkl_part_filename = os.path.join(pkl_output_folder, f'gnn_representation_part_{process_id}.pkl')
    with open(pkl_part_filename, 'wb') as f:
        pickle.dump(local_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    output_q.put((process_id, pkl_part_filename))


class GNNDatasetParallel:
    def __init__(
        self,
        dataset_filename=None,
        pkl_output_folder="gnn_pickles",
        nb_processes=4,
        device="cpu",
        just_load_pickled=False
    ):
        self.data_list = []
        self.attr_list = []
        
        if just_load_pickled:
            print(f"[INFO] Loading existing pickled files from '{pkl_output_folder}'")
            for pkl_part in Path(pkl_output_folder).iterdir():
                print(f"[INFO] Reading {pkl_part}")
                with open(pkl_part, 'rb') as f:
                    local_list = pickle.load(f)
                for fn, data_attr_pairs in local_list:
                    for (data_obj, attrs) in data_attr_pairs:
                        self.data_list.append(data_obj)
                        self.attr_list.append(attrs)
            print(f"[INFO] Loaded {len(self.data_list)} GNN datapoints from pickle files.")
        else:
            print(f"[INFO] Loading dataset from '{dataset_filename}' to generate GNN pickles.")
            if dataset_filename.endswith(".json"):
                with open(dataset_filename, "r") as f:
                    ds_str = f.read()
                    programs_dict = json.loads(ds_str)
                del ds_str
            else:
                with open(dataset_filename, "rb") as f:
                    programs_dict = pickle.load(f)
            
            if os.path.exists(pkl_output_folder):
                shutil.rmtree(pkl_output_folder)
                print(f"[INFO] Removed existing folder '{pkl_output_folder}'")
            os.makedirs(pkl_output_folder, exist_ok=True)
            print(f"[INFO] Created folder '{pkl_output_folder}'")
            
            manager = multiprocessing.Manager()
            input_q = manager.Queue()
            output_q = manager.Queue()
            
            fnames = list(programs_dict.keys())
            random.shuffle(fnames)
            chunk_size = (len(fnames) // nb_processes) + 1
            print(f"[INFO] Spawning {nb_processes} processes with ~{chunk_size} functions each.")
            
            for i in range(nb_processes):
                subset = {k: programs_dict[k] for k in fnames[i * chunk_size: (i + 1) * chunk_size]}
                input_q.put((i, subset, pkl_output_folder, device))
            
            processes = []
            for i in range(nb_processes):
                p = multiprocessing.Process(
                    target=get_func_repr_task_gnn,
                    args=(input_q, output_q)
                )
                p.start()
                processes.append(p)
                print(f"[INFO] Started process {p.pid} (index {i})")
            
            for i in range(nb_processes):
                pid, part_file = output_q.get()
                print(f"[INFO] Process {pid} completed and saved to '{part_file}'")
                with open(part_file, 'rb') as f:
                    local_list = pickle.load(f)
                print(f"[INFO] Loaded {len(local_list)} functions from {part_file}")
                for fn, data_attr_pairs in local_list:
                    for (data_obj, attrs) in data_attr_pairs:
                        self.data_list.append(data_obj)
                        self.attr_list.append(attrs)
            
            for p in processes:
                p.join()
                print(f"[INFO] Process {p.pid} has joined successfully.")
            
            print(f"[INFO] Total GNN datapoints loaded: {len(self.data_list)}")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx], self.attr_list[idx]


# Utility functions (same as before)
def drop_program(prog_dict, prog_name):
    if len(prog_dict["schedules_list"]) < 2:
        return True
    return False

def drop_schedule(prog_dict, schedule_index):
    schedule_json = prog_dict["schedules_list"][schedule_index]
    if (not schedule_json["execution_times"]) or min(schedule_json["execution_times"]) < 0:
        return True
    return False

def speedup_clip(speedup):
    if speedup < 0.01:
        speedup = 0.01
    return speedup

def get_datapoint_attributes(func_name, program_dict, schedule_index, tree_footprint):
    schedule_json = program_dict["schedules_list"][schedule_index]
    sched_id = str(schedule_index).zfill(4)
    sched_str = get_schedule_str(program_dict["program_annotation"], schedule_json)
    exec_time = np.min(schedule_json["execution_times"])
    memory_use = program_dict["program_annotation"]["memory_size"]
    node_name = program_dict["node_name"] if "node_name" in program_dict else "unknown"
    speedup = program_dict["initial_execution_time"] / exec_time
    
    return (
        func_name,
        sched_id,
        sched_str,
        exec_time,
        memory_use,
        node_name,
        tree_footprint,
        speedup,
    )

def get_schedule_str(program_json, sched_json):
    # Simplified version - you can keep your full implementation
    return "schedule_str"


if __name__ == "__main__":
    dataset_path = "datasets/dataset_expr_dataset_batch550000-838143+batch101-1227605_train_part_5_of_22.pkl"
    pkl_folder = "gnn_pickles"
    
    gnn_dataset = GNNDatasetParallel(
        dataset_filename=dataset_path,
        pkl_output_folder=pkl_folder,
        nb_processes=4,
        device="cpu",
        just_load_pickled=False
    )
    
    print("Number of GNN graphs:", len(gnn_dataset))
    
    loader = DataLoader(gnn_dataset.data_list, batch_size=8, shuffle=True)
    for batch in loader:
        print("Batch shapes:")
        print(f"  x: {batch.x.shape}")
        print(f"  edge_index: {batch.edge_index.shape}")
        print(f"  y: {batch.y.shape}")
        print(f"  Feature sample (first node): {batch.x[0]}")
        break