#!/usr/bin/env python3
"""
Create a 250k subset from LOOPer_dataset_val.pkl for validation
"""
import pickle
import random
import os

def create_validation_subset(target_schedules=250000):
    input_file = "/scratch/maa9509/GNN_RL_Pretrain/datasets/LOOPer_dataset_val.pkl"
    output_file = f"/scratch/maa9509/GNN_RL_Pretrain/datasets/LOOPer_dataset_val_{target_schedules//1000}k.pkl"
    
    print(f"Loading full validation dataset from: {input_file}")
    with open(input_file, 'rb') as f:
        full_dataset = pickle.load(f)
    
    print(f"Full dataset contains {len(full_dataset)} programs")
    
    # Calculate how many samples we'll get with target
    total_schedules = 0
    for prog_name, prog_data in full_dataset.items():
        if "schedules_list" in prog_data:
            total_schedules += len(prog_data["schedules_list"])
    
    print(f"Full dataset contains ~{total_schedules} total schedules")
    
    # Calculate what fraction of programs we need to get target schedules
    fraction_needed = target_schedules / total_schedules
    programs_needed = int(len(full_dataset) * fraction_needed)
    
    print(f"Need ~{fraction_needed:.2%} of programs ({programs_needed} programs) to get ~{target_schedules} schedules")
    
    # Randomly sample programs
    program_names = list(full_dataset.keys())
    random.seed(42)  # For reproducibility
    selected_programs = random.sample(program_names, min(programs_needed, len(program_names)))
    
    # Create subset
    subset_dataset = {name: full_dataset[name] for name in selected_programs}
    
    # Count actual schedules in subset
    actual_schedules = 0
    for prog_name, prog_data in subset_dataset.items():
        if "schedules_list" in prog_data:
            actual_schedules += len(prog_data["schedules_list"])
    
    print(f"Created subset with {len(subset_dataset)} programs and ~{actual_schedules} schedules")
    
    # Save subset
    print(f"Saving subset to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(subset_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("✅ Validation subset created successfully!")
    return output_file

if __name__ == "__main__":
    import sys
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 250000
    create_validation_subset(target)







