# Sweep Configuration Comparison

## Problem with Original Sweep
- **Too few epochs**: 200-400 vs your baseline 500 (models still improving at 500!)
- **Too broad exploration**: Including poor models like SimpleGAT (82.25 MAPE)
- **Aggressive early stopping**: Killing runs at 50 epochs when they need 100+ to converge
- **Wide parameter ranges**: Exploring extremes instead of focusing on what works

## Your Baseline Results (500 epochs, no sweep)
| Model | MAPE |
|-------|------|
| PearlGATv2 | **68.54** ✅ |
| DeepResidualGAT | **69.24** ✅ |
| SimpleGraphSAGE | **72.90** ✅ |
| ResidualGIN | 79.75 |
| SimpleGCN | 80.00 |
| SimpleGAT | 82.25 |

## New Sweep Options

### Option 1: Refined Sweep (`sweep_config_refined.yaml`)
**Purpose**: Broad but smarter exploration
- **Models**: Top 4 performers only
- **Epochs**: 500 (matches baseline)
- **LR range**: 0.0005-0.005 (tighter)
- **Hidden sizes**: 256, 384, 512 (removed 128)
- **Early stop**: min_iter=100 (less aggressive)
- **Estimated time**: ~3-4 days for 40 runs

### Option 2: Focused Sweep (`sweep_config_focused.yaml`)
**Purpose**: Fine-tune the best models
- **Models**: Only PearlGATv2 & DeepResidualGAT
- **Epochs**: 500
- **LR**: 6 specific values around 0.001-0.002
- **Batch size**: 512, 1024 only
- **No early termination**: Let everything finish
- **Estimated time**: ~2 days for 24 runs

## Recommendations

### Quick Win (Recommended)
Use the **Focused Sweep** with 24 trials:
```bash
wandb sweep sweep_config_focused.yaml --project gnn_hyperparameter_sweep
sbatch sweep_hpc_job.sh <new_sweep_id>
```
Edit `sweep_hpc_job.sh` to change `--array=1-24%4`

### Thorough Exploration
Use the **Refined Sweep** with 40 trials:
```bash
wandb sweep sweep_config_refined.yaml --project gnn_hyperparameter_sweep
sbatch sweep_hpc_job.sh <new_sweep_id>
```
Edit `sweep_hpc_job.sh` to change `--array=1-40%4`

## Key Changes
1. ✅ **500 epochs** (not 200-400)
2. ✅ **Focus on top models** (not all 6)
3. ✅ **Tighter LR range** (0.0005-0.005 vs 0.0001-0.01)
4. ✅ **Less aggressive early stop** (min_iter=100 vs 50)
5. ✅ **Better batch sizes** (512-1024 vs 256-1024)

These changes should get you **below 70 MAPE** consistently!



























