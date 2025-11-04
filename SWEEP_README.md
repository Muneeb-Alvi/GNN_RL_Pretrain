# WandB Hyperparameter Sweep Setup

This directory contains everything needed to run hyperparameter sweeps for GNN models using WandB.

## Files Created

- `sweep_config.yaml` - Main sweep configuration with hyperparameter ranges
- `launch_sweep.py` - Python script to create and manage sweeps
- `sweep_job.sh` - HPC job script for single-agent sweeps
- `parallel_sweep_job.sh` - HPC job script for parallel multi-agent sweeps
- `test_sweep.sh` - Quick test script to verify setup

## Quick Start

### 1. Create a Sweep
```bash
python launch_sweep.py create
```
This will output a sweep ID like `abc123def456`

### 2. Run Agents on HPC

**Single Agent (20 runs):**
```bash
sbatch sweep_job.sh <sweep_id> 20
```

**Parallel Agents (4 agents, 15 runs each = 60 total):**
```bash
sbatch parallel_sweep_job.sh <sweep_id> 15
```

### 3. Monitor Results
- Visit your WandB dashboard: https://wandb.ai/your-username/gnn_hyperparameter_sweep
- View sweep results, compare runs, and analyze best hyperparameters

## Hyperparameters Being Tuned

- **Model Architecture**: SimpleGraphSAGE, ResidualGIN, SimpleGAT, SimpleGCN
- **Hidden Size**: 128, 256, 384, 512
- **Learning Rate**: 0.0001 to 0.01 (log-uniform)
- **Max Epochs**: 200, 300, 400 (reduced from 500 for faster sweeps)
- **Batch Size**: 256, 512, 1024
- **Weight Decay**: 0.0001 to 0.01 (log-uniform)

## Timing Estimates

Based on your current training time (~50s per epoch):

- **Single run (300 epochs)**: ~4 hours
- **20 runs**: ~3 days (single agent)
- **60 runs**: ~4 days (4 parallel agents)

## Usage Examples

### Test the Setup
```bash
./test_sweep.sh
```

### Create and Run Immediately
```bash
python launch_sweep.py create --run-agent 10
```

### Run More Agents on Existing Sweep
```bash
python launch_sweep.py agent <sweep_id> 25
```

### Manual Agent Commands
```bash
# Run indefinitely
wandb agent <sweep_id>

# Run specific number of experiments
wandb agent <sweep_id> --count 20

# Run multiple agents in parallel
wandb agent <sweep_id> --count 10 &
wandb agent <sweep_id> --count 10 &
wait
```

## Tips

1. **Start Small**: Run 10-15 experiments first to identify promising regions
2. **Focus**: Once you find good configs, run longer experiments (500 epochs) on the best ones
3. **Monitor**: Check WandB dashboard regularly to stop early if needed
4. **Resources**: Use parallel agents to speed up exploration

## Troubleshooting

- **WandB Login**: Run `wandb login` if you get authentication errors
- **GPU Memory**: Reduce batch size if you get OOM errors
- **Time Limits**: Adjust SLURM time limits based on your needs
- **Sweep ID**: Always save the sweep ID when creating sweeps - you'll need it to run agents































