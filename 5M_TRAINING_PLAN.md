# 5M Dataset Training Plan

## ✅ Completed Steps

### 1. Created 1.25M Validation Subset
- **Job ID**: 12706631
- **Output**: `/scratch/maa9509/GNN_RL_Pretrain/datasets/LOOPer_dataset_val_1250k.pkl` (1017MB)
- **Status**: ✅ Complete

### 2. Updated Config for 5M Dataset
- **Config**: `original_pr_files/conf/config-gnn.yaml`
- **Train dataset**: `dataset_expr_dataset_batch550000-838143+batch101-1227605_train_part_5_of_22.pkl` (4.3GB)
- **Val dataset**: `LOOPer_dataset_val_1250k.pkl` (1017MB)
- **Processes**: 32 (for faster pickle generation)

### 3. Submitted Pickle Generation Job
- **Job ID**: 12706653
- **Resources**: 32 CPUs, 512GB RAM, 72 hours
- **Output folders**: 
  - Train: `gnn_pickles/train/`
  - Val: `gnn_pickles/val/`
- **Status**: 🔄 Running

## 📋 Next Steps (After Pickles Complete)

### Submit Training Jobs with Tuned Hyperparameters

Once job 12706653 completes, submit the 4 training jobs:

```bash
# Check if pickle generation is done
squeue -j 12706653

# When complete, submit all 4 models:
sbatch train_5m_deep_residual_gat.sh    # Best model: 67.505 MAPE
sbatch train_5m_gatv2.sh                # 68.99 MAPE
sbatch train_5m_simple_graph_sage.sh    # 68.487 MAPE
sbatch train_5m_residual_gin.sh         # 73.066 MAPE
```

## 🎯 Model Configurations

### 1. Deep Residual GAT (Best Model)
- **Val Loss**: 67.505 (best on 1M dataset)
- **Hyperparameters**:
  - Hidden Size: 512
  - Learning Rate: 0.001707
  - Batch Size: 512
  - Weight Decay: 0.004531

### 2. PearlGATv2
- **Val Loss**: 68.99
- **Hyperparameters**:
  - Hidden Size: 256
  - Learning Rate: 0.001746
  - Batch Size: 512
  - Weight Decay: 0.00005175

### 3. SimpleGraphSAGE
- **Val Loss**: 68.487
- **Hyperparameters**:
  - Hidden Size: 384
  - Learning Rate: 0.002017
  - Batch Size: 1024
  - Weight Decay: 0.001072

### 4. ResidualGIN
- **Val Loss**: 73.066
- **Hyperparameters**:
  - Hidden Size: 384
  - Learning Rate: 0.0007137
  - Batch Size: 512
  - Weight Decay: 0.0002841

## 📊 Expected Results

Training on 5M dataset (vs 1M) should:
- ✅ Improve generalization
- ✅ Reduce overfitting
- ✅ Lower validation loss
- ✅ Better performance on unseen data

## ⏱️ Timeline

- **Pickle Generation**: ~24-48 hours (job 12706653)
- **Each Model Training**: ~12-24 hours (500 epochs on 5M data)
- **Total Time**: ~3-5 days for all 4 models

## 📁 Output Locations

- **Weights**: `/scratch/maa9509/GNN_RL_Pretrain/weights/`
- **Logs**: `/scratch/maa9509/GNN_RL_Pretrain/hpc_logs/`
- **WandB Project**: `gnn_5m_final`















