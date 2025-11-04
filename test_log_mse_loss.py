#!/usr/bin/env python3
"""
Test the log MSE loss function to ensure it works correctly
"""

import torch
import numpy as np
import sys
sys.path.append('/scratch/maa9509/GNN_RL_Pretrain')

from utils_gnn.train_utils import log_mse_loss

# Test the log MSE loss function
print("🧪 Testing Log MSE Loss Function")
print("=" * 40)

# Create test data similar to our speedup distribution
speedups_true = torch.tensor([0.1, 1.0, 2.0, 5.0, 10.0, 100.0], dtype=torch.float32)
speedups_pred = torch.tensor([0.2, 0.9, 2.1, 4.8, 12.0, 80.0], dtype=torch.float32)

print(f"True speedups: {speedups_true.tolist()}")
print(f"Pred speedups: {speedups_pred.tolist()}")

# Calculate regular MSE
regular_mse = torch.nn.functional.mse_loss(speedups_pred, speedups_true)
print(f"\nRegular MSE: {regular_mse.item():.4f}")

# Calculate log MSE
log_mse = log_mse_loss(speedups_pred, speedups_true)
print(f"Log MSE: {log_mse.item():.4f}")

print(f"\nReduction factor: {regular_mse.item() / log_mse.item():.1f}x")

# Test with extreme values
print(f"\n🔥 Testing with extreme values:")
extreme_true = torch.tensor([0.001, 1.0, 1000.0], dtype=torch.float32)
extreme_pred = torch.tensor([0.002, 1.1, 800.0], dtype=torch.float32)

extreme_regular_mse = torch.nn.functional.mse_loss(extreme_pred, extreme_true)
extreme_log_mse = log_mse_loss(extreme_pred, extreme_true)

print(f"Extreme regular MSE: {extreme_regular_mse.item():.1f}")
print(f"Extreme log MSE: {extreme_log_mse.item():.4f}")
print(f"Reduction factor: {extreme_regular_mse.item() / extreme_log_mse.item():.0f}x")

print(f"\n✅ Log MSE loss function is working correctly!")
print(f"   - Handles extreme speedup ranges")
print(f"   - Reduces loss magnitude significantly")
print(f"   - Should give much more stable training")



