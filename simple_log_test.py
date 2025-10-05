#!/usr/bin/env python3
"""
Simple test of log MSE loss function
"""

import torch

def log_mse_loss(predictions, targets, eps=1e-8):
    """
    MSE loss in log space to handle wide-range speedup values.
    """
    # Ensure positive values before taking log
    safe_predictions = torch.clamp(predictions, min=eps)
    safe_targets = torch.clamp(targets, min=eps)
    
    # Apply log transformation
    log_predictions = torch.log(safe_predictions)
    log_targets = torch.log(safe_targets)
    
    # MSE in log space
    return torch.nn.functional.mse_loss(log_predictions, log_targets)

# Test the log MSE loss function
print("🧪 Testing Log MSE Loss Function")
print("=" * 40)

# Test with extreme values (like our speedup data)
extreme_true = torch.tensor([0.001, 1.0, 1000.0], dtype=torch.float32)
extreme_pred = torch.tensor([0.002, 1.1, 800.0], dtype=torch.float32)

extreme_regular_mse = torch.nn.functional.mse_loss(extreme_pred, extreme_true)
extreme_log_mse = log_mse_loss(extreme_pred, extreme_true)

print(f"True speedups: {extreme_true.tolist()}")
print(f"Pred speedups: {extreme_pred.tolist()}")
print(f"Regular MSE: {extreme_regular_mse.item():.1f}")
print(f"Log MSE: {extreme_log_mse.item():.4f}")
print(f"Reduction factor: {extreme_regular_mse.item() / extreme_log_mse.item():.0f}x")

# Test with our expected range
expected_true = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0], dtype=torch.float32)
expected_pred = torch.tensor([0.12, 0.48, 1.1, 1.9, 5.2, 9.8, 48.0], dtype=torch.float32)

expected_regular_mse = torch.nn.functional.mse_loss(expected_pred, expected_true)
expected_log_mse = log_mse_loss(expected_pred, expected_true)

print(f"\n📊 Expected range test:")
print(f"Regular MSE: {expected_regular_mse.item():.2f}")
print(f"Log MSE: {expected_log_mse.item():.4f}")
print(f"Reduction factor: {expected_regular_mse.item() / expected_log_mse.item():.1f}x")

print(f"\n✅ Log MSE loss should give us loss values in the range 0.01-10 instead of 50-300!")
