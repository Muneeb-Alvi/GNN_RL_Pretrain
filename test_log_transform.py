#!/usr/bin/env python3
"""
Test script to see what happens if we log-transform the speedup values
"""

import numpy as np
import torch

# Simulate the speedup distribution from our analysis
np.random.seed(42)

# Create a sample that matches our 1M dataset distribution
# Mean: 3.73, Median: 1.04, Std: 14.5, Max: 592
speedups = []

# Add some typical values around the median
speedups.extend(np.random.lognormal(mean=0.0, sigma=0.8, size=5000))  # Around 1.0

# Add some higher speedups
speedups.extend(np.random.lognormal(mean=1.5, sigma=1.0, size=2000))  # Around 4-5x

# Add some very high speedups (outliers)
speedups.extend(np.random.uniform(10, 100, size=500))  # 10-100x
speedups.extend(np.random.uniform(100, 600, size=50))   # 100-600x

speedups = np.array(speedups)
speedups = np.clip(speedups, 0.0001, 1000)  # Clip to reasonable range

print("🔍 Speedup Analysis - Raw vs Log Transform")
print("=" * 50)

print(f"Raw Speedups:")
print(f"  Mean: {speedups.mean():.2f}")
print(f"  Median: {np.median(speedups):.2f}")  
print(f"  Std: {speedups.std():.2f}")
print(f"  Min: {speedups.min():.6f}")
print(f"  Max: {speedups.max():.2f}")

# Calculate MSE if we always predict median
median_speedup = np.median(speedups)
mse_raw = np.mean((speedups - median_speedup)**2)
print(f"  MSE if predicting median ({median_speedup:.2f}): {mse_raw:.2f}")

# Now try log transform
log_speedups = np.log(speedups + 1e-8)  # Add small epsilon to avoid log(0)

print(f"\nLog-Transformed Speedups:")
print(f"  Mean: {log_speedups.mean():.2f}")
print(f"  Median: {np.median(log_speedups):.2f}")
print(f"  Std: {log_speedups.std():.2f}")
print(f"  Min: {log_speedups.min():.2f}")
print(f"  Max: {log_speedups.max():.2f}")

# Calculate MSE in log space
median_log = np.median(log_speedups)
mse_log = np.mean((log_speedups - median_log)**2)
print(f"  MSE if predicting median ({median_log:.2f}): {mse_log:.2f}")

print(f"\n💡 Impact of Log Transform:")
print(f"  Raw MSE: {mse_raw:.2f}")
print(f"  Log MSE: {mse_log:.2f}")
print(f"  Reduction: {(mse_raw - mse_log) / mse_raw * 100:.1f}%")

print(f"\n🎯 What this means:")
print(f"  - Log transform reduces variance by ~{(speedups.std() - np.exp(log_speedups).std()) / speedups.std() * 100:.0f}%")
print(f"  - MSE in log space is much more manageable")
print(f"  - Model can focus on relative improvements rather than absolute values")
print(f"  - Need to exp() the predictions to get back to speedup space")




