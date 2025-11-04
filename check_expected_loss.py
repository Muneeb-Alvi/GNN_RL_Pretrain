#!/usr/bin/env python3
"""
Quick check: what should we expect for MSE loss given the speedup distribution?
"""

import numpy as np

# From our analysis of the 1M dataset:
# Expected MSE if predicting 1.0: 217.77
# Mean speedup: 3.73
# Median speedup: 1.04

print("🔍 Expected Loss Analysis")
print("=" * 40)

# If we always predict 1.0 and the true distribution has:
expected_mse_baseline = 217.77
print(f"Expected MSE (baseline, pred=1.0): {expected_mse_baseline:.2f}")

# Our model achieved:
model_loss = 98.23
print(f"Actual model loss: {model_loss:.2f}")

# Improvement:
improvement = (expected_mse_baseline - model_loss) / expected_mse_baseline * 100
print(f"Improvement over baseline: {improvement:.1f}%")

print(f"\n💡 Analysis:")
print(f"   - Baseline (always predict 1.0): MSE = {expected_mse_baseline:.1f}")
print(f"   - Our model: MSE = {model_loss:.1f}")
print(f"   - This is actually a {improvement:.1f}% improvement!")

# What would "good" performance look like?
print(f"\n🎯 What would 'good' performance be?")
for target_mse in [50, 25, 10, 5]:
    improvement_needed = (expected_mse_baseline - target_mse) / expected_mse_baseline * 100
    print(f"   MSE = {target_mse:2d} would be {improvement_needed:.1f}% improvement over baseline")

# Compare to 80k dataset
print(f"\n📊 80k Dataset Comparison:")
expected_mse_80k = 303.25  # From our analysis
print(f"   80k expected MSE (baseline): {expected_mse_80k:.1f}")
print(f"   1M expected MSE (baseline): {expected_mse_baseline:.1f}")
print(f"   1M dataset is actually {(expected_mse_80k - expected_mse_baseline)/expected_mse_80k*100:.1f}% easier!")

print(f"\n🤔 So is MSE=98 reasonable?")
print(f"   Given the high variance in speedup data (std ~14.5),")
print(f"   MSE=98 represents meaningful learning from the baseline of ~218")
print(f"   But you're right - it's still quite high for practical use")




