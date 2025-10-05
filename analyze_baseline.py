#!/usr/bin/env python3
"""
Analyze what the baseline MAPE tells us about speedup distribution
"""

import numpy as np

def analyze_baseline_mape():
    """
    If baseline MAPE (predicting 1.0) is 46%, what does this tell us?
    
    MAPE = 100 * |actual - pred| / actual
    If pred = 1.0 and MAPE = 46%, then:
    46 = 100 * |actual - 1.0| / actual
    0.46 = |actual - 1.0| / actual
    """
    
    print("🧮 Analyzing baseline MAPE = 46%")
    print("   Predicting constant 1.0 speedup gives 46% MAPE")
    print()
    
    # Case 1: actual > 1.0 (speedup)
    print("📈 Case 1: If actual speedups > 1.0:")
    print("   46 = 100 * (actual - 1.0) / actual")
    print("   0.46 * actual = actual - 1.0")
    print("   0.46 * actual - actual = -1.0")
    print("   -0.54 * actual = -1.0")
    print("   actual = 1.0 / 0.54 = 1.85")
    print("   → Average speedup ≈ 1.85x")
    print()
    
    # Case 2: actual < 1.0 (slowdown)  
    print("📉 Case 2: If actual speedups < 1.0:")
    print("   46 = 100 * (1.0 - actual) / actual")
    print("   0.46 * actual = 1.0 - actual")
    print("   0.46 * actual + actual = 1.0")
    print("   1.46 * actual = 1.0")
    print("   actual = 1.0 / 1.46 = 0.68")
    print("   → Average speedup ≈ 0.68x (slowdown)")
    print()
    
    # Mixed case - more realistic
    print("🎯 Mixed case (some speedups, some slowdowns):")
    print("   If distribution is mixed around 1.0, average could be:")
    print("   - Some values around 0.5x - 3.0x range")
    print("   - This would give MSE in reasonable range")
    print()
    
    # What MSE should we expect?
    print("💡 Expected MSE calculations:")
    speedups = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]  # Reasonable range
    predictions = [1.0] * len(speedups)  # Model predicting 1.0
    
    mse_values = [(actual - pred)**2 for actual, pred in zip(speedups, predictions)]
    avg_mse = np.mean(mse_values)
    
    print(f"   Sample speedups: {speedups}")
    print(f"   Predictions: {predictions}")
    print(f"   Individual MSE: {[f'{mse:.2f}' for mse in mse_values]}")
    print(f"   Average MSE: {avg_mse:.2f}")
    print()
    print("🚨 But we're seeing MSE ~95, which suggests:")
    print("   - Either speedups are much higher (10-15x range)")
    print("   - Or there's still an issue with the data")

if __name__ == "__main__":
    analyze_baseline_mape()



