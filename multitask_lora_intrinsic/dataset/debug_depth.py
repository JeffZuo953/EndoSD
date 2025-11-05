#!/usr/bin/env python3
"""
Debug script to investigate depth values in a single .pt file
"""

import torch
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python debug_depth.py <path_to_pt_file>")
    sys.exit(1)

pt_path = sys.argv[1]

print(f"Loading: {pt_path}")
data = torch.load(pt_path, map_location='cpu')

depth = data['depth']
if isinstance(depth, torch.Tensor):
    depth_np = depth.numpy()
else:
    depth_np = np.array(depth)

print(f"\nDepth tensor shape: {depth_np.shape}")
print(f"Depth dtype: {depth_np.dtype}")

# Flatten
depth_flat = depth_np.flatten()
print(f"\nFlattened size: {len(depth_flat)}")

# All values
print(f"\nAll values statistics:")
print(f"  Min: {np.min(depth_flat):.6f}")
print(f"  Max: {np.max(depth_flat):.6f}")
print(f"  Mean: {np.mean(depth_flat):.6f}")
print(f"  Median: {np.median(depth_flat):.6f}")
print(f"  Std: {np.std(depth_flat):.6f}")

# Valid values (> 0)
valid_mask = depth_flat > 0
valid_depths = depth_flat[valid_mask]
print(f"\nValid (>0) values statistics:")
print(f"  Count: {len(valid_depths)} / {len(depth_flat)} ({len(valid_depths)/len(depth_flat)*100:.2f}%)")
print(f"  Min: {np.min(valid_depths):.6f}")
print(f"  Max: {np.max(valid_depths):.6f}")
print(f"  Mean: {np.mean(valid_depths):.6f}")
print(f"  Median: {np.median(valid_depths):.6f}")
print(f"  Std: {np.std(valid_depths):.6f}")

# Distribution
print(f"\nValue distribution:")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(valid_depths, p)
    print(f"  P{p:2d}: {val:.6f}")

# Check how many values are exactly 0.3
count_max = np.sum(np.abs(valid_depths - 0.3) < 1e-6)
print(f"\nValues at max_depth (0.3): {count_max} ({count_max/len(valid_depths)*100:.2f}%)")

# Show some sample values
print(f"\nSample of first 20 valid values:")
print(valid_depths[:20])

# Check unique values
unique_vals = np.unique(valid_depths)
print(f"\nNumber of unique values: {len(unique_vals)}")
if len(unique_vals) <= 20:
    print("All unique values:")
    print(unique_vals)
else:
    print("First 10 unique values:")
    print(unique_vals[:10])
    print("Last 10 unique values:")
    print(unique_vals[-10:])
