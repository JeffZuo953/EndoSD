import numpy as np
import os
import sys
from glob import glob
import matplotlib.pyplot as plt

depth_dir = sys.argv[1]
files = sorted(glob(os.path.join(depth_dir, "**/*.npy"), recursive=True))
if not files:
    print("未找到任何 .npy 文件")
    sys.exit(0)

print(f"找到 {len(files)} 个深度文件。")

all_means = []
all_maxs = []

# 初始化整体统计
global_min = float('inf')
global_max = float('-inf')
sum_pixels = 0.0
sum_sq_diff = 0.0
total_count = 0

# 定义直方图 bin（这里先用0-50，后面可以根据实际 max 调整）
hist_bins = 100
hist_range = (0, 50)  # 假设深度最大不会超过50，可根据实际情况调整
global_hist = np.zeros(hist_bins, dtype=np.float64)

# 第一步：计算整体统计 & 单图统计 & 累加直方图
for f in files:
    depth = np.load(f)
    all_means.append(depth.mean())
    all_maxs.append(depth.max())

    # 全局 min/max
    global_min = min(global_min, depth.min())
    global_max = max(global_max, depth.max())
    
    # 全局 sum / count
    sum_pixels += depth.sum()
    total_count += depth.size

    # 累加直方图
    h, _ = np.histogram(depth, bins=hist_bins, range=hist_range)
    global_hist += h

global_mean = sum_pixels / total_count

# 第二步：计算整体 std（逐步计算平方差）
for f in files:
    depth = np.load(f)
    sum_sq_diff += ((depth - global_mean) ** 2).sum()

global_std = np.sqrt(sum_sq_diff / total_count)

# ===================== 输出统计 =====================
print(f"\n=== [整体像素级统计] ===")
print(f"像素总数: {total_count:,}")
print(f"min: {global_min:.6f}")
print(f"max: {global_max:.6f}")
print(f"mean: {global_mean:.6f}")
print(f"std: {global_std:.6f}")

print(f"\n=== [单图统计汇总] ===")
print(f"mean(所有图的mean): {np.mean(all_means):.6f}")
print(f"std(所有图的mean): {np.std(all_means):.6f}")
print(f"min(mean): {np.min(all_means):.6f}, max(mean): {np.max(all_means):.6f}")

print(f"mean(所有图的max): {np.mean(all_maxs):.6f}")
print(f"std(所有图的max): {np.std(all_maxs):.6f}")
print(f"min(max): {np.min(all_maxs):.6f}, max(max): {np.max(all_maxs):.6f}")

# ===================== 可视化 =====================
plt.figure(figsize=(15,4))

# 全局像素分布直方图
plt.subplot(1,3,1)
bin_edges = np.linspace(hist_range[0], hist_range[1], hist_bins+1)
plt.bar(bin_edges[:-1], global_hist, width=bin_edges[1]-bin_edges[0], color='steelblue')
plt.title("全局像素深度分布 (近似)")
plt.xlabel("Depth value")
plt.ylabel("Pixel count")

# 单图 mean 分布
plt.subplot(1,3,2)
plt.hist(all_means, bins=50, color='orange')
plt.title("每张图 mean 分布")
plt.xlabel("Mean depth per image")
plt.ylabel("Image count")

# 单图 max 分布
plt.subplot(1,3,3)
plt.hist(all_maxs, bins=50, color='green')
plt.title("每张图 max 分布")
plt.xlabel("Max depth per image")
plt.ylabel("Image count")

plt.tight_layout()
plt.show()

