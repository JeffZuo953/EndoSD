#!/bin/bash

# Example usage of the cache visualization script

# Visualize a single cache file
# python multitask/dataset/visualize_cache.py --cache_file /path/to/your/cache/file.pt

# Visualize multiple samples from a cache list
# python multitask/dataset/visualize_cache.py --cache_list /path/to/your/cache_list.txt --num_samples 10

# Save visualizations to output directory
# python visualize_cache.py --cache_list /media/ssd2t/jianfu/data/polyp/CVC-EndoScene/TestDataset/cache/train_cache.txt --num_samples 10 --output_dir ./visualizations
python visualize_cache.py --cache_list /media/ssd2t/jianfu/data/polyp/CVC-EndoScene/ValidationDataset/cache/train_cache.txt --num_samples 10 --output_dir ./visualizations

# For depth cache files (if needed)
# python multitask/dataset/visualize_cache.py --cache_file /path/to/depth/cache/file.pt --output_dir ./depth_visualizations