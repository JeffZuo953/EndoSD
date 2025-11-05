import torch
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def visualize_cache_sample(cache_path, output_dir=None):
    """
    Visualize a single cache sample.
    
    Args:
        cache_path (str): Path to the .pt cache file
        output_dir (str): Optional output directory to save visualization
    """
    # Load the cache file
    try:
        data = torch.load(cache_path)
        print(f"Loaded cache file: {cache_path}")
        print(f"Keys: {list(data.keys())}")
    except Exception as e:
        print(f"Error loading cache file: {e}")
        return
    
    # Get image
    if 'image' in data:
        image = data['image']
        # Convert from tensor to numpy and rearrange dimensions
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # If image is in CHW format, convert to HWC
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        
        # Denormalize from ImageNet normalization
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        print(f"Image shape: {image.shape}")
    else:
        print("No image found in cache")
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show segmentation mask if available
    if 'semseg_mask' in data:
        mask = data['semseg_mask']
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        # Create visualization with specified colors:
        # 0 (background) -> black
        # 3 (polyp) -> white
        # 255 (invalid) -> gray
        vis_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
        vis_mask[mask == 0] = [0, 0, 0]      # black for background
        vis_mask[mask == 3] = [1, 1, 1]      # white for polyp
        vis_mask[mask == 255] = [0.5, 0.5, 0.5]  # gray for invalid
        
        axes[1].imshow(vis_mask)
        axes[1].set_title('Segmentation Mask (0=black, 3=white, 255=gray)')
        axes[1].axis('off')
        
        # Print mask statistics
        unique_vals = np.unique(mask)
        print(f"Unique mask values: {unique_vals}")
        print(f"Mask shape: {mask.shape}")
        
        # Show histogram of mask values
        axes[2].hist(mask.flatten(), bins=50, color='blue', alpha=0.7)
        axes[2].set_title('Mask Value Distribution')
        axes[2].set_xlabel('Mask Value')
        axes[2].set_ylabel('Frequency')
    else:
        axes[1].text(0.5, 0.5, 'No segmentation mask', ha='center', va='center')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        axes[2].axis('off')
    
    plt.suptitle(f'Cache Visualization: {os.path.basename(cache_path)}')
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(cache_path).stem}_vis.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.close()
    else:
        plt.show()

def visualize_cache_dataset(cache_list_path, num_samples=5, output_dir=None):
    """
    Visualize multiple samples from a cache dataset.
    
    Args:
        cache_list_path (str): Path to the cache file list (.txt)
        num_samples (int): Number of samples to visualize
        output_dir (str): Optional output directory to save visualizations
    """
    # Read cache file list
    try:
        with open(cache_list_path, 'r') as f:
            cache_files = [line.strip() for line in f.readlines()]
        print(f"Found {len(cache_files)} cache files")
    except Exception as e:
        print(f"Error reading cache list: {e}")
        return
    
    # Visualize first num_samples
    for i, cache_path in enumerate(cache_files[:num_samples]):
        print(f"\nVisualizing sample {i+1}/{min(num_samples, len(cache_files))}")
        visualize_cache_sample(cache_path, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Visualize cache files')
    parser.add_argument('--cache_file', type=str, help='Path to a single .pt cache file')
    parser.add_argument('--cache_list', type=str, help='Path to a cache file list (.txt)')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, help='Output directory for saving visualizations')
    
    args = parser.parse_args()
    
    if args.cache_file:
        visualize_cache_sample(args.cache_file, args.output_dir)
    elif args.cache_list:
        visualize_cache_dataset(args.cache_list, args.num_samples, args.output_dir)
    else:
        print("Please specify either --cache_file or --cache_list")

if __name__ == "__main__":
    main()