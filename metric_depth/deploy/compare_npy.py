import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def compare_npy_files(file1, file2, output_dir='.'):
    """
    Compares two .npy files containing depth maps, calculates metrics, 
    and generates a visual comparison.
    """
    print("=" * 60)
    print("Comparing NumPy arrays")
    print("=" * 60)
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")

    # Load the numpy arrays
    try:
        arr1 = np.load(file1)
        print(f"Loaded File 1 - Shape: {arr1.shape}, Dtype: {arr1.dtype}")
    except FileNotFoundError:
        print(f"Error: File not found at {file1}")
        return

    try:
        arr2 = np.load(file2)
        print(f"Loaded File 2 - Shape: {arr2.shape}, Dtype: {arr2.dtype}")
    except FileNotFoundError:
        print(f"Error: File not found at {file2}")
        return

    # Ensure shapes are identical
    if arr1.shape != arr2.shape:
        print(f"Error: Array shapes do not match! {arr1.shape} vs {arr2.shape}")
        # Optional: try to resize for comparison
        # h, w = min(arr1.shape[0], arr2.shape[0]), min(arr1.shape[1], arr2.shape[1])
        # arr1 = arr1[:h, :w]
        # arr2 = arr2[:h, :w]
        # print(f"Resized arrays to common shape for comparison: {arr1.shape}")
        return

    # --- Calculate Difference Metrics ---
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    
    mae = np.mean(abs_diff)
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    max_abs_err = np.max(abs_diff)
    min_abs_err = np.min(abs_diff)
    std_diff = np.std(diff)

    print("\n--- Difference Metrics ---")
    print(f"Mean Absolute Error (MAE):   {mae:.6f}")
    print(f"Mean Squared Error (MSE):    {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Max Absolute Error:          {max_abs_err:.6f}")
    print(f"Min Absolute Error:          {min_abs_err:.6f}")
    print(f"Standard Deviation of Diff:  {std_diff:.6f}")
    print("--------------------------\n")

    # --- Generate Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Comparison between\n{os.path.basename(file1)} and {os.path.basename(file2)}', fontsize=16)

    # Plot Array 1
    im1 = axes[0, 0].imshow(arr1, cmap='viridis')
    axes[0, 0].set_title(f'Array 1: {os.path.basename(file1)}')
    fig.colorbar(im1, ax=axes[0, 0])

    # Plot Array 2
    im2 = axes[0, 1].imshow(arr2, cmap='viridis')
    axes[0, 1].set_title(f'Array 2: {os.path.basename(file2)}')
    fig.colorbar(im2, ax=axes[0, 1])

    # Plot Absolute Difference
    im3 = axes[1, 0].imshow(abs_diff, cmap='inferno')
    axes[1, 0].set_title(f'Absolute Difference (MAE: {mae:.4f})')
    fig.colorbar(im3, ax=axes[1, 0])
    
    # Plot Histogram of Differences
    axes[1, 1].hist(diff.ravel(), bins=100, color='gray', log=True)
    axes[1, 1].set_title('Log-Histogram of Differences')
    axes[1, 1].set_xlabel('Difference (arr1 - arr2)')
    axes[1, 1].set_ylabel('Frequency (log scale)')
    axes[1, 1].grid(True, alpha=0.5)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    output_filename = os.path.join(output_dir, 'comparison_result.png')
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_filename}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two .npy depth files.')
    parser.add_argument('file1', type=str, help='Path to the first .npy file (e.g., from PyTorch).')
    parser.add_argument('file2', type=str, help='Path to the second .npy file (e.g., from ONNX).')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save the output plot.')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    compare_npy_files(args.file1, args.file2, args.output_dir)