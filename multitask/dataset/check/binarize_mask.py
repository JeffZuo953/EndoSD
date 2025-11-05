import cv2
import numpy as np
import os
import argparse

def binarize_mask(input_path, output_path=None, smooth_edges=True, smooth_strength='medium'):
    """
    Binarize a PNG mask for polyp segmentation:
    - For RGBA images: transparent areas -> black (0), opaque areas keep their value
    - Then: 0-127 -> black (0), 128-255 -> white (255)
    - Advanced edge smoothing to remove jagged edges
    
    Args:
        input_path (str): Path to input PNG mask
        output_path (str): Path to save binarized mask (optional)
        smooth_edges (bool): Whether to smooth edges to remove jagged appearance
        smooth_strength (str): Smoothing strength - 'light', 'medium', 'strong'
    """
    print(f"Loading mask: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
    
    # Load mask with alpha channel support
    mask = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if mask is None:
        print(f"Failed to load mask: {input_path}")
        return
    
    print(f"Original mask shape: {mask.shape}")
    
    # Handle different image types
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        print("Detected 4-channel RGBA image")
        # Handle RGBA image
        b, g, r, alpha = cv2.split(mask)
        
        # Create binary mask from alpha channel
        # Alpha 0 = transparent (should be black/invalid), Alpha 255 = opaque (keep original value)
        transparent_mask = (alpha == 0)
        
        # Convert to grayscale first
        gray = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_BGR2GRAY)
        
        # Set transparent areas to black (0 - will become invalid)
        gray[transparent_mask] = 0
        
        print(f"Grayscale with transparency handled, min/max: {gray.min()}/{gray.max()}")
        mask_to_process = gray
        
    elif len(mask.shape) == 3 and mask.shape[2] == 3:
        print("Detected 3-channel RGB image")
        # Convert RGB to grayscale
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        print(f"Converted to grayscale, min/max: {gray.min()}/{gray.max()}")
        mask_to_process = gray
        
    else:
        print("Detected grayscale image")
        print(f"Grayscale min/max: {mask.min()}/{mask.max()}")
        mask_to_process = mask
    
    # Show original value distribution
    unique, counts = np.unique(mask_to_process, return_counts=True)
    print(f"Values to process - unique count: {len(unique)}")
    
    # Advanced edge smoothing before binarization to reduce jagged edges
    if smooth_edges:
        print(f"\nApplying advanced edge smoothing (strength: {smooth_strength})...")
        
        # Step 1: Apply Gaussian blur to the grayscale mask before binarization
        # This helps smooth the transition areas and reduces hard edges
        if smooth_strength == 'light':
            blur_kernel = 3
            morph_kernel_size = 3
            iterations = 1
        elif smooth_strength == 'medium':
            blur_kernel = 5
            morph_kernel_size = 5
            iterations = 2
        else:  # strong
            blur_kernel = 7
            morph_kernel_size = 7
            iterations = 3
            
        print(f"  - Applying Gaussian blur (kernel: {blur_kernel}x{blur_kernel})")
        blurred_mask = cv2.GaussianBlur(mask_to_process, (blur_kernel, blur_kernel), 0)
        
        # Step 2: Use adaptive threshold for better edge preservation
        print("  - Applying adaptive binarization")
        # Use Otsu's method for automatic threshold selection
        _, binarized_mask = cv2.threshold(blurred_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Step 3: Apply morphological operations with larger kernels
        print(f"  - Applying morphological operations (kernel: {morph_kernel_size}x{morph_kernel_size})")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        
        # Multiple iterations of opening and closing for better smoothing
        for i in range(iterations):
            # Opening: removes small noise and smooths object boundaries
            binarized_mask = cv2.morphologyEx(binarized_mask, cv2.MORPH_OPEN, kernel)
            # Closing: fills small gaps and smooths object boundaries
            binarized_mask = cv2.morphologyEx(binarized_mask, cv2.MORPH_CLOSE, kernel)
            print(f"    Iteration {i+1}/{iterations} completed")
        
        # Step 4: Apply bilateral filter to further smooth while preserving edges
        print("  - Applying bilateral filter for final smoothing")
        # Convert to float for bilateral filtering
        binarized_float = binarized_mask.astype(np.float32)
        bilateral_filtered = cv2.bilateralFilter(binarized_float, 9, 75, 75)
        
        # Convert back to binary
        binarized_mask = np.where(bilateral_filtered > 127, 255, 0).astype(np.uint8)
        
        print("Advanced edge smoothing completed")
    else:
        # Simple binarization without smoothing
        binarized_mask = np.where(mask_to_process <= 127, 0, 255).astype(np.uint8)
    
    print(f"\nBinarized mask min/max values: {binarized_mask.min()}/{binarized_mask.max()}")
    
    # Show binarized value distribution
    unique, counts = np.unique(binarized_mask, return_counts=True)
    print(f"Binarized unique values: {unique}")
    for val, count in zip(unique, counts):
        print(f"  Value {val}: {count} pixels")
    
    # Save binarized mask if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, binarized_mask)
        if success:
            print(f"\nBinarized mask saved to: {output_path}")
        else:
            print(f"\nFailed to save binarized mask to: {output_path}")
    else:
        # Save with "_binarized" suffix
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_binarized{ext}"
        success = cv2.imwrite(output_path, binarized_mask)
        if success:
            print(f"\nBinarized mask saved to: {output_path}")
        else:
            print(f"\nFailed to save binarized mask to: {output_path}")
    
    return binarized_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binarize PNG mask with advanced edge smoothing: transparent -> black (0), 0-127 -> 0, 128-255 -> 255")
    parser.add_argument("input", help="Input PNG mask path")
    parser.add_argument("-o", "--output", help="Output path for binarized mask")
    parser.add_argument("--no-smooth", action="store_true", help="Disable edge smoothing")
    parser.add_argument("--smooth-strength", choices=['light', 'medium', 'strong'],
                       default='medium', help="Edge smoothing strength (default: medium)")
    
    args = parser.parse_args()
    
    smooth_edges = not args.no_smooth
    binarize_mask(args.input, args.output, smooth_edges, args.smooth_strength)