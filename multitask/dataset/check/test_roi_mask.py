import cv2
import numpy as np
import os

def test_roi_mask_loading(roi_mask_path):
    """
    Test ROI mask loading for binary (black/white) images
    """
    print(f"Testing ROI mask: {roi_mask_path}")
    
    if not os.path.exists(roi_mask_path):
        print(f"File not found: {roi_mask_path}")
        return
    
    # Load external ROI mask as grayscale
    external_roi = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if external_roi is None:
        print(f"Failed to load ROI mask: {roi_mask_path}")
        return
    
    print(f"Original ROI shape: {external_roi.shape}")
    print(f"Original ROI unique values: {np.unique(external_roi)}")
    print(f"Original ROI value counts:")
    unique, counts = np.unique(external_roi, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  Value {val}: {count} pixels")
    
    # For binary mask: 255 (white) means valid, 0 (black) means invalid
    # Create binary mask where white areas (255) are valid (1), black areas (0) are invalid (0)
    external_roi_binary = (external_roi == 255).astype(np.uint8)
    
    print(f"\nBinary ROI shape: {external_roi_binary.shape}")
    print(f"Binary ROI unique values: {np.unique(external_roi_binary)}")
    print(f"Binary ROI value counts:")
    unique, counts = np.unique(external_roi_binary, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  Value {val}: {count} pixels")
    
    # Simulate invalid mask detection
    invalid_mask = (external_roi_binary == 0)
    print(f"\nInvalid mask (where ROI mask == 0):")
    print(f"  Invalid pixels count: {np.sum(invalid_mask)}")
    print(f"  Valid pixels count: {np.sum(~invalid_mask)}")

if __name__ == "__main__":
    # Test with the default path from ETISLaribDataset
    roi_mask_path = "/media/ssd2t/jianfu/data/polyp/ETIS-LaribPolypDB/valid_mask.png"
    test_roi_mask_loading(roi_mask_path)