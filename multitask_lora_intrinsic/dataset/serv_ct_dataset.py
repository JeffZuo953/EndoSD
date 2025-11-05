import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import Resize, NormalizeImage, PrepareForNet


class ServCTDataset(Dataset):
    """
    SERV-CT dataset for stereo depth estimation.
    
    Dataset structure:
    - Experiment_1: 8 samples (001-008) with 30° endoscope
    - Experiment_2: 8 samples (009-016) with straight endoscope
    
    Each sample contains:
    - Left rectified image (720x576 RGB PNG)
    - Right rectified image (720x576 RGB PNG)
    - Left depth map (720x576, mm depth scaled by 256, 16-bit PNG)
    - Right depth map (720x576, mm depth scaled by 256, 16-bit PNG)
    - Left-to-right disparity (720x576, pixel disparity scaled by 256, 16-bit PNG)
    - Calibration parameters (JSON)
    """

    def __init__(self, rootpath, mode="val", size=(518, 518), experiment=None):
        """
        Args:
            rootpath: Path to SERV-CT dataset root
            mode: "val" (only validation set available)
            size: Target image size (width, height)
            experiment: "Experiment_1", "Experiment_2", or None (both)
        """
        self.rootpath = rootpath
        self.mode = mode
        self.size = size
        
        # Build file list
        self.samples = []
        
        experiments = [experiment] if experiment else ["Experiment_1", "Experiment_2"]
        
        for exp in experiments:
            exp_path = os.path.join(rootpath, exp)
            if not os.path.exists(exp_path):
                continue
                
            # Get all image indices
            left_rect_path = os.path.join(exp_path, "Left_rectified")
            if os.path.exists(left_rect_path):
                files = sorted([f.split('.')[0] for f in os.listdir(left_rect_path) if f.endswith('.png')])
                for idx in files:
                    self.samples.append({
                        'experiment': exp,
                        'index': idx
                    })
        
        # Setup transforms
        net_w, net_h = size
        self.image_transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=False,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample_info = self.samples[idx]
        experiment = sample_info['experiment']
        index = sample_info['index']
        
        # Build paths
        exp_path = os.path.join(self.rootpath, experiment)
        
        # Load images
        left_img_path = os.path.join(exp_path, "Left_rectified", f"{index}.png")
        right_img_path = os.path.join(exp_path, "Right_rectified", f"{index}.png")
        
        left_image = cv2.imread(left_img_path)
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB) / 255.0
        
        right_image = cv2.imread(right_img_path)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB) / 255.0
        
        # Load depth maps (convert from 16-bit scaled format)
        left_depth_path = os.path.join(exp_path, "Reference_CT", "DepthL", f"{index}.png")
        right_depth_path = os.path.join(exp_path, "Reference_CT", "DepthR", f"{index}.png")
        
        left_depth = cv2.imread(left_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        right_depth = cv2.imread(right_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        
        # Load disparity
        disparity_path = os.path.join(exp_path, "Reference_CT", "Disparity", f"{index}.png")
        disparity = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        
        # Load calibration
        calib_path = os.path.join(exp_path, "Rectified_calibration", f"{index}.json")
        with open(calib_path, 'r') as f:
            calibration = json.load(f)
        
        # Apply transforms
        sample = self.image_transform({
            'image': left_image,
            'image_right': right_image,
            'depth': left_depth,
            'depth_right': right_depth,
            'disparity': disparity
        })
        
        # Convert to tensors
        result = {
            'image': torch.from_numpy(sample['image']),
            'image_right': torch.from_numpy(sample['image_right']),
            'depth': torch.from_numpy(sample['depth']).unsqueeze(0),
            'depth_right': torch.from_numpy(sample['depth_right']).unsqueeze(0),
            'disparity': torch.from_numpy(sample['disparity']).unsqueeze(0),
            'calibration': calibration,
            'experiment': experiment,
            'index': index
        }
        
        return result

    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)

    def get_experiment_info(self):
        """Get information about experiments in the dataset."""
        exp_counts = {}
        for sample in self.samples:
            exp = sample['experiment']
            exp_counts[exp] = exp_counts.get(exp, 0) + 1
        return exp_counts


# Example usage
if __name__ == "__main__":
    dataset = ServCTDataset(rootpath="data/SERV-CT/SERV-CT")
    print(f"Total samples: {len(dataset)}")
    print(f"Experiment info: {dataset.get_experiment_info()}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Depth shape: {sample['depth'].shape}")
    print(f"Disparity shape: {sample['disparity'].shape}")