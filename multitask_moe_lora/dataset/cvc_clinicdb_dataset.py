import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
import os

from .transform import Resize, NormalizeImage, PrepareForNet


class CVCClinicDBDataset(Dataset):
    """
    CVC-ClinicDB dataset for polyp segmentation.
    Images: 612 images
    Crop non-lens areas directly
    """

    def __init__(self, filelist_path, rootpath, mode="train", size=(518, 518), roi_mask_path="/media/ssd2t/jianfu/data/polyp/CVC-ClinicDB/valid_mask.png"):
        self.rootpath = rootpath
        self.mode = mode
        self.size = size
        self.roi_mask_path = roi_mask_path

        with open(filelist_path, "r") as f:
            self.filelist = f.read().splitlines()

        net_w, net_h = size
        self.image_transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=1,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
                downscale_only=True,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def _create_roi_mask(self, raw_image):
        """
        Create default ROI mask (all pixels valid).
        """
        height, width = raw_image.shape[:2]
        # Default: all pixels are valid
        roi_mask = np.ones((height, width), dtype=np.uint8)
        return roi_mask

    def __getitem__(self, item):
        line = self.filelist[item]
        image_path = os.path.join(self.rootpath, "PNG", "Original", line + ".png")
        gt_path = os.path.join(self.rootpath, "PNG", "Ground Truth", line + ".png")

        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
            
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        # Start with default ROI mask (all valid)
        roi_mask = self._create_roi_mask(raw_image)

        # Load external ROI mask if provided
        if self.roi_mask_path is not None:
            # Check if roi_mask_path is a single file or directory
            if os.path.isfile(self.roi_mask_path):
                # Single ROI mask file for all images
                external_roi = cv2.imread(self.roi_mask_path, cv2.IMREAD_GRAYSCALE)
                if external_roi is not None:
                    # Resize to match image size
                    external_roi = cv2.resize(external_roi, (raw_image.shape[1], raw_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # Create binary mask where white areas (>127) are valid
                    external_roi_binary = (external_roi > 127).astype(np.uint8)
                    roi_mask = external_roi_binary
            else:
                # Directory with individual ROI mask files
                external_roi_path = os.path.join(self.roi_mask_path, line + ".png")
                if os.path.exists(external_roi_path):
                    # Load external ROI mask
                    external_roi = cv2.imread(external_roi_path, cv2.IMREAD_GRAYSCALE)
                    if external_roi is not None:
                        # Create binary mask where white areas (>127) are valid
                        external_roi_binary = (external_roi > 127).astype(np.uint8)
                        # Use external ROI mask directly (no combination)
                        roi_mask = external_roi_binary

        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {gt_path}")

        # Apply ROI mask to image (set invalid regions to black)
        image = image * roi_mask[:, :, np.newaxis]

        # Binarize the GT mask properly:
        # - Values 1-127 -> background (class 0)
        # - Values 128-255 -> foreground/polyp (class 3)
        # - Value 0 -> background (class 0)
        # - Invalid regions (ROI mask = 0) -> class 255 (will be ignored by CrossEntropyLoss)
        
        # Create binary mask from GT
        polyp_mask = (mask >= 128)  # High values are polyp
        background_mask = (mask < 128)  # Low values (including 0 and 1-127) are background
        
        # Apply ROI mask to determine invalid regions
        invalid_mask = (roi_mask == 0)
        
        # Create final mask
        mask = np.where(invalid_mask, 255, np.where(polyp_mask, 3, 0)).astype(np.uint8)

        sample = self.image_transform({"image": image, "semseg_mask": mask})

        sample = {"image": torch.from_numpy(sample["image"]), "semseg_mask": torch.from_numpy(sample["semseg_mask"]).long(), "image_path": image_path}

        return sample

    def __len__(self):
        return len(self.filelist)
