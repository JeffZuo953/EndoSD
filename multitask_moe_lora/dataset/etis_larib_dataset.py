import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
import os

from .transform import Resize, NormalizeImage, PrepareForNet


class ETISLaribDataset(Dataset):
    """
    ETIS-Larib dataset for polyp segmentation.
    Images: 196 images
    Use circular ROI mask (difficult to crop directly, so keep mask approach)
    """

    def __init__(self, filelist_path, rootpath, mode="train", size=(518, 518), roi_mask_path="/media/ssd2t/jianfu/data/polyp/ETIS-LaribPolypDB/valid_mask.png"):
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

    def _create_circular_mask(self, image):
        """
        Create default ROI mask (all pixels valid).
        """
        height, width = image.shape[:2]
        # Default: all pixels are valid
        mask = np.ones((height, width), dtype=np.uint8)
        return mask

    def __getitem__(self, item):
        line = self.filelist[item]
        image_path = os.path.join(self.rootpath, "images", line + ".png")
        gt_path = os.path.join(self.rootpath, "masks", line + ".png")

        raw_image = cv2.imread(image_path)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        # Start with default ROI mask (all valid)
        roi_mask = self._create_circular_mask(raw_image)

        # Load external ROI mask if provided
        if self.roi_mask_path is not None:
            # roi_mask_path should be the direct path to the PNG file, not a directory
            if os.path.exists(self.roi_mask_path):
                # Load external ROI mask (PNG with transparency)
                external_roi = cv2.imread(self.roi_mask_path, cv2.IMREAD_GRAYSCALE)
                if external_roi is not None:
                    # ROI mask: white areas (255) are valid, transparent areas (0) are invalid
                    external_roi_binary = (external_roi > 127).astype(np.uint8)
                    # Use external ROI mask directly
                    roi_mask = external_roi_binary
                else:
                    print(f"Failed to load ROI mask: {self.roi_mask_path}")
            else:
                print(f"ROI mask file not found: {self.roi_mask_path}")

        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # Apply circular ROI mask to image only
        image = image * roi_mask[:, :, np.newaxis]

        # Process mask labels first, then apply ROI mask:
        # - Polyp regions (typically white, value > 0) -> class 3
        # - Background (typically black, value = 0) -> class 0
        # - Invalid regions (ROI mask = 0) -> class 255 (will be ignored by CrossEntropyLoss)
        polyp_mask = (mask > 0)
        background_mask = (mask == 0)
        # Apply ROI mask to determine invalid regions
        invalid_mask = (roi_mask == 0)
        mask = np.where(invalid_mask, 255, np.where(polyp_mask, 3, np.where(background_mask, 0, 255))).astype(np.uint8)

        sample = self.image_transform({"image": image, "semseg_mask": mask})

        sample = {"image": torch.from_numpy(sample["image"]), "semseg_mask": torch.from_numpy(sample["semseg_mask"]).long(), "image_path": image_path}

        return sample

    def __len__(self):
        return len(self.filelist)
