import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
import os

from .transform import Resize, NormalizeImage, PrepareForNet


class BKAIPolypDataset(Dataset):
    """
    BKAI-IGH NeoPolyp DB dataset for polyp segmentation.
    Images: 1000 images
    ROI MASK: Crop to (0,0,1280,959), add ROI mask (black areas and text areas set to 0, text is white and gray), then resize to 518x518 without keeping aspect ratio.
    """

    def __init__(self,
                 filelist_path="/media/ssd2t/jianfu/data/polyp/bkai-igh-neopolyp/train.txt",
                 rootpath="/media/ssd2t/jianfu/data/polyp/bkai-igh-neopolyp",
                 mode="train",
                 size=(518, 518),
                 roi_mask_path="/media/ssd2t/jianfu/data/polyp/bkai-igh-neopolyp/valid_mask.png"):
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

    def __getitem__(self, item):
        line = self.filelist[item]
        image_path = os.path.join(self.rootpath, "train", line + ".jpeg")
        gt_path = os.path.join(self.rootpath, "train_gt", line + ".jpeg")

        raw_image = cv2.imread(image_path)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        # Read mask - try color first, fallback to grayscale
        mask_raw = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        if mask_raw is not None:
            # If color image, convert to RGB for easier color detection
            mask_raw = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2RGB)
        else:
            # If color read failed, try grayscale
            mask_raw = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # Crop both image and mask to (0,0,1280,959) first
        crop_h, crop_w = 959, 1280
        if image.shape[0] > crop_h:
            image = image[:crop_h, :, :]
        if image.shape[1] > crop_w:
            image = image[:, :crop_w, :]
        if len(mask_raw.shape) == 3:  # Color image
            if mask_raw.shape[0] > crop_h:
                mask_raw = mask_raw[:crop_h, :, :]
            if mask_raw.shape[1] > crop_w:
                mask_raw = mask_raw[:, :crop_w, :]
        else:  # Grayscale image
            if mask_raw.shape[0] > crop_h:
                mask_raw = mask_raw[:crop_h]
            if mask_raw.shape[1] > crop_w:
                mask_raw = mask_raw[:, :crop_w]

        # Create default ROI mask with same dimensions (all valid)
        roi_mask = np.ones((crop_h, crop_w), dtype=np.uint8)

        # Load external ROI mask if provided
        if self.roi_mask_path is not None:
            # Check if roi_mask_path is a single file or directory
            if os.path.isfile(self.roi_mask_path):
                # Single ROI mask file for all images
                external_roi = cv2.imread(self.roi_mask_path, cv2.IMREAD_GRAYSCALE)
                if external_roi is not None:
                    # Crop external ROI mask to same dimensions
                    if external_roi.shape[0] > crop_h:
                        external_roi = external_roi[:crop_h, :]
                    if external_roi.shape[1] > crop_w:
                        external_roi = external_roi[:, :crop_w]
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
                        # Crop external ROI mask to same dimensions
                        if external_roi.shape[0] > crop_h:
                            external_roi = external_roi[:crop_h, :]
                        if external_roi.shape[1] > crop_w:
                            external_roi = external_roi[:, :crop_w]

                        # Create binary mask where white areas (>127) are valid
                        external_roi_binary = (external_roi > 127).astype(np.uint8)
                        # Use external ROI mask directly (no combination)
                        roi_mask = external_roi_binary

        # Process mask labels based on image type
        if len(mask_raw.shape) == 3:  # Color image with red/green annotations
            # Extract red and green channels for processing
            red_channel = mask_raw[:, :, 0].astype(np.float32)
            green_channel = mask_raw[:, :, 1].astype(np.float32)
            blue_channel = mask_raw[:, :, 2].astype(np.float32)
            
            # Process red channel: values 128-255 are considered red polyp regions
            red_polyp_mask = (red_channel >= 128)
            
            # Process green channel: values 128-255 are considered green polyp regions
            green_polyp_mask = (green_channel >= 128)
            
            # Combine red and green polyp regions (unified as class 3)
            polyp_mask = red_polyp_mask | green_polyp_mask
            
            # Background: areas where all channels are low (< 128)
            background_mask = (red_channel < 128) & (green_channel < 128) & (blue_channel < 128)
            
        else:  # Grayscale image
            # Binarize grayscale GT: 128-255 -> polyp, 0-127 -> background
            polyp_mask = (mask_raw >= 128)
            background_mask = (mask_raw < 128)

        # Apply ROI mask to determine invalid regions
        invalid_mask = (roi_mask == 0)
        
        # Create final mask
        mask = np.where(invalid_mask, 255, np.where(polyp_mask, 3, 0)).astype(np.uint8)

        # Apply ROI mask to image
        image = image * roi_mask[:, :, np.newaxis]

        # Apply image transform (this will resize both image and mask to target size)
        sample = self.image_transform({"image": image, "semseg_mask": mask})

        sample = {"image": torch.from_numpy(sample["image"]), "semseg_mask": torch.from_numpy(sample["semseg_mask"]).long(), "image_path": image_path}

        return sample

    def __len__(self):
        return len(self.filelist)
