import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
import os

from .transform import Resize, NormalizeImage, PrepareForNet


class KvasirSegDataset(Dataset):
    """
    Kvasir-Seg dataset for polyp segmentation.
    Images: 1000 images
    Crop four corners
    """

    def __init__(self, filelist_path, rootpath, mode="train", size=(518, 518)):
        self.rootpath = rootpath
        self.mode = mode
        self.size = size

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
        Create ROI mask for Kvasir-Seg by detecting black pixels with smooth boundaries.
        Mark pixels where all three channels are very low as invalid, then smooth the boundaries.
        """
        # Detect very dark pixels (not just pure black) to capture more boundary pixels
        if len(raw_image.shape) == 3:
            # For BGR image, check if all channels are very low (< 10)
            black_mask = np.all(raw_image < 10, axis=2)
        else:
            # For grayscale image, check if pixel value is very low
            black_mask = (raw_image < 10)
        
        # Apply morphological operations to smooth the boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Closing operation to fill small gaps and smooth boundaries
        black_mask_smooth = cv2.morphologyEx(black_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Opening operation to remove small noise
        black_mask_smooth = cv2.morphologyEx(black_mask_smooth, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur for additional smoothing
        black_mask_float = black_mask_smooth.astype(np.float32)
        blurred = cv2.GaussianBlur(black_mask_float, (7, 7), 1.0)
        
        # Threshold back to binary with a softer threshold
        black_mask_final = (blurred > 0.3).astype(np.uint8)
        
        # Create ROI mask: 1 for valid regions, 0 for invalid (black) regions
        roi_mask = (1 - black_mask_final).astype(np.uint8)
        
        return roi_mask

    def __getitem__(self, item):
        line = self.filelist[item]
        image_path = os.path.join(self.rootpath, "images", line + ".jpg")
        gt_path = os.path.join(self.rootpath, "masks", line + ".jpg")

        raw_image = cv2.imread(image_path)
        if raw_image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
            
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        # Create ROI mask to identify invalid regions (four corners)
        roi_mask = self._create_roi_mask(raw_image)

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
