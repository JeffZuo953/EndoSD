import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import Resize, NormalizeImage, PrepareForNet


class InHouseSegDataset(Dataset):
    """
    Inhouse dataset for segmentation.
    Labels:
    0: background
    1: stone
    2: laser
    3: polyp
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

    def __getitem__(self, item):
        line = self.filelist[item]
        image_path = f"{self.rootpath}/img/{line}.jpg"
        gt_path = f"{self.rootpath}/gt/{line}.png"

        raw_image = cv2.imread(image_path)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        sample = self.image_transform({"image": image, "semseg_mask": mask})

        sample = {"image": torch.from_numpy(sample["image"]), "semseg_mask": torch.from_numpy(sample["semseg_mask"]).long(), "image_path": image_path}

        return sample

    def __len__(self):
        return len(self.filelist)
