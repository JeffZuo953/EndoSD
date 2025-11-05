import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
import OpenEXR
import Imath

from .transform import Resize, NormalizeImage, PrepareForNet
from .utils import compute_valid_mask

# ==== 全局参数 ====
FAR = 4.0
NEAR = 0.01

# 预先计算好全局常量
_X = 1.0 - FAR / NEAR
_Y = FAR / NEAR
Z = _X / FAR
W = _Y / FAR


class Endomapper(Dataset):

    def __init__(self, filelist_path, mode, size=(518, 518), max_depth=0.2):
        self.mode = mode
        self.size = size
        self.max_depth = max_depth
        self.dataset_name = "EndoMapper"
        self.dataset_type = "NO"

        with open(filelist_path, "r") as f:
            self.filelist = f.read().splitlines()

        net_w, net_h = size
        self.transform = Compose([
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
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def read_exr_r(self, filepath):
        """读取OpenEXR文件并返回(H, W, 1)的np.array"""
        exrfile = OpenEXR.InputFile(filepath)
        header = exrfile.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        size = (height, width)

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

        r = np.frombuffer(exrfile.channel("R", FLOAT),
                          dtype=np.float32).reshape(size)

        exr = np.stack([r], axis=-1)  # (H, W, 3)
        return exr

    def __getitem__(self, item):
        img_path = self.filelist[item].split(" ")[0]
        depth_path = self.filelist[item].split(" ")[1]

        # ====== 读取RGB图像 ======
        raw_image = cv2.imread(img_path)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0  # 标准化到[0,1]

        # ====== 读取EXR深度 ======
        exr_r = self.read_exr_r(depth_path)  # (H, W, 1), float32

        # 取第一通道（R通道）
        im = exr_r[..., 0]
        # ====== 深度恢复公式 ======
        depth_dm = 1.0 / (Z * (1.0 - im) + W)  # 分米
        depth = depth_dm * 0.1  # 转换成米（m）

        # 添加深度裁剪以确保范围一致
        depth = np.clip(depth, 0, self.max_depth)

        # ====== Apply transforms ======
        sample = self.transform({"image": image, "depth": depth})

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])

        sample["valid_mask"] = compute_valid_mask(
            sample["image"],
            sample["depth"],
            max_depth=self.max_depth,
            dataset_name="EndoMapper",
        )
        sample["image_path"] = img_path
        sample["max_depth"] = self.max_depth
        sample["dataset_name"] = self.dataset_name
        sample["source_type"] = self.dataset_type

        return sample

    def __len__(self):
        return len(self.filelist)
