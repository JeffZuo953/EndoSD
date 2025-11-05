import os
import warnings
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.

    Args:
        sample (dict): sample
        size (tuple): image size

    Returns:
        tuple: new size
    """
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
        downscale_only=False,
        device_env_var="CACHE_RESIZE_DEVICE",
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method
        self.__downscale_only = downscale_only
        self.__device_env_var = device_env_var
        self.__torch_device = self._resolve_device(device_env_var)
        self.__warned_backend = False

    def _resolve_device(self, env_key: str):
        device_str = os.environ.get(env_key)
        if not device_str:
            return None
        try:
            device = torch.device(device_str)
        except (RuntimeError, ValueError):
            warnings.warn(f"Invalid device string '{device_str}' for {env_key}; falling back to CPU resizing.")
            return None
        if device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn(f"Requested CUDA device '{device_str}' for resizing but CUDA is not available; falling back to CPU.")
            return None
        return device

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def _should_skip_resize(self, width: int, height: int) -> bool:
        if not self.__downscale_only:
            return False
        orig_w, orig_h = width, height
        return orig_w <= self.__width and orig_h <= self.__height

    def _torch_mode_for_image(self) -> str:
        if self.__image_interpolation_method == cv2.INTER_CUBIC:
            return "bicubic"
        return "bilinear"

    def _resize_with_torch(self, array: np.ndarray, width: int, height: int, mode: str, out_dtype: np.dtype | type) -> np.ndarray:
        tensor = torch.from_numpy(array)
        orig_dtype = tensor.dtype
        needs_threshold = False

        if tensor.dtype == torch.bool:
            tensor = tensor.to(torch.float32)
            needs_threshold = True
        elif tensor.dtype not in (torch.float16, torch.float32, torch.float64):
            tensor = tensor.to(torch.float32)

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported array shape for resizing: {array.shape}")

        if self.__torch_device is None:
            if not self.__warned_backend:
                warnings.warn("Torch device unavailable for resizing; falling back to CPU.")
                self.__warned_backend = True
            return self._resize_with_cv2(array, width, height, mode, out_dtype)

        tensor = tensor.to(self.__torch_device, non_blocking=False)
        interpolate_kwargs = {"mode": mode, "size": (height, width)}
        if mode in ("bilinear", "bicubic"):
            interpolate_kwargs["align_corners"] = False

        resized = F.interpolate(tensor, **interpolate_kwargs)
        resized = resized.to("cpu")

        if array.ndim == 2:
            result = resized[0, 0].numpy()
        else:
            result = resized[0].permute(1, 2, 0).numpy()

        if needs_threshold or out_dtype is bool:
            return (result > 0.5)
        result = result.astype(out_dtype if out_dtype is not None else array.dtype, copy=False)
        return result

    def _resize_with_cv2(self, array: np.ndarray, width: int, height: int, mode: str, out_dtype: np.dtype | type) -> np.ndarray:
        if mode == "nearest":
            interpolation = cv2.INTER_NEAREST
        elif mode == "bicubic":
            interpolation = cv2.INTER_CUBIC
        elif mode == "bilinear":
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = self.__image_interpolation_method

        input_array = array
        needs_threshold = False
        if array.dtype == np.bool_:
            input_array = array.astype(np.float32)
            needs_threshold = True
        elif array.dtype not in (np.float16, np.float32, np.float64):
            input_array = array.astype(np.float32)

        resized = cv2.resize(input_array, (width, height), interpolation=interpolation)
        if needs_threshold or out_dtype is bool:
            return resized > 0.5
        return resized.astype(out_dtype if out_dtype is not None else array.dtype, copy=False)

    def _resize_field(self, array: np.ndarray, width: int, height: int, mode: str, out_dtype: np.dtype | type = None) -> np.ndarray:
        backend = "torch" if self.__torch_device is not None else "cv2"
        if backend == "torch":
            return self._resize_with_torch(array, width, height, mode, out_dtype or array.dtype)
        return self._resize_with_cv2(array, width, height, mode, out_dtype or array.dtype)

    def __call__(self, sample):
        orig_height, orig_width = sample["image"].shape[0], sample["image"].shape[1]

        if self._should_skip_resize(orig_width, orig_height):
            target_width, target_height = orig_width, orig_height
        else:
            target_width, target_height = self.get_size(orig_width, orig_height)

        target_width = max(int(round(target_width)), 1)
        target_height = max(int(round(target_height)), 1)

        if target_width == orig_width and target_height == orig_height:
            return sample

        image_mode = self._torch_mode_for_image()
        sample["image"] = self._resize_field(sample["image"], target_width, target_height, image_mode, out_dtype=np.float32)

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = self._resize_field(sample["disparity"], target_width, target_height, "nearest", out_dtype=np.float32)

            if "depth" in sample:
                sample["depth"] = self._resize_field(sample["depth"], target_width, target_height, "nearest", out_dtype=np.float32)

            if "semseg_mask" in sample:
                sample["semseg_mask"] = self._resize_field(sample["semseg_mask"], target_width, target_height, "nearest", out_dtype=np.uint8)

            if "mask" in sample:
                sample["mask"] = self._resize_field(sample["mask"], target_width, target_height, "nearest", out_dtype=bool if sample["mask"].dtype == np.bool_ else np.float32)

            if "valid_mask" in sample:
                sample["valid_mask"] = self._resize_field(sample["valid_mask"], target_width, target_height, "nearest", out_dtype=bool)

            if "depth_valid_mask" in sample:
                sample["depth_valid_mask"] = self._resize_field(sample["depth_valid_mask"], target_width, target_height, "nearest", out_dtype=bool)

            if "seg_valid_mask" in sample:
                sample["seg_valid_mask"] = self._resize_field(sample["seg_valid_mask"], target_width, target_height, "nearest", out_dtype=bool)

        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])
        
        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        if "depth_valid_mask" in sample:
            depth_valid = sample["depth_valid_mask"].astype(bool)
            sample["depth_valid_mask"] = np.ascontiguousarray(depth_valid)

        if "seg_valid_mask" in sample:
            seg_valid = sample["seg_valid_mask"].astype(bool)
            sample["seg_valid_mask"] = np.ascontiguousarray(seg_valid)

        return sample


class Crop(object):
    """Crop sample for batch-wise training. Image is of shape CxHxW
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample['image'].shape[-2:]
        assert h >= self.size[0] and w >= self.size[1], 'Wrong size'
        
        h_start = np.random.randint(0, h - self.size[0] + 1)
        w_start = np.random.randint(0, w - self.size[1] + 1)
        h_end = h_start + self.size[0]
        w_end = w_start + self.size[1]
        
        sample['image'] = sample['image'][:, h_start: h_end, w_start: w_end]
        
        if "depth" in sample:
            sample["depth"] = sample["depth"][h_start: h_end, w_start: w_end]
        
        if "mask" in sample:
            sample["mask"] = sample["mask"][h_start: h_end, w_start: w_end]
            
        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"][h_start: h_end, w_start: w_end]
            
        return sample
