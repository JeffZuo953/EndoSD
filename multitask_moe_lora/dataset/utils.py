import torch
import numpy as np

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def _ensure_3d_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 2:
        return image.unsqueeze(0)
    if image.dim() != 3:
        raise ValueError(f"Expected image tensor with 3 dimensions, got shape {tuple(image.shape)}")
    return image


def _ensure_2d_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 3 and depth.size(0) == 1:
        depth = depth[0]
    if depth.dim() != 2:
        raise ValueError(f"Expected depth tensor with 2 dimensions, got shape {tuple(depth.shape)}")
    return depth


def compute_valid_mask(image: torch.Tensor,
                       depth: torch.Tensor,
                       min_depth: float = 0.0,
                       max_depth: float = 0.4,
                       black_threshold: float = 15.0 / 255.0,
                       dataset_name: str | None = None) -> torch.Tensor:
    """
    Compute a validity mask given an image and depth map.

    A pixel is considered valid only if:
      * Depth is within [min_depth, max_depth]
      * The corresponding RGB pixel is brighter than `black_threshold`

    Args:
        image: Tensor shaped (3, H, W) that has already been normalized with
               ImageNet statistics (mean/std). CPU tensor expected.
        depth: Tensor shaped (H, W) or (1, H, W) with depth values in meters.
        min_depth: Minimum acceptable depth.
        max_depth: Maximum acceptable depth.
        black_threshold: Brightness threshold (range [0, 1]) below which pixels
                         are marked invalid.
        dataset_name: Optional dataset identifier used for dataset-specific
                       invalid-pixel handling (e.g., near-black detection for
                       EndoVis2017).

    Returns:
        torch.BoolTensor of shape (H, W)
    """
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    if not torch.is_tensor(depth):
        depth = torch.as_tensor(depth)

    image = _ensure_3d_image(image).to(torch.float32)
    depth = _ensure_2d_depth(depth).to(torch.float32)

    device = image.device
    mean = IMAGENET_MEAN.to(device).view(-1, 1, 1)
    std = IMAGENET_STD.to(device).view(-1, 1, 1)

    # Denormalize image back to [0, 1] range
    denorm_image = image * std + mean
    brightness = denorm_image.mean(dim=0)

    denorm_255 = torch.clamp((denorm_image * 255.0).round(), 0, 255).to(torch.int16)

    dataset_key = _normalize_dataset_key(dataset_name) if dataset_name else None

    # Dataset-specific thresholds for darkness filtering
    dark_int_threshold = 16
    brightness_threshold = black_threshold
    if dataset_key in {"hamlyn", "c3vd", "c3vdv2"}:
        dark_int_threshold = 3
        brightness_threshold = 3.0 / 255.0

    channel_max = denorm_255.amax(dim=0)
    channel_min = denorm_255.amin(dim=0)
    fully_low_intensity = (denorm_255 <= dark_int_threshold).all(dim=0)
    clustered_low_intensity = ((channel_max - channel_min) <= 2) & (channel_max <= dark_int_threshold + 2)
    near_black = fully_low_intensity | clustered_low_intensity

    if dataset_key in {"endovis2017", "endovis2018"}:
        height, width = map(int, depth.shape[-2:])
        horiz_border = max(1, min(width, int(round(width * 0.30))))
        vert_border = max(1, min(height, int(round(height * 0.10))))
        rows = torch.arange(height, device=denorm_255.device).view(-1, 1)
        cols = torch.arange(width, device=denorm_255.device).view(1, -1)
        edge_mask = (rows < vert_border) | (rows >= height - vert_border) | \
                    (cols < horiz_border) | (cols >= width - horiz_border)
        special_mask = near_black & edge_mask
    elif dataset_key in {"hamlyn", "c3vd", "c3vdv2"}:
        height, width = map(int, depth.shape[-2:])
        third_h = max(1, int(round(height / 3.0)))
        third_w = max(1, int(round(width / 3.0)))
        rows = torch.arange(height, device=denorm_255.device).view(-1, 1)
        cols = torch.arange(width, device=denorm_255.device).view(1, -1)
        border_mask = (rows < third_h) | (rows >= height - third_h) | \
                      (cols < third_w) | (cols >= width - third_w)
        special_mask = near_black & border_mask
    elif dataset_key in {"scared", "stereomis", "simcol", "dvpn"}:
        special_mask = near_black
    else:
        special_mask = torch.zeros_like(denorm_255[0], dtype=torch.bool)

    max_depth_limit = 0.4 if max_depth is None else max(float(max_depth), 0.4)
    depth_valid = torch.isfinite(depth) & (depth > min_depth) & (depth <= max_depth_limit)

    dark_threshold = dark_int_threshold / 255.0
    dark_pixels = (denorm_image <= dark_threshold).all(dim=0)
    brightness_valid = (brightness > brightness_threshold) & (~dark_pixels) & (~special_mask)

    return depth_valid & brightness_valid


def remap_labels(mask: torch.Tensor | np.ndarray,
                 mapping: dict[int, int],
                 dtype=None) -> np.ndarray:
    """
    Remap integer labels in `mask` according to the provided mapping.

    Args:
        mask: Input mask as numpy array or torch tensor.
        mapping: Dict mapping original value -> target value.
        dtype: Optional dtype for output array (defaults to input dtype).

    Returns:
        numpy.ndarray with remapped labels.
    """
    import numpy as np

    if hasattr(mask, "cpu"):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    mask_np = np.asarray(mask_np)

    out = mask_np.copy()
    for src, dst in mapping.items():
        out[mask_np == src] = dst
    return out.astype(dtype or mask_np.dtype, copy=False)


import re


def _to_numpy(mask):
    if hasattr(mask, "cpu"):
        return mask.cpu().numpy()
    return np.asarray(mask)


def _from_numpy_like(reference, array: np.ndarray):
    import torch

    if hasattr(reference, "device"):
        return torch.from_numpy(array).to(reference.device)
    return array


def _normalize_dataset_key(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    # remove trailing qualifiers like _train / _val
    name = re.sub(r"_(train|val|test)$", "", name)
    if "endovis2017" in name:
        return "endovis2017"
    if "endovis2018" in name:
        return "endovis2018"
    if "endosynth" in name:
        return "endosynth"
    if "endonerf" in name:
        return "endonerf"
    if "stereomis" in name:
        return "stereomis"
    if "scared" in name:
        return "scared"
    if "c3vdv2" in name:
        return "c3vdv2"
    if "c3vd" in name:
        return "c3vd"
    if "hamlyn" in name:
        return "hamlyn"
    if "kidney3d" in name:
        return "kidney3d"
    if "kvasir" in name:
        return "kvasir"
    if "bkai" in name:
        return "bkai"
    if "clinic" in name:
        return "clinicdb"
    if "cvc" in name:
        return "cvc_endoscene"
    if "simcol" in name:
        return "simcol"
    if "dvpn" in name or "davinci" in name:
        return "dvpn"
    return name


_LS_10CLASS_MAPPINGS = {
    "endovis2017": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        255: 255,
    },
    "endovis2018": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 6,
        5: 7,
        6: 8,
        7: 9,
        255: 255,
    },
    "endosynth": {
        0: 0,
        1: 2,  # cadiere -> class 2
        2: 1,  # fenestrated bipolar -> class 1
        3: 5,  # double fenestrated -> class 5
        4: 3,  # needle driver -> class 3
        5: 6,  # monopolar curved shears -> class 6
        6: 1,  # maryland bipolar -> class 1
        255: 255,
    },
}


def map_ls_semseg_to_10_classes(mask, dataset_name: str):
    dataset_key = _normalize_dataset_key(dataset_name)
    mapping = _LS_10CLASS_MAPPINGS.get(dataset_key)
    if mapping is None:
        return mask

    mask_np = _to_numpy(mask)
    out = np.full_like(mask_np, 255)
    for src, dst in mapping.items():
        out[mask_np == src] = dst
    return _from_numpy_like(mask, out)


_THREE_CLASS_CONFIG = {
    "endovis2017": {"instrument": list(range(1, 10)), "polyp": []},
    "endovis2018": {"instrument": [1, 2, 3, 6, 7, 8, 9], "polyp": []},
    "endosynth": {"instrument": [1, 2, 3, 5, 6], "polyp": []},
    "endonerf": {"instrument": "nonzero", "polyp": []},
    "kidney3d": {"instrument": "nonzero", "polyp": []},
    "kvasir": {"instrument": [], "polyp": [3]},
    "bkai": {"instrument": [], "polyp": [1, 3]},  # fallback for polyp datasets
    "clinicdb": {"instrument": [], "polyp": [1, 3]},
    "cvc_endoscene": {"instrument": [], "polyp": [1, 3]},
}


def map_semseg_to_three_classes(mask, dataset_name: str):
    dataset_key = _normalize_dataset_key(dataset_name)
    config = _THREE_CLASS_CONFIG.get(dataset_key)
    if config is None:
        return mask

    mask_np = _to_numpy(mask)
    out = np.full_like(mask_np, 255)
    out[mask_np == 0] = 0

    instr_def = config.get("instrument")
    if instr_def == "nonzero":
        instr_mask = (mask_np != 0) & (mask_np != 255)
        out[instr_mask] = 2
    elif isinstance(instr_def, (list, tuple, set)):
        for val in instr_def:
            out[mask_np == val] = 2

    for polyp_val in config.get("polyp", []):
        out[mask_np == polyp_val] = 3

    return _from_numpy_like(mask, out)


__all__ = [
    "compute_valid_mask",
    "remap_labels",
    "map_ls_semseg_to_10_classes",
    "map_semseg_to_three_classes",
]
