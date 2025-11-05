import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .palette import get_palette

def save_depth_prediction(pred: torch.Tensor, filename: str, outdir: str, colormap: str = 'gray'):
    """Saves a depth prediction to a file."""
    if not outdir:
        return
    
    os.makedirs(outdir, exist_ok=True)
    pred_to_save = pred.squeeze().cpu().numpy()
    
    # Ensure the array is 2D
    if pred_to_save.ndim == 3:
        pred_to_save = pred_to_save[0]

    # Normalize to [0, 255] and convert to uint8 for cv2
    normalized = (pred_to_save - pred_to_save.min()) / (pred_to_save.max() - pred_to_save.min() + 1e-8) * 255
    normalized = normalized.astype(np.uint8)
    
    # If colormap is 'gray', save the normalized image directly (it's already single-channel).
    # Otherwise, apply the specified cv2 colormap.
    if colormap.lower() == 'gray':
        cv2.imwrite(os.path.join(outdir, filename), normalized)
    else:
        colormap_cv = cv2.COLORMAP_BONE # Default
        if colormap.upper() in [c for c in dir(cv2) if c.startswith('COLORMAP_')]:
            colormap_cv = getattr(cv2, f'COLORMAP_{colormap.upper()}')
        
        colored_depth = cv2.applyColorMap(normalized, colormap_cv)
        cv2.imwrite(os.path.join(outdir, filename), colored_depth)

def save_seg_prediction(pred: torch.Tensor, filename: str, outdir: str):
    """Saves a segmentation prediction to a file."""
    if not outdir:
        return
        
    os.makedirs(outdir, exist_ok=True)
    pred_to_save = pred.argmax(dim=1)[0].cpu().numpy()
    
    palette = get_palette()
    color_seg = np.zeros((pred_to_save.shape[0], pred_to_save.shape[1], 3), dtype=np.uint8)
    
    for label, color in enumerate(palette):
        if label < len(palette):  # Ensure label is within palette range
            color_seg[pred_to_save == label, :] = color
            
    color_seg = cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(outdir, filename), color_seg)

def _prepare_valid_mask(valid_mask, target_shape):
    """Resize and binarize valid mask to match the target spatial shape."""
    if valid_mask is None:
        return None

    if not isinstance(valid_mask, torch.Tensor):
        valid_mask = torch.as_tensor(valid_mask)

    if valid_mask.dtype != torch.float32:
        valid_mask = valid_mask.float()

    if valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
    elif valid_mask.ndim == 3:
        if valid_mask.shape[0] != 1:
            raise ValueError("valid_mask with 3 dims must have batch size 1")
        valid_mask = valid_mask.unsqueeze(1)
    elif valid_mask.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported valid_mask ndim: {valid_mask.ndim}")

    resized = F.interpolate(valid_mask, size=target_shape, mode='nearest')
    return (resized[0, 0] > 0.5).cpu().numpy()


def save_depth_output(pred, filename, outdir, norm_type, colormap, save_img, save_pt, save_npz, valid_mask=None):
    """Saves depth prediction in multiple formats."""
    if not any([save_img, save_pt, save_npz]):
        return

    base_filename = os.path.splitext(os.path.basename(filename))[0]

    if save_pt:
        pt_dir = os.path.join(outdir, 'pt')
        os.makedirs(pt_dir, exist_ok=True)
        torch.save(pred.clone(), os.path.join(pt_dir, f"{base_filename}.pt"))

    pred_np = pred.squeeze().detach().cpu().numpy()

    # Ensure the array is 2D for visualization
    if pred_np.ndim == 3:
        pred_np = pred_np[0]

    mask_np = _prepare_valid_mask(valid_mask, pred_np.shape) if valid_mask is not None else None

    if save_npz:
        npz_dir = os.path.join(outdir, 'npz')
        os.makedirs(npz_dir, exist_ok=True)
        np.savez_compressed(os.path.join(npz_dir, f"{base_filename}.npz"), depth=pred_np)

    if save_img:
        img_dir = os.path.join(outdir, 'image')
        os.makedirs(img_dir, exist_ok=True)
        if mask_np is not None and mask_np.shape != pred_np.shape:
            raise ValueError("valid_mask shape mismatch after resize")

        if norm_type == 'min-max':
            valid_values = pred_np[mask_np] if mask_np is not None else pred_np
            if valid_values.size == 0:
                normalized = np.zeros_like(pred_np, dtype=np.float32)
            else:
                min_val = float(valid_values.min())
                max_val = float(valid_values.max())
                scale = max(max_val - min_val, 1e-8)
                normalized = (pred_np - min_val) / scale
        elif norm_type == 'max':
            valid_values = pred_np[mask_np] if mask_np is not None else pred_np
            if valid_values.size == 0:
                normalized = np.zeros_like(pred_np, dtype=np.float32)
            else:
                max_val = float(valid_values.max())
                normalized = pred_np / max(max_val, 1e-8)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

        normalized = np.clip(normalized, 0.0, 1.0)

        if mask_np is not None:
            normalized = normalized * mask_np.astype(np.float32)

        if colormap.lower() == 'gray':
            normalized_uint16 = (normalized * 65535.0).round().astype(np.uint16)
            cv2.imwrite(os.path.join(img_dir, f"{base_filename}.png"), normalized_uint16)
        else:
            normalized_uint8 = (normalized * 255.0).round().astype(np.uint8)
            colormap_cv = cv2.COLORMAP_BONE # Default
            if colormap.upper() in [c for c in dir(cv2) if c.startswith('COLORMAP_')]:
                colormap_cv = getattr(cv2, f'COLORMAP_{colormap.upper()}')
            colored_depth = cv2.applyColorMap(normalized_uint8, colormap_cv)
            cv2.imwrite(os.path.join(img_dir, f"{base_filename}.png"), colored_depth)

def save_seg_output(pred, filename, outdir, save_img, save_pt, save_npz, ignore_mask=None):
    """Saves segmentation prediction in multiple formats."""
    if not any([save_img, save_pt, save_npz]):
        return

    base_filename = os.path.splitext(os.path.basename(filename))[0]
    pred_idx = pred.argmax(dim=0)

    # Apply the ignore mask if provided
    if ignore_mask is not None:
        # Ensure mask is on the same device and has the correct shape
        if ignore_mask.shape != pred_idx.shape:
             ignore_mask = F.interpolate(ignore_mask.unsqueeze(0).unsqueeze(0).float(), size=pred_idx.shape, mode='nearest').squeeze().bool()

        pred_idx[ignore_mask] = 0 # Set ignored pixels to background (class 0)

    if save_pt:
        pt_dir = os.path.join(outdir, 'pt')
        os.makedirs(pt_dir, exist_ok=True)
        torch.save(pred_idx.clone().to(torch.uint8), os.path.join(pt_dir, f"{base_filename}.pt"))

    pred_np = pred_idx.cpu().numpy().astype(np.uint8)

    if save_npz:
        npz_dir = os.path.join(outdir, 'npz')
        os.makedirs(npz_dir, exist_ok=True)
        np.savez_compressed(os.path.join(npz_dir, f"{base_filename}.npz"), segmentation=pred_np)

    if save_img:
        img_dir = os.path.join(outdir, 'image')
        os.makedirs(img_dir, exist_ok=True)
        palette = get_palette()
        color_seg = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[pred_np == label, :] = color
        color_seg = cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(img_dir, f"{base_filename}.png"), color_seg)
