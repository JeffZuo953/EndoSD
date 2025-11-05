import argparse
import numpy as np
import os
import torch
import cv2
from typing import List, Tuple

from depth_anything_v2.dpt_live import DepthAnythingV2

from collections import OrderedDict
import matplotlib.cm as cm
import imageio


def save_video(frames,
               output_video_path,
               fps=10,
               is_depths=False,
               grayscale=False):
    writer = imageio.get_writer(output_video_path,
                                fps=fps,
                                macro_block_size=1,
                                codec='libx264',
                                ffmpeg_params=['-crf', '18'])
    if is_depths:
        colormap = np.array(cm.get_cmap("inferno").colors)
        d_min, d_max = frames.min(), frames.max()
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(
                np.uint8)
            depth_vis = (colormap[depth_norm] *
                         255).astype(np.uint8) if not grayscale else depth_norm
            writer.append_data(depth_vis)
    else:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])

    writer.close()


def strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict


def read_image_frames_from_txt(
        filelist_path: str, image_width: int,
        image_height: int) -> Tuple[List[np.ndarray], List[str], int]:
    """
    Reads image paths from a text file, loads images, resizes them, and converts to numpy frames.
    Ignores depth and mask information. Returns frames, original image paths, and a dummy fps.
    """
    frames: List[np.ndarray] = []
    image_paths: List[str] = []
    with open(filelist_path, "r") as f:
        filelist: List[str] = f.read().splitlines()

    for line in filelist:
        img_path: str = line.split(" ")[0]
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found at {img_path}. Skipping.")
            continue

        raw_image: np.ndarray = cv2.imread(img_path)
        if raw_image is None:
            print(f"Warning: Could not read image at {img_path}. Skipping.")
            continue

        # Convert BGR to RGB and normalize to [0, 1]
        image: np.ndarray = cv2.cvtColor(raw_image,
                                         cv2.COLOR_BGR2RGB) / 255.0

        frames.append(image)
        image_paths.append(img_path)

    # Return a dummy fps, as it's not relevant for static image lists
    return frames, image_paths, 30


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Depth Anything')
    parser.add_argument('--input_txt',
                        type=str,
                        required=True,
                        help='Path to the text file containing image paths')
    parser.add_argument('--image_width',
                        type=int,
                        default=960,
                        help='Desired width for input images')
    parser.add_argument('--image_height',
                        type=int,
                        default=540,
                        help='Desired height for input images')
    parser.add_argument("--encoder",
                        default="vitl",
                        choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--load-from",
                        type=str,
                        required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--max-depth", default=20, type=float)

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384]
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    video_depth_anything = DepthAnythingV2(
        **{
            **model_configs[args.encoder], "max_depth": args.max_depth
        })
    video_depth_anything.load_state_dict(strip_module_prefix(
        torch.load(args.load_from, map_location='cpu')["model"]),
                                         strict=True)

    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    frames_list, image_paths, target_fps = read_image_frames_from_txt(
        args.input_txt, args.image_width, args.image_height)

    # Convert list of frames to a single numpy array
    frames_np = np.array(frames_list)
    print(frames_np.shape)

    depths, fps = video_depth_anything.infer_video_depth_live(
        frames_np,
        input_size=(args.image_width, args.image_height),
        device=DEVICE)

    for i, depth_map in enumerate(depths):
        original_image_path = image_paths[i]

        path_parts = original_image_path.split(os.sep)

        # Ensure we have enough parts to extract category and filename
        if len(path_parts) >= 2:
            category_dir = path_parts[-2]  # e.g., 'category'
            image_filename = os.path.splitext(
                path_parts[-1])[0]  # e.g., 'image'
        else:
            # Fallback if path is too short, use a generic name
            category_dir = "misc"
            image_filename = os.path.splitext(
                os.path.basename(original_image_path))[0]

        output_category_dir = os.path.join(args.save_path, category_dir)

        os.makedirs(output_category_dir, exist_ok=True)

        depth_npz_path = os.path.join(output_category_dir,
                                      image_filename + '.npz')
        np.savez_compressed(depth_npz_path, depth=depth_map)

    depth_vis_path = os.path.join(args.save_path, 'depth.mp4')
    save_video(depths,
               depth_vis_path,
               fps=30,
               is_depths=True,
               grayscale=True)
