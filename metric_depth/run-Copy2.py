import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import time  # Import time module

from depth_anything_v2.dpt import DepthAnythingV2
from collections import OrderedDict, deque


def strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict


class TemporalSmoother:
    """
    Manages temporal smoothing of depth maps using a sliding window.
    """

    def __init__(self,
                 window_size: int = 30,
                 sigma: float = 0.01,
                 init_method: str = 'dynamic',
                 device: str = 'cpu'):
        self.N = window_size
        self.sigma = torch.tensor(sigma, device=device, dtype=torch.float32)
        self.init_method = init_method
        self.W_D = deque(maxlen=self.N)
        self.W_w = deque(maxlen=self.N)
        self.frame_count = 0
        self.device = device

    def _solve_scale(self, d_new, d_ref):
        """Solves for the optimal scaling factor s."""
        d_new_flat = d_new.flatten()
        d_ref_flat = d_ref.flatten()

        # Ensure non-zero denominator to avoid division by zero
        denom = torch.dot(d_new_flat, d_new_flat)
        if denom.item() < 1e-6:  # Use .item() for scalar comparison
            return 1.0  # Return a neutral scale factor

        # Solve for scale s that minimizes ||s * d_new - d_ref||^2
        s = torch.dot(d_new_flat, d_ref_flat) / denom
        return s.item()  # Return as a Python float

    def process_frame(self, depth_map: np.ndarray):
        """Processes a new depth map and returns the smoothed version and its confidence."""
        self.frame_count += 1

        # Convert depth_map to torch.Tensor and move to device
        depth_map_tensor = torch.from_numpy(depth_map).to(self.device).float()

        if self.frame_count <= self.N:
            # Initialization Phase
            if self.init_method == 'fixed':
                d_prime = depth_map_tensor
                w = torch.tensor(1.0, device=self.device, dtype=torch.float32)
            elif self.init_method == 'anneal':
                d_prime = depth_map_tensor
                # Sigmoid annealing for smoother transition
                k = torch.tensor(0.3, device=self.device, dtype=torch.float32)
                exponent_input = -k * torch.tensor(
                    (self.frame_count - self.N / 2),
                    device=self.device,
                    dtype=torch.float32)
                w = torch.tensor(
                    1.0, device=self.device,
                    dtype=torch.float32) / (torch.tensor(
                        1.0, device=self.device, dtype=torch.float32) +
                                            torch.exp(exponent_input))
            else:  # 'dynamic'
                if self.frame_count == 1:
                    d_prime = depth_map_tensor
                    w = torch.tensor(1.0,
                                     device=self.device,
                                     dtype=torch.float32)
                else:
                    d_ref = self.W_D[-1]  # d_ref is already a torch.Tensor
                    s = self._solve_scale(depth_map_tensor,
                                          d_ref)  # Pass torch.Tensor
                    d_prime = s * depth_map_tensor
                    residual = torch.mean((d_prime - d_ref)**2)
                    w = torch.exp(-residual / (self.sigma**2))

            self.W_D.append(d_prime)
            self.W_w.append(w)

            return d_prime.cpu().numpy(), w.cpu().item(
            )  # Convert back to numpy and float
        else:
            # Steady-State Phase
            # Ensure all elements in W_w and W_D are tensors on the correct device
            weights_sum = sum(w.item()
                              for w in self.W_w)  # Sum of scalar weights
            if weights_sum < 1e-6:  # Handle case where all weights are near zero
                d_avg = self.W_D[
                    -1]  # Fallback to the last known map (already a tensor)
            else:
                # Perform weighted sum using torch operations
                weighted_sum_tensor = sum(w * d
                                          for w, d in zip(self.W_w, self.W_D))
                d_avg = weighted_sum_tensor / weights_sum

            s = self._solve_scale(depth_map_tensor, d_avg)  # Pass torch.Tensor
            d_prime = s * depth_map_tensor

            residual = torch.mean((d_prime - d_avg)**2)
            w = torch.exp(-residual / (self.sigma**2))

            # The deque will automatically handle removing the oldest element
            self.W_D.append(d_prime)
            self.W_w.append(w)

            return d_prime.cpu().numpy(), w.cpu().item(
            )  # Convert back to numpy and float


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Depth Anything V2 Metric Depth Estimation with Temporal Smoothing')

    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')

    parser.add_argument('--encoder',
                        type=str,
                        default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument(
        '--load-from',
        type=str,
        default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)

    # --- Arguments for temporal smoothing ---
    parser.add_argument('--temporal-smoothing',
                        action='store_true',
                        help='Enable temporal smoothing.')
    parser.add_argument('--init-method',
                        type=str,
                        default='dynamic',
                        choices=['fixed', 'anneal', 'dynamic'],
                        help='Initialization method for the warm-up phase.')
    parser.add_argument('--window-size',
                        type=int,
                        default=30,
                        help='Sliding window size for smoothing.')
    parser.add_argument('--sigma',
                        type=float,
                        default=0.01,
                        help='Sensitivity parameter for weight calculation.')

    # --- Arguments for saving outputs ---
    parser.add_argument('--save-numpy',
                        action='store_true',
                        help='Save the raw depth map as .npy')
    parser.add_argument(
        '--save-npz',
        action='store_true',
        help=
        'Save results (raw, smoothed, confidence) to a compressed .npz file.')

    parser.add_argument('--pred-only',
                        dest='pred_only',
                        action='store_true',
                        help='only display the prediction')
    parser.add_argument('--grayscale',
                        dest='grayscale',
                        action='store_true',
                        help='do not apply colorful palette')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    model_configs = {
        'vits': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384]
        },
        'vitb': {
            'encoder': 'vitb',
            'features': 128,
            'out_channels': [96, 192, 384, 768]
        },
        'vitl': {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024]
        },
        'vitg': {
            'encoder': 'vitg',
            'features': 384,
            'out_channels': [1536, 1536, 1536, 1536]
        }
    }

    depth_anything = DepthAnythingV2(
        **{
            **model_configs[args.encoder], 'max_depth': args.max_depth
        })

    depth_anything.load_state_dict(
        strip_module_prefix(
            torch.load(args.load_from, map_location='cpu').get("model")))

    depth_anything = depth_anything.to(DEVICE).eval()
    print("Model loaded and moved to device. Starting image processing...")

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'),
                              recursive=True)
    # Sort filenames to ensure correct processing order for sequences
    filenames.sort()

    os.makedirs(args.outdir, exist_ok=True)

    # Initialize the temporal smoother if requested
    smoother = None
    if args.temporal_smoothing:
        smoother = TemporalSmoother(window_size=args.window_size,
                                    sigma=args.sigma,
                                    init_method=args.init_method,
                                    device=DEVICE)
        print(
            f"Temporal smoothing enabled with window size {args.window_size} and init method '{args.init_method}'."
        )

    cmap = matplotlib.colormaps.get_cmap('Spectral')

    for k, filename in enumerate(filenames):
        filename = filename.strip().split()[0]

        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        if not os.path.exists(filename):
            print(f"Warning: File does not exist: {filename}")
            continue

        raw_image = cv2.imread(filename)

        if raw_image is None:
            print(f"Warning: Could not read image file: {filename}")
            continue

        # Start timing for prediction
        pred_start_time = time.time()
        try:
            depth_raw = depth_anything.infer_image(raw_image, args.input_size)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
        pred_end_time = time.time()
        print(
            f'  Prediction Time: {pred_end_time - pred_start_time:.4f} seconds'
        )

        # --- Processing and variable assignment ---
        depth_smoothed, confidence = None, None
        if smoother:
            # Start timing for alignment
            align_start_time = time.time()
            depth_smoothed, confidence = smoother.process_frame(depth_raw)
            align_end_time = time.time()
            print(
                f'  Alignment Time: {align_end_time - align_start_time:.4f} seconds'
            )
            depth_for_vis = depth_smoothed
            print(
                f'  Smoothed Frame {smoother.frame_count}, Confidence: {confidence:.8f}'
            )
        else:
            depth_for_vis = depth_raw

        # --- Saving logic ---

        if args.save_npz:
            output_path_npz = os.path.join(
                args.outdir,
                os.path.splitext(os.path.basename(filename))[0] + '.npz')

            if smoother:
                np.savez_compressed(output_path_npz,
                                    raw_depth=depth_raw,
                                    smoothed_depth=depth_smoothed,
                                    confidence=confidence,
                                    frame_count=smoother.frame_count)
            else:
                np.savez_compressed(output_path_npz, raw_depth=depth_raw)
            print(f'  Saved results to {output_path_npz}')
