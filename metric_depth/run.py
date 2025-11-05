import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import time
from collections import OrderedDict, deque
from itertools import product

# Assuming dpt_live is the correct module for your setup
from depth_anything_v2.dpt import DepthAnythingV2


def strip_module_prefix(state_dict):
    """Removes the 'module.' prefix from state dictionary keys."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict


class TemporalSmoother:
    """
    Manages temporal smoothing of depth maps using multiple sliding windows,
    one for each combination of specified initialization and scaling methods.
    """

    def __init__(self,
                 window_size: int = 30,
                 sigma: float = 0.01,
                 init_methods: list[str] = ['dynamic'],
                 scale_methods: list[str] = ['least_squares'],
                 device: str = 'cpu'):
        self.N = window_size
        self.sigma = torch.tensor(sigma, device=device, dtype=torch.float32)
        self.device = device
        self.combinations = list(product(init_methods, scale_methods))
        self.W_D = {combo: deque(maxlen=self.N) for combo in self.combinations}
        self.W_w = {combo: deque(maxlen=self.N) for combo in self.combinations}
        self.frame_count = {combo: 0 for combo in self.combinations}

    def _solve_scale_least_squares(self, d_new, d_ref):
        d_new_flat = d_new.flatten()
        d_ref_flat = d_ref.flatten()
        denom = torch.dot(d_new_flat, d_new_flat)
        if denom.item() < 1e-6:
            return 1.0
        s = torch.dot(d_new_flat, d_ref_flat) / denom
        return s.item()

    def _solve_scale_mean(self, d_new, d_ref):
        mean_new = torch.mean(d_new)
        if mean_new.item() < 1e-6:
            return 1.0
        mean_ref = torch.mean(d_ref)
        return (mean_ref / mean_new).item()

    def _solve_scale_median(self, d_new, d_ref):
        median_new = torch.median(d_new)
        if median_new.item() < 1e-6:
            return 1.0
        median_ref = torch.median(d_ref)
        return (median_ref / median_new).item()

    def _get_scale_solver(self, scale_method: str):
        if scale_method == 'least_squares':
            return self._solve_scale_least_squares
        elif scale_method == 'mean':
            return self._solve_scale_mean
        elif scale_method == 'median':
            return self._solve_scale_median
        else:
            raise ValueError(f"Unknown scale method: {scale_method}")

    def process_frame(self, depth_map: np.ndarray):
        depth_map_tensor = torch.from_numpy(depth_map).to(self.device).float()
        results = {}
        for init_method, scale_method in self.combinations:
            combo_key = (init_method, scale_method)
            self.frame_count[combo_key] += 1
            frame_count = self.frame_count[combo_key]
            W_D_combo = self.W_D[combo_key]
            W_w_combo = self.W_w[combo_key]
            scale_solver = self._get_scale_solver(scale_method)
            if frame_count <= self.N:
                if init_method == 'fixed':
                    d_prime = depth_map_tensor
                    w = torch.tensor(1.0,
                                     device=self.device,
                                     dtype=torch.float32)
                elif init_method == 'anneal':
                    d_prime = depth_map_tensor
                    k = torch.tensor(0.3,
                                     device=self.device,
                                     dtype=torch.float32)
                    exponent = -k * torch.tensor((frame_count - self.N / 2),
                                                 device=self.device,
                                                 dtype=torch.float32)
                    w = 1.0 / (1.0 + torch.exp(exponent))
                else:
                    if frame_count == 1:
                        d_prime = depth_map_tensor
                        w = torch.tensor(1.0,
                                         device=self.device,
                                         dtype=torch.float32)
                    else:
                        d_ref = W_D_combo[-1]
                        s = scale_solver(depth_map_tensor, d_ref)
                        d_prime = s * depth_map_tensor
                        residual = torch.mean((d_prime - d_ref)**2)
                        w = torch.exp(-residual / (self.sigma**2))
                W_D_combo.append(d_prime)
                W_w_combo.append(w)
                results[combo_key] = (d_prime.cpu().numpy(), w.cpu().item())
            else:
                weights_sum = sum(w_val.item() for w_val in W_w_combo)
                d_avg = W_D_combo[-1] if weights_sum < 1e-6 else \
                        sum(w_val * d for w_val, d in zip(W_w_combo, W_D_combo)) / weights_sum
                s = scale_solver(depth_map_tensor, d_avg)
                d_prime = s * depth_map_tensor
                residual = torch.mean((d_prime - d_avg)**2)
                w = torch.exp(-residual / (self.sigma**2))
                W_D_combo.append(d_prime)
                W_w_combo.append(w)
                results[combo_key] = (d_prime.cpu().numpy(), w.cpu().item())
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Depth Anything V2 Metric Depth Estimation with Temporal Smoothing')

    parser.add_argument('--img-path',
                        type=str,
                        required=True,
                        help="Path to a directory containing case*.txt files.")
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth_smoothed')

    parser.add_argument('--encoder',
                        type=str,
                        default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument(
        '--load-from',
        type=str,
        default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)

    parser.add_argument('--temporal-smoothing',
                        action='store_true',
                        help='Enable temporal smoothing.')
    parser.add_argument('--init-methods',
                        type=str,
                        nargs='+',
                        default=['dynamic'],
                        choices=['fixed', 'anneal', 'dynamic'])
    parser.add_argument('--scale-methods',
                        type=str,
                        nargs='+',
                        default=['least_squares'],
                        choices=['least_squares', 'mean', 'median'])
    parser.add_argument('--window-size', type=int, default=30)
    parser.add_argument('--sigma', type=float, default=0.01)

    parser.add_argument('--save-npz',
                        action='store_true',
                        help='Save results to a compressed .npz file.')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true')

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
    print("Model loaded once. Starting batch processing...")

    case_files = sorted(glob.glob(os.path.join(args.img_path, 'case*.txt')))
    print("found following files:")
    print(case_files)
    if not case_files:
        print(
            f"Error: No 'case*.txt' files found in the specified path: {args.img_path}"
        )
        exit()

    sigma_str = str(args.sigma).replace('.', '_')

    for case_filepath in case_files:
        case_name = os.path.splitext(os.path.basename(case_filepath))[0]
        with open(case_filepath, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]

        print(f"\n{'='*80}\nProcessing Case: {case_name}\n{'='*80}")

        smoother = None
        if args.temporal_smoothing:
            smoother = TemporalSmoother(window_size=args.window_size,
                                        sigma=args.sigma,
                                        init_methods=args.init_methods,
                                        scale_methods=args.scale_methods,
                                        device=DEVICE)
            print(f"Temporal smoother re-initialized for {case_name}.")

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

            try:
                depth_raw = depth_anything.infer_image(raw_image,
                                                       args.input_size)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

            base_name = os.path.splitext(os.path.basename(filename))[0]

            if smoother:
                smoothed_results = smoother.process_frame(depth_raw)

                for (init_method,
                     scale_method), (depth_smoothed,
                                     confidence) in smoothed_results.items():
                    # New path structure: /outdir/sigma/init/scale/case/
                    final_outdir = os.path.join(args.outdir, sigma_str,
                                                init_method, scale_method,
                                                case_name)
                    os.makedirs(final_outdir, exist_ok=True)

                    if args.save_npz:
                        output_path_npz = os.path.join(final_outdir,
                                                       f'{base_name}.npz')
                        # Save all relevant data for analysis
                        np.savez_compressed(
                            output_path_npz,
                            depth=depth_smoothed,
                            confidence=confidence,
                            frame_count=smoother.frame_count[(init_method,
                                                              scale_method)])
            else:
                final_outdir = os.path.join(args.outdir, "raw", case_name)
                os.makedirs(final_outdir, exist_ok=True)
                if args.save_npz:
                    output_path_npz = os.path.join(final_outdir,
                                                   f'{base_name}.npz')
                    np.savez_compressed(output_path_npz, depth=depth_raw)

        print(f"Finished processing all frames for {case_name}.")
