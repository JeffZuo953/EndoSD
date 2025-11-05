from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import NormalizeImage, PrepareForNet, Resize
from .utils import compute_valid_mask


@dataclass(frozen=True)
class _StereoMISSample:
    image_path: Path
    depth_path: Path
    sequence: str
    data_format: str  # "disparity_png" or "depth_npy"
    mask_path: Optional[Path] = None
    validity_mask_path: Optional[Path] = None
    depth_scale: float = 1.0


class StereoMISDataset(Dataset):
    """
    StereoMIS 数据集加载器，支持原始的 rectified disparity 目录结构，
    也兼容 LS 数据中按序列(`P1`, `P2_*`, `P3`)存放的 `video_frames` 与
    `depth_npy` 布局。若找不到语义分割标注，则自动生成忽略掩码(全 255)。
    """

    DEFAULT_SEQUENCE_SPLITS: Dict[str, Sequence[str]] = {
        "train": (
            "P2_1",
            "P2_2",
            "P2_3",
            "P2_4",
            "P2_5",
            "P2_8",
            "P3",
        ),
        "val": (
            "P1",
            "P2_0",
            "P2_6",
            "P2_7",
        ),
        "test": (
            "P1",
            "P2_0",
            "P2_6",
            "P2_7",
        ),
    }
    SA_DEPTH_SCALE: float = 1.0 / 1000.0

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        size: Tuple[int, int] = (518, 518),
        max_depth: float = 0.3,
        disparity_scale: float = 256.0,
        calibration_path: str | Path | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.split = split.lower()
        self.size = size
        self.max_depth = max_depth
        self.disparity_scale = disparity_scale
        self._explicit_calibration_path: Optional[Path] = (
            Path(calibration_path).expanduser() if calibration_path is not None else None
        )

        self._samples: List[_StereoMISSample] = self._collect_samples()
        if not self._samples:
            raise RuntimeError(f"StereoMIS 数据集在 split='{split}' 下未找到任何可用样本。")

        # 与缓存脚本保持兼容
        self.file_list: List[Path] = [sample.image_path for sample in self._samples]

        net_w, net_h = size
        self.transform = Compose(
            [
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
            ]
        )

        self._sequence_intrinsics: Dict[str, torch.Tensor] = {}
        self._default_intrinsics: Optional[torch.Tensor] = None
        self._load_intrinsics_information()

    # --------------------------------------------------------------------- #
    # Dataset protocol
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self._samples[idx]

        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {sample.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        semseg_mask = None
        provided_valid_mask = None

        if sample.data_format == "disparity_png":
            depth = self._read_depth_from_disparity(sample.depth_path)
            if sample.mask_path is not None:
                semseg_mask = self._read_mask(sample.mask_path)
        elif sample.data_format == "depth_npy":
            depth = self._read_depth_from_npy(sample.depth_path, scale=sample.depth_scale)
        else:
            raise ValueError(f"未知的数据格式: {sample.data_format}")

        if sample.validity_mask_path is not None:
            provided_valid_mask = self._read_validity_mask(sample.validity_mask_path)

        sample_dict = {
            "image": image,
            "depth": depth,
        }
        if semseg_mask is not None:
            sample_dict["semseg_mask"] = semseg_mask
        if provided_valid_mask is not None:
            sample_dict["valid_mask"] = provided_valid_mask

        transformed = self.transform(sample_dict)

        image_tensor = torch.from_numpy(transformed["image"])
        depth_tensor = torch.from_numpy(transformed["depth"])
        valid_mask = compute_valid_mask(
            image_tensor,
            depth_tensor,
            max_depth=self.max_depth,
            dataset_name="StereoMIS",
        )

        provided_mask_tensor: Optional[torch.Tensor] = None
        if "valid_mask" in transformed:
            provided_mask_tensor = torch.from_numpy(transformed["valid_mask"]).bool()
            valid_mask = valid_mask & provided_mask_tensor

        result: Dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": valid_mask,
            "image_path": str(sample.image_path),
            "depth_path": str(sample.depth_path),
            "max_depth": self.max_depth,
            "source_type": "LS",
        }

        if sample.mask_path is not None:
            mask_tensor = torch.from_numpy(transformed["semseg_mask"]).long()
            result["semseg_mask"] = mask_tensor
            result["mask_path"] = str(sample.mask_path)

        intrinsics_tensor = self._sequence_intrinsics.get(sample.sequence)
        if intrinsics_tensor is None:
            intrinsics_tensor = self._default_intrinsics
        if intrinsics_tensor is not None:
            result["intrinsics"] = intrinsics_tensor

        return result

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    def limit(self, count: Optional[int]) -> None:
        """限制样本数量，主要用于快速调试或缓存脚本。"""
        if count is None:
            return
        if count < 0:
            raise ValueError("count 必须为非负整数。")
        self._samples = self._samples[:count]
        self.file_list = self.file_list[:count]

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _load_intrinsics_information(self) -> None:
        """加载每个序列或全局的相机内参。"""
        sequence_intrinsics: Dict[str, torch.Tensor] = {}
        for calib_txt in self.root_dir.glob("**/camera_calibration.txt"):
            seq_name = calib_txt.parent.name
            intrinsics = self._parse_camera_calibration_txt(calib_txt)
            if intrinsics is not None:
                sequence_intrinsics[seq_name] = intrinsics

        self._sequence_intrinsics = sequence_intrinsics
        if self._sequence_intrinsics:
            return

        calibration_ini = self._resolve_calibration_ini()
        if calibration_ini is None:
            return

        default_intrinsics = self._parse_stereo_calibration_ini(calibration_ini)
        if default_intrinsics is not None:
            self._default_intrinsics = default_intrinsics

    def _resolve_calibration_ini(self) -> Optional[Path]:
        candidates: List[Path] = []
        if self._explicit_calibration_path is not None:
            candidates.append(self._explicit_calibration_path)

        candidate_names = ("StereoCalibration.ini", "stereo_calibration.ini", "stereocalibration.ini")
        dataset_root = Path(__file__).resolve().parent
        for name in candidate_names:
            candidates.extend(
                [
                    self.root_dir / name,
                    self.root_dir.parent / name,
                    dataset_root / name,
                    dataset_root.parent / name,
                ]
            )

        seen: set[Path] = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _parse_camera_calibration_txt(calib_path: Path) -> Optional[torch.Tensor]:
        fx = fy = cx = cy = None
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("//"):
                        continue
                    if line.startswith("Camera-0-F:"):
                        parts = line.split(":", 1)[1].split("//")[0].strip().split()
                        if len(parts) >= 2:
                            fx, fy = map(float, parts[:2])
                    elif line.startswith("Camera-0-C:"):
                        parts = line.split(":", 1)[1].split("//")[0].strip().split()
                        if len(parts) >= 2:
                            cx, cy = map(float, parts[:2])
                    if fx is not None and fy is not None and cx is not None and cy is not None:
                        break
        except OSError:
            return None

        if None in (fx, fy, cx, cy):
            return None

        return torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _parse_stereo_calibration_ini(calibration_ini: Path) -> Optional[torch.Tensor]:
        parser = ConfigParser()
        try:
            with calibration_ini.open("r", encoding="utf-8") as f:
                parser.read_file(f)
        except OSError:
            return None

        section = None
        if parser.has_section("StereoLeft"):
            section = parser["StereoLeft"]
        elif parser.sections():
            section = parser[parser.sections()[0]]

        if section is None:
            return None

        try:
            fx = float(section.get("fc_x"))
            fy = float(section.get("fc_y"))
            cx = float(section.get("cc_x"))
            cy = float(section.get("cc_y"))
        except (TypeError, ValueError):
            return None

        return torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

    def _collect_samples(self) -> List[_StereoMISSample]:
        depth_npy_root = self.root_dir / "depth_npy"
        if depth_npy_root.exists():
            try:
                return self._collect_sequence_layout(depth_npy_root)
            except FileNotFoundError:
                pass
            except RuntimeError:
                pass

        try:
            return self._collect_rectified_left_layout()
        except FileNotFoundError:
            pass
        except RuntimeError:
            pass

        return self._collect_legacy_layout()

    def _collect_sequence_layout(self, depth_root: Path) -> List[_StereoMISSample]:
        available_sequences = self._discover_sequences(depth_root)
        if not available_sequences:
            raise FileNotFoundError(f"在 {depth_root} 下未找到任何有效序列目录。")

        if self.split == "all":
            target_sequences = available_sequences
        else:
            preset = self.DEFAULT_SEQUENCE_SPLITS.get(self.split)
            if preset is None:
                raise ValueError(
                    f"未知的 split '{self.split}'。支持: {list(self.DEFAULT_SEQUENCE_SPLITS.keys()) + ['all']}"
                )
            target_sequences = [seq for seq in preset if seq in available_sequences]
            if not target_sequences:
                raise RuntimeError(f"split='{self.split}' 对应的序列在数据中不存在。")

        samples: List[_StereoMISSample] = []
        missing_depth = 0

        for seq_name in target_sequences:
            rgb_dir = self.root_dir / seq_name / "video_frames"
            depth_dir = depth_root / seq_name / "video_frames"

            if not rgb_dir.is_dir():
                raise FileNotFoundError(f"缺少图像目录: {rgb_dir}")
            if not depth_dir.is_dir():
                raise FileNotFoundError(f"缺少深度目录: {depth_dir}")

            for left_path in sorted(rgb_dir.glob("*l.png")):
                frame_stem = left_path.stem[:-1] if left_path.stem.endswith("l") else left_path.stem
                depth_path = depth_dir / f"{frame_stem}_depth.npy"
                if not depth_path.exists():
                    missing_depth += 1
                    continue
                samples.append(
                    _StereoMISSample(
                        image_path=left_path,
                        depth_path=depth_path,
                        sequence=seq_name,
                        data_format="depth_npy",
                    )
                )

        if not samples:
            raise RuntimeError(f"在序列 {target_sequences} 中未找到匹配的图像与深度对。")

        if missing_depth > 0:
            print(f"警告: 在 StereoMIS {self.split} split 中有 {missing_depth} 个样本缺少深度文件，已被跳过。")

        return samples

    def _collect_rectified_left_layout(self) -> List[_StereoMISSample]:
        sequence_dirs = self._discover_rectified_sequences()
        if not sequence_dirs:
            raise FileNotFoundError(f"在 {self.root_dir} 下未找到包含 rectified_left/pred_rectified_depth 的序列目录。")

        if self.split == "all":
            target_sequences = sorted(sequence_dirs.keys())
        else:
            preset = self.DEFAULT_SEQUENCE_SPLITS.get(self.split)
            if preset is None:
                raise ValueError(
                    f"未知的 split '{self.split}'。支持: {list(self.DEFAULT_SEQUENCE_SPLITS.keys()) + ['all']}"
                )
            target_sequences = [seq for seq in preset if seq in sequence_dirs]
            if not target_sequences:
                raise RuntimeError(f"split='{self.split}' 对应的序列在 rectified_left 布局中不存在。")

        samples: List[_StereoMISSample] = []
        missing_depth = 0
        missing_mask = 0

        for seq_name in target_sequences:
            seq_dir = sequence_dirs[seq_name]
            rgb_dir = seq_dir / "rectified_left"
            depth_dir = seq_dir / "pred_rectified_depth"

            if not rgb_dir.is_dir():
                raise FileNotFoundError(f"缺少 rectified_left 目录: {rgb_dir}")
            if not depth_dir.is_dir():
                raise FileNotFoundError(f"缺少 pred_rectified_depth 目录: {depth_dir}")

            frame_paths = self._iter_sa_rgb_frames(rgb_dir)
            for image_path in frame_paths:
                frame_stem = image_path.stem
                depth_path = self._resolve_sa_depth_path(depth_dir, frame_stem)
                if depth_path is None:
                    missing_depth += 1
                    continue

                validity_mask_path = self._resolve_sa_mask_path(depth_dir, frame_stem)
                if validity_mask_path is None:
                    missing_mask += 1

                samples.append(
                    _StereoMISSample(
                        image_path=image_path,
                        depth_path=depth_path,
                        sequence=seq_name,
                        data_format="depth_npy",
                        validity_mask_path=validity_mask_path,
                        depth_scale=self.SA_DEPTH_SCALE,
                    )
                )

        if not samples:
            raise RuntimeError(f"在序列 {target_sequences} 中未找到 rectified_left 样本。")

        if missing_depth > 0:
            print(f"警告: 在 StereoMIS rectified_left 布局中有 {missing_depth} 个样本缺少深度文件，已被跳过。")
        if missing_mask > 0:
            print(f"警告: 在 StereoMIS rectified_left 布局中有 {missing_mask} 个样本缺少 valid mask，将依赖自动 compute_valid_mask。")

        return samples

    def _collect_legacy_layout(self) -> List[_StereoMISSample]:
        left_dir = self.root_dir / self.split / "Left_rectified"
        disparity_dir = self.root_dir / self.split / "Disparity"
        mask_dir = self.root_dir / self.split / "Masks"

        if not left_dir.exists():
            raise FileNotFoundError(f"StereoMIS 左目目录不存在: {left_dir}")
        if not disparity_dir.exists():
            raise FileNotFoundError(f"StereoMIS disparity 目录不存在: {disparity_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"StereoMIS mask 目录不存在: {mask_dir}")

        samples: List[_StereoMISSample] = []
        for left_path in sorted(left_dir.glob("*.png")):
            disparity_path = disparity_dir / left_path.name
            mask_path = mask_dir / left_path.name
            if not disparity_path.exists() or not mask_path.exists():
                continue
            samples.append(
                _StereoMISSample(
                    image_path=left_path,
                    depth_path=disparity_path,
                    sequence=self.split,
                    data_format="disparity_png",
                    mask_path=mask_path,
                )
            )

        if not samples:
            raise RuntimeError(f"在 {left_dir} 中未找到任何有效图像。")
        return samples

    def _discover_sequences(self, depth_root: Path) -> List[str]:
        sequences: List[str] = []
        for depth_seq_dir in depth_root.iterdir():
            if not depth_seq_dir.is_dir():
                continue
            seq_name = depth_seq_dir.name
            rgb_dir = self.root_dir / seq_name / "video_frames"
            if rgb_dir.is_dir():
                sequences.append(seq_name)
        return sorted(sequences)

    def _discover_rectified_sequences(self) -> Dict[str, Path]:
        candidate_roots = [self.root_dir]
        for sub in ("train", "val", "test"):
            candidate = self.root_dir / sub
            if candidate.is_dir():
                candidate_roots.append(candidate)

        sequences: Dict[str, Path] = {}
        for base_dir in candidate_roots:
            for seq_dir in base_dir.iterdir():
                if not seq_dir.is_dir():
                    continue
                rgb_dir = seq_dir / "rectified_left"
                depth_dir = seq_dir / "pred_rectified_depth"
                if rgb_dir.is_dir() and depth_dir.is_dir():
                    sequences[seq_dir.name] = seq_dir
        return sequences

    @staticmethod
    def _iter_sa_rgb_frames(rgb_dir: Path) -> List[Path]:
        frames: List[Path] = []
        patterns = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.bmp", "*.BMP")
        for pattern in patterns:
            frames.extend(rgb_dir.glob(pattern))
        return sorted(frames)

    @staticmethod
    def _resolve_sa_depth_path(depth_dir: Path, frame_stem: str) -> Optional[Path]:
        candidates = [
            depth_dir / f"{frame_stem}.npz.npy",
            depth_dir / f"{frame_stem}.npy",
            depth_dir / f"{frame_stem}.npz",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _resolve_sa_mask_path(depth_dir: Path, frame_stem: str) -> Optional[Path]:
        candidates = [
            depth_dir / f"{frame_stem}_mask.png",
            depth_dir / f"{frame_stem}_mask.npy",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _read_depth_from_disparity(self, disparity_path: Path) -> np.ndarray:
        disparity_raw = cv2.imread(str(disparity_path), cv2.IMREAD_UNCHANGED)
        if disparity_raw is None:
            raise FileNotFoundError(f"无法读取 disparity: {disparity_path}")
        disparity = disparity_raw.astype(np.float32) / self.disparity_scale
        disparity = np.clip(disparity, a_min=1e-6, a_max=None)
        depth = 1.0 / disparity
        depth = np.clip(depth, 0.0, self.max_depth)
        return depth

    def _read_depth_from_npy(self, depth_path: Path, scale: float = 1.0) -> np.ndarray:
        if not depth_path.exists():
            raise FileNotFoundError(f"深度文件不存在: {depth_path}")
        depth = np.load(str(depth_path)).astype(np.float32)
        if scale != 1.0:
            depth = depth * scale
        depth = np.clip(depth, 0.0, self.max_depth)
        return depth

    def _read_mask(self, mask_path: Optional[Path]) -> np.ndarray:
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            raise FileNotFoundError(f"无法读取分割掩码: {mask_path}")
        if mask_img.ndim == 3:
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        return mask_img.astype(np.uint8)

    def _read_validity_mask(self, mask_path: Path) -> np.ndarray:
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            raise FileNotFoundError(f"无法读取 valid mask: {mask_path}")
        if mask_img.ndim == 3:
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        return (mask_img > 0).astype(np.bool_)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="生成 StereoMIS 数据集缓存。")
    parser.add_argument("--root", type=Path, required=True, help="StereoMIS 数据集根目录")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="需要处理的数据划分",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="缓存列表(.txt)的输出目录")
    parser.add_argument("--filelist-name", type=str, default=None, help="缓存列表文件名")
    parser.add_argument("--cache-root", type=Path, default=None, help="缓存 .pt 文件的目标根目录")
    parser.add_argument("--max-depth", type=float, default=0.3, help="深度截断最大值 (米)")
    parser.add_argument("--disparity-scale", type=float, default=256.0, help="旧版 disparity PNG 的缩放因子")
    parser.add_argument("--limit", type=int, default=None, help="限制样本数，便于快速调试")
    return parser.parse_args()


def _main():
    from dataset.cache_utils import generate_dataset_cache

    args = _parse_args()
    dataset = StereoMISDataset(
        root_dir=str(args.root),
        split=args.split,
        max_depth=args.max_depth,
        disparity_scale=args.disparity_scale,
    )

    dataset.limit(args.limit)

    filelist_name = args.filelist_name if args.filelist_name else f"{args.split}_cache.txt"
    cache_root = args.cache_root if args.cache_root is not None else args.output_dir

    generate_dataset_cache(
        dataset=dataset,
        output_dir=str(args.output_dir),
        filelist_name=filelist_name,
        origin_prefix=str(args.root),
        cache_root_path=str(cache_root),
    )


if __name__ == "__main__":
    _main()
