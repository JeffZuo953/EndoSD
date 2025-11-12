#!/usr/bin/env python3
"""
High-performance cache generator for RGB-Depth-Semantic datasets.

This script replaces the previous ad-hoc implementation and provides:
  * Aspect-ratio preserving resizing with GPU acceleration (via CACHE_RESIZE_DEVICE).
  * Automatic cleanup of stale cache artifacts before regeneration.
  * Batched DataLoader-based prefetch to utilise multiple CPU workers while transforms
    execute on the requested torch device.
  * Strict dtype handling: image=float16, depth=float32, valid_mask=bool, semseg=uint8.
  * Depth statistics (JSON / TXT) emitted alongside cache lists.
  * Rich execution metadata (command, timings, sample counts) per job.

Usage:
    python -m dataset.cache_utils_data --list-jobs
    python -m dataset.cache_utils_data --job stereomis_train --job stereomis_val
    python -m dataset.cache_utils_data --job kvasir_split_train --job kvasir_split_val \
        --job-file configs/cache_jobs.json --num-workers 8 --batch-size 4

By default, job definitions are loaded from ``configs/cache_jobs.json``. Each job entry
defines the dataset constructor arguments, cache root layout, and runtime preferences.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Constants & helpers
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOB_FILE = REPO_ROOT / "configs" / "cache_jobs.json"


def _resolve_path(value: Optional[str | Path]) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(value).expanduser()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_tensor(data: Any) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.detach()
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    return torch.as_tensor(data)


def _to_dtype(data: Any, dtype: torch.dtype) -> torch.Tensor:
    tensor = _as_tensor(data).cpu()
    if dtype == torch.bool:
        if tensor.dtype == torch.bool:
            return tensor.contiguous()
        return tensor.to(torch.bool).contiguous()
    if dtype == torch.uint8:
        if tensor.dtype == torch.uint8:
            return tensor.contiguous()
        return tensor.to(torch.uint8).contiguous()
    return tensor.to(dtype=dtype).contiguous()


def _validate_masks(image: torch.Tensor,
                    depth: Optional[torch.Tensor],
                    valid_mask: Optional[torch.Tensor],
                    max_depth: Optional[float]) -> None:
    if depth is None or valid_mask is None:
        return
    if valid_mask.shape != depth.shape:
        raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} != depth shape {tuple(depth.shape)}")
    if image.shape[-2:] != depth.shape[-2:]:
        raise ValueError(
            f"Image spatial dims {tuple(image.shape[-2:])} do not match depth {tuple(depth.shape)}"
        )
    invalid_depth = valid_mask & (depth <= 0)
    if invalid_depth.any():
        raise ValueError(f"Found {int(invalid_depth.sum())} pixels with valid_mask=True but depth<=0")
    if max_depth is not None:
        outside = valid_mask & (depth > max_depth + 1e-6)
        if outside.any():
            raise ValueError(
                f"Found {int(outside.sum())} pixels with valid_mask=True but depth>max_depth ({max_depth})"
            )


# --------------------------------------------------------------------------- #
# Depth statistics
# --------------------------------------------------------------------------- #

class DepthStatistics:
    """Collect simple global statistics for cached depth maps."""

    def __init__(self) -> None:
        self.depth_values: List[float] = []
        self.valid_pixel_counts: List[int] = []
        self.total_pixel_counts: List[int] = []
        self.min_depths: List[float] = []
        self.max_depths: List[float] = []
        self.mean_depths: List[float] = []
        self.image_paths: List[str] = []

    def add_sample(self, depth: torch.Tensor, image_path: str, valid_mask: Optional[torch.Tensor] = None) -> None:
        depth_np = depth.detach().cpu().numpy()
        if valid_mask is not None:
            mask_np = valid_mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.isfinite(depth_np) & (depth_np > 0)

        total_pixels = depth_np.size
        valid_pixels = int(mask_np.sum())

        self.total_pixel_counts.append(total_pixels)
        self.valid_pixel_counts.append(valid_pixels)
        self.image_paths.append(image_path)

        if valid_pixels == 0:
            self.min_depths.append(0.0)
            self.max_depths.append(0.0)
            self.mean_depths.append(0.0)
            return

        valid_depths = depth_np[mask_np]
        self.depth_values.extend(valid_depths.tolist())
        self.min_depths.append(float(valid_depths.min()))
        self.max_depths.append(float(valid_depths.max()))
        self.mean_depths.append(float(valid_depths.mean()))

    def has_data(self) -> bool:
        return len(self.depth_values) > 0

    def save(self, output_prefix: Path) -> None:
        if not self.has_data():
            print("警告: 深度统计为空，跳过报告生成。")
            return

        depth_array = np.asarray(self.depth_values, dtype=np.float32)
        total_valid = int(sum(self.valid_pixel_counts))
        total_pixels = int(sum(self.total_pixel_counts))

        global_stats = {
            "total_samples": len(self.image_paths),
            "total_pixels": total_pixels,
            "total_valid_pixels": total_valid,
            "valid_pixel_ratio": total_valid / total_pixels if total_pixels else 0.0,
            "global_min_depth": float(depth_array.min()),
            "global_max_depth": float(depth_array.max()),
            "global_mean_depth": float(depth_array.mean()),
            "global_median_depth": float(np.median(depth_array)),
            "global_std_depth": float(depth_array.std()),
        }
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = {f"p{p}": float(np.percentile(depth_array, p)) for p in percentiles}

        per_sample_stats = []
        for idx, path in enumerate(self.image_paths):
            total = self.total_pixel_counts[idx]
            valid = self.valid_pixel_counts[idx]
            per_sample_stats.append({
                "image_path": path,
                "total_pixels": total,
                "valid_pixels": valid,
                "valid_ratio": valid / total if total else 0.0,
                "min_depth": self.min_depths[idx],
                "max_depth": self.max_depths[idx],
                "mean_depth": self.mean_depths[idx],
            })

        report = {
            "global_statistics": global_stats,
            "percentiles": percentile_values,
            "per_sample_statistics": per_sample_stats,
        }

        json_path = output_prefix.with_suffix(".json")
        txt_path = output_prefix.with_suffix(".txt")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("深度数据统计报告\n")
            f.write("=" * 80 + "\n\n")
            f.write("全局统计信息:\n")
            f.write("-" * 80 + "\n")
            f.write(f"总样本数: {global_stats['total_samples']}\n")
            f.write(f"总像素数: {global_stats['total_pixels']:,}\n")
            f.write(f"有效像素数: {global_stats['total_valid_pixels']:,}\n")
            f.write(f"有效像素比例: {global_stats['valid_pixel_ratio']:.2%}\n")
            f.write(f"最小深度值: {global_stats['global_min_depth']:.6f}\n")
            f.write(f"最大深度值: {global_stats['global_max_depth']:.6f}\n")
            f.write(f"平均深度值: {global_stats['global_mean_depth']:.6f}\n")
            f.write(f"中位数深度值: {global_stats['global_median_depth']:.6f}\n")
            f.write(f"深度标准差: {global_stats['global_std_depth']:.6f}\n\n")

            f.write("深度分布百分位数:\n")
            f.write("-" * 80 + "\n")
            for p in percentiles:
                f.write(f"P{p:02d}: {percentile_values[f'p{p}']:.6f}\n")
            f.write("\n前10个样本统计:\n")
            f.write("-" * 80 + "\n")
            for i, stats in enumerate(per_sample_stats[:10]):
                f.write(f"\n样本 {i + 1}: {os.path.basename(stats['image_path'])}\n")
                f.write(f"  有效像素: {stats['valid_pixels']:,} / {stats['total_pixels']:,} ({stats['valid_ratio']:.2%})\n")
                f.write(f"  深度范围: [{stats['min_depth']:.6f}, {stats['max_depth']:.6f}]\n")
                f.write(f"  平均深度: {stats['mean_depth']:.6f}\n")

        print(f"深度统计报告已生成: {json_path} / {txt_path}")


# --------------------------------------------------------------------------- #
# Cache job configuration
# --------------------------------------------------------------------------- #

@dataclass
class CacheJob:
    name: str
    dataset_key: str
    params: Dict[str, Any]
    cache_root_path: Path
    filelist_name: str
    output_dir: Path
    origin_prefix: Optional[Path] = None
    enable_depth_stats: bool = True
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    store_intrinsics_per_sample: bool = True
    intrinsics_map_name: Optional[str] = None
    depth_histogram_mm_max: Optional[int] = None
    depth_histogram_csv_name: Optional[str] = None
    clean: bool = True
    persistent_workers: bool = True
    limit: Optional[int] = None

    def resolve_filelist_path(self) -> Path:
        return self.output_dir / self.filelist_name


def _load_jobs(job_file: Path) -> Dict[str, CacheJob]:
    if not job_file.exists():
        raise FileNotFoundError(f"Job configuration file not found: {job_file}")

    with open(job_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    jobs: Dict[str, CacheJob] = {}
    for name, cfg in data.get("jobs", {}).items():
        dataset_key = cfg["dataset"]
        params = cfg.get("params", {})
        cache_root_path = _resolve_path(cfg["cache_root_path"])
        output_dir = _resolve_path(cfg.get("output_dir")) or cache_root_path.parent
        origin_prefix = _resolve_path(cfg.get("origin_prefix"))
        job = CacheJob(
            name=name,
            dataset_key=dataset_key,
            params=params,
            cache_root_path=cache_root_path,
            filelist_name=cfg.get("filelist_name", f"{name}.txt"),
            output_dir=output_dir,
            origin_prefix=origin_prefix,
            enable_depth_stats=cfg.get("enable_depth_stats", True),
            batch_size=int(cfg.get("batch_size", 1)),
            num_workers=int(cfg.get("num_workers", 0)),
            pin_memory=bool(cfg.get("pin_memory", False)),
            store_intrinsics_per_sample=bool(cfg.get("store_intrinsics_per_sample", True)),
            intrinsics_map_name=cfg.get("intrinsics_map_name"),
            depth_histogram_mm_max=int(cfg["depth_histogram_mm_max"]) if "depth_histogram_mm_max" in cfg and cfg["depth_histogram_mm_max"] is not None else None,
            depth_histogram_csv_name=cfg.get("depth_histogram_csv_name"),
            clean=bool(cfg.get("clean", True)),
            persistent_workers=bool(cfg.get("persistent_workers", True)),
            limit=cfg.get("limit"),
        )
        jobs[name] = job
    return jobs


# --------------------------------------------------------------------------- #
# Dataset factory
# --------------------------------------------------------------------------- #

def _build_dataset(dataset_key: str, params: Dict[str, Any]) -> Dataset:
    key = dataset_key.lower()

    if key == "stereo_mis":
        from .stereo_mis import StereoMISDataset
        dataset = StereoMISDataset(**params)
        return dataset
    if key == "endovis2017":
        from .endovis2017_dataset import EndoVis2017Dataset
        dataset = EndoVis2017Dataset(**params)
        return dataset
    if key == "endovis2018":
        from .endovis2018_dataset import EndoVis2018Dataset
        dataset = EndoVis2018Dataset(**params)
        return dataset
    if key == "endonerf":
        from .endonerf_dataset import EndoNeRFDataset
        dataset = EndoNeRFDataset(**params)
        return dataset
    if key == "endosynth":
        from .endo_synth import EndoSynth
        dataset = EndoSynth(**params)
        return dataset
    if key == "endomapper":
        from .endomapper import Endomapper
        dataset = Endomapper(**params)
        return dataset
    if key == "hamlyn":
        from .hamlyn import HamlynDataset
        dataset = HamlynDataset(**params)
        return dataset
    if key == "dvpn":
        from .dvpn_dataset import DVPNDataset
        dataset = DVPNDataset(**params)
        return dataset
    if key == "filelist_seg_depth":
        from .filelist_seg_depth import FileListSegDepthDataset
        dataset = FileListSegDepthDataset(**params)
        return dataset
    if key == "kvasir_seg":
        from .kvasir_seg_dataset import KvasirSegDataset
        dataset = KvasirSegDataset(**params)
        return dataset
    if key == "bkai_polyp":
        from .bkai_polyp_dataset import BKAIPolypDataset
        dataset = BKAIPolypDataset(**params)
        return dataset
    if key == "clinicdb":
        from .cvc_clinicdb_dataset import CVCClinicDBDataset
        dataset = CVCClinicDBDataset(**params)
        return dataset
    if key == "cvc_endoscene":
        from .cvc_endoscene_still_dataset import CVCEndoSceneStillDataset
        dataset = CVCEndoSceneStillDataset(**params)
        return dataset
    if key == "etis_larib":
        from .etis_larib_dataset import ETISLaribDataset
        dataset = ETISLaribDataset(**params)
        return dataset
    if key == "inhouse_seg_depth":
        from .inhouse_seg_depth import InHouse
        dataset = InHouse(**params)
        return dataset
    if key == "serv_ct":
        from .serv_ct_dataset import ServCTDataset
        dataset = ServCTDataset(**params)
        return dataset
    if key == "synthetic_polyp":
        from .synthetic_polyp_dataset import SyntheticPolypDataset
        dataset = SyntheticPolypDataset(**params)
        return dataset

    raise KeyError(f"Unsupported dataset key: {dataset_key}")


# --------------------------------------------------------------------------- #
# Cache generation logic
# --------------------------------------------------------------------------- #

def _clean_cache_root(cache_root: Path, filelist_path: Path) -> None:
    if cache_root.exists():
        pt_files = list(cache_root.rglob("*.pt"))
        for path in pt_files:
            path.unlink()
        # remove empty directories (bottom-up)
        for directory in sorted({p.parent for p in pt_files}, key=lambda p: len(str(p)), reverse=True):
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
    else:
        cache_root.mkdir(parents=True, exist_ok=True)

    if filelist_path.exists():
        filelist_path.unlink()

    # Remove previous reports
    for report in filelist_path.parent.glob("depth_statistics_report*"):
        if report.is_file():
            report.unlink()


def _prepare_cache_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    image = item.get("image")
    if image is None:
        raise KeyError("Sample is missing 'image' key.")
    payload["image"] = _to_dtype(image, torch.float16)

    depth_tensor: Optional[torch.Tensor] = None
    if "depth" in item:
        depth_tensor = _to_dtype(item["depth"], torch.float32)
        payload["depth"] = depth_tensor

    if "depth_mask" in item:
        payload["depth_mask"] = _to_dtype(item["depth_mask"], torch.bool)

    if "valid_mask" in item:
        payload["valid_mask"] = _to_dtype(item["valid_mask"], torch.bool)
    elif depth_tensor is not None:
        payload["valid_mask"] = depth_tensor > 0

    if "semseg_mask" in item:
        payload["semseg_mask"] = _to_dtype(item["semseg_mask"], torch.uint8)

    if "depth_valid_mask" in item:
        payload["depth_valid_mask"] = _to_dtype(item["depth_valid_mask"], torch.bool)

    if "seg_valid_mask" in item:
        payload["seg_valid_mask"] = _to_dtype(item["seg_valid_mask"], torch.bool)

    if depth_tensor is not None and "valid_mask" in payload:
        depth_clean = depth_tensor.clone()
        depth_clean[~payload["valid_mask"]] = 0.0
        payload["depth"] = depth_clean

    for key in (
        "image_path",
        "depth_path",
        "mask_path",
        "source_type",
        "dataset_name",
        "frame_token",
        "sequence",
        "sequence_path",
        "intrinsics_key",
    ):
        if key in item:
            payload[key] = item[key]

    if "max_depth" in item and item["max_depth"] is not None:
        payload["max_depth"] = float(item["max_depth"])

    if "intrinsics" in item:
        payload["intrinsics"] = _to_dtype(item["intrinsics"], torch.float32)

    return payload


def _resolve_cache_path(original_path: str,
                        origin_prefix: Optional[Path],
                        cache_root_path: Path) -> Path:
    src_path = Path(original_path)
    if origin_prefix and src_path.is_absolute():
        try:
            rel_path = src_path.relative_to(origin_prefix)
            return cache_root_path / rel_path.with_suffix(".pt")
        except ValueError:
            pass
    return cache_root_path / src_path.name.replace(src_path.suffix, ".pt")


def _passthrough_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def _iter_batches(loader: DataLoader) -> Iterator[List[Dict[str, Any]]]:
    for batch in loader:
        if isinstance(batch, list):
            yield batch
        else:
            yield [batch]


def run_job(job: CacheJob,
            batch_size_override: Optional[int] = None,
            num_workers_override: Optional[int] = None,
            limit_override: Optional[int] = None,
            start_index: Optional[int] = None,
            end_index: Optional[int] = None,
            append_mode: bool = False,
            dry_run: bool = False) -> None:
    dataset = _build_dataset(job.dataset_key, job.params)

    limit = limit_override if limit_override is not None else job.limit
    if limit is not None:
        if hasattr(dataset, "limit"):
            dataset.limit(limit)
        elif hasattr(dataset, "samples"):
            dataset.samples = dataset.samples[:limit]
        else:
            raise AttributeError(f"Dataset '{job.dataset_key}' does not support limiting samples.")

    total_len = len(dataset)
    start = start_index or 0
    end = end_index if end_index is not None else total_len
    if start < 0 or start > total_len:
        raise ValueError(f"start_index {start} out of range for dataset length {total_len}")
    if end < start or end > total_len:
        raise ValueError(f"end_index {end} out of range for dataset length {total_len}")
    if start != 0 or end != total_len:
        indices = list(range(start, end))
        dataset = Subset(dataset, indices)

    batch_size = batch_size_override or job.batch_size
    num_workers = num_workers_override if num_workers_override is not None else job.num_workers
    persistent_workers = job.persistent_workers and num_workers > 0

    multiprocessing_context = mp.get_context("spawn") if num_workers > 0 else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=job.pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=_passthrough_collate,
        multiprocessing_context=multiprocessing_context,
    )

    filelist_path = job.resolve_filelist_path()
    _ensure_dir(job.output_dir)
    _ensure_dir(job.cache_root_path)

    if job.clean and not append_mode and not dry_run:
        print(f"[{job.name}] Cleaning previous cache under {job.cache_root_path}")
        _clean_cache_root(job.cache_root_path, filelist_path)

    command = " ".join(sys.argv)
    start_ts = datetime.now()
    start_time = time.time()

    cache_paths: List[Path] = []
    compute_depth_stats = job.enable_depth_stats and not append_mode
    depth_stats = DepthStatistics() if compute_depth_stats else None
    intrinsics_map: Dict[str, torch.Tensor] = {}
    hist_max = job.depth_histogram_mm_max
    histogram_counts = torch.zeros(int(hist_max) + 1, dtype=torch.int64) if hist_max is not None else None
    if job.intrinsics_map_name and append_mode:
        existing_map_path = job.output_dir / job.intrinsics_map_name
        if existing_map_path.exists():
            loaded_map = torch.load(existing_map_path)
            if isinstance(loaded_map, dict):
                intrinsics_map = {
                    str(key): _to_dtype(val, torch.float32).detach().cpu()
                    for key, val in loaded_map.items()
                }
    if histogram_counts is not None and append_mode:
        csv_name = job.depth_histogram_csv_name or f"{job.name}_depth_histogram_mm.csv"
        csv_path = job.output_dir / csv_name
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) < 2:
                        continue
                    try:
                        mm = int(row[0])
                        count = int(row[1])
                    except ValueError:
                        continue
                    if 0 <= mm < histogram_counts.numel():
                        histogram_counts[mm] = count
    processed = 0
    skipped = 0

    progress = tqdm(total=len(dataset), desc=f"{job.name}", unit="sample")
    for batch in _iter_batches(loader):
        for item in batch:
            if item is None:
                skipped += 1
                progress.update(1)
                continue

            original_img_path = item.get("image_path")
            if not original_img_path:
                skipped += 1
                progress.update(1)
                continue

            cache_path = _resolve_cache_path(original_img_path, job.origin_prefix, job.cache_root_path)
            payload = _prepare_cache_payload(item)
            image_tensor = payload["image"]
            depth_tensor = payload.get("depth")
            valid_mask_tensor = payload.get("valid_mask")
            max_depth = payload.get("max_depth")

            intr_key = item.get("intrinsics_key") or item.get("sequence") or payload.get("sequence")
            intrinsics_tensor = payload.get("intrinsics")
            if intrinsics_tensor is None and "intrinsics" in item:
                intrinsics_tensor = _to_dtype(item["intrinsics"], torch.float32)

            if intr_key:
                payload["intrinsics_key"] = intr_key

            if intrinsics_tensor is not None:
                intr_cpu = _to_dtype(intrinsics_tensor, torch.float32).detach().cpu()
                if intr_key and intr_key not in intrinsics_map:
                    intrinsics_map[intr_key] = intr_cpu
                if not job.store_intrinsics_per_sample and "intrinsics" in payload:
                    del payload["intrinsics"]
            else:
                if not job.store_intrinsics_per_sample and "intrinsics" in payload:
                    del payload["intrinsics"]

            _validate_masks(image_tensor, depth_tensor, valid_mask_tensor, max_depth)

            if histogram_counts is not None and depth_tensor is not None and valid_mask_tensor is not None:
                valid_depths = depth_tensor[valid_mask_tensor]
                if valid_depths.numel() > 0:
                    depth_mm = (valid_depths * 1000.0).round().to(torch.int64)
                    depth_mm = torch.clamp(depth_mm, min=0, max=int(hist_max))
                    bincount = torch.bincount(depth_mm, minlength=histogram_counts.numel())
                    histogram_counts += bincount

            if depth_stats is not None and depth_tensor is not None:
                depth_stats.add_sample(depth_tensor, original_img_path, valid_mask_tensor)

            if not dry_run:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, cache_path)
                cache_paths.append(cache_path)

            processed += 1
            progress.update(1)
    progress.close()

    duration = time.time() - start_time
    end_ts = datetime.now()

    if dry_run:
        print(f"[{job.name}] Dry run complete. Processed {processed} samples (skipped {skipped}).")
        return

    cache_paths_sorted = sorted(cache_paths, key=lambda p: str(p))
    mode = "a" if append_mode and filelist_path.exists() else "w"
    with open(filelist_path, mode, encoding="utf-8") as f:
        for path in cache_paths_sorted:
            f.write(str(path) + "\n")

    print(f"[{job.name}] Wrote cache file list: {filelist_path}")

    if depth_stats is not None:
        depth_stats_prefix = job.output_dir / "depth_statistics_report"
        depth_stats.save(depth_stats_prefix)

    intrinsics_map_path: Optional[Path] = None
    if intrinsics_map and job.intrinsics_map_name:
        intrinsics_map_path = job.output_dir / job.intrinsics_map_name
        _ensure_dir(intrinsics_map_path.parent)
        torch.save(intrinsics_map, intrinsics_map_path)
        json_path = intrinsics_map_path.with_suffix(".json")
        serializable = {key: tensor.cpu().tolist() for key, tensor in intrinsics_map.items()}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"[{job.name}] Intrinsics map saved to {intrinsics_map_path} (JSON: {json_path})")

    histogram_csv_path: Optional[Path] = None
    if histogram_counts is not None:
        csv_name = job.depth_histogram_csv_name or f"{job.name}_depth_histogram_mm.csv"
        histogram_csv_path = job.output_dir / csv_name
        _ensure_dir(histogram_csv_path.parent)
        with open(histogram_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["millimeter", "count"])
            for mm, count in enumerate(histogram_counts.tolist()):
                writer.writerow([mm, int(count)])
        print(f"[{job.name}] Depth histogram saved to {histogram_csv_path}")

    meta_suffix = ""
    if start != 0 or end != total_len:
        meta_suffix = f"_s{start}_e{end}"

    metadata = {
        "job": job.name,
        "dataset": job.dataset_key,
        "cache_root_path": str(job.cache_root_path),
        "filelist_path": str(filelist_path),
        "origin_prefix": str(job.origin_prefix) if job.origin_prefix else None,
        "command": command,
        "start_time": start_ts.isoformat(),
        "end_time": end_ts.isoformat(),
        "duration_sec": duration,
        "processed_samples": processed,
        "skipped_samples": skipped,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "start_index": start,
        "end_index": end,
        "append_mode": append_mode,
    }

    if intrinsics_map_path:
        metadata["intrinsics_map_path"] = str(intrinsics_map_path)
        metadata["intrinsics_keys"] = sorted(intrinsics_map.keys())
    if histogram_csv_path:
        metadata["depth_histogram_csv"] = str(histogram_csv_path)
        metadata["depth_histogram_total_valid_pixels"] = int(histogram_counts.sum().item()) if histogram_counts is not None else 0
        if hist_max is not None:
            metadata["depth_histogram_mm_max"] = int(hist_max)

    meta_path = job.output_dir / f"{job.name}_cache_meta{meta_suffix}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[{job.name}] 完成缓存生成，共 {processed} 个样本，用时 {duration:.1f}s。元数据写入 {meta_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate .pt cache files for datasets.")
    parser.add_argument("--job-file", type=str, default=str(DEFAULT_JOB_FILE), help="Path to cache job configuration JSON.")
    parser.add_argument("--job", action="append", dest="jobs", help="Job name(s) to execute. Repeatable.")
    parser.add_argument("--list-jobs", action="store_true", help="List available jobs and exit.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for all jobs.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override worker count for all jobs.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per dataset.")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean existing cache files before generation.")
    parser.add_argument("--start-index", type=int, default=None, help="Optional start index (inclusive) for processing subset.")
    parser.add_argument("--end-index", type=int, default=None, help="Optional end index (exclusive) for processing subset.")
    parser.add_argument("--append", action="store_true", help="Append cache entries and metadata instead of overwriting.")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing any cache files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_file = _resolve_path(args.job_file)
    jobs = _load_jobs(job_file)

    if args.list_jobs:
        print("可用的缓存任务:")
        for name, job in jobs.items():
            print(f"  - {name}: dataset={job.dataset_key}, cache_root={job.cache_root_path}")
        return

    if not args.jobs:
        raise ValueError("No jobs specified. Use --job <name> or --list-jobs.")

    requested_jobs: List[CacheJob] = []
    for job_name in args.jobs:
        if job_name not in jobs:
            raise KeyError(f"Job '{job_name}' not found in {job_file}.")
        job = jobs[job_name]
        if args.no_clean:
            job.clean = False
        if args.append:
            job.clean = False
        requested_jobs.append(job)

    for job in requested_jobs:
        run_job(
            job,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            limit_override=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
            append_mode=args.append,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
