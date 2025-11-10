"""预加载数据以填充 local cache。
运行方式示例：
  LOCAL_CACHE_DIR=/data/ziyi/cache \
  python warm_local_cache.py --dataset-config-name fd_depth_fm_v1 --dataset-modality fd --save-path /tmp/dryrun
"""
import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PARENT = REPO_ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from multitask_moe_lora.dataset.cache_utils import DepthCacheDataset, SegCacheDataset
from multitask_moe_lora.util.config import create_parser, args_to_config, validate_config
from multitask_moe_lora.util.data_utils import setup_dataloaders
from torch.utils.data import Sampler, ConcatDataset, Subset
import torch.utils.data.distributed as torch_dist
import random


def parse_args():
    parser = create_parser()
    parser.add_argument("--warm-steps", type=int, default=None,
                        help="每个 loader 预取多少 batch（默认全量）")
    parser.add_argument("--skip-seg", action="store_true",
                        help="只加载深度数据，跳过分割 loader")
    parser.add_argument("--skip-depth", action="store_true",
                        help="只加载分割数据，跳过深度 loader")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="打印进度的 batch 间隔")
    parser.add_argument("--no-save-path", action="store_true",
                        help="允许 save_path 为空（不会真正训练)")
    args = parser.parse_args()
    if args.no_save_path:
        args.save_path = args.save_path or "warmup"
    return args


def _iter_cache_datasets(dataset):
    stack = [dataset]
    seen = set()
    while stack:
        current = stack.pop()
        if current is None:
            continue
        if isinstance(current, (DepthCacheDataset, SegCacheDataset)):
            identifier = id(current)
            if identifier in seen:
                continue
            seen.add(identifier)
            yield current
            continue
        if isinstance(current, ConcatDataset):
            stack.extend(current.datasets)
            continue
        if isinstance(current, Subset):
            stack.append(current.dataset)
            continue
        nested = getattr(current, 'datasets', None)
        if isinstance(nested, (list, tuple)):
            stack.extend(nested)


def _materialize_filelist(dataset):
    manager = getattr(dataset, "_local_cache", None)
    if manager is None or not manager.enabled:
        return None
    target = manager.filelist_mirror_path(dataset.original_filelist_path)
    if not target:
        return None
    local_entries = []
    missing = 0
    for source_path in dataset.filelist:
        local_path = manager.expected_mirror_path(source_path)
        if local_path and os.path.exists(local_path):
            local_entries.append(local_path)
        else:
            missing += 1
    if not local_entries:
        return None
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp_path = f"{target}.tmp"
    with open(tmp_path, "w") as f:
        f.write("\n".join(local_entries) + "\n")
    os.replace(tmp_path, target)
    return target, missing


def _emit_local_filelists(loaders):
    generated = []
    for loader in loaders:
        if loader is None:
            continue
        for dataset in _iter_cache_datasets(loader.dataset):
            result = _materialize_filelist(dataset)
            if result:
                path, missing = result
                generated.append((dataset.dataset_name, path, missing))
    if not generated:
        print("[warmup] 未生成新的本地缓存文件列表 (可能未启用 LOCAL_CACHE_DIR)。")
        return
    print("[warmup] 新生成的本地缓存文件列表:")
    for name, path, missing in generated:
        miss_note = f", missing={missing}" if missing else ""
        print(f"  - {name}: {path}{miss_note}")


def main():
    args = parse_args()
    config = args_to_config(args)
    errors = validate_config(config)
    if errors:
        logging.warning("Configuration validation produced warnings: %s", errors)

    class _WarmupSampler(Sampler):
        def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, drop_last=False):
            self.dataset = dataset
            self.shuffle = shuffle
            self.drop_last = drop_last

        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)
            if self.drop_last:
                usable = (len(indices) // 1) * 1
                indices = indices[:usable]
            return iter(indices)

        def __len__(self):
            if self.drop_last:
                return len(self.dataset) // 1
            return len(self.dataset)

        def set_epoch(self, epoch):
            pass

    class _WarmupSampler(Sampler):
        def __init__(self, dataset, *_args, **_kwargs):
            self.dataset = dataset
            self.shuffle = _kwargs.get("shuffle", False)
            self.drop_last = _kwargs.get("drop_last", False)

        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)
            if self.drop_last:
                usable = (len(indices) // 1) * 1
                indices = indices[:usable]
            return iter(indices)

        def __len__(self):
            if self.drop_last:
                return len(self.dataset) // 1
            return len(self.dataset)

        def set_epoch(self, epoch):
            pass

    torch_dist.DistributedSampler = _WarmupSampler

    loaders = setup_dataloaders(config)
    depth_loader, seg_loader = loaders[0], loaders[2]
    if args.skip_depth:
        depth_loader = None
    if args.skip_seg:
        seg_loader = None

    for name, loader in (("depth", depth_loader), ("seg", seg_loader)):
        if loader is None:
            continue
        print(f"[warmup] starting {name} loader ...")
        for idx, batch in enumerate(loader, start=1):
            if args.log_interval and idx % args.log_interval == 0:
                print(f"[warmup] {name} batch {idx}")
            if args.warm_steps and idx >= args.warm_steps:
                break
        print(f"[warmup] {name} loader finished (batches={idx if loader else 0})")

    print("[warmup] completed; local cache should contain newly materialized samples.")
    _emit_local_filelists((depth_loader, seg_loader))


if __name__ == "__main__":
    main()
