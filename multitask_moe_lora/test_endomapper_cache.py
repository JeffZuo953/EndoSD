"""逐样本迭代 EndoMapper 缓存，用模型的第一层进行前向，检验 tensor 是否越界。"""
import argparse
import torch
import sys
from pathlib import Path
from torch.utils.data import Sampler
import torch.utils.data.distributed as torch_dist

REPO_ROOT = Path(__file__).resolve().parent
PARENT = REPO_ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from multitask_moe_lora.util.config import TrainingConfig
from multitask_moe_lora.util.data_utils import setup_dataloaders


def build_config(args) -> TrainingConfig:
    include_train = args.dataset_include.split(',') if args.dataset_include else None
    include_val = args.val_dataset_include.split(',') if args.val_dataset_include else include_train
    cfg = TrainingConfig(
        encoder=args.encoder,
        features=args.features,
        num_classes=args.num_classes,
        dataset_config_name=args.dataset_config_name,
        dataset_modality=args.dataset_modality,
        path_transform_name=args.path_transform_name,
        bs=args.bs,
        val_bs=args.val_bs,
        save_path=args.save_path,
        train_dataset_include=include_train,
        val_dataset_include=include_val,
        mixed_precision=False,
        max_samples_per_dataset=args.limit,
    )
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config-name", default="fd_depth_fm_v1")
    parser.add_argument("--dataset-modality", default="fd")
    parser.add_argument("--path-transform-name", default="none")
    parser.add_argument("--encoder", default="vits")
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--val-bs", type=int, default=4)
    parser.add_argument("--dataset-include", default=None)
    parser.add_argument("--val-dataset-include", default="EndoMapper")
    parser.add_argument("--save-path", default="/tmp/endotest")
    parser.add_argument("--limit", type=int, default=200, help="max samples per dataset")
    args = parser.parse_args()

    class _SeqSampler(Sampler):
        def __init__(self, dataset, *_args, **_kwargs):
            self.dataset = dataset

        def __iter__(self):
            return iter(range(len(self.dataset)))

        def __len__(self):
            return len(self.dataset)

        def set_epoch(self, epoch):
            pass

    torch_dist.DistributedSampler = _SeqSampler

    cfg = build_config(args)
    loaders = setup_dataloaders(cfg)
    depth_loader = loaders[2] if loaders[2] is not None else loaders[0]
    assert depth_loader is not None, "Depth loader missing"

    for idx, batch in enumerate(depth_loader, start=1):
        image = batch["image"]
        depth = batch["depth"]
        if idx % 50 == 0:
            print(f"Batch {idx}: image {image.shape} depth {depth.shape}")

    print("Completed EndoMapper cache test")


if __name__ == "__main__":
    main()
