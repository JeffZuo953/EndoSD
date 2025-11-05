#!/usr/bin/env python3
"""
Benchmark script to compare training efficiency between:
1. Image-CPU processing (loading images and processing on-the-fly)
2. Pre-cached pt files (loading pre-processed tensors)
"""
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from util.config import parse_and_validate_config
from util.data_utils import setup_dataloaders
from util.train_utils import setup_training_environment
import argparse


def benchmark_dataloader(loader, num_batches=50, desc=""):
    """Benchmark dataloader throughput"""
    times = []

    print(f"\n{'='*60}")
    print(f"Benchmarking: {desc}")
    print(f"{'='*60}")

    # Warmup
    for i, batch in enumerate(loader):
        if i >= 5:
            break

    # Actual benchmark
    start_time = time.time()
    for i, batch in enumerate(loader):
        batch_start = time.time()

        # Simulate minimal GPU transfer
        if torch.cuda.is_available():
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    v.cuda(non_blocking=True)

        batch_time = time.time() - batch_start
        times.append(batch_time)

        if i >= num_batches - 1:
            break

    total_time = time.time() - start_time
    avg_time = sum(times) / len(times)

    print(f"Total batches: {len(times)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg batch time: {avg_time*1000:.2f}ms")
    print(f"Throughput: {1/avg_time:.2f} batches/s")

    return {
        'total_time': total_time,
        'avg_batch_time': avg_time,
        'throughput': 1/avg_time,
        'num_batches': len(times)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-batches', type=int, default=50, help='Number of batches to benchmark')
    args = parser.parse_args()

    # Setup distributed environment
    config = parse_and_validate_config()
    rank, world_size, logger, writer = setup_training_environment(config)

    # Setup dataloaders (currently using pt-cache)
    train_depth_loader, train_seg_loader, val_depth_loader, val_seg_loader = setup_dataloaders(config)

    if rank == 0:
        print(f"\n{'#'*60}")
        print(f"# Data Loading Benchmark")
        print(f"# GPUs: {world_size}")
        print(f"# Batch size (depth): {config.bs}")
        print(f"# Batch size (seg): {config.seg_bs}")
        print(f"# Workers: {train_depth_loader.num_workers if train_depth_loader else 'N/A'}")
        print(f"{'#'*60}")

        results = {}

        if train_depth_loader:
            results['train_depth'] = benchmark_dataloader(
                train_depth_loader,
                args.num_batches,
                f"Train Depth Loader (bs={config.bs})"
            )

        if train_seg_loader:
            results['train_seg'] = benchmark_dataloader(
                train_seg_loader,
                args.num_batches,
                f"Train Seg Loader (bs={config.seg_bs})"
            )

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for name, metrics in results.items():
            print(f"{name}:")
            print(f"  Throughput: {metrics['throughput']:.2f} batches/s")
            print(f"  Avg batch time: {metrics['avg_batch_time']*1000:.2f}ms")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
