import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Callable, Dict, Any
from collections import defaultdict


class DepthStatistics:
    """
    用于收集和统计深度数据分布的类。
    """

    def __init__(self):
        self.depth_values = []
        self.valid_pixel_counts = []
        self.total_pixel_counts = []
        self.min_depths = []
        self.max_depths = []
        self.mean_depths = []
        self.image_paths = []

    def add_sample(self, depth: torch.Tensor, image_path: str):
        """
        添加一个深度样本的统计信息。

        Args:
            depth (torch.Tensor): 深度张量
            image_path (str): 图像路径
        """
        # 转换为numpy数组以便统计
        if isinstance(depth, torch.Tensor):
            depth_np = depth.cpu().numpy()
        else:
            depth_np = np.array(depth)

        # 展平深度数据
        depth_flat = depth_np.flatten()

        # 计算有效像素（非零值）
        valid_mask = depth_flat > 0
        valid_depths = depth_flat[valid_mask]

        # 记录统计信息
        self.total_pixel_counts.append(len(depth_flat))
        self.valid_pixel_counts.append(len(valid_depths))

        if len(valid_depths) > 0:
            self.min_depths.append(float(np.min(valid_depths)))
            self.max_depths.append(float(np.max(valid_depths)))
            self.mean_depths.append(float(np.mean(valid_depths)))
            self.depth_values.extend(valid_depths.tolist())
        else:
            self.min_depths.append(0.0)
            self.max_depths.append(0.0)
            self.mean_depths.append(0.0)

        self.image_paths.append(image_path)

    def generate_report(self, output_path: str):
        """
        生成深度数据统计报告。

        Args:
            output_path (str): 报告保存路径
        """
        if len(self.depth_values) == 0:
            print("警告: 没有收集到任何深度数据，无法生成报告。")
            return

        depth_array = np.array(self.depth_values)

        # 计算全局统计信息
        global_stats = {
            "total_samples": len(self.image_paths),
            "total_pixels": sum(self.total_pixel_counts),
            "total_valid_pixels": sum(self.valid_pixel_counts),
            "valid_pixel_ratio": sum(self.valid_pixel_counts) / sum(self.total_pixel_counts),
            "global_min_depth": float(np.min(depth_array)),
            "global_max_depth": float(np.max(depth_array)),
            "global_mean_depth": float(np.mean(depth_array)),
            "global_median_depth": float(np.median(depth_array)),
            "global_std_depth": float(np.std(depth_array)),
        }

        # 计算百分位数
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = {f"p{p}": float(np.percentile(depth_array, p)) for p in percentiles}

        # 计算每个样本的统计信息
        per_sample_stats = []
        for i, path in enumerate(self.image_paths):
            per_sample_stats.append({
                "image_path": path,
                "total_pixels": self.total_pixel_counts[i],
                "valid_pixels": self.valid_pixel_counts[i],
                "valid_ratio": self.valid_pixel_counts[i] / self.total_pixel_counts[i] if self.total_pixel_counts[i] > 0 else 0,
                "min_depth": self.min_depths[i],
                "max_depth": self.max_depths[i],
                "mean_depth": self.mean_depths[i],
            })

        # 构建完整报告
        report = {
            "global_statistics": global_stats,
            "percentiles": percentile_values,
            "per_sample_statistics": per_sample_stats,
        }

        # 保存JSON报告
        json_path = output_path.replace('.txt', '.json') if output_path.endswith('.txt') else output_path + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成可读的文本报告
        txt_path = output_path.replace('.json', '.txt') if output_path.endswith('.json') else output_path + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
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
                f.write(f"P{p:2d}: {percentile_values[f'p{p}']:.6f}\n")
            f.write("\n")

            f.write("前10个样本的统计信息:\n")
            f.write("-" * 80 + "\n")
            for i, stats in enumerate(per_sample_stats[:10]):
                f.write(f"\n样本 {i+1}: {os.path.basename(stats['image_path'])}\n")
                f.write(f"  有效像素: {stats['valid_pixels']:,} / {stats['total_pixels']:,} ({stats['valid_ratio']:.2%})\n")
                f.write(f"  深度范围: [{stats['min_depth']:.6f}, {stats['max_depth']:.6f}]\n")
                f.write(f"  平均深度: {stats['mean_depth']:.6f}\n")

            if len(per_sample_stats) > 10:
                f.write(f"\n... (剩余 {len(per_sample_stats) - 10} 个样本的详细信息请查看JSON报告)\n")

        print(f"\n深度统计报告已生成:")
        print(f"  JSON格式: {json_path}")
        print(f"  文本格式: {txt_path}")
        print(f"\n全局统计摘要:")
        print(f"  样本数: {global_stats['total_samples']}")
        print(f"  深度范围: [{global_stats['global_min_depth']:.6f}, {global_stats['global_max_depth']:.6f}]")
        print(f"  平均深度: {global_stats['global_mean_depth']:.6f}")
        print(f"  中位数深度: {global_stats['global_median_depth']:.6f}")


def generate_dataset_cache(dataset: Dataset,
                           output_dir: str,
                           filelist_name: str = "cache_files.txt",
                           origin_prefix: str = "",
                           cache_root_path: str = "",
                           enable_depth_stats: bool = True) -> str:
    """
    遍历数据集，将每个样本的核心张量数据保存为.pt文件，并生成一个包含所有缓存文件路径的txt文件。
    同时收集深度数据的统计信息并生成报告。

    Args:
        dataset (Dataset): 要处理的PyTorch数据集实例。
        output_dir (str): 用于保存最终生成的缓存文件列表（.txt）的目录。
        filelist_name (str): 生成的缓存文件列表的文件名。
        origin_prefix (str): 原始数据路径中需要被替换的根前缀。
        cache_root_path (str): 用于替换 origin_prefix 的新数据根路径。所有.pt缓存文件将存储在此路径下，并保持原始的目录结构。
        enable_depth_stats (bool): 是否启用深度数据统计和报告生成。

    Returns:
        str: 生成的缓存文件列表的完整路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_file_paths: List[str] = []

    # 初始化深度统计对象
    depth_stats = DepthStatistics() if enable_depth_stats else None

    print(f"开始生成数据集缓存到: {cache_root_path}")
    # 创建tqdm迭代器，以便后续更新其描述信息
    tqdm_iterator = tqdm(dataset, desc="Initializing...")
    for i, item in enumerate(tqdm_iterator):
        if "image_path" not in item:
            print(f"警告: 样本 {i} 没有 'image_path' 键，跳过缓存。")
            continue

        original_img_path: str = item['image_path']

        # --- 新增功能：更新tqdm进度条的描述，显示当前正在处理的文件名 ---
        tqdm_iterator.set_description(f"Processing {os.path.basename(original_img_path)}")

        # 核心逻辑：通过替换前缀来生成缓存文件的目标路径
        processed_path: str = original_img_path
        if origin_prefix and original_img_path.startswith(origin_prefix):
            processed_path = original_img_path.replace(origin_prefix, cache_root_path, 1)

        # 将文件扩展名更改为.pt，得到最终的缓存文件路径
        cache_path: str = os.path.splitext(processed_path)[0] + '.pt'

        # 确保缓存文件的父目录存在
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # 根据数据集中是否包含depth字段来决定保存的内容
        if 'depth' in item:
            # 包含深度数据的情况
            essential_data = {
                'image': item['image'],
                'depth': item['depth'],
                'image_path': item['image_path'],
                'max_depth': item.get('max_depth', 1.0),
                'valid_mask': item.get('valid_mask', item['depth'] > 0)
            }

            # 收集深度统计信息
            if depth_stats is not None:
                depth_stats.add_sample(item['depth'], original_img_path)
        elif 'semseg_mask' in item:
            # 包含语义分割掩码的情况
            essential_data = {'image': item['image'], 'semseg_mask': item['semseg_mask'], 'image_path': item['image_path']}
        else:
            print(f"警告: 样本 {i} 既没有 'depth' 也没有 'semseg_mask' 键，跳过缓存。")
            continue

        try:
            torch.save(essential_data, cache_path)
            cache_file_paths.append(cache_path)
        except Exception as e:
            print(f"保存缓存文件 {cache_path} 时发生错误: {e}")

    # 将所有缓存文件的路径写入文件列表
    filelist_path: str = os.path.join(output_dir, filelist_name)
    with open(filelist_path, 'w') as f:
        for path in cache_file_paths:
            f.write(path + '\n')

    print(f"\n缓存生成完成。缓存文件列表已保存到: {filelist_path}")

    # 生成深度统计报告
    if depth_stats is not None and len(depth_stats.depth_values) > 0:
        report_path = os.path.join(output_dir, "depth_statistics_report")
        depth_stats.generate_report(report_path)

    return filelist_path


class DepthCacheDataset(Dataset):
    """
    一个通用的缓存数据集基类，用于从.pt文件加载预先保存的样本。
    """

    def __init__(self, filelist_path: str, dataset_type: str = "unknown", path_transform: Callable[[str], str] = None):
        """
        初始化缓存数据集。

        Args:
            filelist_path (str): 包含缓存文件路径列表的文本文件路径。
            dataset_type (str): 数据集的类型 ('kidney' or 'colon')，用于在验证时区分来源。
            path_transform (Callable[[str], str], optional): 一个函数，用于转换文件列表中的每个路径。默认为 None。
        """
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"缓存文件列表不存在: {filelist_path}")

        with open(filelist_path, "r") as f:
            self.filelist = f.read().splitlines()

        if path_transform:
            self.filelist = [path_transform(path) for path in self.filelist]

        self.dataset_type = dataset_type

    def __getitem__(self, item: int) -> dict:
        """
        根据索引获取缓存数据。

        Args:
            item (int): 数据索引。

        Returns:
            dict: 加载的缓存数据，包含 'source_type' 键。
        """
        cache_path = self.filelist[item]
        try:
            cached_data = torch.load(cache_path)
            if 'c3vd' in cache_path:
                cached_data['max_depth'] = 0.1
            elif 'simcol' in cache_path:
                cached_data['max_depth'] = 0.2
            elif 'endomapper' in cache_path:
                cached_data['max_depth'] = 0.2
            elif 'inhouse' in cache_path:
                cached_data['max_depth'] = 0.05
            else:
                cached_data['max_depth'] = cached_data['max_depth'] | 1.0

            cached_data['image_path'] = cache_path
            cached_data['source_type'] = self.dataset_type  # 添加数据源类型标识
            return cached_data
        except Exception as e:
            print(f"加载缓存文件时发生错误 {cache_path}: {e}")
            return {}

    def __len__(self) -> int:
        """
        返回数据集中缓存文件的总数。
        """
        return len(self.filelist)


class SegCacheDataset(Dataset):
    """
    一个通用的缓存数据集基类，用于从.pt文件加载预先保存的样本。
    """

    def __init__(self, filelist_path: str, dataset_type: str = "unknown", path_transform: Callable[[str], str] = None):
        """
        初始化缓存数据集。

        Args:
            filelist_path (str): 包含缓存文件路径列表的文本文件路径。
            dataset_type (str): 数据集的类型 ('kidney' or 'colon')，用于在验证时区分来源。
            path_transform (Callable[[str], str], optional): 一个函数，用于转换文件列表中的每个路径。默认为 None。
        """
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"缓存文件列表不存在: {filelist_path}")

        with open(filelist_path, "r") as f:
            self.filelist = f.read().splitlines()

        if path_transform:
            self.filelist = [path_transform(path) for path in self.filelist]

        self.dataset_type = dataset_type

    def __getitem__(self, item: int) -> dict:
        """
        根据索引获取缓存数据。

        Args:
            item (int): 数据索引。

        Returns:
            dict: 加载的缓存数据，包含 'source_type' 键。
        """
        cache_path = self.filelist[item]
        try:
            cached_data = torch.load(cache_path)
            cached_data['source_type'] = self.dataset_type  # 添加数据源类型标识
            return cached_data
        except Exception as e:
            print(f"加载缓存文件时发生错误 {cache_path}: {e}")
            return {}

    def __len__(self) -> int:
        """
        返回数据集中缓存文件的总数。
        """
        return len(self.filelist)


def main():
    print("--- generating_dataset_cache ---")

    # 假设 C3VD 数据集和 'JIANFU' 环境变量已正确设置
    # 使用您原来的数据加载方式
    # from .inhouse_seg import InHouseSeg as Dataset
    # from .endomapper import Endomapper as Dataset
    # from .inhouse import InHouse as Dataset
    # from .simcol import Simcol as Dataset

    # Polyp datasets (commented out by default)
    # from .bkai_polyp_dataset import BKAIPolypDataset as Dataset
    # base_dir = '/media/ssd2t/jianfu/data/polyp/bkai-igh-neopolyp'

    # from .cvc_clinicdb_dataset import CVCClinicDBDataset as Dataset
    # base_dir = '/media/ssd2t/jianfu/data/polyp/CVC-ClinicDB'

    # from .cvc_endoscene_still_dataset import CVCEndoSceneStillDataset as Dataset
    # base_dir = '/media/ssd2t/jianfu/data/polyp/CVC-EndoScene/ValidationDataset'

    # from .etis_larib_dataset import ETISLaribDataset as Dataset
    # base_dir = '/media/ssd2t/jianfu/data/polyp/ETIS-LaribPolypDB'

    # from .kvasir_seg_dataset import KvasirSegDataset as Dataset
    # base_dir = "/media/ssd2t/jianfu/data/polyp/kvasir-seg"

    # from .scard import SCardDataset as Dataset
    # base_dir = "/data/ziyi/multitask/data/SCARED"

    from .hamlyn import HamlynDataset as Dataset
    base_dir = "/path/to/hamlyn/dataset"  # 请修改为实际的 Hamlyn 数据集路径

    # --- 路径配置 ---
    # 从环境变量获取数据根目录，若未设置则使用当前目录作为后备

    original_filelist = f"{base_dir}/train.txt"
    origin_prefix = f"{base_dir}"
    cache_root_path = f"{base_dir}/cache"
    output_cache_list_name = "train_cache.txt"

    # --- 创建原始数据集实例 ---
    # 使用原始文件列表初始化数据集
    print(f"正在从文件列表加载原始数据集: {original_filelist}")
    original_dataset = Dataset(
        original_filelist,
        origin_prefix,
        "train",
        size=(518, 518),
        depth_scale=1000.0  # Hamlyn dataset: depth in PNG is stored in millimeters
    )

    print("\n开始生成缓存...")
    generated_filelist_path = generate_dataset_cache(dataset=original_dataset,
                                                     output_dir=cache_root_path,
                                                     filelist_name=output_cache_list_name,
                                                     origin_prefix=origin_prefix,
                                                     cache_root_path=cache_root_path)
    print(f"\n缓存生成完成。文件列表位于: {generated_filelist_path}")


if __name__ == "__main__":
    main()
