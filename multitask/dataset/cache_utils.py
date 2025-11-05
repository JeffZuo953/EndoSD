import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List


def generate_dataset_cache(dataset: Dataset, output_dir: str, filelist_name: str = "cache_files.txt", origin_prefix: str = "", cache_root_path: str = "") -> str:
    """
    遍历数据集，将每个样本的核心张量数据保存为.pt文件，并生成一个包含所有缓存文件路径的txt文件。

    Args:
        dataset (Dataset): 要处理的PyTorch数据集实例。
        output_dir (str): 用于保存最终生成的缓存文件列表（.txt）的目录。
        filelist_name (str): 生成的缓存文件列表的文件名。
        origin_prefix (str): 原始数据路径中需要被替换的根前缀。
        cache_root_path (str): 用于替换 origin_prefix 的新数据根路径。所有.pt缓存文件将存储在此路径下，并保持原始的目录结构。

    Returns:
        str: 生成的缓存文件列表的完整路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_file_paths: List[str] = []

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
        # essential_data = {'image': item['image'], 'depth': item['depth'], 'image_path': item['image_path'], 'max_depth': item['max_depth'], 'valid_mask': item['valid_mask']}
        essential_data = {'image': item['image'], 'semseg_mask': item['semseg_mask'], 'image_path': item['image_path']}

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
    return filelist_path


class DepthCacheDataset(Dataset):
    """
    一个通用的缓存数据集基类，用于从.pt文件加载预先保存的样本。
    """

    def __init__(self, filelist_path: str, dataset_type: str = "unknown"):
        """
        初始化缓存数据集。

        Args:
            filelist_path (str): 包含缓存文件路径列表的文本文件路径。
            dataset_type (str): 数据集的类型 ('kidney' or 'colon')，用于在验证时区分来源。
        """
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"缓存文件列表不存在: {filelist_path}")

        with open(filelist_path, "r") as f:
            self.filelist = f.read().splitlines()
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
                cached_data['max_depth'] = 1.0  # default

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

    def __init__(self, filelist_path: str, dataset_type: str = "unknown"):
        """
        初始化缓存数据集。

        Args:
            filelist_path (str): 包含缓存文件路径列表的文本文件路径。
            dataset_type (str): 数据集的类型 ('kidney' or 'colon')，用于在验证时区分来源。
        """
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"缓存文件列表不存在: {filelist_path}")

        with open(filelist_path, "r") as f:
            self.filelist = f.read().splitlines()
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

    from .kvasir_seg_dataset import KvasirSegDataset as Dataset
    base_dir = "/media/ssd2t/jianfu/data/polyp/kvasir-seg"

    # --- 路径配置 ---
    # 从环境变量获取数据根目录，若未设置则使用当前目录作为后备

    original_filelist = f"{base_dir}/train.txt"
    origin_prefix = f"{base_dir}"
    cache_root_path = f"{base_dir}/cache"
    output_cache_list_name = "train_cache.txt"

    # --- 创建原始数据集实例 ---
    # 使用原始文件列表初始化数据集
    print(f"正在从文件列表加载原始数据集: {original_filelist}")
    original_dataset = Dataset(original_filelist, origin_prefix, "train", size=(518, 518))  # 'train' 模式

    print("\n开始生成缓存...")
    generated_filelist_path = generate_dataset_cache(dataset=original_dataset,
                                                     output_dir=cache_root_path,
                                                     filelist_name=output_cache_list_name,
                                                     origin_prefix=origin_prefix,
                                                     cache_root_path=cache_root_path)
    print(f"\n缓存生成完成。文件列表位于: {generated_filelist_path}")


if __name__ == "__main__":
    main()
