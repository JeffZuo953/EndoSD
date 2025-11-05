import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Callable, Optional

from .camera_utils import get_camera_info
from .utils import compute_valid_mask, map_ls_semseg_to_10_classes, map_semseg_to_three_classes


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
        essential_data = {'image': item['image'], 'image_path': item['image_path']}

        if 'depth' in item:
            essential_data['depth'] = item['depth']
        if 'semseg_mask' in item:
            essential_data['semseg_mask'] = item['semseg_mask']
        if 'valid_mask' in item:
            essential_data['valid_mask'] = item['valid_mask']
        if 'max_depth' in item:
            essential_data['max_depth'] = item['max_depth']

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


def _infer_dataset_name_from_path(path: str) -> str:
    """
    简单根据路径推断数据集名称：
    - 优先使用非 cache/filelists 等中间目录
    - 若未找到合适目录，则使用文件名去除扩展名
    """
    norm_path = os.path.normpath(path)
    directory = os.path.dirname(norm_path)
    ignore_tokens = {"cache", "cache_pt", "filelists", "filelist", "txt", "filelist_txt"}

    while directory:
        base = os.path.basename(directory)
        if base and base.lower() not in ignore_tokens:
            return base
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent

    return os.path.splitext(os.path.basename(norm_path))[0] or "dataset"


class DepthCacheDataset(Dataset):
    """
    一个通用的缓存数据集基类，用于从.pt文件加载预先保存的样本。
    """

    def __init__(self, filelist_path: str, dataset_type: str = "unknown", path_transform: Callable[[str], str] = None, dataset_name: Optional[str] = None):
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
            raw_paths = [line.strip() for line in f if line.strip()]

        if path_transform:
            raw_paths = [path_transform(path) for path in raw_paths]

        existing_paths: List[str] = []
        missing_count = 0
        for path in raw_paths:
            if os.path.exists(path):
                existing_paths.append(path)
            else:
                missing_count += 1
        if missing_count > 0:
            print(f"[DepthCacheDataset] Skipped {missing_count} missing cache files for {dataset_name or filelist_path}.")

        self.filelist = existing_paths

        self.dataset_type = dataset_type
        self.dataset_name = dataset_name or _infer_dataset_name_from_path(filelist_path)

    def __getitem__(self, item: int) -> dict:
        """
        根据索引获取缓存数据。

        Args:
            item (int): 数据索引。

        Returns:
            dict: 加载的缓存数据，包含 'source_type' 键。
        """
        cache_path = self.filelist[item]
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
            cached_data['max_depth'] = cached_data.get('max_depth', 1.0)

        original_image_path = cached_data.get('image_path', cache_path)
        if isinstance(original_image_path, bytes):
            original_image_path = original_image_path.decode("utf-8", errors="ignore")
        original_image_path = str(original_image_path)
        cached_data['original_image_path'] = original_image_path
        cached_data['image_path'] = cache_path
        cached_data['source_type'] = self.dataset_type  # 添加数据源类型标识
        cached_data['dataset_name'] = self.dataset_name

        if 'image' in cached_data and torch.is_tensor(cached_data['image']):
            cached_data['image'] = cached_data['image'].to(torch.float32)

        if 'semseg_mask' in cached_data:
            semseg_mask = cached_data['semseg_mask']
            if torch.is_tensor(semseg_mask):
                cached_data['semseg_mask'] = semseg_mask.to(torch.long)
            else:
                cached_data['semseg_mask'] = torch.as_tensor(semseg_mask, dtype=torch.long)

        if 'seg_valid_mask' in cached_data:
            seg_valid = cached_data['seg_valid_mask']
            if torch.is_tensor(seg_valid):
                cached_data['seg_valid_mask'] = seg_valid.to(torch.bool)
            else:
                cached_data['seg_valid_mask'] = torch.as_tensor(seg_valid, dtype=torch.bool)

        if 'depth' in cached_data and 'image' in cached_data:
            depth = cached_data['depth']
            if not torch.is_tensor(depth):
                depth = torch.as_tensor(depth, dtype=torch.float32)
            else:
                depth = depth.to(torch.float32)
            cached_data['depth'] = depth

            max_depth = float(cached_data.get('max_depth', 0.3))

            depth_valid_data = cached_data.get('depth_valid_mask')
            if depth_valid_data is not None:
                if torch.is_tensor(depth_valid_data):
                    depth_valid = depth_valid_data.to(torch.bool)
                else:
                    depth_valid = torch.as_tensor(depth_valid_data, dtype=torch.bool)
            else:
                depth_valid = None

            valid_mask_data = cached_data.get('valid_mask')
            if valid_mask_data is not None:
                if torch.is_tensor(valid_mask_data):
                    valid_mask = valid_mask_data.to(torch.bool)
                else:
                    valid_mask = torch.as_tensor(valid_mask_data, dtype=torch.bool)
            else:
                valid_mask = None

            if depth_valid is None and valid_mask is not None:
                depth_valid = valid_mask.clone()

            if valid_mask is None:
                if depth_valid is not None:
                    valid_mask = depth_valid.clone()
                else:
                    image = cached_data['image']
                    computed_valid = compute_valid_mask(image, depth, min_depth=1e-3, max_depth=max_depth)
                    if torch.is_tensor(computed_valid):
                        valid_mask = computed_valid.to(torch.bool)
                    else:
                        valid_mask = torch.as_tensor(computed_valid, dtype=torch.bool)

            if depth_valid is None:
                depth_valid = valid_mask.clone()

            depth_clean = depth.clone()
            depth_clean[~valid_mask] = 0.0
            cached_data['depth'] = depth_clean
            cached_data['valid_mask'] = valid_mask
            cached_data['depth_valid_mask'] = depth_valid

        camera_info = get_camera_info(self.dataset_name or "", original_image_path)
        if camera_info is None:
            camera_info = get_camera_info(self.dataset_type or "", original_image_path)
        if camera_info is not None:
            cached_data['camera_intrinsics'] = camera_info.intrinsics.clone()
            cached_data['camera_intrinsics_norm'] = camera_info.intrinsics_norm.clone()
            cached_data['camera_size'] = torch.tensor(
                [camera_info.width, camera_info.height],
                dtype=torch.float32,
            )

        return cached_data

    def __len__(self) -> int:
        """
        返回数据集中缓存文件的总数。
        """
        return len(self.filelist)


class SegCacheDataset(Dataset):
    """
    一个通用的缓存数据集基类，用于从.pt文件加载预先保存的样本。
    """

    def __init__(
        self,
        filelist_path: str,
        dataset_type: str = "unknown",
        path_transform: Callable[[str], str] = None,
        dataset_name: Optional[str] = None,
        label_mode: str = "raw",
    ):
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
        self.dataset_name = dataset_name or _infer_dataset_name_from_path(filelist_path)
        self.label_mode = label_mode

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
            cached_data['dataset_name'] = self.dataset_name

            if 'image' in cached_data and torch.is_tensor(cached_data['image']):
                cached_data['image'] = cached_data['image'].to(torch.float32)

            if 'semseg_mask' in cached_data and self.label_mode != "raw":
                mask = cached_data['semseg_mask']
                if self.label_mode == "10class":
                    mapped = map_ls_semseg_to_10_classes(mask, self.dataset_name)
                elif self.label_mode == "3class":
                    mapped = map_semseg_to_three_classes(mask, self.dataset_name)
                else:
                    mapped = mask
                if mapped is not mask:
                    if not torch.is_tensor(mapped):
                        mapped = torch.as_tensor(mapped, dtype=torch.long)
                    else:
                        mapped = mapped.to(torch.long)
                    cached_data['semseg_mask'] = mapped
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

    from .scard import SCardDataset as Dataset
    base_dir = "/data/ziyi/multitask/data/SCARED"

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
