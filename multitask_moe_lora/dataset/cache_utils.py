import hashlib
import logging
import os
from typing import Callable, List, Optional, Set

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .camera_utils import get_camera_info, normalize_intrinsics
from .utils import compute_valid_mask, map_ls_semseg_to_10_classes, map_semseg_to_three_classes
from ..util.local_cache import LocalCacheManager

logger = logging.getLogger(__name__)

_DEFAULT_SKIP_CAMERA = {"endosynth"}
_env_skip = os.environ.get("SKIP_CAMERA_DATASETS", "")
if _env_skip.strip():
    _DEFAULT_SKIP_CAMERA.update(
        name.strip().lower() for name in _env_skip.split(",") if name.strip()
    )
SKIP_CAMERA_DATASETS = frozenset(_DEFAULT_SKIP_CAMERA)

_FORCE_DEPTH_POSITIVE_MASK_TOKENS = (
    "endosynth",
    "c3vd",
    "c3vdv2",
    "endomapper",
    "simcol",
    "kidney3d",
)


def _expand_path(value: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(value)))


_PREFIX_ENV_MAP = (
    ("/data/ziyi/multitask", "BASE_DATA_PATH"),
    ("/home/ziyi/ssde", "HOME_SSD_PATH"),
    ("/media/ssd2t/jianfu", "SSD2T_DATA_PATH"),
    ("/media/ExtHDD1/jianfu", "EXT_HDD_DATA_PATH"),
)


def _rewrite_cache_path(path: str) -> str:
    if not isinstance(path, str):
        return path
    normalized = _expand_path(path)
    for prefix, env_key in _PREFIX_ENV_MAP:
        if normalized.startswith(prefix):
            replacement = os.environ.get(env_key)
            if not replacement:
                continue
            replacement = _expand_path(replacement)
            suffix = normalized[len(prefix):].lstrip("/\\")
            normalized = os.path.join(replacement, suffix)
            break
    return normalized


def _resolve_progress_step(default: int = 3000) -> int:
    env_value = os.environ.get("DATASET_PROGRESS_STEP", "").strip()
    if env_value:
        try:
            step = int(env_value)
            if step > 0:
                return step
            return 0
        except ValueError:
            return default
    return default


def _build_cache_namespace(prefix: str, dataset_name: str, filelist_path: str) -> str:
    normalized = os.path.abspath(os.path.expanduser(filelist_path))
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    safe_name = (dataset_name or "dataset").strip().replace("/", "_")
    return f"{prefix}/{safe_name}/{digest}"


def _should_force_depth_positive_mask(value: Optional[str]) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return any(token in lowered for token in _FORCE_DEPTH_POSITIVE_MASK_TOKENS)


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

    def __init__(
        self,
        filelist_path: str,
        dataset_type: str = "unknown",
        path_transform: Callable[[str], str] = None,
        dataset_name: Optional[str] = None,
        local_cache_dir: Optional[str] = None,
    ):
        """
        初始化缓存数据集。

        Args:
            filelist_path (str): 包含缓存文件路径列表的文本文件路径。
            dataset_type (str): 数据集的类型 ('kidney' or 'colon')，用于在验证时区分来源。
            path_transform (Callable[[str], str], optional): 一个函数，用于转换文件列表中的每个路径。默认为 None。
        """
        filelist_path = _rewrite_cache_path(filelist_path)
        filelist_path = _rewrite_cache_path(filelist_path)
        self.original_filelist_path = filelist_path
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name or _infer_dataset_name_from_path(filelist_path)
        cache_namespace = _build_cache_namespace("depth_cache", self.dataset_name, filelist_path)
        self._local_cache = LocalCacheManager(local_cache_dir, namespace=cache_namespace)
        self.local_filelist_path = None

        candidate_filelist = filelist_path
        if self._local_cache.enabled:
            candidate = self._local_cache.filelist_mirror_path(filelist_path)
            if candidate and os.path.exists(candidate):
                candidate_filelist = candidate
                self.local_filelist_path = candidate

        if not os.path.exists(candidate_filelist):
            raise FileNotFoundError(f"缓存文件列表不存在: {candidate_filelist}")

        with open(candidate_filelist, "r") as f:
            raw_paths = [_rewrite_cache_path(line.strip()) for line in f if line.strip()]

        if path_transform:
            raw_paths = [path_transform(path) for path in raw_paths]

        expected_count = len(raw_paths)
        if self._local_cache.enabled:
            local_count = self._local_cache.namespace_file_count(suffix=".pt")
            print(f"[DepthCacheDataset] {self.dataset_name}: txt={expected_count}, local_cache={local_count}")
            if local_count != expected_count:
                raise RuntimeError(
                    f"[DepthCacheDataset] 本地缓存文件数量({local_count})与文件列表({expected_count})不一致: {self.dataset_name}. "
                    "请先使用 warm_local_cache.py 同步缓存。"
                )
            self.filelist = raw_paths
        else:
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
        self.dataset_name_lower = (self.dataset_name or "unknown").lower()
        base_normalized = self.dataset_name_lower.split("[", 1)[0]
        self.dataset_name_base_lower = base_normalized or self.dataset_name_lower
        self.dataset_name_base = (self.dataset_name or "unknown").split("[", 1)[0] or self.dataset_name
        self._force_depth_positive_valid_mask = _should_force_depth_positive_mask(self.dataset_name)
        self._missing_camera_datasets: Set[str] = set()
        self.active_filelist_path = candidate_filelist
        self._progress_step = _resolve_progress_step()
        self._progress_counter = 0

    def _log_progress(self):
        if not self._progress_step:
            return
        self._progress_counter += 1
        if self._progress_counter % self._progress_step == 0:
            print(f"[DepthCacheDataset] {self.dataset_name}: processed {self._progress_counter} samples.")

    def __getitem__(self, item: int) -> dict:
        """
        根据索引获取缓存数据。

        Args:
            item (int): 数据索引。

        Returns:
            dict: 加载的缓存数据，包含 'source_type' 键。
        """
        cache_path = self.filelist[item]
        load_path = cache_path
        if self._local_cache.enabled:
            load_path = self._local_cache.ensure_copy(cache_path, cache_path)
        cached_data = torch.load(load_path, map_location="cpu")
        self._log_progress()
        force_depth_positive_mask = self._force_depth_positive_valid_mask or _should_force_depth_positive_mask(cache_path)
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

        img_tensor = cached_data.get('image')
        if torch.is_tensor(img_tensor):
            cached_data['image'] = img_tensor.to(torch.float32)
            img_h, img_w = int(img_tensor.shape[-2]), int(img_tensor.shape[-1])
        else:
            img_h = img_w = None

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

        img_tensor = cached_data.get('image')
        if torch.is_tensor(img_tensor):
            cached_data['image'] = img_tensor.to(torch.float32)
            img_h, img_w = int(img_tensor.shape[-2]), int(img_tensor.shape[-1])
        else:
            img_h = img_w = None

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
            dataset_name_lower = self.dataset_name_lower or ""
            is_stereomis = "stereomis" in dataset_name_lower
            stereo_max_depth = max_depth if max_depth > 0 else 0.3
            default_depth_mask = torch.isfinite(depth) & (depth > 0)
            if is_stereomis:
                default_depth_mask = default_depth_mask & (depth <= stereo_max_depth + 1e-6)
            default_depth_mask = default_depth_mask.to(torch.bool)

            depth_valid = None
            valid_mask = None
            depth_valid_data = None
            if not force_depth_positive_mask:
                valid_mask_data = cached_data.get('valid_mask')
                if valid_mask_data is not None:
                    if torch.is_tensor(valid_mask_data):
                        valid_mask = valid_mask_data.to(torch.bool)
                    else:
                        valid_mask = torch.as_tensor(valid_mask_data, dtype=torch.bool)

                depth_valid_data = cached_data.get('depth_valid_mask')
                if depth_valid_data is not None:
                    if torch.is_tensor(depth_valid_data):
                        depth_valid = depth_valid_data.to(torch.bool)
                    else:
                        depth_valid = torch.as_tensor(depth_valid_data, dtype=torch.bool)

            fallback_mask = default_depth_mask
            depth_clean = depth.clone()
            depth_clean[~fallback_mask] = 0.0
            cached_data['depth'] = depth_clean

            if force_depth_positive_mask:
                cached_data['valid_mask'] = fallback_mask.clone()
                cached_data['depth_valid_mask'] = fallback_mask.clone()
            else:
                if valid_mask is not None:
                    cached_data['valid_mask'] = valid_mask
                else:
                    cached_data.pop('valid_mask', None)
                if depth_valid is not None:
                    cached_data['depth_valid_mask'] = depth_valid
                else:
                    cached_data.pop('depth_valid_mask', None)

            if depth_valid_data is not None:
                if torch.is_tensor(depth_valid_data):
                    depth_valid = depth_valid_data.to(torch.bool)
                else:
                    depth_valid = torch.as_tensor(depth_valid_data, dtype=torch.bool)
            else:
                depth_valid = None

            valid_mask_data = None if force_depth_positive_mask else cached_data.get('valid_mask')
            if valid_mask_data is not None:
                if torch.is_tensor(valid_mask_data):
                    valid_mask = valid_mask_data.to(torch.bool)
                else:
                    valid_mask = torch.as_tensor(valid_mask_data, dtype=torch.bool)
            else:
                valid_mask = None

            if force_depth_positive_mask:
                positive_mask = torch.isfinite(depth) & (depth > 0)
                valid_mask = positive_mask.to(torch.bool)
                depth_valid = valid_mask.clone()
            else:
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

        # --- Camera intrinsics metadata ---
        intrinsic_tensor: Optional[torch.Tensor] = None
        base_width: Optional[float] = None
        base_height: Optional[float] = None

        existing_intrinsics = cached_data.get('camera_intrinsics')
        if existing_intrinsics is not None:
            if torch.is_tensor(existing_intrinsics):
                intrinsic_tensor = existing_intrinsics.to(torch.float32)
            else:
                intrinsic_tensor = torch.as_tensor(existing_intrinsics, dtype=torch.float32)
            cached_data['camera_intrinsics'] = intrinsic_tensor

        size_value = cached_data.get('camera_size')
        if size_value is not None:
            if torch.is_tensor(size_value):
                base_width = float(size_value[0].item())
                base_height = float(size_value[1].item())
            elif isinstance(size_value, (list, tuple)) and len(size_value) >= 2:
                base_width = float(size_value[0])
                base_height = float(size_value[1])

        dataset_key = self.dataset_name_lower
        dataset_base_key = getattr(self, "dataset_name_base_lower", dataset_key)
        skip_camera = dataset_base_key in SKIP_CAMERA_DATASETS
        camera_info = None
        camera_lookup_name = getattr(self, "dataset_name_base", self.dataset_name or "")
        if not skip_camera:
            camera_info = get_camera_info(camera_lookup_name, original_image_path)
        if camera_info is None:
            if not skip_camera and dataset_base_key not in self._missing_camera_datasets:
                logger.error(
                    "[DepthCacheDataset] Missing camera metadata for dataset '%s'; skipping camera intrinsics for %s.",
                    camera_lookup_name or (self.dataset_name or "unknown"),
                    original_image_path,
                )
                self._missing_camera_datasets.add(dataset_base_key)
        else:
            intrinsic_tensor = camera_info.intrinsics.clone().to(torch.float32)
            base_width = float(camera_info.width)
            base_height = float(camera_info.height)
            cached_data['camera_intrinsics'] = intrinsic_tensor

        if base_width is not None and base_height is not None:
            size_tensor = torch.tensor([base_width, base_height], dtype=torch.float32)
            cached_data['camera_size'] = size_tensor
            cached_data['camera_size_original'] = size_tensor.clone()
            cached_data['camera_original_image_size'] = size_tensor.clone()

        if img_h is not None and img_w is not None:
            cached_data['camera_image_size'] = torch.tensor([float(img_w), float(img_h)], dtype=torch.float32)

        if intrinsic_tensor is not None and base_width is not None and base_height is not None:
            norm = normalize_intrinsics(intrinsic_tensor, base_width, base_height)
            cached_data['camera_intrinsics_norm'] = norm

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
        local_cache_dir: Optional[str] = None,
    ):
        """
        初始化缓存数据集。

        Args:
            filelist_path (str): 包含缓存文件路径列表的文本文件路径。
            dataset_type (str): 数据集的类型 ('kidney' or 'colon')，用于在验证时区分来源。
            path_transform (Callable[[str], str], optional): 一个函数，用于转换文件列表中的每个路径。默认为 None。
        """
        self.original_filelist_path = filelist_path
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name or _infer_dataset_name_from_path(filelist_path)
        cache_namespace = _build_cache_namespace("seg_cache", self.dataset_name, filelist_path)
        self._local_cache = LocalCacheManager(local_cache_dir, namespace=cache_namespace)
        self.local_filelist_path = None

        candidate_filelist = filelist_path
        if self._local_cache.enabled:
            candidate = self._local_cache.filelist_mirror_path(filelist_path)
            if candidate and os.path.exists(candidate):
                candidate_filelist = candidate
                self.local_filelist_path = candidate

        if not os.path.exists(candidate_filelist):
            raise FileNotFoundError(f"缓存文件列表不存在: {candidate_filelist}")

        with open(candidate_filelist, "r") as f:
            raw_paths = [_rewrite_cache_path(line.strip()) for line in f if line.strip()]

        if path_transform:
            raw_paths = [path_transform(path) for path in raw_paths]

        expected_count = len(raw_paths)
        if self._local_cache.enabled:
            local_count = self._local_cache.namespace_file_count(suffix=".pt")
            print(f"[SegCacheDataset] {self.dataset_name}: txt={expected_count}, local_cache={local_count}")
            if local_count != expected_count:
                raise RuntimeError(
                    f"[SegCacheDataset] 本地缓存文件数量({local_count})与文件列表({expected_count})不一致: {self.dataset_name}. "
                    "请先使用 warm_local_cache.py 同步缓存。"
                )
            self.filelist = raw_paths
        else:
            self.filelist = raw_paths

        self.label_mode = label_mode
        self.active_filelist_path = candidate_filelist
        self._progress_step = _resolve_progress_step()
        self._progress_counter = 0

    def _log_progress(self):
        if not self._progress_step:
            return
        self._progress_counter += 1
        if self._progress_counter % self._progress_step == 0:
            print(f"[SegCacheDataset] {self.dataset_name}: processed {self._progress_counter} samples.")

    def __getitem__(self, item: int) -> dict:
        """
        根据索引获取缓存数据。

        Args:
            item (int): 数据索引。

        Returns:
            dict: 加载的缓存数据，包含 'source_type' 键。
        """
        cache_path = self.filelist[item]
        load_path = cache_path
        if self._local_cache.enabled:
            load_path = self._local_cache.ensure_copy(cache_path, cache_path)
        try:
            cached_data = torch.load(load_path, map_location="cpu")
            self._log_progress()
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
