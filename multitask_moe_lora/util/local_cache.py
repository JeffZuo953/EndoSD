import hashlib
import os
import shutil
import threading
from typing import Optional

import torch

_ENV_KEYS = ("LOCAL_CACHE_DIR", "LOCAL_CACHE_PATH")
_ENV_DISABLE_MIRROR = "LOCAL_CACHE_FLAT_LAYOUT"


def _resolve_root(preferred: Optional[str]) -> Optional[str]:
    path = preferred
    if not path:
        for key in _ENV_KEYS:
            candidate = os.environ.get(key)
            if candidate:
                path = candidate
                break
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    return path


def _hashed_subpath(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(digest[:2], digest[2:4], digest[4:])


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path))) if path else ""


def _mirror_subpath(path: str) -> Optional[str]:
    if not path:
        return None
    expanded = _abs_path(path)
    drive, tail = os.path.splitdrive(expanded)
    tail = tail.lstrip(os.sep).strip()
    if not tail:
        return None
    parts = [segment for segment in tail.split(os.sep) if segment not in ("", ".", "..")]
    if not parts:
        return None
    if drive:
        drive_token = drive.rstrip(":\\/")
        if drive_token:
            parts.insert(0, drive_token)
    return os.path.join(*parts)


def _atomic_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp_path = f"{dst}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(src, "rb") as src_f, open(tmp_path, "wb") as dst_f:
        shutil.copyfileobj(src_f, dst_f)
    try:
        os.replace(tmp_path, dst)
    except FileExistsError:
        os.remove(tmp_path)


def _atomic_torch_save(obj, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp_path = f"{dst}.tmp.{os.getpid()}.{threading.get_ident()}"
    torch.save(obj, tmp_path)
    try:
        os.replace(tmp_path, dst)
    except FileExistsError:
        os.remove(tmp_path)


class LocalCacheManager:
    """
    负责在本地目录下管理缓存文件（.pt），根据 key 生成稳定的路径。
    - `ensure_copy` 用于镜像已有 .pt 文件
    - `load_obj` / `save_obj` 用于缓存任意 Python 对象（通常是样本 dict）
    """

    def __init__(self, root: Optional[str], namespace: str):
        self.root = _resolve_root(root)
        self.namespace = namespace.strip("/ ")
        disable_mirror = os.environ.get(_ENV_DISABLE_MIRROR, "").strip().lower() in {"1", "true", "yes"}
        self._mirror_paths = not disable_mirror
        self._root_real = os.path.realpath(self.root) if self.root else None

    @property
    def enabled(self) -> bool:
        return bool(self.root)

    def _is_within_root(self, path: str) -> bool:
        if not (self._root_real and path):
            return False
        try:
            path_real = os.path.realpath(_abs_path(path))
            return os.path.commonpath([self._root_real, path_real]) == self._root_real
        except ValueError:
            return False

    def _resolve_cache_path(self, key: str, original_path: Optional[str] = None, suffix: str = ".pt") -> str:
        rel = None
        if self._mirror_paths and original_path:
            original_abs = _abs_path(original_path)
            if self._is_within_root(original_abs):
                return original_abs
            rel = _mirror_subpath(original_abs)
            if rel:
                return os.path.join(self.root, self.namespace, rel)
        hashed = _hashed_subpath(key)
        return os.path.join(self.root, self.namespace, hashed) + suffix

    def expected_mirror_path(self, original_path: str) -> Optional[str]:
        if not self.enabled:
            return None
        if not self._mirror_paths:
            return None
        if self._is_within_root(original_path):
            return _abs_path(original_path)
        rel = _mirror_subpath(original_path)
        if not rel:
            return None
        return os.path.join(self.root, self.namespace, rel)

    def filelist_mirror_path(self, filelist_path: str) -> Optional[str]:
        if not self.enabled:
            return None
        if self._mirror_paths and self._is_within_root(filelist_path):
            return _abs_path(filelist_path)
        rel = None
        if self._mirror_paths:
            rel = _mirror_subpath(filelist_path)
        if not rel:
            rel = _hashed_subpath(filelist_path)
        return os.path.join(self.root, "filelists", self.namespace, rel)

    def ensure_copy(self, key: str, src_path: str) -> str:
        if not self.enabled:
            return src_path
        if self._is_within_root(src_path):
            return _abs_path(src_path)
        dst = self._resolve_cache_path(key, original_path=src_path, suffix=os.path.splitext(src_path)[1] or ".pt")
        if not os.path.exists(dst):
            _atomic_copy(src_path, dst)
        return dst

    def exists(self, key: str) -> bool:
        if not self.enabled:
            return False
        path = self._resolve_cache_path(key, suffix=".pt")
        return os.path.exists(path)

    def load_obj(self, key: str):
        if not self.enabled:
            return None
        path = self._resolve_cache_path(key, suffix=".pt")
        if not os.path.exists(path):
            return None
        return torch.load(path, map_location="cpu")

    def save_obj(self, key: str, obj) -> Optional[str]:
        if not self.enabled:
            return None
        path = self._resolve_cache_path(key, suffix=".pt")
        if os.path.exists(path):
            return path
        _atomic_torch_save(obj, path)
        return path

    def namespace_file_count(self, suffix: str = ".pt") -> int:
        if not (self.enabled and self.root and self.namespace):
            return 0
        namespace_dir = os.path.join(self.root, self.namespace)
        if not os.path.exists(namespace_dir):
            return 0
        total = 0
        for _dirpath, _dirnames, filenames in os.walk(namespace_dir):
            if suffix:
                total += sum(1 for name in filenames if name.endswith(suffix))
            else:
                total += len(filenames)
        return total
