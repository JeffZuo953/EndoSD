import hashlib
import os
import shutil
import threading
from typing import Optional

import torch

_ENV_KEYS = ("LOCAL_CACHE_DIR", "LOCAL_CACHE_PATH")


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
    """Utility to mirror .pt files or processed samples under a hashed directory tree."""

    def __init__(self, root: Optional[str], namespace: str):
        self.root = _resolve_root(root)
        self.namespace = namespace.strip("/ ")

    @property
    def enabled(self) -> bool:
        return bool(self.root)

    def _path(self, key: str, suffix: str = ".pt") -> str:
        sub = _hashed_subpath(key)
        rel = os.path.join(self.namespace, sub)
        return os.path.join(self.root, rel) + suffix

    def ensure_copy(self, key: str, src_path: str) -> str:
        if not self.enabled:
            return src_path
        dst = self._path(key)
        if not os.path.exists(dst):
            _atomic_copy(src_path, dst)
        return dst

    def load_obj(self, key: str):
        if not self.enabled:
            return None
        path = self._path(key)
        if not os.path.exists(path):
            return None
        return torch.load(path, map_location="cpu")

    def save_obj(self, key: str, obj) -> Optional[str]:
        if not self.enabled:
            return None
        path = self._path(key)
        if os.path.exists(path):
            return path
        _atomic_torch_save(obj, path)
        return path
