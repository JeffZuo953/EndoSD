"""Compatibility helpers for backbone implementations."""

from __future__ import annotations

import importlib
import inspect
import logging
from types import ModuleType
from typing import Iterable, Sequence


_DEFAULT_BACKBONE_MODULES: Sequence[str] = (
    "depth_anything_v2.dinov2",
    "depth_anything_v2.dinov3",
    "multitask_moe_lora.depth_anything_v2.dinov2",
    "multitask_moe_lora.depth_anything_v2.dinov3",
)


def _patch_module(module: ModuleType, logger: logging.Logger | None = None) -> None:
    cls = getattr(module, "DinoVisionTransformer", None)
    if cls is None:
        return
    if getattr(cls, "_extra_token_compat_patched", False):
        return
    method = getattr(cls, "get_intermediate_layers", None)
    if method is None:
        return
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        setattr(cls, "_extra_token_compat_patched", True)
        return
    if "extra_tokens" in signature.parameters:
        setattr(cls, "_extra_token_compat_patched", True)
        return
    original = method

    def wrapper(self, *args, **kwargs):
        kwargs.pop("extra_tokens", None)
        return original(self, *args, **kwargs)

    wrapper.__name__ = original.__name__
    wrapper.__doc__ = original.__doc__
    setattr(cls, "get_intermediate_layers", wrapper)
    setattr(cls, "_extra_token_compat_patched", True)
    if logger is not None:
        logger.warning(
            "Patched %s.get_intermediate_layers to ignore unsupported `extra_tokens` kwarg.",
            cls.__name__,
        )


def ensure_backbone_extra_token_support(
    logger: logging.Logger,
    module_names: Iterable[str] | None = None,
) -> None:
    """Patch Dino backbones so `extra_tokens` kwargs are ignored when unsupported."""

    modules = module_names or _DEFAULT_BACKBONE_MODULES
    for name in modules:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to import %s for extra token patch: %s", name, exc)
            continue
        _patch_module(module, logger)
