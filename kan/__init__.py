# -*- coding: utf-8 -*-
"""
@file   kan/__init__.py
@brief  KAN package entry point. Provides stable top-level API exposure.
@date   2025-09-21

@zh
  KAN（Knowledge-aware Attention Network）顶层入口：
  - 暴露 data/modules/interfaces/pipelines/utils 子包；
  - 提供集中式日志初始化方法；
  - 定义版本号。
@en
  KAN top-level entry point:
  - Expose subpackages: data/modules/interfaces/pipelines/utils
  - Provide centralized logging setup
  - Define version string
"""

from __future__ import annotations

__all__ = [
    "data",
    "modules",
    "interfaces",
    "pipelines",
    "utils",
    "get_version",
]

__version__ = "0.1.0"

# ---- Subpackages (lazy import to reduce startup overhead) ----
import importlib
import sys


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"kan.{name}")
    raise AttributeError(f"module 'kan' has no attribute '{name}'")


# ---- Logging Utilities ----
from pathlib import Path
from typing import Optional
from kan.utils import logging as _logging


def configure_logging(
    cfg_path: Optional[Path] = None, *, log_dir: Path = Path("logs")
) -> None:
    """
    @zh 初始化日志系统（集中式入口）。
    @en Configure logging system (centralized entry).
    """
    _logging.configure_logging(cfg_path=cfg_path, log_dir=log_dir)


def get_version() -> str:
    """
    @zh 返回当前 KAN 包版本号。
    @en Return current KAN package version string.
    """
    return __version__
