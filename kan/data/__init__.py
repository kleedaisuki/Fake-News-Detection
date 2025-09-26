# -*- coding: utf-8 -*-
"""
@file   kan/data/__init__.py
@brief  数据层稳定入口（加载/批处理/知识抓取/向量化） / Stable entrypoints for KAN data layer.
@date   2025-09-26
@version 0.1.1

@zh
  提供上游 pipelines 的**唯一数据入口**，聚合以下契约并保持向后兼容（We don't break userspace!）：
  - 统一样本架构（NewsRecord）与数据加载（build_loader）
  - 批处理（Batcher & BatcherConfig）
  - 知识图谱一跳邻居获取（fetch_context & KGConfig）
  - 文本向量化（build_vectorizer & VectorizerConfig）
  - 历史别名兼容：`loader_from_config`、`Dataset`

@en
  Single point-of-entry for upstream pipelines with backward compatibility:
  - Unified sample schema (NewsRecord) & dataset building (build_loader)
  - Batching (Batcher & BatcherConfig)
  - KG one-hop fetching (fetch_context & KGConfig)
  - Text vectorization (build_vectorizer & VectorizerConfig)
  - Legacy aliases supported: `loader_from_config`, `Dataset`
"""

from __future__ import annotations

from typing import Any
import warnings
import logging

__all__ = [
    # loaders
    "NewsRecord",
    "FieldMap",
    "DatasetConfig",
    "BaseLoader",
    "build_loader",
    # batcher
    "TextConfig",
    "EntityConfig",
    "ContextConfig",
    "BatcherConfig",
    "Batcher",
    "EntityVocab",
    "PropertyVocab",
    # kg fetcher
    "KGConfig",
    "fetch_context",
    # vectorizer
    "VectorizerConfig",
    "BaseVectorizer",
    "build_vectorizer",
    # legacy aliases (kept for compatibility)
    "loader_from_config",
    "Dataset",
]

__version__ = "0.1.1"

LOGGER = logging.getLogger("kan.data.__init__")

# ---------------------------
# Loaders (schema & builder)
# ---------------------------
from .loaders import (  # type: ignore
    NewsRecord,
    FieldMap,
    DatasetConfig,
    BaseLoader,
    build_loader,
)

# -------------
# Batcher APIs
# -------------
from .batcher import (  # type: ignore
    TextConfig,
    EntityConfig,
    ContextConfig,
    BatcherConfig,
    Batcher,
    EntityVocab,
    PropertyVocab,
)

# ------------------------
# KG one-hop fetcher APIs
# ------------------------
from .kg_fetcher import (  # type: ignore
    KGConfig,
    fetch_context,
)

# ----------------
# Vectorizer APIs
# ----------------
from .vectorizer import (  # type: ignore
    VectorizerConfig,
    BaseVectorizer,
    build_vectorizer,
)

# ---------------------------------------------------------
# Legacy aliases (compatibility shim with deprecation note)
# ---------------------------------------------------------


def loader_from_config(cfg: DatasetConfig, /, **overrides: Any) -> BaseLoader:
    """@brief 兼容别名：转调 build_loader（中文）/ Legacy alias to build_loader (English).
    @param cfg: 数据集配置 / dataset config
    @param overrides: 额外覆盖参数 / extra overrides
    @return BaseLoader
    @note
      @zh 本函数已弃用，请改用 `build_loader`；当前版本仍保留并仅提示一次警告。
      @en Deprecated; use `build_loader`. Kept with a one-time warning.
    """
    warnings.warn(
        "kan.data.loader_from_config is deprecated; use kan.data.build_loader instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # 允许调用方以 kwargs 形式补充配置的实现差异
    if overrides:
        try:
            # 尝试浅拷贝后更新（保持最小惊讶原则）
            cfg = type(cfg)(**{**cfg.__dict__, **overrides})  # dataclass-friendly
        except Exception:
            LOGGER.debug(
                "loader_from_config: ignore overrides merge failure.", exc_info=False
            )
    return build_loader(cfg)


# 历史名称 Dataset → BaseLoader（保持 import 兼容）
Dataset = BaseLoader
