# -*- coding: utf-8 -*-
"""
@file   kan/data/__init__.py
@brief  数据加载与批处理入口（KAN 数据层）
@date   2025-09-21

@zh
  提供 KAN 的稳定数据接口：NewsRecord 数据结构、加载器（loaders）、批处理器（batcher）。
  这些 API 构成上游 pipelines 的唯一依赖，确保向后兼容。
@en
  Stable data entrypoints for KAN: NewsRecord schema, loaders, and batcher.
  These APIs are the sole dependencies for upstream pipelines, ensuring backward compatibility.
"""

from .loaders import (
    NewsRecord,
    FieldMap,
    DatasetConfig,
    loader_from_config,
    Dataset,
)
from .batcher import (
    TextConfig,
    EntityConfig,
    ContextConfig,
    BatcherConfig,
    Batcher,
    EntityVocab,
    PropertyVocab,
)

__all__ = [
    # loaders
    "NewsRecord",
    "FieldMap",
    "DatasetConfig",
    "loader_from_config",
    "Dataset",
    # batcher
    "TextConfig",
    "EntityConfig",
    "ContextConfig",
    "BatcherConfig",
    "Batcher",
    "EntityVocab",
    "PropertyVocab",
]
