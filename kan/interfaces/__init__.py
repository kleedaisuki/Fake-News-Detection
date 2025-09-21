# -*- coding: utf-8 -*-
"""
@file   kan/interfaces/__init__.py
@brief  KAN 稳定契约层（Interfaces）
@date   2025-09-21

@zh
  定义 KAN 的核心接口（Interface/Protocol）：实体链接、知识图谱、编码器。
  所有上游 pipelines/下游模块仅依赖这些接口，不直接耦合具体实现。
  遵循 "We don’t break userspace" 原则，保证向后兼容。
@en
  Stable contract layer for KAN: entity linking, knowledge graph, and encoders.
  Pipelines and modules depend only on these interfaces, not concrete implementations.
  Follow "We don’t break userspace" to ensure backward compatibility.
"""

# ---- 实体链接 (Entity Linking) ----
from .linker import (
    ILinker,
    Mention,
    LinkerError,
    RateLimitError,
)

# ---- 知识图谱 (Knowledge Graph) ----
from .kg import (
    IKG,
    KGNode,
    KGEdge,
    KGRelation,
    KGError,
    KGNotFound,
)

# ---- 编码器 (Encoders) ----
from .encoders import (
    ITextEncoder,
    IKnowledgeEncoder,
    SequenceBatch,
    EncoderOutput,
)

__all__ = [
    # linker
    "ILinker",
    "Mention",
    "LinkerError",
    "RateLimitError",
    # kg
    "IKG",
    "KGNode",
    "KGEdge",
    "KGRelation",
    "KGError",
    "KGNotFound",
    # encoders
    "ITextEncoder",
    "IKnowledgeEncoder",
    "SequenceBatch",
    "EncoderOutput",
]
