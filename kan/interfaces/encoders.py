# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   encoders.py
@brief  Encoder protocols for text, entities, and entity contexts.
@date   2025-09-16

@zh
  三类编码器的协议定义：TextEncoder / EntityEncoder / ContextEncoder。
  统一批处理输入/输出契约，输出序列级隐藏状态与池化向量，便于
  N-E 与 N-E2C 注意力分别取 Q/K/V。

@en
  Protocols for three encoders: text, entity, and entity-context encoders.
  Unify batched I/O contracts; return both sequence-level hidden states and
  pooled vectors, which are convenient for N-E and N-E2C attention as Q/K/V.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor


# ----------------------------- Batch Schema -----------------------------
@dataclass(frozen=True)
class SequenceBatch:
    """
    @zh
      通用序列批：兼容 HuggingFace/自定义 Tokenizer。
      - ids: 形如 [B, L] 的 token/索引 ID。
      - mask: [B, L] 的 0/1 掩码，1 表示有效。
      - type_ids: （可选）segment ids。
    @en
      Generic sequence batch compatible with HF/custom tokenizers.
      - ids: [B, L] token/index ids.
      - mask: [B, L] attention mask (1 = valid).
      - type_ids: optional segment ids.
    """

    ids: Tensor  # LongTensor [B, L]
    mask: Tensor  # LongTensor/BoolTensor [B, L]
    type_ids: Optional[Tensor] = None  # LongTensor [B, L] (optional)


@dataclass(frozen=True)
class EncoderOutput:
    """
    @zh
      编码器输出：包含序列最后一层隐藏状态与池化表示。
      注意：具体池化策略由实现决定（[CLS]、mean-pool等）。
    @en
      Encoder output with last hidden states and a pooled vector.
      Pooling strategy is implementation-defined (CLS/mean-pool/etc.).
    """

    last_hidden_state: Tensor  # [B, L, D]
    pooled_state: Tensor  # [B, D]


# ------------------------------ Protocols -------------------------------
@runtime_checkable
class ITextEncoder(Protocol):
    """
    @zh 文本编码器：S → p / P，用于生成新闻文本表示。
    @en Text encoder for news content representations.
    """

    @property
    def d_model(self) -> int:
        """@zh 模型隐层维度 D；@en hidden size D."""
        ...

    @property
    def device(self) -> torch.device:
        """@zh 当前设备；@en current device."""
        ...

    def encode(self, batch: SequenceBatch, *, train: bool = False) -> EncoderOutput:
        """
        @zh 将文本批编码为隐藏序列与池化向量。
        @en Encode text batch into hidden sequence and pooled vector.
        """
        ...

    def to(self, device: torch.device) -> "ITextEncoder":
        """@zh 迁移设备（就地或返回 self）；@en Move to device (in-place or return self)."""
        ...

    def close(self) -> None:
        """@zh 释放资源；@en Release resources."""
        ...


@runtime_checkable
class IKnowledgeEncoder(Protocol):
    """
    @zh
      知识编码器：实体序列 / 实体上下文序列 → 表征（q' 或 r'）。
      与 ITextEncoder 共享相同批接口，以统一 Q/K/V 维度匹配。
    @en
      Knowledge encoder for entity or entity-context sequences (q' or r').
      Shares the same batch interface as ITextEncoder to align Q/K/V dims.
    """

    @property
    def d_model(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    def encode(self, batch: SequenceBatch, *, train: bool = False) -> EncoderOutput: ...

    def to(self, device: torch.device) -> "IKnowledgeEncoder": ...

    def close(self) -> None: ...
