# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   linker.py
@brief  Entity Linking interface (Protocol) and related data models.
@date   2025-09-16

@zh
  实体链接（Entity Linking, EL）协议定义。将文本中的实体提及映射到
  知识库唯一实体 ID（如 Wikidata QID）。提供单条与批量接口、缓存、
  关闭资源等钩子。

@en
  Entity Linking (EL) protocol. Map surface mentions in text to canonical
  entity identifiers (e.g., Wikidata QIDs). Batch API, caching hooks, and
  resource cleanup are included.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)


# ----------------------------- Type Aliases -----------------------------
EntityID = str  # e.g., "Q76" (Wikidata QID)
LanguageCode = str  # e.g., "en", "zh", "de"


# ----------------------------- Data Models -----------------------------
@dataclass(frozen=True)
class Mention:
    """
    @zh
      文本中的实体提及与链接结果。
    @en
      A surface mention and its linking result.
    """

    text: str  # 原文片段
    start_char: int  # 起始字符索引（含）
    end_char: int  # 结束字符索引（不含）
    entity_id: Optional[EntityID]  # 解析到的实体ID，可能为 None
    score: Optional[float] = None  # 置信度（实现自定义）
    provider: Optional[str] = None  # 实现方标识（如 "tagme"）
    extra: Mapping[str, Any] = field(default_factory=dict)  # 其他元数据


# ----------------------------- Exceptions ------------------------------
class LinkerError(RuntimeError):
    """Base error for linker implementations."""

    pass


class RateLimitError(LinkerError):
    """Raised when upstream EL service rate limit is hit."""

    pass


# ------------------------------ Protocol -------------------------------
@runtime_checkable
class ILinker(Protocol):
    """
    @zh
      实体链接协议。实现应是**无副作用**的纯函数式接口（同输入应得同输出），
      可通过 `set_cache_dir` 启用本地缓存（可选）。
    @en
      Entity linking protocol. Implementations should be functionally pure
      (same input -> same output) and may enable local caching via
      `set_cache_dir`.
    """

    def set_cache_dir(self, cache_dir: Optional[Path]) -> None:
        """
        @zh 设置（或清空）缓存目录；None 表示禁用缓存。
        @en Set (or clear) cache directory; None disables caching.
        """
        ...

    def link(
        self,
        text: str,
        *,
        lang: Optional[LanguageCode] = None,
        max_mentions: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Mention]:
        """
        @zh
          对单条文本做实体链接。
          - lang: 语言代码；None 由实现自行检测或使用默认。
          - max_mentions: 限制最多返回多少个提及（优先高分）。
          - score_threshold: 过滤低于阈值的结果。
        @en
          Link entities in a single text.
          - lang: language code; None to auto-detect or use default.
          - max_mentions: cap the number of mentions (score-desc sorted).
          - score_threshold: drop results below threshold.
        """
        ...

    def link_batch(
        self,
        texts: Sequence[str],
        *,
        lang: Optional[LanguageCode] = None,
        max_mentions: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[List[Mention]]:
        """
        @zh 批量实体链接；返回与输入等长的二维列表。
        @en Batch entity linking; returns a 2D list aligned with inputs.
        """
        ...

    def close(self) -> None:
        """
        @zh 释放外部资源（例如 HTTP 会话、进程池等）；幂等。
        @en Release external resources (HTTP sessions, pools, etc.); idempotent.
        """
        ...
