# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kg.py
@brief  Knowledge Graph (KG) access protocol and graph data models.
@date   2025-09-16

@zh
  知识图谱访问协议定义。提供“一跳邻居”查询与批量版本，既可返回
  仅实体ID的轻量视图，也可返回包含关系/方向的结构化视图，便于后续
  做消融或改进（如关系过滤、度裁剪等）。

@en
  Knowledge Graph access protocol. Provides one-hop neighbor query and batch
  variant, either as a light-weight ID list or fully-structured view with
  relations/directions—useful for ablations and future extensions.
"""

from dataclasses import dataclass, field
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
EntityID = str  # e.g., "Q76"
RelationID = str  # e.g., "P31"
Lang = str  # e.g., "en", "zh"


# ----------------------------- Data Models -----------------------------
@dataclass(frozen=True)
class KGNode:
    """
    @zh 知识图谱节点（实体）。
    @en Knowledge graph node (entity).
    """

    id: EntityID
    label: Optional[str] = None
    types: Tuple[EntityID, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KGRelation:
    """
    @zh 知识图谱关系（谓词）。
    @en Knowledge graph relation (predicate).
    """

    id: RelationID
    label: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KGEdge:
    """
    @zh 有向/无向边（取决于实现）；用于需要保留关系信息的场景。
    @en Edge (directed or not depending on implementation).
    """

    src: EntityID
    rel: RelationID
    dst: EntityID
    directed: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)


# ----------------------------- Exceptions ------------------------------
class KGError(RuntimeError):
    """Base error for KG access."""

    pass


class KGNotFound(KGError):
    """Raised when an entity is not found in KG."""

    pass


# ------------------------------ Protocol -------------------------------
@runtime_checkable
class IKG(Protocol):
    """
    @zh
      知识图谱协议。核心用于 KAN 的“实体上下文（entity context）”
      抽取：给定实体，返回其一跳邻居。
    @en
      Knowledge graph protocol. Focuses on extracting one-hop neighbors
      as "entity contexts" used by KAN.
    """

    def one_hop(
        self,
        entity_id: EntityID,
        *,
        max_neighbors: Optional[int] = None,
        return_ids_only: bool = True,
        relation_whitelist: Optional[Iterable[RelationID]] = None,
        relation_blacklist: Optional[Iterable[RelationID]] = None,
    ) -> List[EntityID] | List[KGEdge]:
        """
        @zh
          返回实体的一跳邻居。
          - return_ids_only=True：返回邻居实体ID列表（方向忽略，满足论文默认“取一跳邻居集合”的做法）。
          - return_ids_only=False：返回包含关系/方向的 KGEdge 列表，供拓展实验使用。
          - relation_white/blacklist：可选关系过滤（实现可忽略）。
        @en
          Return one-hop neighbors of an entity.
          - return_ids_only=True: return neighbor entity IDs (direction-agnostic).
          - return_ids_only=False: return KGEdge list with relations/directions.
          - relation filters are optional.
        """
        ...

    def one_hop_many(
        self,
        entity_ids: Sequence[EntityID],
        *,
        max_neighbors: Optional[int] = None,
        return_ids_only: bool = True,
        relation_whitelist: Optional[Iterable[RelationID]] = None,
        relation_blacklist: Optional[Iterable[RelationID]] = None,
    ) -> Dict[EntityID, List[EntityID] | List[KGEdge]]:
        """
        @zh 批量一跳邻居查询；返回字典映射。
        @en Batch one-hop neighbor query; returns a mapping.
        """
        ...

    def close(self) -> None:
        """
        @zh 释放资源（连接池、HTTP会话等）；幂等。
        @en Release resources (pools, sessions, etc.); idempotent.
        """
        ...
