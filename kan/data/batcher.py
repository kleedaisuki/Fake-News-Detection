# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kan/data/batcher.py
@brief  Collate KAN NewsRecord into tensors for Text/Entity/Context branches.
@date   2025-09-16

@zh
  统一批处理（batching）模块：将 `List[NewsRecord]`（来自 loaders）整理为三路张量：
  - Text branch:  可选 HuggingFace 分词张量（input_ids/attention_mask/...）与/或向量器特征 text_vec
  - Entity branch: 实体 QID -> 索引化并按样本维度填充（[B, E])
  - Context branch: 每实体的一跳邻居（可含属性标签），填充至 ([B, E, N])，并生成掩码

@en
  Collate `List[NewsRecord]` into tensors for three branches: Text / Entity / Context.
  Text can be tokenized (HF) and/or vectorized; Entities and 1-hop neighbors are indexed
  with per-batch padding and masks.

@notes
  - 该模块不做模型前向；仅负责把稳定 Schema v1 组织成可喂给 encoders/NE/NE2C 的张量。
  - Windows 友好；仅依赖 torch 与可选 transformers/Vectorizer。
  - 针对超长（E/N 超限）采取截断策略，保证 **不破坏用户态接口**（We don't break userspace）。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from collections import defaultdict
import logging

from pathlib import Path

LOGGER = logging.getLogger("kan.data.batcher")

# Optional deps ---------------------------------------------------------------
try:
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = Any  # type: ignore

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore

# Project schema --------------------------------------------------------------
from kan.data.loaders import NewsRecord
from kan.data.vectorizer import BaseVectorizer, VectorizerConfig, build_vectorizer

# =============================================================================
# Vocabularies (Entity / Property)
# =============================================================================


class EntityVocab:
    """Bidirectional mapping for entity QIDs.

    Reserved indices:
      0: <PAD>
      1: <UNK>
    """

    PAD = 0
    UNK = 1

    def __init__(self) -> None:
        self.q2i: Dict[str, int] = {}
        self.i2q: List[str] = ["<PAD>", "<UNK>"]

    def add(self, qid: str) -> int:
        if not qid:
            return self.UNK
        if qid in self.q2i:
            return self.q2i[qid]
        idx = len(self.i2q)
        self.q2i[qid] = idx
        self.i2q.append(qid)
        return idx

    def get(self, qid: Optional[str]) -> int:
        if not qid:
            return self.UNK
        return self.q2i.get(qid, self.UNK)

    def __len__(self) -> int:
        return len(self.i2q)


class PropertyVocab:
    """Bidirectional mapping for relation properties (Pxxxx).

    Reserved indices:
      0: <PAD>
      1: <NONE>   # when neighbor has no explicit property
    """

    PAD = 0
    NONE = 1

    def __init__(self) -> None:
        self.p2i: Dict[str, int] = {}
        self.i2p: List[str] = ["<PAD>", "<NONE>"]

    def add(self, pid: str) -> int:
        if not pid:
            return self.NONE
        if pid in self.p2i:
            return self.p2i[pid]
        idx = len(self.i2p)
        self.p2i[pid] = idx
        self.i2p.append(pid)
        return idx

    def get(self, pid: Optional[str]) -> int:
        if pid is None:
            return self.NONE
        return self.p2i.get(pid, self.NONE)

    def __len__(self) -> int:
        return len(self.i2p)


# =============================================================================
# Config
# =============================================================================


@dataclass
class TextConfig:
    tokenizer_backend: Optional[str] = "hf"  # 'hf' or None
    tokenizer_name: Optional[str] = None  # e.g., 'bert-base-uncased'
    max_length: int = 256
    pad_to_max: bool = True
    truncation: bool = True
    return_token_type_ids: bool = False
    # vectorizer path
    vectorizer: Optional[VectorizerConfig] = None  # if provided -> compute text_vec


@dataclass
class EntityConfig:
    max_entities: int = 64  # per record


@dataclass
class ContextConfig:
    max_neighbors: int = 32  # per entity
    keep_properties: bool = True  # whether to carry relation property ids


@dataclass
class BatcherConfig:
    text: TextConfig = field(default_factory=TextConfig)
    entity: EntityConfig = field(default_factory=EntityConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    device: Optional[str] = None  # collate output device ('cpu' recommended)


# =============================================================================
# Batcher
# =============================================================================


class Batcher:
    """Batcher to collate NewsRecord list into tensors for encoders/attention.

    Output dict contains (keys may be omitted if corresponding feature disabled):
      - ids: List[str]
      - text: List[str]
      - text_tok: {input_ids, attention_mask, (token_type_ids)}  # torch.LongTensor [B, T]
      - text_vec: torch.FloatTensor [B, D]                        # if vectorizer configured
      - ent_ids: torch.LongTensor [B, E]
      - ent_mask: torch.BoolTensor  [B, E]
      - ctx_ids: torch.LongTensor [B, E, N]
      - ctx_prop: torch.LongTensor [B, E, N]  # if keep_properties else omitted
      - ctx_mask: torch.BoolTensor  [B, E, N]
    """

    def __init__(
        self,
        cfg: BatcherConfig,
        entity_vocab: Optional[EntityVocab] = None,
        property_vocab: Optional[PropertyVocab] = None,
        tokenizer: Optional[Any] = None,
        vectorizer: Optional[BaseVectorizer] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("Batcher requires 'torch'. Please install PyTorch.")
        self.cfg = cfg
        self.entity_vocab = entity_vocab or EntityVocab()
        self.property_vocab = property_vocab or PropertyVocab()
        self.device = cfg.device or "cpu"
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        # Lazy init tokenizer/vectorizer if configs provided
        if (
            self.tokenizer is None
            and cfg.text.tokenizer_backend == "hf"
            and cfg.text.tokenizer_name
        ):
            if AutoTokenizer is None:
                raise RuntimeError(
                    "HuggingFace tokenizer requested but 'transformers' not installed."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.text.tokenizer_name)
        if self.vectorizer is None and cfg.text.vectorizer is not None:
            self.vectorizer = build_vectorizer(cfg.text.vectorizer)
        LOGGER.info(
            "Batcher ready: device=%s, tok=%s, vec=%s",
            self.device,
            getattr(self.tokenizer, "__class__", type(None)).__name__,
            getattr(self.vectorizer, "__class__", type(None)).__name__,
        )

    # ---------------------------------------------------------------------
    # Vocabulary utilities
    # ---------------------------------------------------------------------
    def build_vocabs(self, records: Sequence[NewsRecord]) -> None:
        """Populate entity/property vocabs from a corpus (usually training split)."""
        for rec in records:
            # entities
            for q in rec.entities or []:
                if q and isinstance(q, str):
                    self.entity_vocab.add(q)
            # contexts
            for q, neighs in (rec.contexts or {}).items():
                self.entity_vocab.add(q)
                if not neighs:
                    continue
                for item in neighs:
                    n_q, p = _parse_neighbor(item)
                    if n_q:
                        self.entity_vocab.add(n_q)
                    if self.cfg.context.keep_properties and p:
                        self.property_vocab.add(p)

    # ---------------------------------------------------------------------
    # Collate
    # ---------------------------------------------------------------------
    def collate(self, records: Sequence[NewsRecord]) -> Dict[str, Any]:
        B = len(records)
        ids = [r.id for r in records]
        texts = [r.text for r in records]
        out: Dict[str, Any] = {"ids": ids, "text": texts}

        # Text tokenization -------------------------------------------------
        if self.tokenizer is not None:
            tok = self.tokenizer(
                texts,
                padding=("max_length" if self.cfg.text.pad_to_max else True),
                truncation=self.cfg.text.truncation,
                max_length=self.cfg.text.max_length,
                return_tensors="pt",
            )
            text_tok = {
                "input_ids": tok["input_ids"].to(self.device),
                "attention_mask": tok["attention_mask"].to(self.device),
            }
            if self.cfg.text.return_token_type_ids and "token_type_ids" in tok:
                text_tok["token_type_ids"] = tok["token_type_ids"].to(self.device)
            out["text_tok"] = text_tok

        # Text vectorization -----------------------------------------------
        if self.vectorizer is not None:
            vec = self.vectorizer.encode_texts(texts)  # CPU tensor by contract
            out["text_vec"] = vec.to(self.device)

        # Entities ----------------------------------------------------------
        E = self.cfg.entity.max_entities
        ent_ids = torch.full(
            (B, E), fill_value=EntityVocab.PAD, dtype=torch.long, device=self.device
        )
        ent_mask = torch.zeros((B, E), dtype=torch.bool, device=self.device)
        # for contexts construction per batch
        batch_entities: List[List[str]] = []
        for i, rec in enumerate(records):
            ents = [
                q
                for q in (rec.entities or [])
                if isinstance(q, str) and q.startswith("Q")
            ]
            if len(ents) > E:
                ents = ents[:E]
            batch_entities.append(ents)
            for j, q in enumerate(ents):
                ent_ids[i, j] = self.entity_vocab.get(q)
                ent_mask[i, j] = True
        out["ent_ids"], out["ent_mask"] = ent_ids, ent_mask

        # Contexts ----------------------------------------------------------
        N = self.cfg.context.max_neighbors
        ctx_ids = torch.full(
            (B, E, N), fill_value=EntityVocab.PAD, dtype=torch.long, device=self.device
        )
        ctx_mask = torch.zeros((B, E, N), dtype=torch.bool, device=self.device)
        ctx_prop: Optional[Tensor] = None
        if self.cfg.context.keep_properties:
            ctx_prop = torch.full(
                (B, E, N),
                fill_value=PropertyVocab.PAD,
                dtype=torch.long,
                device=self.device,
            )
        for i, rec in enumerate(records):
            # build map for quick access; if contexts missing, try defaults
            cdict = rec.contexts or {}
            for j, q in enumerate(batch_entities[i]):
                neighs = cdict.get(q, [])
                # fallback: if no explicit contexts for q, leave zeros
                if not neighs:
                    continue
                # normalize and clip
                pairs: List[Tuple[str, Optional[str]]] = []
                for item in neighs:
                    n_q, p = _parse_neighbor(item)
                    if n_q and n_q.startswith("Q"):
                        pairs.append((n_q, p))
                if len(pairs) > N:
                    pairs = pairs[:N]
                for k, (n_q, p) in enumerate(pairs):
                    ctx_ids[i, j, k] = self.entity_vocab.get(n_q)
                    ctx_mask[i, j, k] = True
                    if ctx_prop is not None:
                        ctx_prop[i, j, k] = self.property_vocab.get(p)
        out["ctx_ids"], out["ctx_mask"] = ctx_ids, ctx_mask
        if ctx_prop is not None:
            out["ctx_prop"] = ctx_prop

        return out


# =============================================================================
# Utilities
# =============================================================================


def _parse_neighbor(item: Any) -> Tuple[Optional[str], Optional[str]]:
    """Parse neighbor spec which may be:
       - 'Qxxxx'
       - 'Pxxxx=Qyyyy'
       - other nonconforming values -> (None, None)
    Returns (neighbor_qid, property_id)
    """
    if not isinstance(item, str):
        return None, None
    s = item.strip()
    if not s:
        return None, None
    if "=" in s:
        p, q = s.split("=", 1)
        p = p.strip()
        q = q.strip()
        # 严格模式：只有 "Pxxxx=Qyyyy" 才算合规；否则整体视为无效
        if p.startswith("P") and q.startswith("Q"):
            return q, p
        return None, None
    else:
        return (s if s.startswith("Q") else None), None


# =============================================================================
# Doxygen examples (bilingual)
# =============================================================================

__doc_examples__ = r"""
/**
 * @zh 用法（最小可运行）：
 * ```python
 * from kan.data.loaders import NewsRecord
 * from kan.data.batcher import Batcher, BatcherConfig, TextConfig, EntityConfig, ContextConfig
 * # 1) 构造 batcher（可同时启用 tokenizer 与 vectorizer；都可缺省）
 * cfg = BatcherConfig(
 *   text=TextConfig(tokenizer_backend='hf', tokenizer_name='bert-base-uncased', max_length=128,
 *                   vectorizer=None),  # 或传入 VectorizerConfig
 *   entity=EntityConfig(max_entities=8),
 *   context=ContextConfig(max_neighbors=16, keep_properties=True),
 *   device='cpu'
 * )
 * batcher = Batcher(cfg)
 * # 2) 用训练集先构建词表（推荐）
 * train_records = [
 *   NewsRecord(id='a', text='Barack Obama visited Berlin.', label=1,
 *              entities=['Q76'], contexts={'Q76': ['Q183','P27=Q30']}, meta={}),
 * ]
 * batcher.build_vocabs(train_records)
 * # 3) collate
 * batch = batcher.collate(train_records)
 * print(batch['ent_ids'].shape, batch['ctx_ids'].shape)
 * ```
 *
 * @en Usage with sentence embeddings:
 * ```python
 * from kan.modules.vectorizer import VectorizerConfig
 * cfg = BatcherConfig(text=TextConfig(tokenizer_backend=None,
 *                                     vectorizer=VectorizerConfig(backend='st', model_name='all-mpnet-base-v2')))
 * batcher = Batcher(cfg)
 * batch = batcher.collate([NewsRecord(id='x', text='hello world', label=0, entities=[], contexts={}, meta={})])
 * print(batch['text_vec'].shape)  # -> [1, D]
 * ```
 */
"""
