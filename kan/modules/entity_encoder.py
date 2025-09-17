# -*- coding: utf-8 -*-
"""
@file: entity_encoder.py

中英双语文档 / Bilingual Docstring
===================================

目的 / Purpose
--------------
实现 **实体编码器（Entity Encoder）**：将实体 ID 及（可选）其一跳邻居上下文编码为统一维度
的序列表示与聚合表示，供上层 NE / NE2C 注意力模块消费。

契约 / Contract（与 interfaces 对齐）
-----------------------------------
- 输入键（全部为 keyword-only）：
  - `entity_ids: LongTensor[B, E]`（实体 ID，含 padding_idx）
  - `entity_mask: Bool/Long[B, E]`（1 有效）可选
  - `context_ids: LongTensor[B, E, Lc]`（每个实体的一跳邻居 ID）可选
  - `context_mask: Bool/Long[B, E, Lc]`（1 有效）可选
- 输出键：
  - `entities_last_hidden: FloatTensor[B, E, D]`（q'）
  - `entities_pooled: FloatTensor[B, D]`（跨实体聚合）
  - `entities_mask: LongTensor[B, E]`
  - `contexts_last_hidden: FloatTensor[B, E, D]`（r'；对每个实体的上下文先聚合到一个向量再编码）
  - `contexts_pooled: FloatTensor[B, D]`（跨实体聚合）
  - `contexts_mask: LongTensor[B, E]`（由 context_mask 聚合得到）
  - 另外暴露（可选，用于调试/增强）：`raw_contexts_last_hidden: FloatTensor[B, E, Lc, D]`

设计要点 / Design Highlights
----------------------------
1) **嵌入来源**：`nn.Embedding` 查表，支持从 `.pt/.bin/.npy` 载入预训练实体向量；`
   padding_idx/unk_idx` 可配；可冻结或训练。
2) **知识编码器（TransformerEncoder 可选）**：对实体序列（维度 E）做自注意力；
   对聚合后的上下文序列做自注意力；`layers=0` 时跳过（恒等映射）。
3) **上下文聚合**：按论文 KAN 的做法，先对每个实体的邻居取平均得到 `ec'(ei)`，
   再对整条序列编码；也支持 `max` 聚合；并可返回未聚合的 `raw_contexts_last_hidden` 以供实验。
4) **日志**：命名 logger `kan.entity_encoder`；复用项目集中式配置。
5) **Windows 友好**：不依赖 fork；仅用 PyTorch/标准库。

安全与兼容 / Stability & Compatibility
--------------------------------------
- **We don't break userspace!**
  - 默认值齐全；新增字段仅扩展，不改变现有键名与张量形状。
  - `D = cfg.embedding_dim` 与其它编码器对齐。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
import logging
from pathlib import Path

import torch
from torch import nn, Tensor

logger = logging.getLogger("kan.entity_encoder")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class EntityEncoderConfig:
    """实体编码器配置 / Configuration for Entity Encoder.

    Attributes
    ----------
    vocab_size : int
        实体词表大小（含 padding/unk）。
    embedding_dim : int
        向量维度（即 D）。
    padding_idx : int
        padding 的索引；mask=0 时会被忽略。
    unk_idx : int
        未登录实体的索引（可与 padding 相同）。
    embeddings_path : Optional[str]
        预训练向量文件（.pt/.bin 为 `state_dict` 或张量，.npy 为矩阵）。若为空则随机初始化。
    trainable : bool
        嵌入是否参与训练。
    xformer_layers : int
        TransformerEncoder 层数；0 表示跳过编码层（恒等映射）。
    xformer_heads : int
        Multi-head 数；需整除 `embedding_dim`。
    xformer_ffn_dim : Optional[int]
        前馈隐层维度，默认 `4 * embedding_dim`。
    dropout : float
        dropout 概率。
    entity_pooling : Literal["mean", "max"]
        跨实体的聚合策略（用于 `entities_pooled` / `contexts_pooled`）。
    context_inner_pooling : Literal["mean", "max"]
        对每个实体的邻居（维 Lc）如何聚合为一个向量 `ec'(ei)`。
    return_raw_contexts : bool
        是否回传 `raw_contexts_last_hidden` 便于调试。
    device : Optional[str]
        计算设备；None 时自动选择。
    """

    vocab_size: int
    embedding_dim: int
    padding_idx: int = 0
    unk_idx: int = 0
    embeddings_path: Optional[str] = None
    trainable: bool = False
    xformer_layers: int = 1
    xformer_heads: int = 4
    xformer_ffn_dim: Optional[int] = None
    dropout: float = 0.1
    entity_pooling: Literal["mean", "max"] = "mean"
    context_inner_pooling: Literal["mean", "max"] = "mean"
    return_raw_contexts: bool = False
    device: Optional[str] = None


# -----------------------------------------------------------------------------
# Entity Encoder
# -----------------------------------------------------------------------------
class EntityEncoder(nn.Module):
    """实体 + 上下文编码器 / Entities & Contexts encoder.

    - 输入：实体 ID `[B,E]`，以及可选的邻居 ID `[B,E,Lc]`；
    - 输出：`entities_last_hidden: [B,E,D]`，`contexts_last_hidden: [B,E,D]` 等。
    """

    def __init__(self, cfg: EntityEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # ---- Embedding table ----
        self.emb = nn.Embedding(
            cfg.vocab_size, cfg.embedding_dim, padding_idx=cfg.padding_idx
        )
        if cfg.embeddings_path:
            self._load_pretrained(Path(cfg.embeddings_path))
        self.emb.weight.requires_grad = bool(cfg.trainable)

        # ---- Positional encoding for set/sequence over E ----
        # 这里采用可学习位置向量（对实体序列维度 E）；如果不需要，影响也接近恒等
        self.pos_entity = nn.Embedding(
            1024, cfg.embedding_dim
        )  # 支持至多 1024 个实体/样本

        # ---- Transformer for entities ----
        self.ent_encoder = self._maybe_build_encoder()
        # ---- Transformer for contexts (after inner pooling) ----
        self.ctx_encoder = self._maybe_build_encoder()

        self.dropout = nn.Dropout(cfg.dropout)

        logger.info(
            "EntityEncoder init: vocab=%d, D=%d, trainable=%s, layers=%d, heads=%d",
            cfg.vocab_size,
            cfg.embedding_dim,
            cfg.trainable,
            cfg.xformer_layers,
            cfg.xformer_heads,
        )

    # ----------------------------- Forward -----------------------------
    def forward(
        self,
        *,
        entity_ids: Tensor,  # [B,E]
        entity_mask: Optional[Tensor] = None,  # [B,E]
        context_ids: Optional[Tensor] = None,  # [B,E,Lc]
        context_mask: Optional[Tensor] = None,  # [B,E,Lc]
    ) -> Dict[str, Tensor]:
        cfg = self.cfg
        device = entity_ids.device

        # -------- Entities --------
        B, E = entity_ids.shape
        ent_mask = self._ensure_mask(
            entity_mask, entity_ids != cfg.padding_idx
        )  # [B,E]
        ent_tok = self.emb(entity_ids)  # [B,E,D]
        ent_tok = ent_tok + self.pos_entity(torch.arange(E, device=device))[None, :, :]
        ent_tok = self.dropout(ent_tok)

        if self.ent_encoder is not None:
            # TransformerEncoder(batch_first=True) 接收 [B,E,D]，key_padding_mask=True 表示位置被 mask
            ent_last = self.ent_encoder(ent_tok, src_key_padding_mask=(ent_mask == 0))
        else:
            ent_last = ent_tok
        ent_pooled = self._pool_over_len(
            ent_last, ent_mask, mode=cfg.entity_pooling
        )  # [B,D]

        # -------- Contexts (neighbors per entity) --------
        ctx_last: Tensor
        ctx_mask_over_E: Tensor
        raw_ctx_last: Optional[Tensor] = None

        if context_ids is not None:
            # raw contexts embedding
            ctx_emb_raw = self.emb(context_ids)  # [B,E,Lc,D]
            raw_ctx_last = ctx_emb_raw if cfg.return_raw_contexts else None
            if context_mask is None:
                context_mask = context_ids != cfg.padding_idx

            # inner pooling over Lc -> [B,E,D]
            ctx_vec = self._pool_over_Lc(
                ctx_emb_raw, context_mask, mode=cfg.context_inner_pooling
            )
            ctx_vec = self.dropout(ctx_vec)

            # contexts mask over E：某实体若无任何有效邻居，则置 0
            ctx_has = (context_mask.long().sum(dim=-1) > 0).long()  # [B,E]

            if self.ctx_encoder is not None:
                ctx_last = self.ctx_encoder(
                    ctx_vec, src_key_padding_mask=(ctx_has == 0)
                )  # [B,E,D]
            else:
                ctx_last = ctx_vec
            ctx_mask_over_E = ctx_has
        else:
            # 若未提供上下文，返回零向量但给出掩码 0（避免影响下游）
            ctx_last = torch.zeros_like(ent_last)
            ctx_mask_over_E = torch.zeros_like(ent_mask)

        ctx_pooled = self._pool_over_len(
            ctx_last, ctx_mask_over_E, mode=cfg.entity_pooling
        )

        out = {
            "entities_last_hidden": ent_last,
            "entities_pooled": ent_pooled,
            "entities_mask": ent_mask.long(),
            "contexts_last_hidden": ctx_last,
            "contexts_pooled": ctx_pooled,
            "contexts_mask": ctx_mask_over_E.long(),
        }
        if raw_ctx_last is not None:
            out["raw_contexts_last_hidden"] = raw_ctx_last

        # 形状日志（DEBUG 级别，避免刷屏）
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "shapes: ent_last=%s, ent_pooled=%s, ctx_last=%s, ctx_pooled=%s",
                tuple(ent_last.shape),
                tuple(ent_pooled.shape),
                tuple(ctx_last.shape),
                tuple(ctx_pooled.shape),
            )
        return out

    # --------------------------- Helpers ---------------------------
    @property
    def d_model(self) -> int:
        return self.cfg.embedding_dim

    def _maybe_build_encoder(self) -> Optional[nn.Module]:
        cfg = self.cfg
        if cfg.xformer_layers <= 0:
            return None
        nhead = cfg.xformer_heads
        d_model = cfg.embedding_dim
        if d_model % nhead != 0:
            # 自动修正 head 数，避免报错
            for h in (8, 4, 2, 1):
                if d_model % h == 0:
                    nhead = h
                    logger.warning("xformer_heads 不整除 D，自动调整为 %d", nhead)
                    break
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=cfg.xformer_ffn_dim or (4 * d_model),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=cfg.xformer_layers)

    def _pool_over_len(
        self, x: Tensor, mask: Tensor, *, mode: Literal["mean", "max"]
    ) -> Tensor:
        """跨长度维度（实体维 E）做聚合，mask=1 有效。
        x: [B,E,D], mask: [B,E] -> [B,D]
        """
        mask = mask.to(dtype=x.dtype)
        if mode == "mean":
            denom = mask.sum(dim=1).clamp(min=1e-6).unsqueeze(-1)
            return (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        elif mode == "max":
            very_small = torch.finfo(x.dtype).min
            masked = x.masked_fill(mask.unsqueeze(-1) == 0, very_small)
            return masked.max(dim=1).values
        else:
            raise ValueError(f"未知的 pooling 策略: {mode}")

    def _pool_over_Lc(
        self, x: Tensor, mask: Tensor, *, mode: Literal["mean", "max"]
    ) -> Tensor:
        """对 Lc 维（邻居）聚合到每个实体一个向量。
        x: [B,E,Lc,D], mask: [B,E,Lc] -> [B,E,D]
        """
        mask = mask.to(dtype=x.dtype)
        if mode == "mean":
            denom = mask.sum(dim=2).clamp(min=1e-6).unsqueeze(-1)
            return (x * mask.unsqueeze(-1)).sum(dim=2) / denom
        elif mode == "max":
            very_small = torch.finfo(x.dtype).min
            masked = x.masked_fill(mask.unsqueeze(-1) == 0, very_small)
            return masked.max(dim=2).values
        else:
            raise ValueError(f"未知的 context_inner_pooling: {mode}")

    @staticmethod
    def _ensure_mask(mask: Optional[Tensor], default: Tensor) -> Tensor:
        if mask is None:
            return default.to(dtype=torch.long)
        return mask.to(dtype=torch.long)

    # ---- Embedding loader ----
    def _load_pretrained(self, path: Path) -> None:
        """从磁盘加载预训练实体向量（.pt/.bin/.npy）。形状需为 [vocab_size, embedding_dim]。
        若权重为 dict/`state_dict`，则尝试键 `weight` / `embeddings` / 第一项。
        """
        if not path.exists():
            raise FileNotFoundError(f"预训练向量文件不存在: {path}")
        if path.suffix.lower() == ".npy":
            import numpy as np  # 延迟依赖

            arr = np.load(path)
            wt = torch.tensor(arr, dtype=self.emb.weight.dtype)
        else:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, torch.Tensor):
                wt = obj
            elif isinstance(obj, dict):
                if "weight" in obj:
                    wt = obj["weight"]
                elif "embeddings" in obj:
                    wt = obj["embeddings"]
                else:
                    # 取第一个张量型值
                    val = next(
                        (v for v in obj.values() if isinstance(v, torch.Tensor)), None
                    )
                    if val is None:
                        raise ValueError("无法从 state_dict 中找到权重张量")
                    wt = val
            else:
                raise ValueError("不支持的预训练权重文件格式")
        if wt.shape != self.emb.weight.shape:
            raise ValueError(
                f"预训练形状不匹配: got {tuple(wt.shape)} expect {tuple(self.emb.weight.shape)}"
            )
        with torch.no_grad():
            self.emb.weight.copy_(wt)
        logger.info("Loaded pretrained entity embeddings from %s", path)


# -----------------------------------------------------------------------------
# Factory Helper
# -----------------------------------------------------------------------------


def build_entity_encoder(cfg: EntityEncoderConfig) -> EntityEncoder:
    enc = EntityEncoder(cfg)
    # 将模块移动到设备（若指定）
    if cfg.device:
        enc = enc.to(torch.device(cfg.device))
    else:
        enc = enc.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Entity encoder built: %s", cfg)
    return enc


# -----------------------------------------------------------------------------
# Self-test (optional manual)
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    cfg = EntityEncoderConfig(
        vocab_size=1000, embedding_dim=128, xformer_layers=1, xformer_heads=4
    )
    enc = build_entity_encoder(cfg)
    B, E, Lc = 2, 5, 7
    entity_ids = torch.randint(low=1, high=cfg.vocab_size, size=(B, E))
    entity_ids[:, -1] = cfg.padding_idx  # pad 一个位置
    context_ids = torch.randint(low=1, high=cfg.vocab_size, size=(B, E, Lc))
    context_ids[:, :, -2:] = cfg.padding_idx  # 每实体 pad 两个邻居
    out = enc(entity_ids=entity_ids, context_ids=context_ids)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(k, tuple(v.shape))
