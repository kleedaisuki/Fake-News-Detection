# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kan/modules/attention/ne2c.py
@brief  NE2C（News→Entities & Entity-Contexts）分层注意力实现（多头、带掩码、稳定契约）。

@zh
  对齐 `kan.interfaces.encoders.EncoderOutput` 约定：
  - news.last_hidden_state: [B, Lt, D]; news.pooled_state: [B, D]
  - entities.last_hidden_state: [B, Le, D]
  - contexts_last_hidden: [B, E, Lc, D]（其中 E==Le）
  - 掩码：news_mask [B,Lt] / entity_mask [B,Le] / contexts_mask [B,E,Lc]; 1 表示有效。
  本模块实现两级注意力：
    1) N-E（token→entity）：Q 来自 news.token 级表示，K/V 来自实体序列 q'；输出 token 级聚合 q_tok
    2) N-E2C（doc→context）：Q 来自 news.pooled p，K 来自实体序列 q'，V 来自“按上下文序列 r' 的掩码均值池化后的每实体向量”
  最终融合：fused = LN(news_tokens + Drop(q_tok_proj) + Drop(r_doc_proj broadcast))；
  同时返回权重 heatmaps：
    - weights['ne'] : [B, H, Lt, Le]
    - weights['e2c']: [B, H, 1, Le]（按论文定义，文级 Q 对实体打分）。

@en
  Implementation of NE2C hierarchical attention with masks and multi-head MHA.
  Shapes follow EncoderOutput contract. Two attentions:
    (1) N-E: token-level Q over entity keys/values → token-level aggregated q_tok.
    (2) N-E2C: doc-level Q over entity keys and per-entity context pooled values → r_doc.
  Fusion: fused = LN(tokens + Drop(q_tok_proj) + Drop(r_doc_proj broadcast)).
  Returns attention weights for observability.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
from torch import Tensor, nn

from kan.interfaces.encoders import EncoderOutput


# -----------------------------------------------------------------------------
# Output dataclass (保持与占位版一致)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class NE2CAttentionOutput:
    """@brief 结果容器 / Output container (bilingual).

    @zh
      - fused_states: 融合后的 token 序列表征 [B, Lt, D]
      - pooled: 文级池化（融合后）[B, D]
      - weights: 可选可视化权重 {'ne': [B,H,Lt,Le], 'e2c': [B,H,1,Le]}
    @en
      - fused_states: fused token representations [B, Lt, D]
      - pooled: fused pooled representation [B, D]
      - weights: optional attentions
    """

    fused_states: Tensor
    pooled: Tensor
    weights: Optional[Dict[str, Tensor]] = None


# -----------------------------------------------------------------------------
# NE2C Module
# -----------------------------------------------------------------------------


class NE2CAttention(nn.Module):
    r"""
    @brief NE2C 分层注意力（实现版）/ NE2C hierarchical attention (implemented).

    @zh
      符合 `kan.interfaces` 契约，提供：
      - 形状与掩码校验；
      - 多头注意力（可投影），dropout；
      - N-E（token→entity）与 N-E2C（doc→context）两级注意力；
      - 稳健的 masked softmax 与上下文均值池化；
      - 融合与 LayerNorm；
      - 可选返回权重 heatmaps 以便调试/可视化。

    @en
      Contract-aligned implementation with robust masking and logging-friendly weights.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        context_pooling: str = "attn",  # 目前实现使用 masked-mean；保留接口兼容
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dropout_p = float(dropout)
        self.context_pooling = str(context_pooling)
        self.use_bias = bool(use_bias)

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model({self.d_model}) must be divisible by n_heads({self.n_heads})"
            )
        self.d_head = self.d_model // self.n_heads

        # ---- Projections for N-E (token→entity) ----
        self.q_ne = nn.Linear(self.d_model, self.d_model, bias=use_bias)
        self.k_ne = nn.Linear(self.d_model, self.d_model, bias=use_bias)
        self.v_ne = nn.Linear(self.d_model, self.d_model, bias=use_bias)
        self.o_ne = nn.Linear(self.d_model, self.d_model, bias=use_bias)

        # ---- Projections for N-E2C (doc→context) ----
        self.q_e2c = nn.Linear(self.d_model, self.d_model, bias=use_bias)
        self.k_e2c = nn.Linear(self.d_model, self.d_model, bias=use_bias)
        self.v_e2c = nn.Linear(self.d_model, self.d_model, bias=use_bias)
        self.o_e2c = nn.Linear(self.d_model, self.d_model, bias=use_bias)

        # ---- Fusion & pooling ----
        self.dropout = nn.Dropout(self.dropout_p)
        self.ln_fuse = nn.LayerNorm(self.d_model)
        self.pooler = nn.Linear(self.d_model, self.d_model, bias=True)

        # 小门控，避免外部知识强行覆盖（sigmoid ∈ [0,1]）
        self.gate_ne = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.Sigmoid()
        )
        self.gate_ctx = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.Sigmoid()
        )

    # ------------------------ helpers ------------------------
    def _shape(self, x: Tensor | None) -> str:
        return "None" if x is None else "x".join(map(str, x.shape))

    @staticmethod
    def _mask_density(mask: Optional[Tensor]) -> Optional[float]:
        if mask is None:
            return None
        total = mask.numel()
        if total == 0:
            return 0.0
        valid = mask.to(dtype=torch.float32).sum().item()
        return float(valid / total)

    @staticmethod
    def _expand_mask(
        mask: Optional[Tensor], *, to_len: int, dim: int = -1
    ) -> Optional[Tensor]:
        """@brief 扩展 2D/3D 掩码到目标长度（沿指定 dim 重复）。/ Repeat mask along a dim to target length."""
        if mask is None:
            return None
        reps = [1] * mask.dim()
        if mask.shape[dim] == to_len:
            return mask
        reps[dim] = to_len
        return mask.unsqueeze(dim).repeat(reps).squeeze(dim)

    def _split_heads(self, x: Tensor) -> Tensor:
        # [B, L, D] -> [B, H, L, Dh]
        B, L, _ = x.shape
        x = x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        return x

    def _merge_heads(self, x: Tensor) -> Tensor:
        # [B, H, L, Dh] -> [B, L, D]
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def _masked_mean(self, x: Tensor, mask: Optional[Tensor], dim: int) -> Tensor:
        """@brief 带掩码的均值池化 / Masked mean along dim.
        @param x [B, ..., D]
        @param mask same shape as x without D (1 for valid)
        """
        if mask is None:
            return x.mean(dim=dim)
        m = mask.to(dtype=x.dtype)
        while m.dim() < x.dim():
            m = m.unsqueeze(-1)
        s = (x * m).sum(dim=dim)
        denom = m.sum(dim=dim).clamp_min(1e-6)
        return s / denom

    def _attend(
        self,
        q: Tensor,  # [B, Lq, D]
        k: Tensor,  # [B, Lk, D]
        v: Tensor,  # [B, Lk, D]
        *,
        key_mask: Optional[Tensor] = None,  # [B, Lk] (1 valid)
        scale: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """@brief 多头缩放点积注意力 / Scaled dot-product attention with masks.
        @return (out:[B,Lq,D], attn:[B,H,Lq,Lk])
        """
        B, Lq, _ = q.shape
        Bk, Lk, _ = k.shape
        assert B == Bk, "batch mismatch in attention"
        Dh = self.d_head
        qh = self._split_heads(q)  # [B,H,Lq,Dh]
        kh = self._split_heads(k)  # [B,H,Lk,Dh]
        vh = self._split_heads(v)  # [B,H,Lk,Dh]
        # [B,H,Lq,Dh] @ [B,H,Dh,Lk] -> [B,H,Lq,Lk]
        scale = scale or (Dh**-0.5)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
        if key_mask is not None:
            # 扩展到 [B,1,1,Lk]
            m = (key_mask == 0).view(B, 1, 1, Lk)
            scores = scores.masked_fill(m, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # [B,H,Lq,Lk] @ [B,H,Lk,Dh] -> [B,H,Lq,Dh]
        out_h = torch.matmul(attn, vh)
        out = self._merge_heads(out_h)  # [B,Lq,D]
        return out, attn

    # ------------------------ public API ------------------------
    def forward(
        self,
        *,
        news: EncoderOutput,
        entities: EncoderOutput,
        contexts_last_hidden: Tensor,  # [B, E, Lc, D]
        news_mask: Optional[Tensor] = None,
        entity_mask: Optional[Tensor] = None,
        contexts_mask: Optional[Tensor] = None,
        return_weights: bool = False,
    ) -> NE2CAttentionOutput:
        """@brief 前向：两级注意力 + 融合。
        @param news 文本编码输出（含 token & pooled）
        @param entities 实体序列编码输出（含 token & pooled；只用 token）
        @param contexts_last_hidden 实体上下文序列编码的 last_hidden_state [B,E,Lc,D]
        @param news_mask token 有效位 [B,Lt]
        @param entity_mask 实体有效位 [B,Le]
        @param contexts_mask 上下文有效位 [B,E,Lc]
        @param return_weights 是否返回注意力权重热力图
        @return NE2CAttentionOutput
        """
        # 形状
        B, Lt, Dn = news.last_hidden_state.shape
        Be, Le, De = entities.last_hidden_state.shape
        Bc, E, Lc, Dc = contexts_last_hidden.shape
        if not (Dn == De == Dc == self.d_model):
            raise ValueError(
                f"[NE2C] d_model mismatch: {self.d_model} vs (news {Dn}, entities {De}, contexts {Dc})"
            )
        if not (B == Be == Bc):
            raise ValueError(
                f"[NE2C] batch mismatch: news {B}, entities {Be}, contexts {Bc}"
            )
        if E != Le:
            raise ValueError(
                f"[NE2C] E({E}) must equal Le({Le}) for per-entity context alignment"
            )
        if news_mask is not None and tuple(news_mask.shape) != (B, Lt):
            raise ValueError("[NE2C] news_mask shape mismatch")
        if entity_mask is not None and tuple(entity_mask.shape) != (B, Le):
            raise ValueError("[NE2C] entity_mask shape mismatch")
        if contexts_mask is not None and tuple(contexts_mask.shape) != (B, E, Lc):
            raise ValueError("[NE2C] contexts_mask shape mismatch")

        # ---------------- N-E: token→entity ----------------
        # Q: token-level news, K/V: entity tokens
        q_tok = self.q_ne(news.last_hidden_state)  # [B,Lt,D]
        k_ent = self.k_ne(entities.last_hidden_state)  # [B,Le,D]
        v_ent = self.v_ne(entities.last_hidden_state)  # [B,Le,D]
        ne_out, ne_w = self._attend(q_tok, k_ent, v_ent, key_mask=entity_mask)
        ne_out = self.o_ne(ne_out)  # [B,Lt,D]

        # ---------------- N-E2C: doc→context ----------------
        # 先对每实体的上下文序列做 masked-mean → [B,E,D]
        ctx_mask_e_lc = contexts_mask  # [B,E,Lc] or None
        ctx_pooled = self._masked_mean(
            contexts_last_hidden, ctx_mask_e_lc, dim=2
        )  # [B,E,D]
        # Q: doc pooled news；K: entity tokens pooled(=平均) 或直接 token 平均（更稳健）
        ent_pooled = self._masked_mean(
            entities.last_hidden_state, entity_mask, dim=1
        )  # [B,D]
        k_e2c = self.k_e2c(
            entities.last_hidden_state
        )  # 仍使用 token 序列键，随后将 Lq=1 的 Q 对其打分
        # 为了稳定与直观，我们对实体 K 采用 token-mean 的等效实现：直接在注意力层使用 token 级 K；
        # 输出的权重对每个实体是“token 权重求和后”的效果。这里采用实体级 K 近似：对 k_e2c 做 masked-mean。
        k_e2c_ent = self._masked_mean(k_e2c, entity_mask, dim=1)  # [B,D]
        v_e2c_ent = self.v_e2c(ctx_pooled)  # [B,E,D]（E=Le）

        q_doc = self.q_e2c(news.pooled_state).unsqueeze(1)  # [B,1,D]
        # 将 K、V 对齐到实体级：K:[B,Le,D], V:[B,Le,D]
        # 这里 K 用 entities 的 token-mean 复制到 Le 维度（与 V 对齐）。
        k_doc = k_e2c_ent.unsqueeze(1).repeat(1, Le, 1)  # [B,Le,D]
        r_doc, e2c_w = self._attend(
            q_doc, k_doc, v_e2c_ent, key_mask=entity_mask
        )  # out:[B,1,D]
        r_doc = self.o_e2c(r_doc).squeeze(1)  # [B,D]

        # ---------------- 融合与池化 ----------------
        # 门控可抑制信息泄漏：q_tok、r_doc 各自一个门
        g_ne = self.gate_ne(ne_out)
        g_ctx = self.gate_ctx(r_doc)
        fused_tokens = self.ln_fuse(
            news.last_hidden_state
            + self.dropout(g_ne * ne_out)
            + self.dropout(g_ctx.unsqueeze(1) * r_doc.unsqueeze(1))
        )  # [B,Lt,D]

        # 文级 pooled：在融合 token 上 masked-mean，再与 doc-level上下文残差
        pooled_tokens = self._masked_mean(fused_tokens, news_mask, dim=1)  # [B,D]
        pooled = self.ln_fuse(
            pooled_tokens + 0.5 * r_doc + 0.5 * ent_pooled
        )  # 混合一部分实体摘要
        pooled = self.pooler(pooled)

        weights: Optional[Dict[str, Tensor]] = None
        if return_weights:
            # ne_w: [B,H,Lt,Le]
            # e2c_w: [B,H,1,Le]
            weights = {
                "ne": ne_w,
                "e2c": e2c_w,
            }

        return NE2CAttentionOutput(
            fused_states=fused_tokens, pooled=pooled, weights=weights
        )
