# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import logging
import torch
from torch import Tensor, nn

from kan.interfaces.encoders import EncoderOutput


@dataclass(frozen=True)
class NEAttentionOutput:
    """@zh/EN 见注：仅保留数据容器；实现方决定是否返回权重。"""

    fused_states: Tensor  # [B, Lt, D]
    pooled: Tensor  # [B, D]
    attn_weights: Optional[Tensor] = None  # [B, H, Lt, Le]


class NEAttention(nn.Module):
    r"""
    @zh
      新闻-实体（N-E）跨注意力（cross-attention）。
      Q ← news.last_hidden_state，[B,Lt,D]
      K,V ← entities.last_hidden_state，[B,Le,D]
      输出 fused_states 为在文本每个位置融合实体后的序列表征（[B,Lt,D]），并对其做掩码池化得到 pooled（[B,D]）。
      - 支持可选返回多头注意力权重（[B,H,Lt,Le]）；
      - 对齐接口契约与 mask 语义：mask=1 表示有效，内部会按 PyTorch 语义取反给 key_padding_mask。
      - 强化鲁棒：当某样本实体全 padding 时降级为“无 key mask”并写告警，避免 NaN。
    @en
      News→Entities cross-attention using PyTorch MultiheadAttention.
      Q from news [B,Lt,D]; K,V from entities [B,Le,D]. Returns fused states and a masked-mean pooled vector.
      Supports optional per-head attention weights and robust fallbacks when all-keys are masked.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.use_bias = bool(use_bias)
        self.log = logging.getLogger("kan.modules.attention.ne.NEAttention")
        self.log.info(
            "NEAttention init d_model=%d n_heads=%d dropout=%.3f use_bias=%s",
            self.d_model,
            self.n_heads,
            self.dropout,
            self.use_bias,
        )

        # 内置 Q/K/V 投影的多头注意力（batch_first）
        self.mha = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            bias=self.use_bias,
            batch_first=True,
            kdim=self.d_model,
            vdim=self.d_model,
        )

        # 输出层可按需再做线性调整（保持恒等以最大兼容）
        self.out = nn.Identity()

    @staticmethod
    def _shape(x: Tensor | None) -> str:
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

    def _validate(
        self,
        *,
        news: EncoderOutput,
        entities: EncoderOutput,
        news_mask: Optional[Tensor],
        entity_mask: Optional[Tensor],
    ) -> None:
        Bn, Lt, Dn = news.last_hidden_state.shape
        Be, Le, De = entities.last_hidden_state.shape
        if Dn != self.d_model or De != self.d_model:
            self.log.error(
                "Hidden size mismatch: d_model=%d, news.D=%d, entities.D=%d",
                self.d_model,
                Dn,
                De,
            )
            raise ValueError(
                f"[NE] d_model mismatch: expected {self.d_model}, got news {Dn}, entities {De}"
            )

        if news_mask is not None and tuple(news_mask.shape) != (Bn, Lt):
            self.log.error(
                "news_mask shape mismatch: expect [%d,%d], got %s",
                Bn,
                Lt,
                self._shape(news_mask),
            )
            raise ValueError(
                f"[NE] news_mask shape mismatch: expected {(Bn, Lt)}, got {tuple(news_mask.shape)}"
            )

        if entity_mask is not None and tuple(entity_mask.shape) != (Be, Le):
            self.log.error(
                "entity_mask shape mismatch: expect [%d,%d], got %s",
                Be,
                Le,
                self._shape(entity_mask),
            )
            raise ValueError(
                f"[NE] entity_mask shape mismatch: expected {(Be, Le)}, got {tuple(entity_mask.shape)}"
            )

        if Bn != Be:
            self.log.error("Batch size mismatch: news.B=%d entities.B=%d", Bn, Be)
            raise ValueError(f"[NE] batch mismatch: news {Bn} vs entities {Be}")

    @staticmethod
    def _masked_mean(x: Tensor, mask: Optional[Tensor]) -> Tensor:
        """x: [B,L,D]; mask: [B,L] with 1=valid."""
        if mask is None:
            return x.mean(dim=1)
        m = mask.to(dtype=x.dtype).unsqueeze(-1)  # [B,L,1]
        denom = m.sum(dim=1).clamp_min(1.0)  # [B,1]
        num = (x * m).sum(dim=1)  # [B,D]
        # 若全 0，等效退化为普通 mean（通过 denom=1 保证稳定；再检测补上 mean）
        zero_rows = m.sum(dim=1).squeeze(-1) == 0  # [B]
        if zero_rows.any():
            fallback = x.mean(dim=1)  # [B,D]
            num = torch.where(zero_rows.unsqueeze(-1), fallback, num)
        return num / denom

    def forward(
        self,
        *,
        news: EncoderOutput,
        entities: EncoderOutput,
        news_mask: Optional[Tensor] = None,  # [B, Lt], 1=valid
        entity_mask: Optional[Tensor] = None,  # [B, Le], 1=valid
        return_weights: bool = False,
    ) -> NEAttentionOutput:
        # ---- instrumentation (debug-friendly) ----
        self.log.debug(
            "forward(news=[B=%d,Lt=%d,D=%d], entities=[B=%d,Le=%d,D=%d], "
            "news_mask=%s entity_mask=%s, device=%s/%s dtype=%s/%s)",
            news.last_hidden_state.shape[0],
            news.last_hidden_state.shape[1],
            news.last_hidden_state.shape[2],
            entities.last_hidden_state.shape[0],
            entities.last_hidden_state.shape[1],
            entities.last_hidden_state.shape[2],
            self._shape(news_mask),
            self._shape(entity_mask),
            str(news.last_hidden_state.device),
            str(entities.last_hidden_state.device),
            str(news.last_hidden_state.dtype),
            str(entities.last_hidden_state.dtype),
        )
        nd = self._mask_density(news_mask)
        ed = self._mask_density(entity_mask)
        if nd is None:
            self.log.warning(
                "news_mask is None; attention may treat all tokens as valid"
            )
        if ed is None:
            self.log.warning(
                "entity_mask is None; attention may treat all entities as valid"
            )
        if nd is not None:
            self.log.debug("news_mask density=%.4f", nd)
        if ed is not None:
            self.log.debug("entity_mask density=%.4f", ed)

        # ---- strict validation ----
        self._validate(
            news=news, entities=entities, news_mask=news_mask, entity_mask=entity_mask
        )

        # ---- prepare tensors ----
        q = news.last_hidden_state  # [B,Lt,D]
        k = entities.last_hidden_state  # [B,Le,D]
        v = entities.last_hidden_state  # [B,Le,D]

        # PyTorch 语义：key_padding_mask=True 表示“需要屏蔽（padding）”
        # 我们接口：mask=1 表示“有效” → 需要取反
        key_padding_mask = None
        if entity_mask is not None:
            # 检查是否存在某 batch 样本全 padding 的极端情况
            inv = ~entity_mask.to(dtype=torch.bool)  # True=padding
            all_pad = inv.all(dim=1)  # [B]
            if all_pad.any():
                # 对这些样本降级为“不提供 key mask”，避免 softmax(-inf) → NaN
                self.log.warning(
                    "entity_mask has all-padding rows; fallback to no key mask for those samples"
                )
                # 将这些行设为全 False（等价于不过滤），其它样本正常使用
                inv = inv.clone()
                inv[all_pad] = False
            key_padding_mask = inv  # [B,Le] with True=padding

        need_w = bool(return_weights)
        avg_w = False  # 我们需要每个头的权重 [B,H,Lt,Le]

        # ---- cross-attention ----
        # attn_output: [B,Lt,D]; attn_weights: [B,H,Lt,Le]（当 need_w=True & avg_w=False）
        attn_out, attn_w = self.mha(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            need_weights=need_w,
            average_attn_weights=avg_w,
        )

        fused = self.out(attn_out)  # [B,Lt,D]

        # ---- pooled（masked mean over Lt；如无 mask 则普通 mean）----
        pooled = self._masked_mean(fused, news_mask)  # [B,D]

        return NEAttentionOutput(
            fused_states=fused,
            pooled=pooled,
            attn_weights=attn_w if need_w else None,
        )


# -------------------- 可选：注册到 Registry（attention 命名空间） --------------------
try:
    from kan.utils.registry import HUB

    _ATTN = HUB.get_or_create("attention")

    @_ATTN.register("ne", alias=["NE", "news_entities"])
    def _build_ne(**cfg):
        """
        @brief 构建 NEAttention（Chinese）/ Build NEAttention (English).
        @note 透传参数：d_model/n_heads/dropout/use_bias
        """
        return NEAttention(**cfg)

except Exception:
    # registry 不可用时静默跳过，保持无侵入
    pass
