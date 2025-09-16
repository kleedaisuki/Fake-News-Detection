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
    pooled: Tensor        # [B, D]
    attn_weights: Optional[Tensor] = None  # [B, H, Lt, Le]

class NEAttention(nn.Module):
    r"""
    @zh
      新闻-实体（NE）注意力算子——**带日志与校验的签名声明**。
      本类不含实际注意力实现，但在 forward 中会：
        1) 记录输入形状、mask 密度、dtype/device；
        2) 严格校验 D 维一致性与形状匹配，失败则 ERROR 并抛出异常。
    @en
      News-Entity attention (signature with logging & validation). No real
      attention is implemented; forward logs inputs and validates shapes.
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
            self.d_model, self.n_heads, self.dropout, self.use_bias,
        )

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

    def _validate(self, *, news: EncoderOutput, entities: EncoderOutput,
                  news_mask: Optional[Tensor], entity_mask: Optional[Tensor]) -> None:
        Bn, Lt, Dn = news.last_hidden_state.shape
        Be, Le, De = entities.last_hidden_state.shape
        if Dn != self.d_model or De != self.d_model:
            self.log.error("Hidden size mismatch: d_model=%d, news.D=%d, entities.D=%d",
                           self.d_model, Dn, De)
            raise ValueError(f"[NE] d_model mismatch: expected {self.d_model}, got news {Dn}, entities {De}")

        if news_mask is not None and tuple(news_mask.shape) != (Bn, Lt):
            self.log.error("news_mask shape mismatch: expect [%d,%d], got %s", Bn, Lt, self._shape(news_mask))
            raise ValueError(f"[NE] news_mask shape mismatch: expected {(Bn, Lt)}, got {tuple(news_mask.shape)}")

        if entity_mask is not None and tuple(entity_mask.shape) != (Be, Le):
            self.log.error("entity_mask shape mismatch: expect [%d,%d], got %s", Be, Le, self._shape(entity_mask))
            raise ValueError(f"[NE] entity_mask shape mismatch: expected {(Be, Le)}, got {tuple(entity_mask.shape)}")

        if Bn != Be:
            self.log.error("Batch size mismatch: news.B=%d entities.B=%d", Bn, Be)
            raise ValueError(f"[NE] batch mismatch: news {Bn} vs entities {Be}")

    def forward(
        self,
        *,
        news: EncoderOutput,
        entities: EncoderOutput,
        news_mask: Optional[Tensor] = None,    # [B, Lt]
        entity_mask: Optional[Tensor] = None,  # [B, Le]
        return_weights: bool = False,
    ) -> NEAttentionOutput:
        # ---- instrumentation (debug-friendly) ----
        self.log.debug(
            "forward(news=[B=%d,Lt=%d,D=%d], entities=[B=%d,Le=%d,D=%d], "
            "news_mask=%s entity_mask=%s, device=%s/%s dtype=%s/%s)",
            news.last_hidden_state.shape[0], news.last_hidden_state.shape[1], news.last_hidden_state.shape[2],
            entities.last_hidden_state.shape[0], entities.last_hidden_state.shape[1], entities.last_hidden_state.shape[2],
            self._shape(news_mask), self._shape(entity_mask),
            str(news.last_hidden_state.device), str(entities.last_hidden_state.device),
            str(news.last_hidden_state.dtype), str(entities.last_hidden_state.dtype),
        )
        nd = self._mask_density(news_mask); ed = self._mask_density(entity_mask)
        if nd is None:
            self.log.warning("news_mask is None; attention may treat all tokens as valid")
        if ed is None:
            self.log.warning("entity_mask is None; attention may treat all entities as valid")
        if nd is not None:
            self.log.debug("news_mask density=%.4f", nd)
        if ed is not None:
            self.log.debug("entity_mask density=%.4f", ed)

        # ---- strict validation ----
        self._validate(news=news, entities=entities, news_mask=news_mask, entity_mask=entity_mask)

        # ---- not implemented yet ----
        self.log.info("NEAttention.forward called; implementation not provided yet")
        raise NotImplementedError("NEAttention.forward is not implemented yet (logging/validation active).")
