# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

import logging
import torch
from torch import Tensor, nn

from kan.interfaces.encoders import EncoderOutput

@dataclass(frozen=True)
class NE2CAttentionOutput:
    fused_states: Tensor   # [B, Lt, D]
    pooled: Tensor         # [B, D]
    weights: Optional[Dict[str, Tensor]] = None  # {"ne": [B,H,Lt,Le], "e2c": [B,H,Le,Lc] or [B,H,E,Lc]}

class NE2CAttention(nn.Module):
    r"""
    @zh
      NE2C 分层注意力（签名+日志+校验）。未实现注意力逻辑；
      但 forward 会记录形状/掩码密度并严格校验。
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        context_pooling: str = "attn",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.context_pooling = str(context_pooling)
        self.use_bias = bool(use_bias)
        self.log = logging.getLogger("kan.modules.attention.ne2c.NE2CAttention")
        self.log.info(
            "NE2CAttention init d_model=%d n_heads=%d dropout=%.3f pooling=%s use_bias=%s",
            self.d_model, self.n_heads, self.dropout, self.context_pooling, self.use_bias
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

    def _validate(
        self,
        *,
        news: EncoderOutput,
        entities: EncoderOutput,
        contexts_last_hidden: Tensor,   # [B, E, Lc, D]
        news_mask: Optional[Tensor],
        entity_mask: Optional[Tensor],
        contexts_mask: Optional[Tensor]
    ) -> None:
        Bn, Lt, Dn = news.last_hidden_state.shape
        Be, Le, De = entities.last_hidden_state.shape
        Bc, E, Lc, Dc = contexts_last_hidden.shape

        # D 维对齐
        if not (Dn == De == Dc == self.d_model):
            self.log.error("Hidden size mismatch: expect D=%d, got news=%d, entities=%d, contexts=%d",
                           self.d_model, Dn, De, Dc)
            raise ValueError(f"[NE2C] d_model mismatch: {self.d_model} vs (news {Dn}, entities {De}, contexts {Dc})")
        # B 对齐
        if not (Bn == Be == Bc):
            self.log.error("Batch mismatch: news.B=%d entities.B=%d contexts.B=%d", Bn, Be, Bc)
            raise ValueError(f"[NE2C] batch mismatch: news {Bn}, entities {Be}, contexts {Bc}")

        # 掩码形状
        if news_mask is not None and tuple(news_mask.shape) != (Bn, Lt):
            self.log.error("news_mask shape mismatch: expect [%d,%d], got %s", Bn, Lt, self._shape(news_mask))
            raise ValueError(f"[NE2C] news_mask shape mismatch: expected {(Bn, Lt)}, got {tuple(news_mask.shape)}")
        if entity_mask is not None and tuple(entity_mask.shape) != (Be, Le):
            self.log.error("entity_mask shape mismatch: expect [%d,%d], got %s", Be, Le, self._shape(entity_mask))
            raise ValueError(f"[NE2C] entity_mask shape mismatch: expected {(Be, Le)}, got {tuple(entity_mask.shape)}")
        if contexts_mask is not None and tuple(contexts_mask.shape) != (Bc, E, Lc):
            self.log.error("contexts_mask shape mismatch: expect [%d,%d,%d], got %s", Bc, E, Lc, self._shape(contexts_mask))
            raise ValueError(f"[NE2C] contexts_mask shape mismatch: expected {(Bc, E, Lc)}, got {tuple(contexts_mask.shape)}")

    def forward(
        self,
        *,
        news: EncoderOutput,
        entities: EncoderOutput,
        contexts_last_hidden: Tensor,      # [B, E, Lc, D]
        news_mask: Optional[Tensor] = None,
        entity_mask: Optional[Tensor] = None,
        contexts_mask: Optional[Tensor] = None,
        return_weights: bool = False,
    ) -> NE2CAttentionOutput:
        # ---- instrumentation ----
        self.log.debug(
            "forward(news=[B=%d,Lt=%d,D=%d], entities=[B=%d,Le=%d,D=%d], contexts=[B=%d,E=%d,Lc=%d,D=%d], "
            "news_mask=%s entity_mask=%s contexts_mask=%s, device=%s/%s/%s dtype=%s/%s/%s)",
            news.last_hidden_state.shape[0], news.last_hidden_state.shape[1], news.last_hidden_state.shape[2],
            entities.last_hidden_state.shape[0], entities.last_hidden_state.shape[1], entities.last_hidden_state.shape[2],
            contexts_last_hidden.shape[0], contexts_last_hidden.shape[1], contexts_last_hidden.shape[2], contexts_last_hidden.shape[3],
            self._shape(news_mask), self._shape(entity_mask), self._shape(contexts_mask),
            str(news.last_hidden_state.device), str(entities.last_hidden_state.device), str(contexts_last_hidden.device),
            str(news.last_hidden_state.dtype), str(entities.last_hidden_state.dtype), str(contexts_last_hidden.dtype),
        )
        for name, m in (("news_mask", news_mask), ("entity_mask", entity_mask), ("contexts_mask", contexts_mask)):
            d = self._mask_density(m)
            if d is None:
                self.log.warning("%s is None; downstream attention may treat all tokens as valid", name)
            else:
                self.log.debug("%s density=%.4f", name, d)

        # ---- strict validation ----
        self._validate(
            news=news,
            entities=entities,
            contexts_last_hidden=contexts_last_hidden,
            news_mask=news_mask,
            entity_mask=entity_mask,
            contexts_mask=contexts_mask,
        )

        # ---- not implemented yet ----
        self.log.info("NE2CAttention.forward called; implementation not provided yet")
        raise NotImplementedError("NE2CAttention.forward is not implemented yet (logging/validation active).")
