# -*- coding: utf-8 -*-
"""
@file: context_encoder.py

中英双语文档 / Bilingual Docstring
===================================

目的 / Purpose
--------------
实现 **Context Encoder（上下文编码器）**：对每个实体的若干条**文本化上下文**（如实体描述、邻接三元组
的自然语言化、别名、维基抽象等）进行编码，输出与 `entity_encoder` 对齐的形状，供 NE2C 等模块作为 V 使用。

为什么需要单独的 Context Encoder？ / Why a separate module
-----------------------------------------------------------
- `entity_encoder` 处理 **ID 型上下文**（邻居 ID → 嵌入 → Lc 聚合）。
- `context_encoder` 处理 **文本型上下文**（上下文文本 → HF 文本编码器 → Lc 聚合）。
二者契约保持一致，方便在实验中互换或并联（ensemble）。

输入 / Inputs（keyword-only）
---------------------------
- `context_input_ids: LongTensor[B,E,Lc,Lt]`
- `context_attention_mask: LongTensor[B,E,Lc,Lt]`
- `context_token_type_ids: Optional[LongTensor[B,E,Lc,Lt]]`
- `contexts_mask: Optional[LongTensor[B,E,Lc]]`  # 1 有效；若缺省则由 `attention_mask` 推断

输出 / Outputs
--------------
- `contexts_last_hidden: FloatTensor[B,E,D]`     # 对每个实体的若干条文本上下文聚合后的表示 r'
- `contexts_pooled: FloatTensor[B,D]`            # 跨实体聚合
- `contexts_mask: LongTensor[B,E]`               # 某实体是否至少有一条有效上下文
- （可选）`raw_contexts_last_hidden: FloatTensor[B,E,Lc,D]`  # 每条上下文（文本）单独的向量

设计要点 / Design Highlights
----------------------------
1) 复用 `kan.modules.text_encoder` 的统一契约：`sequence_output/pooled_output/attention_mask`。
2) 通过展平维度（B*E*Lc, Lt）→ 文本编码器 → 还原形状（B,E,Lc,D），再在 Lc 聚合（mean/max）。
3) 完整的 mask 流：基于 `contexts_mask` 与 `attention_mask` 推断有效性，避免 padding 污染。
4) 命名 logger：`kan.context_encoder`；复用全局 logging 配置；Windows 友好。

安全与兼容 / Stability & Compatibility
--------------------------------------
- **We don't break userspace!** 新增字段均有默认值；输出键名与形状稳定。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Literal, List
import logging

import torch
from torch import nn, Tensor

try:  # 本地相对导入（保持解耦）
    from kan.modules.text_encoder import (
        BaseTextEncoder,
        HFTextEncoder,
        TextEncoderConfig,
        build_text_encoder,
    )
except Exception:  # pragma: no cover
    # 允许相对导入失败时再尝试绝对路径（取决于包布局）
    from .text_encoder import BaseTextEncoder, HFTextEncoder, TextEncoderConfig, build_text_encoder  # type: ignore

logger = logging.getLogger("kan.context_encoder")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class ContextEncoderConfig:
    """文本上下文编码器配置 / Configuration.

    Attributes
    ----------
    text_encoder : Optional[TextEncoderConfig]
        若提供则内部构建 HF 文本编码器；否则需要外部注入 `text_encoder_module`。
    freeze_text_encoder : bool
        是否冻结内部文本编码器参数（默认冻结，作为特征抽取器）。
    inner_pooling : Literal["mean", "max"]
        Lc 维度上的聚合策略（将多条上下文聚成每实体一个向量）。
    entity_pooling : Literal["mean", "max"]
        E 维度上的聚合策略（跨实体聚合到样本向量）。
    return_raw_contexts : bool
        是否返回每条上下文的向量（[B,E,Lc,D]）。
    device : Optional[str]
        计算设备；None 自动选择。
    """

    text_encoder: Optional[TextEncoderConfig] = None
    freeze_text_encoder: bool = True
    inner_pooling: Literal["mean", "max"] = "mean"
    entity_pooling: Literal["mean", "max"] = "mean"
    return_raw_contexts: bool = False
    device: Optional[str] = None


# -----------------------------------------------------------------------------
# Context Encoder
# -----------------------------------------------------------------------------
class ContextEncoder(nn.Module):
    """将每个实体的**文本上下文**编码为与实体同维度的向量，用于 NE2C 的 V。

    - 依赖一个 `BaseTextEncoder` 实例（内部构建或外部注入）。
    - 将四维张量 `[B,E,Lc,Lt]` 通过展平为 `[B*E*Lc, Lt]` 输入文本编码器，再聚合回 `[B,E,D]`。
    """

    def __init__(
        self,
        cfg: ContextEncoderConfig,
        *,
        text_encoder_module: Optional[BaseTextEncoder] = None,
    ):
        super().__init__()
        self.cfg = cfg

        if text_encoder_module is not None:
            self.text_encoder: BaseTextEncoder = text_encoder_module
        elif cfg.text_encoder is not None:
            self.text_encoder = build_text_encoder(cfg.text_encoder)
        else:
            raise ValueError(
                "必须提供 TextEncoderConfig 或已构建的 text_encoder_module / Need a text encoder"
            )

        # 冻结策略
        if cfg.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(0.1)

        # 从文本编码器推导隐藏维度 D
        if hasattr(self.text_encoder, "hidden_size"):
            self.hidden_size = getattr(self.text_encoder, "hidden_size")
        else:
            # 兜底值：大多数 HF 模型都有 config.hidden_size；若没有，运行一次 dummy 推断也可（此处不做）
            self.hidden_size = 768

        logger.info(
            "ContextEncoder init: D=%s, freeze_text_encoder=%s, inner_pooling=%s",
            self.hidden_size,
            cfg.freeze_text_encoder,
            cfg.inner_pooling,
        )

    # ----------------------------- Forward -----------------------------
    def forward(
        self,
        *,
        context_input_ids: Tensor,  # [B,E,Lc,Lt]
        context_attention_mask: Tensor,  # [B,E,Lc,Lt]
        context_token_type_ids: Optional[Tensor] = None,  # [B,E,Lc,Lt]
        contexts_mask: Optional[Tensor] = None,  # [B,E,Lc]
    ) -> Dict[str, Tensor]:
        cfg = self.cfg
        device = context_input_ids.device

        B, E, Lc, Lt = context_input_ids.shape
        # 若未传 contexts_mask，则根据 attention_mask 判定每条上下文是否有 token（sum>0）
        if contexts_mask is None:
            contexts_mask = (context_attention_mask.sum(dim=-1) > 0).long()  # [B,E,Lc]

        # 展平为 [N, Lt]
        N = B * E * Lc
        ids_flat = context_input_ids.view(N, Lt)
        attn_flat = context_attention_mask.view(N, Lt)
        tok_flat = (
            context_token_type_ids.view(N, Lt)
            if context_token_type_ids is not None
            else None
        )
        mask_flat = contexts_mask.view(N)

        # 仅对有效条目编码（减少无用计算）
        valid_idx = torch.nonzero(mask_flat > 0, as_tuple=False).squeeze(
            -1
        )  # [N_valid]
        num_valid = int(valid_idx.numel())
        if num_valid == 0:
            # 没有任何有效上下文 -> 返回零张量
            zeros_be = context_input_ids.new_zeros(
                (B, E, self.hidden_size), dtype=torch.float32
            )
            zeros_b = context_input_ids.new_zeros(
                (B, self.hidden_size), dtype=torch.float32
            )
            return {
                "contexts_last_hidden": zeros_be,
                "contexts_pooled": zeros_b,
                "contexts_mask": context_input_ids.new_zeros((B, E), dtype=torch.long),
                "raw_contexts_last_hidden": (
                    context_input_ids.new_zeros(
                        (B, E, Lc, self.hidden_size), dtype=torch.float32
                    )
                    if cfg.return_raw_contexts
                    else None
                ),
            }

        # 构造子批，仅编码有效上下文
        batch_sub = {
            "input_ids": ids_flat.index_select(0, valid_idx),
            "attention_mask": attn_flat.index_select(0, valid_idx),
        }
        if tok_flat is not None:
            batch_sub["token_type_ids"] = tok_flat.index_select(0, valid_idx)

        # 调用文本编码器：我们仅需要 pooled 向量
        enc_out = self.text_encoder(**batch_sub)  # type: ignore[arg-type]
        pooled_valid: Tensor = enc_out["pooled_output"].detach()  # [N_valid, D]
        D = pooled_valid.shape[-1]

        # 回填到全量位置
        ctx_vec_flat = context_input_ids.new_zeros((N, D), dtype=pooled_valid.dtype)
        ctx_vec_flat.index_copy_(0, valid_idx, pooled_valid)

        # 还原形状 [B,E,Lc,D]
        ctx_vec = ctx_vec_flat.view(B, E, Lc, D)
        if cfg.return_raw_contexts:
            raw_ctx = ctx_vec.clone()
        else:
            raw_ctx = None

        # Lc 聚合 -> [B,E,D]
        contexts_mask_f = contexts_mask.to(dtype=ctx_vec.dtype)  # [B,E,Lc]
        if cfg.inner_pooling == "mean":
            denom = contexts_mask_f.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B,E,1]
            ctx_be = (ctx_vec * contexts_mask_f.unsqueeze(-1)).sum(
                dim=2
            ) / denom  # [B,E,D]
        elif cfg.inner_pooling == "max":
            very_small = torch.finfo(ctx_vec.dtype).min
            masked = ctx_vec.masked_fill(contexts_mask_f.unsqueeze(-1) == 0, very_small)
            ctx_be = masked.max(dim=2).values
        else:
            raise ValueError(f"未知的 inner_pooling: {cfg.inner_pooling}")

        ctx_be = self.dropout(ctx_be)

        # E 聚合 -> [B,D]
        has_ctx = (contexts_mask.long().sum(dim=-1) > 0).long()  # [B,E]
        contexts_pooled = self._pool_over_entities(
            ctx_be, has_ctx, mode=cfg.entity_pooling
        )

        out = {
            "contexts_last_hidden": ctx_be,  # [B,E,D]
            "contexts_pooled": contexts_pooled,  # [B,D]
            "contexts_mask": has_ctx,  # [B,E]
        }
        if raw_ctx is not None:
            out["raw_contexts_last_hidden"] = raw_ctx  # [B,E,Lc,D]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "shapes: ctx_be=%s, pooled=%s, has_ctx=%s",
                tuple(ctx_be.shape),
                tuple(contexts_pooled.shape),
                tuple(has_ctx.shape),
            )
        return out

    # --------------------------- Helpers ---------------------------
    def _pool_over_entities(
        self, x: Tensor, mask: Tensor, *, mode: Literal["mean", "max"]
    ) -> Tensor:
        """跨实体维 E 聚合到样本向量。
        x: [B,E,D], mask: [B,E] -> [B,D]
        """
        m = mask.to(dtype=x.dtype)
        if mode == "mean":
            denom = m.sum(dim=1).clamp(min=1e-6).unsqueeze(-1)
            return (x * m.unsqueeze(-1)).sum(dim=1) / denom
        elif mode == "max":
            very_small = torch.finfo(x.dtype).min
            masked = x.masked_fill(m.unsqueeze(-1) == 0, very_small)
            return masked.max(dim=1).values
        else:
            raise ValueError(f"未知的 entity_pooling: {mode}")

    # ------------------------ Convenience -------------------------
    @torch.no_grad()
    def batch_encode_contexts(
        self,
        contexts: List[List[List[str]]],  # 形如：batch × entities × contexts_per_entity
        *,
        max_length: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        pad_to_max_length: bool = True,
    ) -> Dict[str, Tensor]:
        """将嵌套的文本列表编码为 `[B,E,Lc,Lt]` 形式的张量 batch，便于直接 `forward`。
        注意：这里使用内部文本编码器的 tokenizer。
        """
        # 统计最大 E 与 Lc
        B = len(contexts)
        E_max = max((len(es) for es in contexts), default=1)
        Lc_max = max((len(c) for es in contexts for c in es), default=1)
        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # 展平为一维列表做分词
        texts: List[str] = []
        # 记录位置：[(b, e, j)]
        coords: List[tuple[int, int, int]] = []
        for b, es in enumerate(contexts):
            for e, cs in enumerate(es):
                for j, t in enumerate(cs):
                    texts.append(t)
                    coords.append((b, e, j))
        if len(texts) == 0:
            # 返回空 batch（占位）
            shape_ids = (
                B,
                E_max,
                Lc_max,
                max_length
                or getattr(self.text_encoder, "cfg", TextEncoderConfig()).max_length,
            )
            return {
                "context_input_ids": torch.zeros(
                    shape_ids, dtype=torch.long, device=device
                ),
                "context_attention_mask": torch.zeros(
                    shape_ids, dtype=torch.long, device=device
                ),
                "contexts_mask": torch.zeros(
                    (B, E_max, Lc_max), dtype=torch.long, device=device
                ),
            }

        # 使用内部 tokenizer
        if isinstance(self.text_encoder, HFTextEncoder):
            tok = self.text_encoder.tokenizer
            pad = "max_length" if pad_to_max_length else False
            enc = tok(
                texts,
                padding=pad,
                truncation=True,
                max_length=max_length
                or getattr(self.text_encoder, "cfg", TextEncoderConfig()).max_length,
                add_special_tokens=True,
                return_tensors="pt",
            )
        else:
            raise RuntimeError("batch_encode_contexts 需要 HFTextEncoder 内部实现")

        # 将一维编码还原到 [B,E,Lc,Lt]
        Lt = int(enc["input_ids"].shape[-1])
        ids = torch.zeros((B, E_max, Lc_max, Lt), dtype=torch.long)
        attn = torch.zeros((B, E_max, Lc_max, Lt), dtype=torch.long)
        mask = torch.zeros((B, E_max, Lc_max), dtype=torch.long)
        tok_type = torch.zeros_like(ids) if "token_type_ids" in enc else None

        for k, (b, e, j) in enumerate(coords):
            ids[b, e, j] = enc["input_ids"][k]
            attn[b, e, j] = enc["attention_mask"][k]
            mask[b, e, j] = 1
            if tok_type is not None:
                tok_type[b, e, j] = enc["token_type_ids"][k]

        out = {
            "context_input_ids": ids.to(device),
            "context_attention_mask": attn.to(device),
            "contexts_mask": mask.to(device),
        }
        if tok_type is not None:
            out["context_token_type_ids"] = tok_type.to(device)
        return out


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def build_context_encoder(
    cfg: ContextEncoderConfig, *, text_encoder_module: Optional[BaseTextEncoder] = None
) -> ContextEncoder:
    enc = ContextEncoder(cfg, text_encoder_module=text_encoder_module)
    if cfg.device:
        enc = enc.to(torch.device(cfg.device))
    else:
        enc = enc.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Context encoder built: %s", cfg)
    return enc


# -----------------------------------------------------------------------------
# Self-test (optional manual)
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    # 使用一个极小的 BERT 以便快速自测
    te_cfg = TextEncoderConfig(
        model_name_or_path="prajjwal1/bert-tiny", max_length=24, pooling="mean"
    )
    ctx_cfg = ContextEncoderConfig(
        text_encoder=te_cfg, inner_pooling="mean", entity_pooling="mean"
    )
    enc = build_context_encoder(ctx_cfg)

    contexts = [
        [
            [
                "Barack Obama was the 44th President of the United States.",
                "Born in Hawaii.",
            ],
            ["White House is the official residence."],
        ],
        [["OpenAI created ChatGPT."], []],
    ]
    batch = enc.batch_encode_contexts(contexts, pad_to_max_length=True)
    out = enc(**batch)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(k, tuple(v.shape))
