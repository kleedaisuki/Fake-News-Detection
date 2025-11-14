# -*- coding: utf-8 -*-
"""
@file: text_encoder.py

中英双语文档 / Bilingual Docstring
===================================

目的 / Purpose
--------------
实现一个可插拔的文本编码器（Text Encoder）模块，对接上层 `kan/modules/*` 与下层
Transformer/RNN 等实现，提供统一契约：**输入批文本或已分词张量，输出序列表示与聚合表示**。

设计要点 / Design Highlights
----------------------------
1. 统一接口（契约）/ Unified Contract
   - `forward(**batch)`：接收 `input_ids`, `attention_mask`, （可选）`token_type_ids`。
   - 返回 `{"sequence_output": Tensor[B, L, H], "pooled_output": Tensor[B, H], "attention_mask": Tensor[B, L]}`。
   - 聚合策略（pooling）支持 `"cls" | "mean" | "max"`。

2. 易用性 / Ergonomics
   - `from_pretrained()` 一行加载 Hugging Face 模型与分词器。
   - `batch_encode(texts)` 对原始字符串进行分词并返回可直接 `forward` 的 batch。
   - `encode(texts)` 端到端：原文 → embedding（pooled）。

3. 训练与部署兼容 / Train & Inference Friendly
   - `trainable` 决定是否冻结 backbone 参数。
   - 支持 Windows 平台与 CPU/GPU 自动选择（`device` 参数）。
   - 仅依赖 `torch` 与 `transformers`，两者缺失时给出友好错误。

4. 日志 / Logging
   - 通过命名 logger `kan.text_encoder` 输出调试信息。
   - 若项目根目录存在自定义 `logging` 配置（例如 `logging.py` 已经设置 `basicConfig` 或 `dictConfig`），
     本模块将**自动复用**全局配置；否则退化到标准库默认行为。

5. 可扩展性 / Extensibility
   - 通过 `BaseTextEncoder` 约定接口，新模型（如 BiGRU/CNN）只需复用 `pooling` 与输出契约即可。

安全与兼容 / Stability & Compatibility
--------------------------------------
- **We don't break userspace!** 遵循向后兼容：新增参数采用默认值；保存/加载的 state dict 不改变关键名。

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Dict
import logging

try:
    import torch
    from torch import nn, Tensor
except Exception as e:  # pragma: no cover - 清晰错误提示
    raise RuntimeError(
        "PyTorch 未安装或导入失败，请先安装 pytorch。/ PyTorch is missing."
    ) from e

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Transformers 未安装或导入失败，请先安装 transformers。/ transformers is missing."
    ) from e

from kan.utils.registry import HUB

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
# 命名空间式的 logger，便于在全局 logging 配置中精确控制该模块的日志级别
logger = logging.getLogger("kan.modules.text_encoder")

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
TEXT = HUB.get_or_create("text_encoder")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class TextEncoderConfig:
    """文本编码器配置 / Configuration for Text Encoder

    Attributes
    ----------
    model_name_or_path : str
        Hugging Face 模型名或本地路径 / HF id or local path (e.g., "bert-base-uncased").
    max_length : int
        分词最大长度 / Max sequence length for tokenization.
    pooling : Literal["cls", "mean", "max"]
        序列聚合策略 / Aggregation strategy for pooled output.
    trainable : bool
        是否反向传播更新 backbone / Whether to unfreeze backbone parameters.
    use_fast_tokenizer : bool
        是否优先使用 fast tokenizer（若可用）/ Prefer fast tokenizer.
    pad_to_max_length : bool
        是否 pad 到 max_length，便于静态 shape / Pad to max_length for static shapes.
    """

    model_name_or_path: str = "bert-base-uncased"
    max_length: int = 512
    pooling: Literal["cls", "mean", "max"] = "cls"
    trainable: bool = False
    use_fast_tokenizer: bool = True
    pad_to_max_length: bool = True


# -----------------------------------------------------------------------------
# Base Interface
# -----------------------------------------------------------------------------
class BaseTextEncoder(nn.Module):
    """抽象基类：定义统一契约 / Abstract base class that defines the contract.

    输出约定（Output Contract）
    ---------------------------
    - sequence_output: Tensor[B, L, H]
    - pooled_output:   Tensor[B, H]
    - attention_mask:  Tensor[B, L]
    """

    def forward(self, **batch) -> Dict[str, Tensor]:  # type: ignore[override]
        raise NotImplementedError

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        *,
        device: Optional[torch.device | str] = None,
        batch_size: int = 32,
        progress: bool = False,
    ) -> torch.Tensor:
        """端到端：原文 → pooled embedding（B, H）/ Texts to pooled embeddings.

        Parameters
        ----------
        texts : List[str]
            原始文本列表 / Raw text inputs.
        device : Optional[torch.device | str]
            计算设备，默认自动选择 / Device; auto-detected if None.
        batch_size : int
            批大小 / Batch size.
        progress : bool
            是否显示进度（简易版）/ Show naive progress bar.
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# HF Backbone Implementation
# -----------------------------------------------------------------------------
class HFTextEncoder(BaseTextEncoder):
    """基于 Hugging Face Transformers 的文本编码器。

    使用方式 / Usage
    -----------------
    >>> enc = HFTextEncoder.from_pretrained(TextEncoderConfig("bert-base-uncased"))
    >>> batch = enc.batch_encode(["hello world", "kan!"], device="cuda")
    >>> out = enc(**batch)
    >>> out["pooled_output"].shape  # (B, H)
    torch.Size([2, 768])
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cfg: TextEncoderConfig,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.hidden_size = getattr(model.config, "hidden_size", None) or getattr(
            model.config, "d_model", None
        )

        # 冻结与否
        for p in self.model.parameters():
            p.requires_grad = bool(cfg.trainable)

        logger.info(
            "Initialized HFTextEncoder: model=%s, hidden_size=%s, trainable=%s, pooling=%s",
            type(self.model).__name__,
            self.hidden_size,
            cfg.trainable,
            cfg.pooling,
        )

    # -------------------------- Factory --------------------------
    @classmethod
    def from_pretrained(
        cls, cfg: TextEncoderConfig, *, trust_remote_code: bool = False
    ) -> "HFTextEncoder":
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path,
            use_fast=cfg.use_fast_tokenizer,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModel.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=trust_remote_code
        )
        return cls(model=model, tokenizer=tokenizer, cfg=cfg)

    # ----------------------- Tokenization ------------------------
    def batch_encode(
        self,
        texts: List[str],
        *,
        device: Optional[torch.device | str] = None,
        add_special_tokens: bool = True,
        return_tensors: Literal["pt"] = "pt",
        truncation: bool = True,
    ) -> Dict[str, Tensor]:
        """分词并返回可喂给 `forward` 的批字典 / Tokenize texts into a forward-able batch.
        返回键包含：`input_ids`, `attention_mask`, 可能的 `token_type_ids`。
        """
        if device is None:
            device = self._auto_device()
        pad = "max_length" if self.cfg.pad_to_max_length else False
        enc = self.tokenizer(
            texts,
            padding=pad,
            truncation=truncation,
            max_length=self.cfg.max_length,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
        batch = {k: v.to(device) for k, v in enc.items()}
        return batch  # type: ignore[return-value]

    # -------------------------- Forward -------------------------
    def forward(self, **batch) -> Dict[str, Tensor]:  # type: ignore[override]
        outputs = self.model(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids", None),
            return_dict=True,
        )
        sequence_output: Tensor = outputs.last_hidden_state  # (B, L, H)
        attention_mask: Tensor = batch.get("attention_mask")
        pooled_output: Tensor = self._pool(sequence_output, attention_mask)

        return {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
            "attention_mask": attention_mask,
        }

    # --------------------------- API ----------------------------
    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        *,
        device: Optional[torch.device | str] = None,
        batch_size: int = 32,
        progress: bool = False,
    ) -> torch.Tensor:  # type: ignore[override]
        device = device or self._auto_device()
        self.eval()
        pooled_list: List[Tensor] = []

        total = len(texts)
        rng = range(0, total, batch_size)
        for i in rng:
            sub = texts[i : i + batch_size]
            batch = self.batch_encode(sub, device=device)
            out = self(**batch)
            pooled_list.append(out["pooled_output"].detach().cpu())
            if progress:
                pct = min(100, int((i + len(sub)) * 100 / max(1, total)))
                print(
                    f"[TextEncoder] progress: {pct:3d}% ({i + len(sub)}/{total})",
                    end="\r",
                )
        if progress:
            print()
        return torch.cat(pooled_list, dim=0)

    # ------------------------ Utilities -------------------------
    def _auto_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _pool(
        self, sequence_output: Tensor, attention_mask: Optional[Tensor]
    ) -> Tensor:
        """根据配置进行聚合 / Pooling by configuration.

        - cls: 取位置 0 的表示；若模型没有明确 CLS 语义（如某些 RoBERTa 变体），仍按索引 0 取值。
        - mean: 按 attention_mask 做加权平均（去除 padding）。
        - max: 按 attention_mask 做 masked max（去除 padding）。
        """
        pooling = self.cfg.pooling
        if pooling == "cls":
            return sequence_output[:, 0]
        if attention_mask is None:
            raise ValueError(
                "mean/max pooling 需要 attention_mask / attention_mask required for mean/max pooling"
            )

        mask = attention_mask.unsqueeze(-1).type_as(sequence_output)  # (B, L, 1)
        if pooling == "mean":
            summed = (sequence_output * mask).sum(dim=1)  # (B, H)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        if pooling == "max":
            # 将 padding 位置置为极小值，避免参与 max
            very_small = torch.finfo(sequence_output.dtype).min
            masked = sequence_output.masked_fill(mask == 0, very_small)
            return masked.max(dim=1).values
        raise ValueError(f"未知的 pooling 策略 / Unknown pooling: {pooling}")


# -----------------------------------------------------------------------------
# Factory Helper
# -----------------------------------------------------------------------------
def build_text_encoder(
    cfg: TextEncoderConfig, *, trust_remote_code: bool = False
) -> HFTextEncoder:
    """根据配置构建文本编码器 / Build text encoder from config.

    Notes / 说明
    -------------
    - 如需支持 RNN/CNN 编码器，可在此处扩展分支但**保持返回类型不变**。
    - 当前仅提供 HF backbone 实现，满足论文/实用需求（BERT-family, RoBERTa, DeBERTa, E5 等）。
    """
    enc = HFTextEncoder.from_pretrained(cfg, trust_remote_code=trust_remote_code)
    logger.info("Text encoder built: %s", cfg)
    return enc


@TEXT.register("hf_text", alias=["hf", "transformer"])
def build_hf_text_encoder(
    model_name_or_path: str = "bert-base-uncased",
    max_length: int = 512,
    pooling: str = "cls",
    trainable: bool = False,
    use_fast_tokenizer: bool = True,
    pad_to_max_length: bool = True,
) -> HFTextEncoder:
    """\
    @brief 注册到 Registry 的 HF 文本编码器工厂。
           HF-based text encoder factory registered to 'text_encoder' registry.
    """
    cfg = TextEncoderConfig(
        model_name_or_path=model_name_or_path,
        max_length=max_length,
        pooling=pooling,  # type: ignore[arg-type]
        trainable=trainable,
        use_fast_tokenizer=use_fast_tokenizer,
        pad_to_max_length=pad_to_max_length,
    )
    return build_text_encoder(cfg)


# -----------------------------------------------------------------------------
# Self-test (optional manual)
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    cfg = TextEncoderConfig(
        model_name_or_path="prajjwal1/bert-tiny", max_length=32, pooling="mean"
    )
    enc = build_text_encoder(cfg)
    texts = [
        "Hello world!",
        "Knowledge-aware Attention Network (KAN) for Fake News Detection.",
    ]
    vec = enc.encode(texts, batch_size=2, progress=True)
    print("Embeddings:", vec.shape)
