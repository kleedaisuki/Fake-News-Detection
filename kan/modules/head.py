# -*- coding: utf-8 -*-
"""
@file: head.py

中英双语文档 / Bilingual Docstring
===================================

目的 / Purpose
--------------
实现最终分类/回归头（Head），将新闻文本向量 `p`、实体注意力结果 `q`、上下文注意力结果 `r`
进行融合（fusion），随后经 MLP 输出 logits/数值。该契约与 NE/NE2C 一致：通常以
**concat([p, q, r]) → MLP → logits** 作为默认路径（与 KAN 论文式(14)(15) 的拼接-分类器一致）。

输入（keyword-only） / Inputs (keyword-only)
-------------------------------------------
- `p: FloatTensor[B, Hp]`   来自 text_encoder 的新闻向量（可选，按 use_p 控制）
- `q: FloatTensor[B, Hq]`   来自 NE 的实体聚合向量（可选，按 use_q 控制）
- `r: FloatTensor[B, Hr]`   来自 NE2C 的上下文聚合向量（可选，按 use_r 控制）
- `labels: Optional[Tensor]` 监督信号（shape 依任务而定，见下）

输出 / Outputs
--------------
- `logits: FloatTensor[B, C]`  (分类) 或 `[B] / [B, C]` (回归)
- `loss`（可选，当传入 labels 时返回）
- `fused`（可选，融合后的中间向量，便于可解释/调试）

任务类型 / Problem Types
------------------------
- `single_label_classification`（单标签分类，默认）：CrossEntropyLoss（支持 class_weights / label_smoothing）
- `multi_label_classification`（多标签分类）：BCEWithLogitsLoss（labels 形状 [B,C]）
- `regression`：MSELoss（`num_labels==1` 时输出 squeeze 到 [B]）

工程特性 / Engineering Notes
----------------------------
- **Lazy 线性层**：首层使用 `nn.LazyLinear`，无需在构造时显式提供输入维度，首个 forward 自动绑定。
- **可选投影**：支持为 `p/q/r` 各自添加线性投影至统一维度 `proj_dim`，便于 `sum/mean` 融合。
- **日志**：命名 logger `kan.head`；复用全局 logging 配置。
- **We don't break userspace!** 默认参数全覆盖，输出键名稳定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
import logging

import torch
from torch import nn, Tensor
import torch.nn.functional as F

logger = logging.getLogger("kan.head")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class HeadConfig:
    """最终预测头配置 / Configuration for the prediction head.

    Attributes
    ----------
    num_labels : int
        类别数（分类）或输出维度（回归）。
    problem_type : Literal['single_label_classification','multi_label_classification','regression']
        任务类型。
    use_p : bool
        是否使用文本向量 p。
    use_q : bool
        是否使用实体向量 q。
    use_r : bool
        是否使用上下文向量 r。
    fusion : Literal['concat','sum','mean']
        融合策略。`sum/mean` 需要先投影到统一维度（见 `proj_dim`）。
    proj_dim : Optional[int]
        若不为 None，则为每个启用的分支（p/q/r）添加 `LazyLinear(proj_dim)` 以便统一维度。
    hidden_sizes : List[int]
        融合后 MLP 的隐藏层大小列表。空列表表示直接线性到输出。
    activation : Literal['relu','gelu']
        激活函数。
    dropout : float
        Dropout 概率（用于融合后与各隐藏层之间）。
    layernorm : bool
        是否在融合后做 LayerNorm（训练参数在首次 forward 时动态创建）。
    label_smoothing : float
        分类的标签平滑，默认 0.0。
    class_weights : Optional[List[float]]
        类别权重（CE 用），长度需等于 `num_labels`。
    bias : bool
        线性层是否带偏置。
    """

    num_labels: int = 2
    problem_type: Literal[
        "single_label_classification", "multi_label_classification", "regression"
    ] = "single_label_classification"
    use_p: bool = True
    use_q: bool = True
    use_r: bool = True
    fusion: Literal["concat", "sum", "mean"] = "concat"
    proj_dim: Optional[int] = None
    hidden_sizes: List[int] = field(default_factory=lambda: [256])
    activation: Literal["relu", "gelu"] = "gelu"
    dropout: float = 0.1
    layernorm: bool = True
    label_smoothing: float = 0.0
    class_weights: Optional[List[float]] = None
    bias: bool = True


# -----------------------------------------------------------------------------
# Head Module
# -----------------------------------------------------------------------------
class Head(nn.Module):
    """融合 p/q/r 后进行分类/回归的头。

    - 支持 concat/sum/mean 融合；
    - 支持分支投影到统一维度（proj_dim）；
    - 首层使用 LazyLinear，免去在构造时显式提供输入维度。
    """

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg

        # 分支投影（lazy），仅在对应分支启用时生效
        self.proj_p = (
            nn.LazyLinear(cfg.proj_dim)
            if (cfg.proj_dim and cfg.use_p)
            else nn.Identity()
        )
        self.proj_q = (
            nn.LazyLinear(cfg.proj_dim)
            if (cfg.proj_dim and cfg.use_q)
            else nn.Identity()
        )
        self.proj_r = (
            nn.LazyLinear(cfg.proj_dim)
            if (cfg.proj_dim and cfg.use_r)
            else nn.Identity()
        )

        # 融合后 LayerNorm（延迟初始化，见 forward）
        self._post_fuse_ln: Optional[nn.LayerNorm] = None

        # 构造 MLP（全部 LazyLinear，直到最终输出）
        layers: List[nn.Module] = []
        if len(cfg.hidden_sizes) == 0:
            layers.append(nn.LazyLinear(cfg.num_labels, bias=cfg.bias))
        else:
            # 首层到 hidden_sizes[0]
            layers.append(nn.LazyLinear(cfg.hidden_sizes[0], bias=cfg.bias))
            for i in range(len(cfg.hidden_sizes) - 1):
                layers.extend(
                    [
                        self._activation_layer(cfg.activation),
                        nn.Dropout(cfg.dropout),
                        nn.LazyLinear(cfg.hidden_sizes[i + 1], bias=cfg.bias),
                    ]
                )
            # 输出层
            layers.extend(
                [
                    self._activation_layer(cfg.activation),
                    nn.Dropout(cfg.dropout),
                    nn.LazyLinear(cfg.num_labels, bias=cfg.bias),
                ]
            )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(cfg.dropout)

        # 损失函数占位（延迟绑定 weights）
        self._ce_weight: Optional[Tensor] = None

        logger.info(
            "Head init: problem=%s, fusion=%s, proj_dim=%s, hidden=%s, use=[p:%s q:%s r:%s]",
            cfg.problem_type,
            cfg.fusion,
            cfg.proj_dim,
            cfg.hidden_sizes,
            cfg.use_p,
            cfg.use_q,
            cfg.use_r,
        )

    # ----------------------------- Forward -----------------------------
    def forward(
        self,
        *,
        p: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        r: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        cfg = self.cfg

        feats: List[Tensor] = []
        if cfg.use_p:
            if p is None:
                raise ValueError("use_p=True 但未提供张量 p")
            feats.append(self._maybe_project(self.proj_p, p))
        if cfg.use_q:
            if q is None:
                raise ValueError("use_q=True 但未提供张量 q")
            feats.append(self._maybe_project(self.proj_q, q))
        if cfg.use_r:
            if r is None:
                raise ValueError("use_r=True 但未提供张量 r")
            feats.append(self._maybe_project(self.proj_r, r))
        if len(feats) == 0:
            raise ValueError("Head 至少需要一个输入分支（p/q/r）")

        # 融合
        if cfg.fusion == "concat":
            fused = torch.cat(feats, dim=-1)
        elif cfg.fusion in ("sum", "mean"):
            # 为安全起见，确保它们最后维度一致（通常通过 proj_dim 保证）
            last_dims = {int(t.shape[-1]) for t in feats}
            if len(last_dims) != 1:
                raise ValueError(
                    "sum/mean 融合要求各分支最后维度一致；请设置 proj_dim 或手动对齐"
                )
            stacked = torch.stack(feats, dim=0)  # [K,B,D]
            fused = stacked.sum(dim=0) if cfg.fusion == "sum" else stacked.mean(dim=0)
        else:
            raise ValueError(f"未知的 fusion 策略: {cfg.fusion}")

        # 融合后 LN（首次 forward 时动态创建，随后复用）
        if cfg.layernorm:
            if self._post_fuse_ln is None:
                self._post_fuse_ln = nn.LayerNorm(fused.shape[-1])
                # 将动态创建的层注册进模块，以便保存/加载
                self.add_module("post_fuse_ln", self._post_fuse_ln)
            fused = self._post_fuse_ln(fused)

        fused = self.dropout(fused)
        logits = self.mlp(fused)

        out: Dict[str, Tensor] = {"logits": logits, "fused": fused}

        # 训练：计算损失
        if labels is not None:
            if cfg.problem_type == "single_label_classification":
                if self._ce_weight is None and cfg.class_weights is not None:
                    w = torch.tensor(
                        cfg.class_weights, dtype=logits.dtype, device=logits.device
                    )
                    self._ce_weight = w
                loss = F.cross_entropy(
                    logits,
                    labels.long(),
                    weight=self._ce_weight,
                    label_smoothing=cfg.label_smoothing,
                )
            elif cfg.problem_type == "multi_label_classification":
                # labels 形状 [B, C]，数值 0/1 或任意概率
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            elif cfg.problem_type == "regression":
                # 若 num_labels==1，支持 [B] / [B,1] 两种标签形状
                if cfg.num_labels == 1 and labels.dim() == 1:
                    loss = F.mse_loss(logits.squeeze(-1), labels.float())
                else:
                    loss = F.mse_loss(logits, labels.float())
            else:
                raise ValueError(f"未知的 problem_type: {cfg.problem_type}")
            out["loss"] = loss

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "head: fused=%s, logits=%s", tuple(fused.shape), tuple(logits.shape)
            )
        return out

    # --------------------------- Helpers ---------------------------
    def _maybe_project(self, proj: nn.Module, x: Tensor) -> Tensor:
        y = proj(x)
        return y

    @staticmethod
    def _activation_layer(name: Literal["relu", "gelu"]) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"未知的激活函数: {name}")


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def build_head(cfg: HeadConfig) -> Head:
    head = Head(cfg)
    logger.info("Head built: %s", cfg)
    return head


# -----------------------------------------------------------------------------
# Self-test (optional manual)
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    cfg = HeadConfig(
        num_labels=2, fusion="concat", proj_dim=None, hidden_sizes=[128, 64]
    )
    head = build_head(cfg)
    B, Hp, Hq, Hr = 4, 768, 256, 256
    p = torch.randn(B, Hp)
    q = torch.randn(B, Hq)
    r = torch.randn(B, Hr)
    out = head(p=p, q=q, r=r, labels=torch.randint(0, 2, (B,)))
    print(out["logits"].shape, out["loss"].item())
