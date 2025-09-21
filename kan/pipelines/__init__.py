# -*- coding: utf-8 -*-
"""
@file   kan/pipelines/__init__.py
@brief  训练/评估编排层稳定入口（prepare/train/evaluate）
@date   2025-09-21

@zh
  本包提供 KAN 的流程（pipeline）级对外 API：数据准备、训练（Trainer/Accelerate 两种）、
  以及评估聚合。此层只负责**装配与调度**，不承载模型/算法细节（见 kan.modules.*, kan.data.*）。
  遵循“配置即契约（Config-as-Contract）”与“We don’t break userspace”原则，保证向后兼容。
@en
  Stable entrypoints for KAN orchestration: data preparation, training (Trainer/Accelerate),
  and evaluation aggregation. This layer performs assembly & scheduling only; model logic
  lives in kan.modules.* / kan.data.*. Follows Config-as-Contract and “We don’t break userspace”.
"""

from __future__ import annotations
from typing import Any, Callable


# ---- 内部：优雅降级的占位器（实现文件缺失时不破坏 import） -----------------
def _missing(name: str) -> Callable[..., Any]:
    def _raiser(*_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise ImportError(
            f"[kan.pipelines] '{name}' is not available. "
            "Please ensure the corresponding module file exists and is importable."
        )

    return _raiser


# ---- 数据准备（prepare_data.py） -----------------------------------------
# 目标与产物：缓存 datasets/EL/KG/向量化、词表与 manifest.json【README】
try:
    from .prepare_data import (
        run as run_prepare_data,  # 统一入口：合并配置→执行→写清单
    )
except Exception:  # pragma: no cover
    run_prepare_data = _missing("run_prepare_data")


# ---- 训练（train_trainer.py：🤗 Trainer 版） ------------------------------
# 特点：高复用、少代码；按 metric_for_best 管理 best/last checkpoints【README】
try:
    from .train_trainer import (
        run as run_train_trainer,  # 统一入口：build dataloader/model→Trainer.train/eval
    )
except Exception:  # pragma: no cover
    run_train_trainer = _missing("run_train_trainer")


# ---- 训练（train_accelerate.py：Accelerate 版） ---------------------------
# 特点：自定义 loop、灵活断点/调度、可接自定义日志后端【README】
try:
    from .train_accelerate import (
        run as run_train_accelerate,  # 统一入口：accelerator.prepare→自定义循环
    )
except Exception:  # pragma: no cover
    run_train_accelerate = _missing("run_train_accelerate")


# ---- 评估聚合（evaluate.py） ---------------------------------------------
# 产物：macro/micro 指标、混淆矩阵 CSV、ROC/PR 曲线点【README】
try:
    from .evaluate import (
        run as run_evaluate,  # 统一入口：读取 pred_*.jsonl → 聚合导出
    )
except Exception:  # pragma: no cover
    run_evaluate = _missing("run_evaluate")


__all__ = [
    # Stable pipeline entrypoints
    "run_prepare_data",  # 数据准备：EL/KG/向量化/词表/manifest
    "run_train_trainer",  # 训练（Trainer 版）
    "run_train_accelerate",  # 训练（Accelerate 版）
    "run_evaluate",  # 评估聚合（单 run / 多 run / k-fold）
]
