# -*- coding: utf-8 -*-
"""
@file   kan/utils/__init__.py
@brief  KAN 实用工具稳定入口（日志/指标/注册表/随机性） | Stable utilities entrypoints
@date   2025-09-21

@zh
  对外暴露四类核心工具：
    1) logging：集中式日志配置与上下文（configure_logging, log_context）
    2) metrics：分类指标计算与折次聚合（compute_classification_metrics, FoldAccumulator, …）
    3) registry：强类型命名空间注册表与构建器（Registry, RegistryHub, HUB, build_from_config）
    4) seed：随机性控制与确定性工具（set_seed, seed_worker, with_seed, rng_state, …）
  遵循 “We don't break userspace” 原则，保证 API 向后兼容。
@en
  Public, stable utility API surface for:
    (1) logging, (2) metrics, (3) registry, and (4) seed/determinism helpers.
  Follow “We don't break userspace” to keep backward compatibility.
"""
from __future__ import annotations
from typing import Any


# --------- 小工具：占位降级，避免实现文件暂缺导致 import 失败 ---------
def _missing(name: str):
    class _Missing:
        """@brief 占位符；访问即抛清晰错误 | Placeholder that raises clear ImportError on access."""

        def __getattr__(self, *_: Any) -> Any:  # pragma: no cover
            raise ImportError(
                f"[kan.utils] symbol '{name}' is not available yet. "
                "Ensure its implementation module exists and is importable."
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise ImportError(f"[kan.utils] callable '{name}' is not available yet.")

    return _Missing()


# --------- logging（集中式配置与上下文）---------
# 说明：此处仅导出函数，不创建 handler，避免重复句柄与噪声。
try:
    from .logging import configure_logging, log_context  # noqa: F401
except Exception:  # pragma: no cover
    configure_logging = _missing("configure_logging")
    log_context = _missing("log_context")

# --------- metrics（分类指标与折次聚合）---------
try:
    from .metrics import (  # noqa: F401
        infer_task_type,
        compute_classification_metrics,
        safe_confusion_matrix,
        FoldAccumulator,
    )
except Exception:  # pragma: no cover
    infer_task_type = _missing("infer_task_type")
    compute_classification_metrics = _missing("compute_classification_metrics")
    safe_confusion_matrix = _missing("safe_confusion_matrix")
    FoldAccumulator = _missing("FoldAccumulator")

# --------- registry（注册表与工厂）---------
try:
    from .registry import (  # noqa: F401
        Registry,
        RegistryHub,
        HUB,
        build_from_config,
    )
except Exception:  # pragma: no cover
    Registry = _missing("Registry")
    RegistryHub = _missing("RegistryHub")
    HUB = _missing("HUB")
    build_from_config = _missing("build_from_config")

# --------- seed（随机性与确定性工具）---------
try:
    from .seed import (  # noqa: F401
        set_seed,
        seed_worker,
        with_seed,
        rng_state,
        restore_rng_state,
        derive_seed,
    )
except Exception:  # pragma: no cover
    set_seed = _missing("set_seed")
    seed_worker = _missing("seed_worker")
    with_seed = _missing("with_seed")
    rng_state = _missing("rng_state")
    restore_rng_state = _missing("restore_rng_state")
    derive_seed = _missing("derive_seed")

__all__ = [
    # logging
    "configure_logging",
    "log_context",
    # metrics
    "infer_task_type",
    "compute_classification_metrics",
    "safe_confusion_matrix",
    "FoldAccumulator",
    # registry
    "Registry",
    "RegistryHub",
    "HUB",
    "build_from_config",
    # seed
    "set_seed",
    "seed_worker",
    "with_seed",
    "rng_state",
    "restore_rng_state",
    "derive_seed",
]
