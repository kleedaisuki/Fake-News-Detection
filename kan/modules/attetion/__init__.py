# -*- coding: utf-8 -*-
"""
@file   kan/modules/attention/__init__.py
@brief  知识感知注意力（NE / NE2C）稳定入口
@date   2025-09-21

@zh
  暴露 NE 与 NE2C 两类注意力算子（类名约定：NEAttention / NE2CAttention）。
  当前实现可仅含“签名 + 日志 + 形状校验”，forward() 抛 NotImplementedError 亦可，
  但契约（张量形状/参数命名）须保持稳定。
@en
  Public entrypoints for knowledge-aware attention operators (NE / NE2C).
  Implementations may be stubs (signature + logging + shape checks), with forward() raising
  NotImplementedError; contracts must remain stable.
"""
from __future__ import annotations
from typing import Any


def _missing(name: str):
    class _Missing:
        def __getattr__(self, *_: Any) -> Any:  # pragma: no cover
            raise ImportError(
                f"[kan.modules.attention] symbol '{name}' is not available yet. "
                "Ensure 'ne.py'/'ne2c.py' exist and are importable."
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise ImportError(
                f"[kan.modules.attention] callable '{name}' is not available yet."
            )

    return _missing


# ---- NE（News → Entities）----
try:
    from .ne import (
        NEAttention,
    )  # 约定类名；logger 名称：kan.modules.attention.ne.NEAttention
except Exception:  # pragma: no cover
    NEAttention = _missing("NEAttention")()

# ---- NE2C（News → Entities → Contexts）----
try:
    from .ne2c import (
        NE2CAttention,
    )  # 约定类名；logger 名称：kan.modules.attention.ne2c.NE2CAttention
except Exception:  # pragma: no cover
    NE2CAttention = _missing("NE2CAttention")()

__all__ = [
    "NEAttention",
    "NE2CAttention",
]
