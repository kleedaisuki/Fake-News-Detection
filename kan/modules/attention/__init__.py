# -*- coding: utf-8 -*-
"""
@file   kan/modules/attention/__init__.py
@brief  知识感知注意力（NE / NE2C）稳定入口：懒加载 + 全局注册（Registry）
@brief  Stable entry for knowledge-aware attention (NE / NE2C): lazy import + global registry

@zh
  - 暴露 NE 与 NE2C 两类注意力算子（NEAttention / NE2CAttention）
  - 采用“懒加载（lazy import）”避免循环依赖和半实现期 ImportError
  - 自动将可用算子注册到全局 Registry 命名空间 "attention"
  - 若子模块缺失/异常，提供占位符（MissingSymbol）并在实例化时给出清晰错误

@en
  - Expose two attention operators (NEAttention / NE2CAttention)
  - Use lazy import to avoid cyclic imports and half-implemented crashes
  - Auto-register available operators to global Registry under "attention"
  - If a submodule is missing, provide a MissingSymbol placeholder that raises on call
"""
from __future__ import annotations

from typing import Any, Optional, Type, TYPE_CHECKING
import logging

# ---- Optional type-checking only imports (no runtime dependency) ----
if TYPE_CHECKING:
    from .ne import NEAttention as _NEAttentionType
    from .ne2c import NE2CAttention as _NE2CAttentionType

# ---- Logger ----
logger = logging.getLogger("kan.modules.attention.__init__")

# ---- Global Registry HUB ----
try:
    from kan.utils.registry import HUB  # 全局注册中心 / Global registry hub
except Exception as e:  # pragma: no cover
    HUB = None  # type: ignore[assignment]
    logger.warning("[kan.modules.attention] registry HUB unavailable: %s", e)


# =============================================================================
# Missing symbol placeholder
# =============================================================================
class MissingSymbol:
    """
    @brief 缺失占位符：在实例化或访问属性时抛出清晰错误
    @brief Placeholder for missing symbol: raises informative errors on use

    @param name  符号名（中文）/ Symbol name (EN)
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, *_: Any) -> Any:  # pragma: no cover
        raise ImportError(
            f"[kan.modules.attention] symbol '{self._name}' is not available. "
            "Ensure 'ne.py'/'ne2c.py' exist and are importable."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise ImportError(
            f"[kan.modules.attention] callable '{self._name}' is not available. "
            "This operator has not been implemented or failed to import."
        )


def _try_import(module: str, symbol: str) -> Any:
    """
    @brief 安全导入工具：失败时返回 MissingSymbol，而非直接崩溃
    @brief Safe import helper: return MissingSymbol instead of crashing on failure

    @param module  子模块名（如 'ne', 'ne2c'）/ submodule name (e.g., 'ne', 'ne2c')
    @param symbol  目标符号名（如 'NEAttention'）/ target symbol name
    @return 已导入的符号或 MissingSymbol / imported symbol or MissingSymbol
    """
    try:
        mod = __import__(f"{__name__}.{module}", fromlist=[symbol])
        obj = getattr(mod, symbol)
        logger.debug("Imported %s.%s successfully.", module, symbol)
        return obj
    except Exception as e:  # pragma: no cover
        logger.warning(
            "[kan.modules.attention] lazy import failed for %s.%s: %s",
            module,
            symbol,
            e,
        )
        return MissingSymbol(symbol)


# =============================================================================
# Lazy symbols (exported API)
# =============================================================================
NEAttention: "Type[_NEAttentionType] | MissingSymbol" = _try_import("ne", "NEAttention")
NE2CAttention: "Type[_NE2CAttentionType] | MissingSymbol" = _try_import(
    "ne2c", "NE2CAttention"
)

__all__ = [
    "NEAttention",
    "NE2CAttention",
    "register_all",
]


# =============================================================================
# Registry integration
# =============================================================================
def _is_available(obj: Any) -> bool:
    """
    @brief 判断符号是否可用（非 MissingSymbol）
    @brief Check whether a symbol is available (not a MissingSymbol)

    @param obj  待检查对象 / object to check
    @return True 如果可用 / True if available
    """
    return not isinstance(obj, MissingSymbol)


def register_all() -> None:
    """
    @brief 将可用注意力模块注册到全局 Registry（命名空间 "attention"）
    @brief Register available attention operators into the global Registry ("attention" namespace)

    @note 若 HUB 不可用，仅记录警告；不会抛出异常
    @note If HUB is unavailable, only warn; do not raise.

    @example
    @zh
      >>> from kan.modules.attention import register_all
      >>> register_all()
      >>> from kan.utils.registry import HUB
      >>> attn = HUB.get("attention").build({"type": "ne", "d_model": 256})
    @en
      >>> from kan.modules.attention import register_all
      >>> register_all()
      >>> from kan.utils.registry import HUB
      >>> attn = HUB.get("attention").build({"type": "ne", "d_model": 256})
    """
    if HUB is None:  # pragma: no cover
        logger.warning("[kan.modules.attention] skip register_all(): HUB is None.")
        return

    registry = HUB.get_or_create("attention")

    # 注册可用算子 / register available operators
    if _is_available(NEAttention):
        registry.register("ne", NEAttention)  # 新闻→实体 / News→Entities
        logger.debug("Registered attention.ne -> NEAttention")
    else:
        logger.info("NEAttention unavailable; skipped registry binding.")

    if _is_available(NE2CAttention):
        registry.register(
            "ne2c", NE2CAttention
        )  # 新闻→实体→上下文 / News→Entities→Contexts
        logger.debug("Registered attention.ne2c -> NE2CAttention")
    else:
        logger.info("NE2CAttention unavailable; skipped registry binding.")


# ---- Auto-register on import (safe) ----
try:
    register_all()
except Exception as e:  # pragma: no cover
    logger.warning("[kan.modules.attention] auto register failed: %s", e)
