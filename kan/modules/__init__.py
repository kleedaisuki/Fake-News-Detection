# -*- coding: utf-8 -*-
"""
@file   kan/modules/__init__.py
@brief  模型模块稳定入口（文本/实体/上下文编码器与分类头）
@date   2025-09-21

@zh
  对上层只暴露稳定 API：构造器(build_*), 配置(Config), 与核心类(Encoder/Head)。
  不在此处添加日志 Handler，保持“干净导出”与可观测性的一致入口（见 kan.utils.logging）。
@en
  Stable entrypoints for model modules: public builders (build_*), Configs, and core classes.
  Do not attach logging handlers here; keep exports clean and rely on kan.utils.logging for setup.
"""
from __future__ import annotations
from typing import Any


# -------- 小工具：占位符（当实现文件暂未到位时不破坏 import） --------
def _missing(name: str):
    class _Missing:
        """占位符：访问即抛出清晰错误；用于实现文件尚未落地的阶段。"""

        def __getattr__(self, *_: Any) -> Any:  # pragma: no cover
            raise ImportError(
                f"[kan.modules] symbol '{name}' is not available yet. "
                "Make sure the corresponding implementation file exists and is importable."
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise ImportError(
                f"[kan.modules] callable '{name}' is not available yet. "
                "Please implement or ensure module is on PYTHONPATH."
            )

    return _Missing()


# -------- 文本编码器 (Text Encoder) --------
try:
    from .text_encoder import (
        TextEncoderConfig,
        TextEncoder,  # 具体类（若提供）
        build_text_encoder,  # 推荐对外构造器
    )
except Exception:  # pragma: no cover
    TextEncoderConfig = _missing("TextEncoderConfig")
    TextEncoder = _missing("TextEncoder")
    build_text_encoder = _missing("build_text_encoder")

# -------- 实体编码器 (Entity Encoder) --------
try:
    from .entity_encoder import (
        EntityEncoderConfig,
        EntityEncoder,
        build_entity_encoder,
    )
except Exception:  # pragma: no cover
    EntityEncoderConfig = _missing("EntityEncoderConfig")
    EntityEncoder = _missing("EntityEncoder")
    build_entity_encoder = _missing("build_entity_encoder")

# -------- 上下文编码器 (Context Encoder) --------
try:
    from .context_encoder import (
        ContextEncoderConfig,
        ContextEncoder,
        build_context_encoder,
    )
except Exception:  # pragma: no cover
    ContextEncoderConfig = _missing("ContextEncoderConfig")
    ContextEncoder = _missing("ContextEncoder")
    build_context_encoder = _missing("build_context_encoder")

# -------- 预测头 (Prediction Head) --------
try:
    from .head import (
        HeadConfig,
        PredictionHead,  # 若实现文件使用别名 Head 也可在此导出
        build_head,
    )
except Exception:  # pragma: no cover
    HeadConfig = _missing("HeadConfig")
    PredictionHead = _missing("PredictionHead")
    build_head = _missing("build_head")

__all__ = [
    # text
    "TextEncoderConfig",
    "TextEncoder",
    "build_text_encoder",
    # entity
    "EntityEncoderConfig",
    "EntityEncoder",
    "build_entity_encoder",
    # context
    "ContextEncoderConfig",
    "ContextEncoder",
    "build_context_encoder",
    # head
    "HeadConfig",
    "PredictionHead",
    "build_head",
]
