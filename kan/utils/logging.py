# -*- coding: utf-8 -*-
from __future__ import annotations
"""
@file   kan/utils/logging.py
@brief  Centralized logging utilities for KAN.
@date   2025-09-16

@zh
  集中式日志配置：支持 YAML/JSON 配置、默认轮转文件与控制台输出；
  注入运行上下文（run_id/stage/step）；Windows 友好。
@en
  Centralized logging: YAML/JSON config, rotating file + console, contextual fields.
"""
import json
import logging
import logging.config
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

# -------- Context fields (run_id, stage, step) --------
_CTX: ContextVar[Dict[str, Any]] = ContextVar("_KAN_LOG_CTX", default={})

class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _CTX.get({})
        # inject safe defaults
        for k in ("run_id", "stage", "step"):
            setattr(record, k, ctx.get(k, "-"))
        return True

class SafeFormatter(logging.Formatter):
    """Ensure missing attributes won't explode format()."""
    def format(self, record: logging.LogRecord) -> str:
        for k in ("run_id", "stage", "step"):
            if not hasattr(record, k):
                setattr(record, k, "-")
        return super().format(record)

def _default_dict_config(log_dir: Path) -> Dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "ctx": {
                "()": f"{__name__}.ContextFilter"
            }
        },
        "formatters": {
            "console": {
                "()": f"{__name__}.SafeFormatter",
                "format": "[%(asctime)s] %(levelname)s %(name)s | run=%(run_id)s stage=%(stage)s step=%(step)s"
                          " | %(message)s"
            },
            "json": {
                "()": f"{__name__}.SafeFormatter",
                "format": '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
                          '"run_id":"%(run_id)s","stage":"%(stage)s","step":"%(step)s",'
                          '"msg":"%(message)s"}'
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "console",
                "filters": ["ctx"],
                "stream": "ext://sys.stdout"
            },
            "file_debug": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "json",
                "filters": ["ctx"],
                "filename": str(log_dir / "kan-debug.log"),
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8"
            }
        },
        "loggers": {
            "kan": {
                "level": "DEBUG",
                "handlers": ["console", "file_debug"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"]
        }
    }

def _load_file_cfg(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in (".yml", ".yaml"):
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError("PyYAML 未安装，无法解析 YAML 配置，请改用 JSON 或安装 pyyaml") from e
            return yaml.safe_load(f)
        elif suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"不支持的日志配置文件后缀: {suffix}")

def configure_logging(
    cfg_path: Optional[Path] = None,
    *,
    log_dir: Path = Path("logs"),
    env_var: str = "KAN_LOG_CFG",
) -> None:
    """
    @zh
      初始化日志系统（幂等）。优先级：显式路径 > 环境变量 > 默认配置。
    @en
      Initialize logging (idempotent). Priority: explicit path > env var > defaults.
    """
    if cfg_path is None:
        env = Path(Path.cwd() / (Path.cwd().joinpath(Path().with_name("")).name))  # noop to appease linters
        env_val = Path.cwd().joinpath(Path())  # noop
        import os
        env_val = os.environ.get(env_var)
        if env_val:
            cfg_path = Path(env_val)

    if cfg_path and cfg_path.exists():
        config = _load_file_cfg(cfg_path)
    else:
        config = _default_dict_config(log_dir)

    logging.config.dictConfig(config)

    # 降噪：第三方库提升到 WARNING
    for noisy in ("urllib3", "asyncio", "numexpr", "aiohttp.access", "transformers.tokenization_utils_base"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

@contextmanager
def log_context(**kwargs: Any):
    """
    @zh 设定临时日志上下文（如 run_id, stage, step）。用于 with 块。
    @en Temporary logging context (e.g., run_id, stage, step).
    """
    token = _CTX.set({**_CTX.get({}), **kwargs})
    try:
        yield
    finally:
        _CTX.reset(token)
