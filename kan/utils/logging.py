# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kan/utils/logging.py
@brief  集中式日志工具（Chinese）/ Centralized logging utilities (English).
@date   2025-09-21

@zh
  目标：在**不破坏用户空间**前提下增强稳健性与可观测性，仅修改此文件即可生效。
  - 保持原公开 API：configure_logging(), log_context()（幂等，不在 import 期做副作用）
  - 结构化日志：新增 JsonFormatter2（dict→json.dumps，避免模板转义风险）
  - 掩码（masking）：MaskFilter 支持对敏感字段自动打码
  - 限流（rate limit）：RateLimitFilter 防刷屏
  - 多进程安全：可选 QueueHandler/QueueListener（env 或参数开启）
  - 异常采集：可选安装 sys.excepthook / asyncio 异常处理（默认关闭）
  - 快速开关：支持环境变量切换等级/格式/限流/多进程（开发、CI 友好）

@en
  Goal: strengthen robustness & observability without breaking userspace; changes localized
  to this file only. Keeps public API (configure_logging, log_context). Adds structured JSON
  formatter, masking, rate limiting, optional multiprocess-safe queue logging, and optional
  exception hooks. Controlled by parameters or env vars.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import atexit
import json
import logging
import logging.config
import sys
import os
import time
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import multiprocessing as mp

# 注：避免在多进程路径下误用线程队列（queue.Queue）
try:
    import queue as _thread_queue  # 仅用于单进程或线程场景
except Exception:  # pragma: no cover
    _thread_queue = None

__all__ = [
    "configure_logging",
    "log_context",
    "ContextFilter",
    "SafeFormatter",
    "JsonFormatter2",
    "MaskFilter",
    "RateLimitFilter",
]

# ----------------------------- 环境变量约定 / Env Keys -----------------------------

MASK_KEYS_ENV = "KAN_LOG_MASK_KEYS"  # comma list, e.g. "password,token,email"
FORMAT_ENV = "KAN_LOG_FORMAT"  # "human" | "json"
LEVEL_ENV = "KAN_LOG_LEVEL"  # "DEBUG" | "INFO" | "WARNING" | ...
MULTIPROC_ENV = "KAN_LOG_MULTIPROC"  # "1" to enable QueueHandler/Listener
EXHOOK_ENV = "KAN_LOG_EXCEPT_HOOK"  # "1" to install excepthook
RATE_ENV = "KAN_LOG_RATE"  # e.g. "msg_per_sec=5;window_sec=10"

# ----------------------------- 上下文字段 / Log Context -----------------------------

_CTX: ContextVar[Dict[str, Any]] = ContextVar("_KAN_LOG_CTX", default={})
_QL: Optional[QueueListener] = None  # global QueueListener if enabled
_Q: Optional[Queue] = None  # global Queue if enabled


class ContextFilter(logging.Filter):
    """@brief 注入上下文字段（Chinese）/ Inject context fields (English).

    @zh
      将 ContextVar 中的 run_id/stage/step 注入记录，若缺失则填 "-".
    @en
      Inject run_id/stage/step from ContextVar into LogRecord with safe defaults.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _CTX.get({})
        for k in ("run_id", "stage", "step"):
            setattr(record, k, ctx.get(k, "-"))
        return True


class SafeFormatter(logging.Formatter):
    """@brief 缺字段容错格式化（Chinese）/ Tolerant formatter for missing fields (English)."""

    def format(self, record: logging.LogRecord) -> str:
        for k in ("run_id", "stage", "step"):
            if not hasattr(record, k):
                setattr(record, k, "-")
        return super().format(record)


class JsonFormatter2(logging.Formatter):
    """@brief 结构化 JSON 格式化（Chinese）/ Structured JSON formatter (English).

    @zh
      以 dict→json.dumps 生成，避免字符串模板转义陷阱；包含源位置与异常详情。
    @en
      Build a dict and json.dumps it to avoid string-template escape pitfalls; includes source
      location and exception details.
    """

    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "stage": getattr(record, "stage", "-"),
            "step": getattr(record, "step", "-"),
            "msg": record.getMessage(),
            "file": record.pathname,
            "line": record.lineno,
            "func": record.funcName,
            "process": record.process,
            "thread": record.thread,
        }
        if record.exc_info:
            base["err"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "",
                "msg": str(record.exc_info[1]),
                "stack": self.formatException(record.exc_info),
            }
        return json.dumps(base, ensure_ascii=False)


class MaskFilter(logging.Filter):
    """@brief 字段掩码过滤器（Chinese）/ Field masking filter (English).

    @param keys
        @zh 需要掩码的 record 属性名列表；若为 None，则仅使用环境变量。
        @en attribute names on LogRecord to be masked; if None, only env is used.

    @note
        @zh 读取 KAN_LOG_MASK_KEYS（逗号分隔），将对应字段值替换为 "***"。
        @en reads KAN_LOG_MASK_KEYS and replaces listed fields with "***".
    """

    def __init__(self, keys: Optional[List[str]] = None):
        super().__init__()
        env_keys = os.environ.get(MASK_KEYS_ENV, "")
        env_list = [k.strip() for k in env_keys.split(",") if k.strip()]
        self.keys = set((keys or []) + env_list)

    def filter(self, record: logging.LogRecord) -> bool:
        for k in self.keys:
            if hasattr(record, k) and getattr(record, k) not in (None, "-"):
                setattr(record, k, "***")
        return True


class RateLimitFilter(logging.Filter):
    """@brief 简单双令牌桶限流（Chinese）/ Simple token-bucket-ish rate limit (English).

    @param capacity
        @zh 每窗口允许的最大日志条数（0 表示不启用）。
        @en max messages per window (0 disables limiting).
    @param window_s
        @zh 限流窗口秒数。
        @en window length in seconds.

    @example
        @zh 设置环境变量：KAN_LOG_RATE="msg_per_sec=5;window_sec=1"。
        @en set env: KAN_LOG_RATE="msg_per_sec=5;window_sec=1".
    """

    def __init__(self, capacity: int = 0, window_s: int = 10):
        super().__init__()
        self.capacity = max(0, capacity)
        self.window_s = max(1, window_s)
        self.bucket: Dict[str, Tuple[int, float]] = {}  # key -> (count, window_start)

    def _key(self, record: logging.LogRecord) -> str:
        # 合并维度：logger + level + 原始消息模板（不含变量），降低误伤
        return f"{record.name}|{record.levelno}|{record.msg}"

    def filter(self, record: logging.LogRecord) -> bool:
        if self.capacity == 0:
            return True
        now = time.time()
        key = self._key(record)
        count, start = self.bucket.get(key, (0, now))
        if now - start >= self.window_s:
            self.bucket[key] = (1, now)
            return True
        if count < self.capacity:
            self.bucket[key] = (count + 1, start)
            return True
        return False


class _LiveSysStdout:
    """@brief 面向 capsys 的“活的 stdout 代理”（Chinese）/ Live proxy to current sys.stdout (English).
    @zh 每次 write/flush 都转发到“此刻的 sys.stdout”，避免 pytest 重置 stdout 后出现
        I/O on closed file。close() 为 no-op，防止误关真实 stdout。
    @en Forward write/flush to the current sys.stdout at call time so that pytest capsys
        resets won't break the handler; close() is a no-op.
    """

    def write(self, s: str) -> int:
        return sys.stdout.write(s)

    def flush(self) -> None:
        sys.stdout.flush()

    def close(self) -> None:
        # never close the real stdout
        pass

    # 某些库会探测 isatty；保守返回 False
    def isatty(self) -> bool:
        try:
            return bool(getattr(sys.stdout, "isatty", lambda: False)())
        except Exception:
            return False


# ============================================================================
# 多进程日志辅助 (_mp) —— 显式“主进程监听 / 子进程投递”两步式
# Multiprocessing logging helpers (_mp)
# ============================================================================

# 这些全局仅在主进程内有效（持有监听器与队列）
_MP_CTX: Optional[mp.context.BaseContext] = None
_MP_Q: Optional["mp.queues.Queue"] = None
_MP_QL: Optional["QueueListener"] = None


def _get_ctx(start_method: Optional[str]) -> "mp.context.BaseContext":
    """
    @brief 返回指定 start_method 的 multiprocessing 上下文（Chinese）/ Get multiprocessing context with given start method (English).
    @param start_method 启动方式："spawn" | "fork" | "forkserver" 或 None / Start method or None
    @return 对应的 BaseContext / The BaseContext instance
    """
    return mp.get_context(start_method) if start_method else mp.get_context()


def _is_main_process() -> bool:
    """
    @brief 判断当前是否主进程（Chinese）/ Check if current process is MainProcess (English).
    @return True 表示主进程 / True if main process
    """
    try:
        return mp.current_process().name == "MainProcess"
    except Exception:
        return True


def start_listener_main(
    *,
    log_dir: "Path",
    level_name: str,
    fmt: str,
    rate_limit: Optional[tuple],
    start_method: Optional[str] = None,
) -> "mp.queues.Queue":
    """
    @brief 启动主进程日志监听器（单点写入），并返回可跨进程共享的 mp.Queue（Chinese）
           Start main-process QueueListener (single sink) and return mp.Queue (English).
    @param log_dir 日志目录 / Directory for log files
    @param level_name 控制台等级名（INFO/DEBUG...）/ Console level name
    @param fmt "human" | "json"
    @param rate_limit 节流配置 (capacity, window_s) 或 None / Rate limit tuple or None
    @param start_method 进程启动方式（可选）/ Optional start method
    @return 用于子进程投递的 mp.Queue / The mp.Queue to pass to workers
    @note 仅应在主进程调用一次；重复调用返回同一队列 / Should be called once in main; re-entrant.
    """
    global _MP_CTX, _MP_Q, _MP_QL
    if _MP_QL is not None and _MP_Q is not None:
        return _MP_Q

    _MP_CTX = _get_ctx(start_method)
    _MP_Q = _MP_CTX.Queue(-1)

    # === 监听端的“单写” handlers（控制台 + 文件）===
    console = logging.StreamHandler(_LiveSysStdout())
    level_no = getattr(logging, level_name.upper(), logging.INFO)
    console.setLevel(level_no)
    if fmt == "json":
        console.setFormatter(JsonFormatter2())
    else:
        console.setFormatter(
            SafeFormatter(
                "[%(asctime)s] %(levelname)s %(name)s | run=%(run_id)s "
                "stage=%(stage)s step=%(step)s | %(message)s"
            )
        )

    file_h = RotatingFileHandler(
        str(Path(log_dir) / "kan-debug.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(JsonFormatter2())

    # 过滤器保持与原实现一致（集中生效在监听端）
    for h in (console, file_h):
        h.addFilter(ContextFilter())
        h.addFilter(MaskFilter())
        if rate_limit:
            cap, win = rate_limit
            h.addFilter(RateLimitFilter(capacity=cap, window_s=win))

    _MP_QL = QueueListener(_MP_Q, console, file_h, respect_handler_level=True)
    _MP_QL.start()

    import atexit

    atexit.register(lambda: (_MP_QL.stop() if _MP_QL else None))
    return _MP_Q


def install_queue_handler_worker(
    q: "mp.queues.Queue", logger_name: str = "kan"
) -> None:
    """
    @brief 在当前进程为指定 logger 安装 QueueHandler（只投递，不直写）（Chinese）
           Install QueueHandler for given logger in this process (producer only) (English).
    @param q 由主进程创建的 mp.Queue / The mp.Queue created in main process
    @param logger_name 目标 logger 名称 / Target logger name
    @note 幂等：重复调用不会产生多重 QueueHandler（Chinese & English）
    """
    log = logging.getLogger(logger_name)
    # 移除非队列 handler，避免直写到控制台/文件
    for h in list(log.handlers):
        if not isinstance(h, QueueHandler):
            log.removeHandler(h)
    if not any(isinstance(h, QueueHandler) for h in log.handlers):
        qh = QueueHandler(q)
        qh.setLevel(logging.DEBUG)
        log.addHandler(qh)
    log.propagate = False
    log.setLevel(logging.DEBUG)


# ----------------------------- 默认配置 / Default DictConfig -----------------------------


def _default_dict_config(
    log_dir: Path,
    *,
    fmt: str = "human",
    level: str = "INFO",
    use_queue: bool = False,
    rate_limit: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """@brief 生成默认 DictConfig（Chinese）/ Build default DictConfig (English)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    json_formatter = f"{__name__}.JsonFormatter2"
    console_formatter = f"{__name__}.SafeFormatter"
    formatter_name = "json" if fmt == "json" else "console"
    rl_capacity, rl_window = rate_limit or (0, 10)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "ctx": {"()": f"{__name__}.ContextFilter"},
            "mask": {"()": f"{__name__}.MaskFilter"},
            "rate": {
                "()": f"{__name__}.RateLimitFilter",
                "capacity": rl_capacity,
                "window_s": rl_window,
            },
        },
        "formatters": {
            "console": {
                "()": console_formatter,
                "format": "[%(asctime)s] %(levelname)s %(name)s | run=%(run_id)s stage=%(stage)s "
                "step=%(step)s | %(message)s",
            },
            "json": {"()": json_formatter},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": formatter_name,
                "filters": ["ctx", "mask", "rate"],
                "stream": "ext://sys.stdout",
            },
            "file_debug": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "json",
                "filters": ["ctx", "mask", "rate"],
                "filename": str(log_dir / "kan-debug.log"),
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "kan": {
                "level": "DEBUG",
                # 当 use_queue=True 时，handlers 会由 QueueListener 接管，因此此处留空并在后续安装
                "handlers": [] if use_queue else ["console", "file_debug"],
                "propagate": False,
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"],
        },
    }


def _load_file_cfg(path: Path) -> Dict[str, Any]:
    """@brief 从 YAML/JSON 文件加载配置（Chinese）/ Load DictConfig from YAML/JSON (English)."""
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in (".yml", ".yaml"):
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "PyYAML 未安装，无法解析 YAML 配置，请改用 JSON 或安装 pyyaml"
                ) from e
            return yaml.safe_load(f)
        elif suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"不支持的日志配置文件后缀: {suffix}")


# ----------------------------- 配置入口 / Public API -----------------------------


def configure_logging(
    cfg_path: Optional[Path] = None,
    *,
    log_dir: Path = Path("logs"),
    env_var: str = "KAN_LOG_CFG",
    level: Optional[str] = None,
    fmt: Optional[str] = None,
    multiproc: Optional[bool] = None,
    rate_limit: Optional[Tuple[int, int]] = None,
) -> None:
    """@brief 初始化日志系统（Chinese）/ Initialize logging system (English).

    @zh
      优先级：显式参数 > 环境变量 > 默认值。幂等；不在 import 期默认调用。
      - 等级：level 或 KAN_LOG_LEVEL
      - 格式：fmt 或 KAN_LOG_FORMAT（human/json）
      - 多进程：multiproc 或 KAN_LOG_MULTIPROC=1
      - 限流：rate_limit=(cap,win) 或 KAN_LOG_RATE="msg_per_sec=5;window_sec=1"
      - 掩码：KAN_LOG_MASK_KEYS="token,email"

    @en
      Priority: explicit args > env vars > defaults. Idempotent; not auto-invoked on import.
    """
    # 1) 解析 cfg 路径
    if cfg_path is None:
        env_val = os.environ.get(env_var)
        if env_val:
            cfg_path = Path(env_val)

    # 2) 有效等级/格式/并发设置
    eff_level = (level or os.environ.get(LEVEL_ENV) or "INFO").upper()
    eff_fmt = (fmt or os.environ.get(FORMAT_ENV) or "human").lower()
    if multiproc is None:
        multiproc = os.environ.get(MULTIPROC_ENV, "0") == "1"
    if rate_limit is None and (rv := os.environ.get(RATE_ENV)):
        # 支持形如 "msg_per_sec=5;window_sec=10"
        try:
            parts = dict(p.split("=", 1) for p in rv.split(";") if "=" in p)
            rate_limit = (
                int(parts.get("msg_per_sec", "0")),
                int(parts.get("window_sec", "10")),
            )
        except Exception:
            rate_limit = None

    # 3) 组装配置
    if cfg_path and cfg_path.exists():
        config = _load_file_cfg(cfg_path)
        # 注入必要 filters（若外部配置缺失），尽可能保持增强能力
        config.setdefault("filters", {})
        config["filters"].setdefault("ctx", {"()": f"{__name__}.ContextFilter"})
        config["filters"].setdefault("mask", {"()": f"{__name__}.MaskFilter"})
        # 限流按需
        if rate_limit:
            cap, win = rate_limit
            config["filters"].setdefault(
                "rate",
                {"()": f"{__name__}.RateLimitFilter", "capacity": cap, "window_s": win},
            )
        # 确保至少有 root console
        config.setdefault("root", {"level": "WARNING", "handlers": ["console"]})
        config.setdefault("formatters", {})
        config["formatters"].setdefault(
            "console",
            {
                "()": f"{__name__}.SafeFormatter",
                "format": "[%(asctime)s] %(levelname)s %(name)s | run=%(run_id)s stage=%(stage)s "
                "step=%(step)s | %(message)s",
            },
        )
        config["formatters"].setdefault("json", {"()": f"{__name__}.JsonFormatter2"})
        config.setdefault("handlers", {})
        config["handlers"].setdefault(
            "console",
            {
                "class": "logging.StreamHandler",
                "level": eff_level,
                "formatter": "json" if eff_fmt == "json" else "console",
                "filters": ["ctx", "mask"] + (["rate"] if rate_limit else []),
                "stream": "ext://sys.stdout",
            },
        )
    else:
        config = _default_dict_config(
            log_dir,
            fmt=eff_fmt,
            level=eff_level,
            use_queue=bool(multiproc),
            rate_limit=rate_limit,
        )

    # 4) 应用配置
    logging.config.dictConfig(config)

    # 5) 多进程处理（新实现：主进程单写，所有进程只投递）
    if multiproc:
        # 可选：允许通过环境变量明确指定 start method（默认按解释器策略）
        start_method = os.environ.get("KAN_LOG_MP_START", None)  # "spawn"|"fork"|"forkserver"|None

        if _is_main_process():
            # 主进程启动监听器并获取跨进程队列
            q = start_listener_main(
                log_dir=log_dir or Path("./logs"),
                level_name=(level or "INFO"),
                fmt=fmt,
                rate_limit=rate_limit,
                start_method=start_method,
            )
            # 主进程自身也统一通过队列投递（路径一致，便于切换/测试）
            install_queue_handler_worker(q, logger_name="kan")
            # 将队列暴露给调用方（若你已有全局 _Q，可同步一下）
            globals()["_Q"] = q  # 向后兼容：如项目内其它模块会读取 _Q
        else:
            # 子进程不应自启监听器。建议：由调用方把 q 显式传参进来，
            # 并在子进程入口处调用 install_queue_handler_worker(q)。
            # 这里仅保证当前进程不会直写：
            log = logging.getLogger("kan")
            for h in list(log.handlers):
                if not isinstance(h, QueueHandler):
                    log.removeHandler(h)
            log.propagate = False
            log.setLevel(logging.DEBUG)

        # 兜底：确保 "kan" 不再直写任何 handler（主/子一致走队列）
        _kan = logging.getLogger("kan")
        for h in list(_kan.handlers):
            if not isinstance(h, QueueHandler):
                _kan.removeHandler(h)
        _kan.propagate = False
        _kan.setLevel(logging.DEBUG)
    # 6) 降噪：第三方库提升到 WARNING（保持原行为）
    for noisy in (
        "urllib3",
        "asyncio",
        "numexpr",
        "aiohttp.access",
        "transformers.tokenization_utils_base",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 7) 可选：安装 excepthook（默认关闭）
    if os.environ.get(EXHOOK_ENV, "0") == "1":
        _install_exception_hooks()


def _install_exception_hooks() -> None:
    """@brief 安装同步/异步异常钩子（Chinese）/ Install sync/async exception hooks (English)."""
    import sys
    import asyncio

    def _excepthook(tp, val, tb):
        logging.getLogger("kan").exception("Uncaught exception", exc_info=(tp, val, tb))

    sys.excepthook = _excepthook

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def _async_handler(loop, ctx):
            # 尽量保留原上下文，但避免不可序列化对象
            msg = ctx.get("message") or "Async exception"
            exc = ctx.get("exception")
            if exc:
                logging.getLogger("kan").exception(
                    msg, exc_info=(type(exc), exc, exc.__traceback__)
                )
            else:
                logging.getLogger("kan").error(
                    msg,
                    extra={
                        "err": {k: str(v) for k, v in ctx.items() if k != "exception"}
                    },
                )

        loop.set_exception_handler(_async_handler)
    except Exception:
        # 无事件循环时静默跳过
        pass


@contextmanager
def log_context(**kwargs: Any):
    """@brief 临时日志上下文（Chinese）/ Temporary logging context (English).

    @zh
      在 with 作用域内注入附加字段（如 run_id, stage, step）。嵌套合并，退出后还原。
    @en
      Inject additional fields (e.g., run_id, stage, step) within a with-scope; nested merges;
      restored on exit.
    """
    token = _CTX.set({**_CTX.get({}), **kwargs})
    try:
        yield
    finally:
        _CTX.reset(token)
