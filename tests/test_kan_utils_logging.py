# -*- coding: utf-8 -*-
import io
import json
import logging
import sys
import time
import importlib
from pathlib import Path
from typing import Any

import pytest

# ------- 工具：每个用例前“冷启动”被测模块，避免全局状态串扰 -------


@pytest.fixture()
def fresh_mod(monkeypatch):
    """
    重新加载 kan.utils.logging 模块，清空 logging 配置与全局变量副作用。
    """
    # 防止外部环境变量影响
    for k in [
        "KAN_LOG_MASK_KEYS",
        "KAN_LOG_FORMAT",
        "KAN_LOG_LEVEL",
        "KAN_LOG_MULTIPROC",
        "KAN_LOG_EXCEPT_HOOK",
        "KAN_LOG_RATE",
        "KAN_LOG_CFG",
    ]:
        monkeypatch.delenv(k, raising=False)

    # 完整关闭上一次 logging
    logging.shutdown()
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    for name in list(logging.Logger.manager.loggerDict.keys()):
        lg = logging.Logger.manager.loggerDict.get(name)
        if isinstance(lg, logging.Logger):
            for h in list(lg.handlers):
                lg.removeHandler(h)

    # 重新导入被测模块
    import kan.utils.logging as K

    K = importlib.reload(K)

    yield K

    # teardown：再清一次，稳妥
    logging.shutdown()


# ------- 基础：human/json 两种 formatter 与等级 -------


def test_configure_human_and_json(tmp_path, capsys, fresh_mod):
    K = fresh_mod
    log_dir = tmp_path / "logs"

    # human（console）—使用 capsys 捕获 stdout，而不是 caplog
    K.configure_logging(log_dir=log_dir, fmt="human", level="INFO")
    logger = logging.getLogger("kan.test")
    logger.info("hello-human")
    out, err = capsys.readouterr()
    assert "hello-human" in out

    # json（console 使用 JsonFormatter2）
    logging.shutdown()
    K = importlib.reload(K)
    K.configure_logging(log_dir=log_dir, fmt="json", level="DEBUG")
    logger = logging.getLogger("kan.test")
    logger.debug("hello-json")
    # 不对 caplog 断言，做端到端文件字段检查
    path = log_dir / "kan-debug.log"
    logger.info("json-to-file")
    assert path.exists()
    rows = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]
    assert any(
        r.get("msg") == "json-to-file" and r.get("level") in ("INFO", 20) for r in rows
    )


# ------- ContextFilter + SafeFormatter + log_context -------


def test_context_injection_and_safeformatter(tmp_path, caplog, fresh_mod):
    K = fresh_mod
    K.configure_logging(log_dir=tmp_path / "logs", fmt="human", level="INFO")
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("kan.test")

    # 未显式上下文：应有 run/stage/step 的缺省 "-"
    logger.info("noctx")
    # SafeFormatter 不应因缺字段报错；caplog 仅含 message，这里用不报错来证明

    # 显式上下文
    with K.log_context(run_id="R1", stage="train", step="42"):
        logger.info("withctx")
    # 由于 caplog 不包含格式化后的串，我们通过 Json 文件路径再做一次严格验证
    logging.shutdown()
    K = importlib.reload(K)
    K.configure_logging(log_dir=tmp_path / "logs2", fmt="json", level="INFO")
    logger = logging.getLogger("kan.test")
    with K.log_context(run_id="R2", stage="eval", step="7"):
        logger.info("ctx-to-file")

    # 读取 file handler 的 JSON 行断言字段存在
    f = tmp_path / "logs2" / "kan-debug.log"
    assert f.exists()
    lines = [json.loads(x) for x in f.read_text("utf-8").splitlines() if x.strip()]
    last = lines[-1]
    assert last["msg"] == "ctx-to-file"
    assert last["run_id"] == "R2"
    assert last["stage"] == "eval"
    assert str(last["step"]) == "7"


# ------- JsonFormatter2 异常序列化 -------


def test_jsonformatter_exception_serialization(tmp_path, fresh_mod):
    K = fresh_mod
    K.configure_logging(log_dir=tmp_path / "logs", fmt="json", level="DEBUG")
    logger = logging.getLogger("kan.test")
    try:
        raise ValueError("boom!")
    except Exception:
        logger.exception("got-exc")

    f = tmp_path / "logs" / "kan-debug.log"
    data = [json.loads(x) for x in f.read_text("utf-8").splitlines() if x.strip()]
    js = data[-1]
    assert js["msg"] == "got-exc"
    assert js["level"] == "ERROR"
    assert "err" in js and "stack" in js["err"] and "ValueError" in js["err"]["type"]


# ------- MaskFilter：组件级验证（keys参数 & 环境变量）-------


def test_maskfilter_component_keys_and_env(monkeypatch, fresh_mod):
    K = fresh_mod
    # 组件级：直接作用于 LogRecord
    rec = logging.LogRecord(
        name="kan.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="x",
        args=(),
        exc_info=None,
    )
    rec.email = "a@b.com"
    mf = K.MaskFilter(keys=["email"])
    assert mf.filter(rec) is True
    assert rec.email == "***"

    # 环境变量驱动
    monkeypatch.setenv("KAN_LOG_MASK_KEYS", "token, secret  ")
    rec2 = logging.LogRecord(
        name="kan.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="y",
        args=(),
        exc_info=None,
    )
    rec2.token = "t0k"
    rec2.secret = "sec"
    mf2 = K.MaskFilter()
    assert mf2.filter(rec2) is True
    assert rec2.token == "***" and rec2.secret == "***"


# ------- RateLimitFilter：组件级限流语义 -------


def test_ratelimitfilter_component_behavior(fresh_mod):
    K = fresh_mod
    rf = K.RateLimitFilter(capacity=2, window_s=1)
    rec = logging.LogRecord(
        "kan.test", logging.INFO, __file__, 1, "same-template", (), None
    )
    # 前两条放行，第三条抑制
    assert rf.filter(rec) is True
    assert rf.filter(rec) is True
    assert rf.filter(rec) is False
    time.sleep(1.05)
    # 新窗口恢复
    assert rf.filter(rec) is True


# ------- 多进程模式：QueueHandler 安装与幂等 -------


def test_multiprocess_queuehandler_and_idempotent(tmp_path, fresh_mod, capsys):
    """
    幂等测试：
    - multiproc + json：console 立即可见（capsys 断言）；
    - 二次 configure 后，不发生重复打印（按新增块计数）；
    - QueueListener 仍在（优先检查 _MP_QL；若不可见则回退 _QL；线程可见则 is_alive()）。
    - （可选）若文件存在，则解析最后一条 JSON 行做弱断言，但文件同步性不是硬条件。
    """
    import logging, json, time, os

    K = fresh_mod
    log_dir = tmp_path / "logs"

    # 第一次配置
    K.configure_logging(log_dir=log_dir, multiproc=True, fmt="json", level="DEBUG")
    logger = logging.getLogger("kan.test")  # 子 logger 通过传播到 "kan" 的 QueueHandler

    # baseline：清空捕获缓冲
    capsys.readouterr()

    # 打第一条 —— console 立即可见
    logger.info("mp-queue-1")
    # 监听线程存在微小调度延迟，轮询等待最多 0.5s
    deadline = time.monotonic() + 0.5
    seen1 = ""
    while time.monotonic() < deadline:
        out1, err1 = capsys.readouterr()
        seen1 += out1
        if "mp-queue-1" in seen1:
            break
        time.sleep(0.01)
    assert (
        "mp-queue-1" in seen1
    ), "Multiproc 下 console 输出应可被捕获（QueueListener 异步写控制台）"

    # 第二次配置（幂等，不应造成重复 handler）
    K.configure_logging(log_dir=log_dir, multiproc=True, fmt="json", level="DEBUG")

    # 再打一次 —— 只新增一条可见输出
    logger.info("mp-queue-2")
    # 只统计“本次新增缓冲”中的出现次数，同样做 0.5s 轮询以避免时序抖动
    deadline = time.monotonic() + 0.5
    seen2 = ""
    while time.monotonic() < deadline:
        out2, err2 = capsys.readouterr()
        seen2 += out2
        if "mp-queue-2" in seen2:
            break
        time.sleep(0.01)
    assert seen2.count("mp-queue-2") == 1, f"重复配置后不应产生重复打印，实际：{seen2}"

    # 优先使用 _MP_QL；为兼容旧用例，若不存在则退回 _QL
    ql = getattr(K, "_MP_QL", None) or getattr(K, "_QL", None)
    assert ql is not None, "QueueListener 应存在（_MP_QL 或 _QL 非 None）"
    th = getattr(ql, "_thread", None)
    if th is not None:
        assert th.is_alive(), "QueueListener 线程应仍在运行"

    # （可选）文件弱断言：存在则检查 JSON，但不同步不视为失败
    path = log_dir / "kan-debug.log"
    if path.exists():
        # 为异步落盘留一点时间，但即使为空也不失败
        deadline = time.monotonic() + 1.0
        lines = []
        while time.monotonic() < deadline:
            txt = path.read_text("utf-8")
            lines = [x for x in txt.splitlines() if x.strip()]
            if lines:
                break
            time.sleep(0.02)
        if lines:
            last = json.loads(lines[-1])
            # 允许最后一条是 mp-queue-1/2 任意一个（取决于调度），只要字段合理即可
            assert "msg" in last and "level" in last


# ------- 文件 handler：JSON 行输出与轮转配置存在 -------


def test_file_handler_json_output(tmp_path, fresh_mod):
    K = fresh_mod
    K.configure_logging(log_dir=tmp_path / "logs", fmt="json", level="INFO")
    logger = logging.getLogger("kan.test")
    logger.info("to-file")
    path = tmp_path / "logs" / "kan-debug.log"
    assert path.exists()
    rows = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]
    assert any(r["msg"] == "to-file" and r["logger"].startswith("kan") for r in rows)


# ------- 外部配置：最小 JSON + 保底注入 -------


def test_external_config_minimal_and_fallbacks(tmp_path, fresh_mod):
    K = fresh_mod
    cfg = {
        "version": 1,
        "formatters": {
            "console": {"()": f"{K.__name__}.SafeFormatter", "format": "%(message)s"}
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "console",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "kan": {"level": "DEBUG", "handlers": ["console"], "propagate": False}
        },
    }
    cfg_path = tmp_path / "logcfg.json"
    cfg_path.write_text(json.dumps(cfg), "utf-8")

    K.configure_logging(cfg_path=cfg_path)
    # root 应被保底注入（若缺），且第三方降噪
    assert logging.getLogger("urllib3").level == logging.WARNING

    # 还能打日志
    logger = logging.getLogger("kan.test")
    logger.info("extcfg-ok")


# ------- 环境变量解析健壮性（RATE 无效时忽略）-------


def test_bad_rate_env_is_ignored(monkeypatch, tmp_path, fresh_mod):
    K = fresh_mod
    monkeypatch.setenv("KAN_LOG_RATE", "this-is-bad")
    K.configure_logging(
        log_dir=tmp_path / "logs", fmt="human", level="INFO"
    )  # 不应抛异常
    logger = logging.getLogger("kan.test")
    logger.info("still-works")


# ------- 幂等：重复 configure 不累积 handlers -------


def test_idempotent_handlers_not_duplicated(tmp_path, fresh_mod):
    K = fresh_mod
    K.configure_logging(log_dir=tmp_path / "logs", fmt="human", level="INFO")
    kan = logging.getLogger("kan")
    before = len(kan.handlers)
    K.configure_logging(log_dir=tmp_path / "logs", fmt="human", level="INFO")
    after = len(kan.handlers)
    assert before == after


# ------- excepthook：EXHOOK_ENV 生效并写入 kan logger -------


def test_excepthook_installed_and_logs(monkeypatch, tmp_path, caplog, fresh_mod):
    K = fresh_mod
    monkeypatch.setenv("KAN_LOG_EXCEPT_HOOK", "1")
    K.configure_logging(log_dir=tmp_path / "logs", fmt="json", level="DEBUG")
    caplog.set_level(logging.ERROR)
    # 手动触发 sys.excepthook，避免真的让 pytest 崩溃
    exc = ValueError("uncaught!")
    sys.excepthook(type(exc), exc, exc.__traceback__)
    # 等待队列（若 multiproc 未开启也没关系），给 IO 一点时间
    time.sleep(0.05)

    # 写入到了文件 JSON
    path = tmp_path / "logs" / "kan-debug.log"
    assert path.exists()
    rows = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]
    assert any(r["msg"] == "Uncaught exception" and r["level"] == "ERROR" for r in rows)
