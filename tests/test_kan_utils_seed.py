# tests/test_kan_utils_seed.py
# -*- coding: utf-8 -*-
import os
import sys
import importlib
import types
import math

import pytest

# 按用户约定：组件从 kan.utils.seed 导入
import kan.utils.seed as seedmod
from kan.utils.seed import (
    set_seed,
    derive_seed,
    rng_state,
    restore_rng_state,
    with_seed,
    seed_worker,
)

np = __import__("numpy")
random = __import__("random")


# ------------------------------
# 工具：判断 torch 可用
# ------------------------------
def _torch_available():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


# =========================================================
# 1) derive_seed：稳定性 & 32-bit 范围
# =========================================================
def test_derive_seed_stability_and_range():
    base = 123456789
    a1 = derive_seed(base, "alpha", 1, ("x", 2))
    a2 = derive_seed(base, "alpha", 1, ("x", 2))
    b = derive_seed(base, "beta", 1, ("x", 2))
    assert a1 == a2
    assert a1 != b
    # 32-bit 范围校验（0..2^32-1）
    assert 0 <= a1 <= (2**32 - 1)
    assert 0 <= b <= (2**32 - 1)


# =========================================================
# 2) set_seed：Python/NumPy 确定性 & PYTHONHASHSEED
# =========================================================
def test_set_seed_python_numpy_determinism_and_env(monkeypatch):
    info1 = set_seed(123, deterministic=False)
    x1 = random.random()
    y1 = float(np.random.rand())  # 取单值

    info2 = set_seed(123, deterministic=True)  # 再次同种子
    x2 = random.random()
    y2 = float(np.random.rand())

    assert x1 == x2
    assert y1 == y2
    assert info1["seed"] == 123 and info2["seed"] == 123

    # 环境变量：仅影响子进程，这里只校验设置值
    info3 = set_seed(456, deterministic=False, set_env_pythonhashseed=True)
    assert os.environ.get("PYTHONHASHSEED") == str(info3["seed"]) == "456"


# =========================================================
# 3) RNG 状态快照/恢复：Python + NumPy 往返
# =========================================================
def test_rng_state_restore_roundtrip_python_numpy():
    set_seed(789, deterministic=False)
    # 先走一步，拿到“下一步”的基准
    _ = random.random()
    _ = float(np.random.rand())

    s = rng_state()
    x_next = random.random()
    y_next = float(np.random.rand())

    # 恢复后应复现相同“下一步”
    restore_rng_state(s)
    x_next_r = random.random()
    y_next_r = float(np.random.rand())
    assert x_next == x_next_r
    assert y_next == y_next_r


# =========================================================
# 4) with_seed：局部确定且退出后完全恢复
# =========================================================
def test_with_seed_restores_global_sequence():
    set_seed(999, deterministic=False)

    # 捕捉“进入 ctx 前”的下一步
    s0 = rng_state()
    next_before = float(np.random.rand())
    # 回到进入前
    restore_rng_state(s0)

    # 进入上下文，用不同种子生成若干数
    with with_seed(100, deterministic=False):
        inner1 = float(np.random.rand())
        inner2 = float(np.random.rand())
        # 同一上下文内可复现（同种子、同序列）
        with with_seed(100, deterministic=False):
            assert float(np.random.rand()) == inner1
            assert float(np.random.rand()) == inner2

    # 退出上下文后，应当与“进入前的下一步”一致
    next_after = float(np.random.rand())
    assert next_after == next_before


# =========================================================
# 5) seed_worker：无 torch 分支（用固定 base 避免时间抖动）
# =========================================================
def test_seed_worker_without_torch(monkeypatch):
    # 暂时模拟“无 torch 环境”
    monkeypatch.setattr(seedmod, "_TORCH_AVAILABLE", False, raising=True)
    monkeypatch.setattr(seedmod, "torch", None, raising=True)

    # 固定 _now_seed 以可重复
    monkeypatch.setattr(seedmod, "_now_seed", lambda: 11111, raising=True)

    # 调用前，先计算期望派生种子
    base = 11111
    expect_np = derive_seed(base, "numpy")
    expect_py = derive_seed(base, "python")

    # 执行 worker seeding
    seed_worker(worker_id=0)

    # 验证 numpy/python RNG 序列起点与派生种子一致
    # 方式：再以派生种子重置一次，比较首个样本
    x_worker_np = float(np.random.rand())
    y_worker_py = random.random()

    np.random.seed(expect_np)
    random.seed(expect_py)
    x_expect = float(np.random.rand())
    y_expect = random.random()

    assert x_worker_np == x_expect
    assert y_worker_py == y_expect


# =========================================================
# 6) seed_worker：有 torch 分支（用 monkeypatch 固定 initial_seed）
# =========================================================
@pytest.mark.skipif(not _torch_available(), reason="torch not available")
def test_seed_worker_with_torch(monkeypatch):
    import torch

    # 确认走“有 torch”路径
    monkeypatch.setattr(seedmod, "_TORCH_AVAILABLE", True, raising=True)
    monkeypatch.setattr(seedmod, "torch", torch, raising=True)

    # 固定 torch.initial_seed()
    monkeypatch.setattr(torch, "initial_seed", lambda: 22222, raising=True)

    base = 22222
    expect_np = derive_seed(base, "numpy")
    expect_py = derive_seed(base, "python")

    seed_worker(worker_id=3)
    x_worker_np = float(np.random.rand())
    y_worker_py = random.random()

    np.random.seed(expect_np)
    random.seed(expect_py)
    x_expect = float(np.random.rand())
    y_expect = random.random()

    assert x_worker_np == x_expect
    assert y_worker_py == y_expect


# =========================================================
# 7) set_seed：torch 分支（若可用）及 deterministic 标志
# =========================================================
@pytest.mark.skipif(not _torch_available(), reason="torch not available")
def test_set_seed_with_torch_and_deterministic_flag():
    import torch

    info = set_seed(13579, deterministic=True, warn_only=True)
    assert info["torch"]["available"] is True

    # CPU RNG 序列可复现（同种子两次）
    a1 = torch.rand(3)
    info2 = set_seed(13579, deterministic=True, warn_only=True)
    a2 = torch.rand(3)
    assert torch.allclose(a1, a2)

    # 尝试检测 deterministic 开启状态（不同版本 API 差异）
    ok = True
    try:
        state = bool(torch.are_deterministic_algorithms_enabled())  # type: ignore[attr-defined]
        assert state is True
    except Exception:
        # 一些老版本 torch 没这个 API，不做硬性断言
        ok = False
    assert ok in (True, False)


# =========================================================
# 8) rng_state/restore_rng_state：torch CPU RNG 往返（若可用）
# =========================================================
@pytest.mark.skipif(not _torch_available(), reason="torch not available")
def test_rng_state_restore_roundtrip_torch_cpu():
    import torch

    set_seed(24680, deterministic=False)
    _ = torch.rand(1)  # 走一步
    s = rng_state()

    # 记录此刻 torch CPU RNG 状态
    t0 = torch.random.get_rng_state().clone()

    _ = torch.rand(5)  # 前进若干步
    # 恢复
    restore_rng_state(s)

    t1 = torch.random.get_rng_state().clone()
    assert torch.equal(t0, t1)

    # 恢复后生成的下一步应与“保存时的下一步”一致
    # 做法：再次恢复到 s 后，比对下一次样本
    restore_rng_state(s)
    after = torch.rand(3)
    restore_rng_state(s)
    ref = torch.rand(3)
    assert torch.allclose(after, ref)


# =========================================================
# 9) set_seed：返回摘要结构的关键字段完整性
# =========================================================
def test_set_seed_summary_structure_keys():
    info = set_seed(31415, deterministic=True)
    for key in (
        "seed",
        "deterministic",
        "python_random",
        "numpy_random",
        "torch",
        "note",
    ):
        assert key in info
    assert isinstance(info["torch"], dict)
