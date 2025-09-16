# -*- coding: utf-8 -*-
from __future__ import annotations
"""
@file   kan/utils/seed.py
@brief  Reproducibility utilities: set global seeds, deterministic guards, and worker seeding.
@date   2025-09-16

@zh
  - 统一设置 Python / NumPy / PyTorch 的随机种子；
  - 可选启用确定性（deterministic）算子并给出安全回退；
  - 提供 DataLoader worker 的种子函数与上下文管理器，便于局部复现；
  - Windows 友好，日志命名空间：`kan.utils.seed`。

@en
  Unified reproducibility helpers across Python/NumPy/PyTorch with deterministic toggles,
  worker seeding hooks, and context managers. Windows friendly. Logger namespace: `kan.utils.seed`.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os
import random
import logging
import time
import hashlib

import numpy as np

LOGGER = logging.getLogger("kan.utils.seed")

try:  # optional torch
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

__all__ = [
    "set_seed",
    "seed_worker",
    "with_seed",
    "derive_seed",
    "SeedState",
    "rng_state",
    "restore_rng_state",
]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _now_seed() -> int:
    # 32-bit range for broad compatibility with various APIs
    return int(time.time_ns() % (2**32 - 1))


def derive_seed(base_seed: int, *names: Any) -> int:
    """Derive a deterministic child seed from a base seed and arbitrary names.

    @zh 通过 SHA-256 派生，结果落入 32bit。
    """
    h = hashlib.sha256()
    h.update(str(int(base_seed)).encode("utf-8"))
    for n in names:
        h.update(str(n).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big")  # 32-bit


# -----------------------------------------------------------------------------
# RNG state capture/restore
# -----------------------------------------------------------------------------

@dataclass
class SeedState:
    """Snapshot of RNG states for Python/NumPy/Torch (CPU & CUDA)."""

    py_state: object
    np_state: tuple
    torch_cpu_state: Optional[bytes] = None
    torch_cuda_states: Optional[Tuple[bytes, ...]] = None


def rng_state() -> SeedState:
    py = random.getstate()
    npst = np.random.get_state()
    t_cpu = None
    t_cuda = None
    if _TORCH_AVAILABLE:
        try:
            t_cpu = torch.random.get_rng_state().clone().numpy().tobytes()  # bytes for portability
            if torch.cuda.is_available():
                t_cuda = tuple(st.clone().numpy().tobytes() for st in torch.cuda.get_rng_state_all())
        except Exception as e:  # pragma: no cover
            LOGGER.debug("torch rng_state capture failed: %s", e)
    return SeedState(py_state=py, np_state=npst, torch_cpu_state=t_cpu, torch_cuda_states=t_cuda)


def restore_rng_state(state: SeedState) -> None:
    random.setstate(state.py_state)
    np.random.set_state(state.np_state)
    if _TORCH_AVAILABLE and state.torch_cpu_state is not None:
        try:
            cpu = torch.frombuffer(bytearray(state.torch_cpu_state), dtype=torch.uint8)
            torch.random.set_rng_state(cpu)
            if state.torch_cuda_states is not None and torch.cuda.is_available():
                cuda_states = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in state.torch_cuda_states]
                torch.cuda.set_rng_state_all(cuda_states)
        except Exception as e:  # pragma: no cover
            LOGGER.debug("torch rng_state restore failed: %s", e)


# -----------------------------------------------------------------------------
# Core API
# -----------------------------------------------------------------------------

def set_seed(
    seed: Optional[int] = None,
    *,
    deterministic: bool = True,
    warn_only: bool = True,
    set_env_pythonhashseed: bool = True,
) -> Dict[str, Any]:
    """Set global seeds across Python / NumPy / (optionally) PyTorch.

    Parameters
    ----------
    seed : Optional[int]
        If None, a seed is derived from time_ns().
    deterministic : bool
        Try to enforce deterministic kernels (PyTorch); may reduce performance.
    warn_only : bool
        For `torch.use_deterministic_algorithms`, pass warn_only=True when possible.
    set_env_pythonhashseed : bool
        Set `PYTHONHASHSEED` env var for subprocesses. (Note: changing it at runtime does NOT
        affect current process's hash randomization.)

    Returns
    -------
    Dict[str, Any]
        A summary dict of applied settings.
    """
    s = int(seed) if seed is not None else _now_seed()

    # Python & NumPy
    random.seed(s)
    np.random.seed(s)

    # PyTorch (optional)
    torch_info: Dict[str, Any] = {"available": _TORCH_AVAILABLE}
    if _TORCH_AVAILABLE:
        try:
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

            # Determinism toggles
            if deterministic:
                try:
                    # torch >= 1.8
                    torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))  # type: ignore[arg-type]
                    torch_info["use_deterministic_algorithms"] = True
                except TypeError:
                    # older API without warn_only
                    torch.use_deterministic_algorithms(True)  # type: ignore
                    torch_info["use_deterministic_algorithms"] = True
                except Exception as e:  # pragma: no cover
                    LOGGER.warning("use_deterministic_algorithms failed: %s", e)

                # cuDNN flags
                try:
                    import torch.backends.cudnn as cudnn
                    cudnn.deterministic = True
                    cudnn.benchmark = False
                    torch_info["cudnn_deterministic"] = True
                    torch_info["cudnn_benchmark"] = False
                except Exception as e:  # pragma: no cover
                    LOGGER.debug("cuDNN flags set failed: %s", e)

                # cuBLAS workspace config for determinism on CUDA (if honored)
                try:
                    # Setting this env var makes some GEMM paths deterministic on CUDA 10.2+
                    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
                    torch_info["CUBLAS_WORKSPACE_CONFIG"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
                except Exception as e:  # pragma: no cover
                    LOGGER.debug("CUBLAS_WORKSPACE_CONFIG not set: %s", e)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("PyTorch seeding failed: %s", e)

    # pythonhashseed for subprocesses
    if set_env_pythonhashseed:
        try:
            os.environ["PYTHONHASHSEED"] = str(s)
        except Exception as e:  # pragma: no cover
            LOGGER.debug("setting PYTHONHASHSEED failed: %s", e)

    # Summary
    summary: Dict[str, Any] = {
        "seed": s,
        "deterministic": bool(deterministic),
        "python_random": True,
        "numpy_random": True,
        "torch": torch_info,
        "note": "PYTHONHASHSEED only affects new processes after this call.",
    }

    LOGGER.info("Global seed set to %d (deterministic=%s)", s, deterministic)
    return summary


def seed_worker(worker_id: int) -> None:
    """Torch DataLoader worker_init_fn compatible seeding.

    Usage
    -----
    ```python
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(12345)
    DataLoader(dataset, num_workers=4, worker_init_fn=seed_worker, generator=g)
    ```
    """
    # In a worker, torch.initial_seed() returns a distinct seed per worker
    base = torch.initial_seed() if _TORCH_AVAILABLE else _now_seed()
    s_np = derive_seed(base, "numpy")
    s_py = derive_seed(base, "python")
    np.random.seed(s_np)
    random.seed(s_py)
    # CUDA RNG per worker is handled by PyTorch via initial_seed()
    LOGGER.debug("worker %d seeded (py=%d, np=%d, base=%d)", worker_id, s_py, s_np, base)


# -----------------------------------------------------------------------------
# Context manager for local determinism
# -----------------------------------------------------------------------------

class with_seed:
    """Context manager that temporarily sets global RNG to a fixed seed and restores later.

    Example
    -------
    ```python
    with with_seed(7):
        # deterministic block
        ...
    ```
    """

    def __init__(self, seed: int, *, deterministic: bool = False) -> None:
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self._state: Optional[SeedState] = None
        self._prev_det: Optional[bool] = None

    def __enter__(self):
        self._state = rng_state()
        if _TORCH_AVAILABLE and self.deterministic:
            self._prev_det = torch.are_deterministic_algorithms_enabled()  # type: ignore[attr-defined]
        set_seed(self.seed, deterministic=self.deterministic)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._state is not None:
            restore_rng_state(self._state)
        if _TORCH_AVAILABLE and self._prev_det is not None:
            try:
                torch.use_deterministic_algorithms(self._prev_det)  # type: ignore
            except Exception:
                pass
        return False  # don't suppress exceptions


# -----------------------------------------------------------------------------
# Mini CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="KAN reproducibility helper")
    parser.add_argument("--seed", type=int, default=None, help="seed to set (default: time-based)")
    parser.add_argument("--no-det", action="store_true", help="disable deterministic toggles")
    args = parser.parse_args()

    info = set_seed(args.seed, deterministic=(not args.no_det))
    print("Applied:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # quick demo of with_seed
    a = np.random.rand(3)
    with with_seed(info["seed"]):
        b = np.random.rand(3)
    c = np.random.rand(3)
    print("np demo:")
    print(" a:", a)
    print(" b (inside ctx):", b)
    print(" c:", c)
