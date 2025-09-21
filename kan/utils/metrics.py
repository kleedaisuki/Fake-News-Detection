# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kan/utils/metrics.py
@brief  Classification metrics for KAN (Precision/Recall/F1/Accuracy/AUC, PR-AUC),
        with fold-wise aggregation and safe fallbacks.
@date   2025-09-16

@zh
  为二分类/多分类/多标签任务提供统一的指标计算：Precision/Recall/F1/Accuracy/ROC-AUC/PR-AUC，
  支持 micro/macro/weighted，支持按 fold 汇总均值/标准差；在只有单一类别的极端情形下提供安全回退（返回 NaN 并日志告警）。

@en
  Unified metrics for binary / multiclass / multilabel tasks: Precision/Recall/F1/Accuracy/ROC-AUC/PR-AUC,
  with micro/macro/weighted averaging and fold-wise mean/std aggregation. Handles degenerate cases (single-class) safely.

@notes
  - Windows friendly. No fork dependency. Only relies on numpy & scikit-learn; PyTorch is optional.
  - Logger namespace: "kan.utils.metrics" (see kan/utils/logging.py for centralized config).
"""

from dataclasses import dataclass, asdict
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import logging

import numpy as np
from numpy.typing import ArrayLike

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False

# scikit-learn is pinned in requirements.txt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)


LOGGER = logging.getLogger("kan.utils.metrics")

__all__ = [
    "TaskType",
    "infer_task_type",
    "compute_classification_metrics",
    "FoldAccumulator",
    "safe_confusion_matrix",
]


# -----------------------------------------------------------------------------
# Task type inference
# -----------------------------------------------------------------------------


class TaskType:
    """任务类型（Task Type）。"""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


def _to_numpy(x: ArrayLike | None) -> Optional[np.ndarray]:
    if x is None:
        return None
    if _TORCH_AVAILABLE and isinstance(x, getattr(__import__("torch"), "Tensor")):  # type: ignore
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x


def infer_task_type(y_true: ArrayLike) -> str:
    """Infer task type by inspecting ``y_true``.

    @zh
      规则：
        * 1D 且取值仅 {0,1} → binary；
        * 1D 且类别数 > 2 → multiclass；
        * 2D（shape=[N,C] 且为 0/1）→ multilabel。

    @en
      Rules:
        * 1D with values in {0,1} → binary;
        * 1D with n_classes > 2 → multiclass;
        * 2D (N×C with 0/1) → multilabel.
    """
    y = _to_numpy(y_true)
    if y is None:
        raise ValueError("y_true is None")
    y = np.asarray(y)
    if y.ndim == 1:
        uniq = np.unique(y)
        if set(uniq.tolist()) <= {0, 1}:
            return TaskType.BINARY
        return TaskType.MULTICLASS
    elif y.ndim == 2:
        return TaskType.MULTILABEL
    raise ValueError(f"Unsupported y_true.ndim={y.ndim}")


# -----------------------------------------------------------------------------
# Core metrics computation
# -----------------------------------------------------------------------------


@dataclass
class ClassificationMetrics:
    """单次评估的指标（per-run metrics）。

    All fields are floats; may be ``np.nan`` if undefined.
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    roc_auc_ovr: float
    roc_auc_ovo: float
    pr_auc_micro: float
    pr_auc_macro: float

    def to_dict(self) -> Dict[str, float]:
        return {
            k: float(v) if v == v else float("nan") for k, v in asdict(self).items()
        }  # NaN-safe


def _safe_roc_auc(
    y_true: np.ndarray,
    y_score: Optional[np.ndarray],
    *,
    multi_class: str = "ovr",
    average: Optional[str] = None,
) -> float:
    """Safe ROC-AUC wrapper.

    Returns ``np.nan`` when AUC is undefined (e.g., y_true has single class or y_score is None).
    """
    if y_score is None:
        return float("nan")
    y_true = np.asarray(y_true)
    try:
        return float(
            roc_auc_score(y_true, y_score, multi_class=multi_class, average=average)
        )
    except Exception as e:  # pragma: no cover
        LOGGER.warning("roc_auc_score failed: %s", e)
        return float("nan")


def _safe_pr_auc(
    y_true: np.ndarray,
    y_score: Optional[np.ndarray],
    *,
    average: str = "micro",
) -> float:
    if y_score is None:
        return float("nan")
    y_true = np.asarray(y_true)
    try:
        return float(average_precision_score(y_true, y_score, average=average))
    except Exception as e:  # pragma: no cover
        LOGGER.warning("average_precision_score failed: %s", e)
        return float("nan")


def _ensure_probabilities(
    task: str, y_pred: Optional[np.ndarray], y_score: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Normalize scores to probabilities when possible.

    - binary: allow shape (N,) or (N,2) or (N,1) → convert to (N,) positive-class probs
    - multiclass/multilabel: expect (N,C)
    """
    if y_score is None:
        return None
    y_score = np.asarray(y_score)
    if task == TaskType.BINARY:
        # If y_score is logits/probas of positive class shape (N,) or (N,1) or (N,2)
        if y_score.ndim == 1:
            return y_score
        if y_score.ndim == 2:
            if y_score.shape[1] == 1:
                return y_score[:, 0]
            if y_score.shape[1] == 2:
                # assume [:, 1] is positive class
                return y_score[:, 1]
    return y_score


def safe_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Optional[Sequence[int]] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Confusion matrix with safe fallback when classes missing in a fold.

    @param normalize: {None, 'true', 'pred', 'all'}
    """
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred)
    if y_t is None or y_p is None:
        raise ValueError("y_true/y_pred is None")
    try:
        return confusion_matrix(y_t, y_p, labels=labels, normalize=normalize)
    except Exception as e:  # pragma: no cover
        LOGGER.warning("confusion_matrix failed: %s", e)
        # Fallback: compute on the union of observed labels
        return confusion_matrix(
            y_t,
            y_p,
            labels=np.unique(np.concatenate([np.unique(y_t), np.unique(y_p)])),
            normalize=normalize,
        )


def compute_classification_metrics(
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    y_score: Optional[ArrayLike] = None,
    *,
    labels: Optional[Sequence[int]] = None,
    average_for_pr: Tuple[str, str] = ("micro", "macro"),
    average_for_roc: Tuple[str, str] = ("ovr", "ovo"),
    topk: Sequence[int] = (1, 3),
    zero_division: Union[str, int] = 0,
) -> Dict[str, float]:
    """Compute a **comprehensive** set of classification metrics.

    @zh
      - 自动判断任务类型（binary/multiclass/multilabel）；
      - 若提供 ``y_score``（概率或 logits），会额外计算 ROC-AUC（OvR/OvO）与 PR-AUC（micro/macro）；
      - 返回字典键稳定，缺失场景返回 ``NaN``。

    @en
      - Auto-detect task type;
      - If ``y_score`` is provided (probas or logits), computes ROC-AUC (OvR/OvO) and PR-AUC (micro/macro);
      - Returns a flat dict with stable keys; missing metrics use ``NaN``.
    """
    y_t = _to_numpy(y_true)
    y_p = _to_numpy(y_pred) if y_pred is not None else None
    y_s = _to_numpy(y_score)

    if y_t is None:
        raise ValueError("y_true is None")
    task = infer_task_type(y_t)
    y_s = _ensure_probabilities(task, y_p, y_s)

    # Derive y_pred from y_score if absent
    if y_p is None and y_s is not None:
        if task in (TaskType.BINARY, TaskType.MULTICLASS):
            if y_s.ndim == 1:
                y_p = (y_s >= 0.5).astype(int)
            else:
                y_p = np.argmax(y_s, axis=1)
        elif task == TaskType.MULTILABEL:
            y_p = (y_s >= 0.5).astype(int)

    if y_p is None:
        raise ValueError("y_pred is None and y_score is None; cannot compute metrics.")

    # Accuracy
    acc = float(accuracy_score(y_t, y_p))

    # Precision/Recall/F1
    def _prf(avg: str) -> Tuple[float, float, float]:
        p, r, f1, _ = precision_recall_fscore_support(
            y_t, y_p, average=avg, zero_division=zero_division
        )
        return float(p), float(r), float(f1)

    p_macro, r_macro, f1_macro = _prf("macro")
    p_micro, r_micro, f1_micro = _prf("micro")
    p_weighted, r_weighted, f1_weighted = _prf("weighted")

    # AUCs
    roc_auc_ovr = _safe_roc_auc(
        y_t,
        y_s,
        multi_class=average_for_roc[0],
        average=None if task != TaskType.MULTILABEL else "micro",
    )
    roc_auc_ovo = _safe_roc_auc(
        y_t,
        y_s,
        multi_class=average_for_roc[1],
        average=None if task != TaskType.MULTILABEL else "micro",
    )

    # PR-AUC
    pr_auc_micro = _safe_pr_auc(y_t, y_s, average=average_for_pr[0])
    pr_auc_macro = _safe_pr_auc(y_t, y_s, average=average_for_pr[1])

    # Top-k (only meaningful for multiclass with probabilities)
    out: Dict[str, float] = {}
    if y_s is not None and task == TaskType.MULTICLASS and y_s.ndim == 2:
        for k in topk:
            try:
                out[f"top{k}_accuracy"] = float(
                    top_k_accuracy_score(y_t, y_s, k=k, labels=labels)
                )
            except Exception as e:  # pragma: no cover
                LOGGER.debug("top-%d accuracy failed: %s", k, e)
                out[f"top{k}_accuracy"] = float("nan")

    metrics = ClassificationMetrics(
        accuracy=acc,
        precision_macro=p_macro,
        recall_macro=r_macro,
        f1_macro=f1_macro,
        precision_micro=p_micro,
        recall_micro=r_micro,
        f1_micro=f1_micro,
        precision_weighted=p_weighted,
        recall_weighted=r_weighted,
        f1_weighted=f1_weighted,
        roc_auc_ovr=roc_auc_ovr,
        roc_auc_ovo=roc_auc_ovo,
        pr_auc_micro=pr_auc_micro,
        pr_auc_macro=pr_auc_macro,
    ).to_dict()

    metrics.update(out)
    return metrics


# -----------------------------------------------------------------------------
# Fold-wise accumulator
# -----------------------------------------------------------------------------


class FoldAccumulator:
    """@zh 折次（fold）指标累加与统计；@en Fold-wise metrics aggregator.

    用法：
    ```python
    acc = FoldAccumulator()
    for fold in range(5):
        m = compute_classification_metrics(y_true, y_pred, y_score)
        acc.add(m)
    print(acc.summary())  # → {metric_mean, metric_std}
    ```
    """

    def __init__(self) -> None:
        self._rows: List[Dict[str, float]] = []

    def add(self, metrics: Mapping[str, float]) -> None:
        self._rows.append(dict(metrics))

    def as_table(self) -> Dict[str, List[float]]:
        if not self._rows:
            return {}
        keys = sorted({k for row in self._rows for k in row.keys()})
        table: Dict[str, List[float]] = {k: [] for k in keys}
        for row in self._rows:
            for k in keys:
                v = row.get(k, float("nan"))
                table[k].append(v)
        return table

    def summary(
        self, *, prefix_mean: str = "mean_", prefix_std: str = "std_"
    ) -> Dict[str, float]:
        table = self.as_table()
        out: Dict[str, float] = {}
        for k, col in table.items():
            arr = np.asarray(col, dtype=float)
            out[prefix_mean + k] = float(np.nanmean(arr))
            out[prefix_std + k] = float(np.nanstd(arr, ddof=0))
        return out


# -----------------------------------------------------------------------------
# Mini CLI (optional)
# -----------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute classification metrics from CSV files or arrays."
    )
    parser.add_argument("--demo", action="store_true", help="Run a tiny demo.")
    args = parser.parse_args()

    if args.demo:
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=64)
        y_score = rng.random(size=(64, 2))
        y_pred = y_score.argmax(axis=1)
        m = compute_classification_metrics(y_true, y_pred=y_pred, y_score=y_score)
        for k in sorted(m.keys()):
            print(f"{k:>20s}: {m[k]:.4f}")
        acc = FoldAccumulator()
        for _ in range(5):
            acc.add(m)
        print("\nSummary:")
        s = acc.summary()
        for k in sorted(s.keys()):
            print(f"{k:>20s}: {s[k]:.4f}")
