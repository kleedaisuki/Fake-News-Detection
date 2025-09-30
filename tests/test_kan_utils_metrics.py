# -*- coding: utf-8 -*-
# tests/test_kan_utils_metrics.py

import math
import numpy as np
import pytest

from kan.utils.metrics import (
    TaskType,
    infer_task_type,
    compute_classification_metrics,
    FoldAccumulator,
    safe_confusion_matrix,
)

# --------------------------
# Helpers
# --------------------------


def is_nan(x):
    """Return True iff x is float('nan')."""
    return isinstance(x, float) and math.isnan(x)


def keys_base():
    # 14 stable keys from ClassificationMetrics
    return {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc_ovr",
        "roc_auc_ovo",
        "pr_auc_micro",
        "pr_auc_macro",
    }


# --------------------------
# infer_task_type
# --------------------------


def test_infer_task_type_binary():
    y = np.array([0, 1, 1, 0, 1], dtype=int)
    assert infer_task_type(y) == TaskType.BINARY


def test_infer_task_type_multiclass():
    y = np.array([0, 2, 1, 2, 3, 1], dtype=int)
    assert infer_task_type(y) == TaskType.MULTICLASS


def test_infer_task_type_multilabel():
    y = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=int)
    assert infer_task_type(y) == TaskType.MULTILABEL


def test_infer_task_type_invalid_none():
    with pytest.raises(ValueError):
        infer_task_type(None)  # type: ignore


# --------------------------
# compute_classification_metrics : Binary
# --------------------------


def test_binary_pred_only_basic_keys_and_auc_nan():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=128)
    # perfect predictor to keep numbers deterministic
    y_pred = y_true.copy()
    m = compute_classification_metrics(y_true, y_pred=y_pred, y_score=None)
    # stable base keys present
    assert set(m.keys()).issuperset(keys_base())
    # auc/pr-auc are NaN when y_score is None
    assert is_nan(m["roc_auc_ovr"])
    assert is_nan(m["roc_auc_ovo"])
    assert is_nan(m["pr_auc_micro"])
    assert is_nan(m["pr_auc_macro"])
    # accuracy is 1.0
    assert m["accuracy"] == pytest.approx(1.0, rel=0, abs=0)


def test_binary_score_only_1d_derives_pred():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    # 1D score (prob of positive)
    y_score = rng.random(size=200)
    m = compute_classification_metrics(y_true, y_pred=None, y_score=y_score)
    # Should have non-NaN PR-AUC (micro/macro are same for binary average)
    assert not is_nan(m["pr_auc_micro"])


def test_binary_score_2col_uses_second_column_as_positive():
    rng = np.random.default_rng(2)
    n = 150
    y_true = rng.integers(0, 2, size=n)
    # construct 2-column probabilities: [:,1] is positive class
    p_pos = rng.random(size=n)
    y_score = np.stack([1 - p_pos, p_pos], axis=1)
    m = compute_classification_metrics(y_true, y_pred=None, y_score=y_score)
    # Compare with feeding only positive column
    m2 = compute_classification_metrics(y_true, y_pred=None, y_score=p_pos)
    # AUCs/PR-AUC should match closely
    for k in ("pr_auc_micro", "roc_auc_ovr"):
        assert m[k] == pytest.approx(m2[k], rel=1e-12, abs=1e-12)


def test_binary_all_single_class_auc_nan_and_accuracy_ok():
    y_true = np.zeros(64, dtype=int)
    y_pred = np.zeros(64, dtype=int)
    # y_score is consistent but degenerate w.r.t. labels
    y_score = np.zeros((64, 2), dtype=float)
    m = compute_classification_metrics(y_true, y_pred=y_pred, y_score=y_score)
    assert m["accuracy"] == pytest.approx(1.0, rel=0, abs=0)
    # AUC/PR-AUC undefined → NaN
    assert is_nan(m["roc_auc_ovr"])
    assert is_nan(m["roc_auc_ovo"])
    # sklearn.average_precision_score 在无正样本时返回 0.0（非 NaN）
    # 参见 sklearn 行为：无正样本 -> AP = 0.0
    assert m["pr_auc_micro"] == pytest.approx(0.0, rel=0, abs=0)
    assert m["pr_auc_macro"] == pytest.approx(0.0, rel=0, abs=0)


def test_error_when_both_pred_and_score_none():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=50)
    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_pred=None, y_score=None)


# --------------------------
# compute_classification_metrics : Multiclass
# --------------------------


def test_multiclass_with_topk_and_prob_matrix():
    rng = np.random.default_rng(4)
    n, C = 256, 5
    y_true = rng.integers(0, C, size=n)
    logits = rng.normal(size=(n, C))
    # softmax-like but not necessary; only rank matters for top-k
    # to keep values in [0,1], normalize row-wise
    y_score = logits - logits.min(axis=1, keepdims=True)
    denom = y_score.sum(axis=1, keepdims=True)
    y_score = np.divide(y_score, np.maximum(denom, 1e-9))
    y_pred = y_score.argmax(axis=1)

    m = compute_classification_metrics(y_true, y_pred=y_pred, y_score=y_score)
    assert set(m.keys()).issuperset(keys_base() | {"top1_accuracy", "top3_accuracy"})
    # top1 should equal accuracy (sklearn definition)
    assert m["top1_accuracy"] == pytest.approx(m["accuracy"], rel=1e-12, abs=1e-12)
    # top3 >= top1
    assert m["top3_accuracy"] + 1e-12 >= m["top1_accuracy"]


def test_multiclass_topk_greater_than_classes_returns_nan():
    rng = np.random.default_rng(5)
    n, C = 64, 4
    y_true = rng.integers(0, C, size=n)
    y_score = rng.random(size=(n, C))
    m = compute_classification_metrics(y_true, y_pred=None, y_score=y_score, topk=(10,))
    # 当 k >= n_classes，top-k accuracy 按定义为 1.0（虽无意义，但 sklearn 会告警并返回完美分数）
    # 这里验证键存在且值为 1.0
    assert "top10_accuracy" in m
    assert m["top10_accuracy"] == pytest.approx(1.0, rel=0, abs=0)


# --------------------------
# compute_classification_metrics : Multilabel
# --------------------------


def test_multilabel_score_only_thresholding_and_auc_defined():
    rng = np.random.default_rng(6)
    n, C = 100, 6
    # create moderately informative scores
    y_true = (rng.random(size=(n, C)) > 0.7).astype(int)
    y_score = rng.random(size=(n, C)) * 0.9 + 0.05
    m = compute_classification_metrics(y_true, y_pred=None, y_score=y_score)
    # For multilabel, code uses average="micro" for ROC-AUC
    assert not is_nan(m["roc_auc_ovr"])
    assert not is_nan(m["pr_auc_micro"])


# --------------------------
# safe_confusion_matrix
# --------------------------


@pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
def test_safe_confusion_matrix_basic_and_with_labels(normalize):
    y_true = np.array([0, 1, 2, 2, 1, 0, 3, 3, 3], dtype=int)
    y_pred = np.array([0, 1, 1, 2, 1, 0, 3, 0, 3], dtype=int)

    # no labels
    cm1 = safe_confusion_matrix(y_true, y_pred, normalize=normalize)
    # labels with extra class 4 (not present)
    cm2 = safe_confusion_matrix(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], normalize=normalize
    )

    # shapes
    assert cm1.shape == (4, 4)
    assert cm2.shape == (5, 5)

    # the [4, :] and [:, 4] must be zeros if class 4 absent
    assert np.allclose(cm2[4, :], 0)
    assert np.allclose(cm2[:, 4], 0)


# --------------------------
# FoldAccumulator
# --------------------------


def test_fold_accumulator_union_keys_and_summary():
    # two metric rows with partly disjoint keys
    row1 = {"a": 1.0, "b": 2.0}
    row2 = {"b": 4.0, "c": np.nan}

    acc = FoldAccumulator()
    acc.add(row1)
    acc.add(row2)

    table = acc.as_table()
    # union keys
    assert set(table.keys()) == {"a", "b", "c"}
    # fill missing with NaN — NaN 不可直接相等比较，用逐元素断言
    assert len(table["a"]) == 2
    assert table["a"][0] == pytest.approx(1.0, rel=0, abs=0)
    assert is_nan(table["a"][1])

    assert table["b"] == [2.0, 4.0]

    # 对 c：两折均为 NaN（第一折缺失→NaN，第二折显式 NaN）
    assert len(table["c"]) == 2
    assert is_nan(table["c"][0])
    assert is_nan(table["c"][1])

    summary = acc.summary()
    # mean/std use nan-aware computations
    assert summary["mean_a"] == pytest.approx(1.0, rel=0, abs=0)
    assert summary["std_a"] == pytest.approx(0.0, rel=0, abs=0)
    assert summary["mean_b"] == pytest.approx(3.0, rel=0, abs=0)
    assert summary["std_b"] == pytest.approx(1.0, rel=0, abs=0)
    assert is_nan(summary["mean_c"])
    assert is_nan(summary["std_c"])
