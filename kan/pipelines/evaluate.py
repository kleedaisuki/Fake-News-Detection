# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kan/pipelines/evaluate.py
@brief  Pipeline: 聚合/评估已训练 run 的预测，生成指标、混淆矩阵与曲线数据；支持 k-fold 汇总。
@date   2025-09-16

# 设计哲学（与其它 pipelines 一致）
- **流程编排**：不实现分类算法本身，仅负责读入 `runs/<run_id>/pred_*.jsonl` 等产物，聚合、计算指标、落盘。
- **超参来源**：从 `configs/` 合并（多 YAML 覆盖 + `--override key=value`），最终快照写入 `runs/<eval_run_id>/configs_merged.yaml`。
- **scripts 驱动**：由 `scripts/` 下脚本调用。不要在此写死业务路径。
- **过程 vs 结果**：日志与中间表在 `runs/<eval_run_id>/logs/`，评估结果写入 `runs/<eval_run_id>/reports/`。
- **可复用/可重入**：允许一次评估多个训练 run（如 k-fold），同时输出 micro 聚合与 macro 平均+方差。
- **Windows 友好**：路径 `pathlib`，不强制多进程。

# 输入契约
- 每个被评估的训练目录（`run_dir`）应包含：`pred_<split>.jsonl`，行格式：
  {"y_true": <int 或 List[int]>, "y_pred": <int 或 List[int]>, "y_score": <List[float] 或 List[List[float]] 或 float>}
  注意：
    * 单标签分类：`y_true:int`，`y_score: List[float]=softmax 概率`。
    * 多标签分类：`y_true: List[int]`（0/1），`y_score: List[float]`（每标签 sigmoid 后概率）。
    * 回归：`y_true: float`，`y_score: float` 或 `y_pred: float`。

# 输出
- reports/
  - metrics_<split>.json                 # 聚合后的指标（micro 全量、macro 均值/标准差、各 run 明细）
  - confusion_matrix_<split>.csv         # 单/多类分类的混淆矩阵（micro 全量）
  - classification_report_<split>.json   # （若可用）sklearn 的 per-class 报告（micro 全量）
  - curves/<split>/roc_class<i>.csv      # （单标签多类）一对多 ROC 曲线点
  - curves/<split>/pr_class<i>.csv       # （单标签多类）一对多 PR 曲线点
"""

import argparse
import copy
import csv
import gzip
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
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
)

import numpy as np

# ------------------------------
# 日志（集中式，若不可用则回退）
# ------------------------------
try:
    from kan.utils.logging import configure_logging, log_context
    import logging

    LOGGER = logging.getLogger("kan.pipelines.evaluate")
except Exception:
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s | %(message)s"
    )
    LOGGER = logging.getLogger("kan.pipelines.evaluate")

    def configure_logging(*args, **kwargs):  # type: ignore
        pass

    from contextlib import contextmanager

    @contextmanager
    def log_context(**kwargs):  # type: ignore
        yield


# ------------------------------
# YAML 合并与覆盖
# ------------------------------
try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("缺少 PyYAML，请 `pip install pyyaml`。") from e


def _deep_update(
    base: MutableMapping[str, Any], other: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for k, v in other.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            _deep_update(base[k], v)  # type: ignore
        else:
            base[k] = copy.deepcopy(v)
    return base


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_overrides(cfg: MutableMapping[str, Any], overrides: Sequence[str]) -> None:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"覆盖项格式错误：{ov}，应为 key.subkey=VALUE")
        key, val = ov.split("=", 1)
        try:
            val_conv = json.loads(val)
        except Exception:
            val_conv = val
        cur: MutableMapping[str, Any] = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], MutableMapping):
                cur[p] = {}
            cur = cur[p]  # type: ignore
        cur[parts[-1]] = val_conv


# ------------------------------
# 工具函数
# ------------------------------


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_jsonl(path: Path) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _discover_pred(run_dir: Path, split: str) -> Optional[Path]:
    # 优先非步数版本
    p = run_dir / f"pred_{split}.jsonl"
    if p.exists():
        return p
    # 兼容 gz
    p = run_dir / f"pred_{split}.jsonl.gz"
    if p.exists():
        return p
    # 回落：扫描 step/epoch 标注文件，取最近修改
    cands = list(run_dir.glob(f"pred_{split}@*.json")) + list(
        run_dir.glob(f"pred_{split}@*.jsonl")
    )
    if cands:
        return sorted(cands, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return None


def _as_numpy_labels(row_labels: Any) -> np.ndarray:
    if isinstance(row_labels, (list, tuple)):
        return np.asarray(row_labels, dtype=np.int64)
    if row_labels is None:
        return np.asarray([], dtype=np.int64)
    return np.asarray([row_labels], dtype=np.int64)


def _stack_list_of_arrays(xs: List[np.ndarray]) -> np.ndarray:
    if not xs:
        return np.empty((0,), dtype=np.float32)
    return np.stack(xs, axis=0)


def _to_2d_scores(y_score: Any, num_labels: Optional[int]) -> np.ndarray:
    """把 score 转成二维 [N, C]（单标签）或 [N, L]（多标签）。回归返回 [N, 1]。"""
    arr = np.asarray(y_score)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr  # [C] or [L]


# ------------------------------
# 指标计算（桥接 kan.utils.metrics，必要时回退 sklearn）
# ------------------------------


def _compute_metrics_bridge(
    problem_type: str,
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    y_score: Optional[np.ndarray],
) -> Dict[str, float]:
    # 首选仓库自带的 metrics
    try:
        from kan.utils.metrics import compute_classification_metrics

        if problem_type == "single_label_classification":
            assert y_score is not None and y_score.ndim == 2
            return compute_classification_metrics(
                y_true, y_pred=y_score.argmax(axis=1), y_score=y_score
            )
        elif problem_type == "multilabel_classification":
            assert y_score is not None and y_score.ndim == 2
            return compute_classification_metrics(
                y_true, y_pred=(y_score >= 0.5).astype("i4"), y_score=y_score
            )
        else:
            from sklearn.metrics import mean_squared_error

            assert y_score is not None and y_score.ndim >= 1
            return {
                "mse": float(
                    mean_squared_error(
                        y_true.astype(float), y_score.squeeze(-1).astype(float)
                    )
                )
            }
    except Exception:
        # 轻量回退
        try:
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

            if problem_type == "single_label_classification":
                yp = y_pred if y_pred is not None else y_score.argmax(axis=1)
                return {
                    "acc": float(accuracy_score(y_true, yp)),
                    "f1_macro": float(f1_score(y_true, yp, average="macro")),
                    "precision_macro": float(
                        precision_score(y_true, yp, average="macro")
                    ),
                    "recall_macro": float(recall_score(y_true, yp, average="macro")),
                }
            elif problem_type == "multilabel_classification":
                yp = y_pred if y_pred is not None else (y_score >= 0.5).astype("i4")
                return {
                    "f1_micro": float(f1_score(y_true, yp, average="micro")),
                    "f1_macro": float(f1_score(y_true, yp, average="macro")),
                }
            else:
                from sklearn.metrics import mean_squared_error

                return {
                    "mse": float(
                        mean_squared_error(
                            y_true.astype(float), y_score.squeeze(-1).astype(float)
                        )
                    )
                }
        except Exception:
            return {}


# ------------------------------
# 聚合评估
# ------------------------------


def _load_split_from_run(
    run_dir: Path, split: str, problem_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_path = _discover_pred(run_dir, split)
    if pred_path is None:
        raise FileNotFoundError(f"{run_dir} 缺少 {split} 预测文件。")
    rows = _read_jsonl(pred_path)

    y_trues: List[np.ndarray] = []
    y_preds: List[np.ndarray] = []
    y_scores: List[np.ndarray] = []
    for r in rows:
        y_trues.append(_as_numpy_labels(r.get("y_true")))
        if problem_type == "single_label_classification":
            y_scores.append(_to_2d_scores(r.get("y_score"), None))
            if r.get("y_pred") is not None:
                y_preds.append(_as_numpy_labels(r.get("y_pred")))
        elif problem_type == "multilabel_classification":
            # y_true: [L]；score: [L]
            yt = np.asarray(r.get("y_true"), dtype=np.int64).reshape(1, -1)
            y_trues[-1] = yt  # 覆盖为 1xL
            y_scores.append(_to_2d_scores(r.get("y_score"), None))
            if r.get("y_pred") is not None:
                y_preds.append(
                    np.asarray(r.get("y_pred"), dtype=np.int64).reshape(1, -1)
                )
        else:  # regression
            ys = np.asarray(
                [r.get("y_score") if r.get("y_score") is not None else r.get("y_pred")],
                dtype=float,
            ).reshape(1, 1)
            y_scores.append(ys)
            # y_pred 对回归没必要强求
    y_true = (
        np.concatenate(y_trues, axis=0) if y_trues else np.empty((0,), dtype=np.int64)
    )
    y_pred = np.concatenate(y_preds, axis=0) if y_preds else None
    y_score = np.concatenate(y_scores, axis=0) if y_scores else None
    return (
        y_true,
        (y_pred if y_pred is not None else np.empty((0,), dtype=np.int64)),
        y_score,
    )


def _confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_labels: Optional[int] = None
) -> np.ndarray:
    if y_true.ndim != 1:
        return np.zeros((0, 0), dtype=np.int64)
    if num_labels is None:
        K = (
            int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
            if y_true.size and y_pred.size
            else 0
        )
    else:
        K = int(num_labels)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < K and 0 <= p < K:
            cm[t, p] += 1
    return cm


def _write_csv(path: Path, rows: Iterable[Iterable[Any]]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


# ------------------------------
# 主流程：run_from_configs
# ------------------------------


def run_from_configs(
    config_paths: Sequence[str], overrides: Sequence[str] = ()
) -> Path:
    # 1) 合并配置
    cfg: MutableMapping[str, Any] = {}
    for p in config_paths:
        cp = Path(p)
        assert cp.exists(), f"配置文件不存在：{cp}"
        _deep_update(cfg, _read_yaml(cp))
    _apply_overrides(cfg, list(overrides))

    eval_cfg = cfg.get("eval") or cfg
    inputs: List[str] = list(eval_cfg.get("inputs", []))
    if not inputs and eval_cfg.get("inputs_glob"):
        inputs = [
            str(p)
            for p in Path(eval_cfg.get("inputs_glob")).parent.glob(
                Path(eval_cfg.get("inputs_glob")).name
            )
        ]
    assert inputs, "请在 eval.inputs 或 eval.inputs_glob 指定需要评估的 run 目录。"

    splits: List[str] = list(eval_cfg.get("splits", ["validation", "test"]))
    problem_type = eval_cfg.get(
        "problem_type",
        cfg.get("head", {}).get("problem_type", "single_label_classification"),
    )
    num_labels = (
        int(eval_cfg.get("num_labels", cfg.get("head", {}).get("num_labels", 2)))
        if problem_type != "regression"
        else 1
    )

    now = time.strftime("%Y%m%d-%H%M%S")
    run_id = cfg.get("run_id") or f"eval-{now}"
    out_dir = Path(cfg.get("output_dir", "runs")) / run_id
    reports_dir = _ensure_dir(out_dir / "reports")
    logs_dir = _ensure_dir(out_dir / "logs")
    try:
        configure_logging(log_dir=logs_dir)
    except Exception:
        pass

    with log_context(run_id=str(run_id), stage="evaluate", step=0):
        (out_dir / "configs_merged.yaml").write_text(
            yaml.safe_dump(dict(cfg), allow_unicode=True), encoding="utf-8"
        )
        LOGGER.info("评估 runs：%s", inputs)

        for split in splits:
            run_metrics: Dict[str, Dict[str, float]] = {}
            ys_true: List[np.ndarray] = []
            ys_pred: List[np.ndarray] = []
            ys_score: List[np.ndarray] = []

            for run_path in inputs:
                rd = Path(run_path)
                try:
                    y_true, y_pred, y_score = _load_split_from_run(
                        rd, split, problem_type
                    )
                except FileNotFoundError:
                    LOGGER.warning("%s 缺少 %s 预测文件，跳过。", rd, split)
                    continue
                # 单 run 指标
                m = _compute_metrics_bridge(
                    problem_type, y_true, (y_pred if y_pred.size else None), y_score
                )
                run_metrics[rd.name] = m
                # 收集用于 micro
                if y_true.size:
                    ys_true.append(y_true)
                if y_score is not None and y_score.size:
                    ys_score.append(y_score)
                if y_pred is not None and y_pred.size:
                    ys_pred.append(y_pred)

            # macro 均值/标准差
            if run_metrics:
                keys = sorted({k for m in run_metrics.values() for k in m.keys()})
                means = {
                    k: float(
                        np.mean(
                            [m.get(k, np.nan) for m in run_metrics.values() if k in m]
                        )
                    )
                    for k in keys
                }
                stds = {
                    k: float(
                        np.std(
                            [m.get(k, np.nan) for m in run_metrics.values() if k in m]
                        )
                    )
                    for k in keys
                }
            else:
                means, stds = {}, {}

            # micro（全量拼接后重算）
            micro: Dict[str, float] = {}
            if ys_true and (ys_pred or ys_score):
                y_true_all = np.concatenate(ys_true, axis=0)
                y_pred_all = np.concatenate(ys_pred, axis=0) if ys_pred else None
                y_score_all = np.concatenate(ys_score, axis=0) if ys_score else None
                micro = _compute_metrics_bridge(
                    problem_type, y_true_all, y_pred_all, y_score_all
                )
            # 写 metrics
            metrics_out = {
                "split": split,
                "problem_type": problem_type,
                "num_labels": num_labels,
                "micro": micro,
                "macro_mean": means,
                "macro_std": stds,
                "runs": run_metrics,
            }
            (reports_dir / f"metrics_{split}.json").write_text(
                json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # 混淆矩阵（仅适用于单标签分类）
            if (
                problem_type == "single_label_classification"
                and ys_true
                and (ys_pred or ys_score)
            ):
                y_true_all = np.concatenate(ys_true, axis=0).reshape(-1)
                if ys_pred:
                    y_pred_all = np.concatenate(ys_pred, axis=0).reshape(-1)
                else:
                    y_score_all = np.concatenate(ys_score, axis=0)
                    y_pred_all = y_score_all.argmax(axis=1)
                cm = _confusion_matrix(y_true_all, y_pred_all, num_labels=num_labels)
                _write_csv(
                    reports_dir / f"confusion_matrix_{split}.csv",
                    [["true\\pred"] + list(range(num_labels))]
                    + [[i] + list(map(int, row)) for i, row in enumerate(cm)],
                )

                # 试图输出 per-class ROC/PR 曲线
                try:
                    from sklearn.metrics import roc_curve, precision_recall_curve

                    curves_dir = _ensure_dir(reports_dir / "curves" / split)
                    # 拿到全量概率
                    if ys_score:
                        y_score_all = np.concatenate(ys_score, axis=0)
                    else:
                        # 若没有概率，仅能跳过
                        y_score_all = None
                    if y_score_all is not None and y_score_all.ndim == 2:
                        for c in range(num_labels):
                            y_true_bin = (y_true_all == c).astype(int)
                            y_score_c = y_score_all[:, c]
                            fpr, tpr, _ = roc_curve(y_true_bin, y_score_c)
                            prec, rec, _ = precision_recall_curve(y_true_bin, y_score_c)
                            _write_csv(curves_dir / f"roc_class{c}.csv", zip(fpr, tpr))
                            _write_csv(curves_dir / f"pr_class{c}.csv", zip(rec, prec))
                except Exception as e:
                    LOGGER.warning("跳过曲线导出：%s", e)

        LOGGER.info("评估完成，输出位于：%s", out_dir)
        return out_dir


# ------------------------------
# CLI 入口
# ------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate and aggregate KAN runs")
    p.add_argument(
        "-c",
        "--config",
        nargs="+",
        required=True,
        help="YAML 配置文件列表，后者覆盖前者",
    )
    p.add_argument(
        "-o",
        "--override",
        nargs="*",
        default=[],
        help="点号覆盖，如 eval.inputs_glob='runs/kan-*/' eval.splits='[\"validation\",\"test\"]'",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover
    args = build_argparser().parse_args(argv)
    run_from_configs(args.config, args.override)


if __name__ == "__main__":  # pragma: no cover
    main()
