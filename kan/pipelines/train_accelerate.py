# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kan/pipelines/train_accelerate.py
@brief  Pipeline: 使用 Hugging Face Accelerate 自定义训练循环（替代 Trainer）。
@date   2025-09-16

# 设计哲学 / Design Philosophy
- **这是流程编排（pipeline），不是具体模块实现**：所有算法细节（编码器/注意力/损失/张量化）交由 `kan.modules.*` 与 `kan.data.*` 完成；本文件只做装配、调度、日志与落盘。
- **超参从 configs/ 合并**：支持多 YAML 顺序覆盖 + `--override key=value` 点号覆盖；最终快照写入 `runs/<run_id>/configs_merged.yaml`。
- **被 scripts/ 下的脚本调用**：不要在 pipeline 中硬编码任何绝对路径或环境敏感逻辑。
- **过程 vs 结果分离**：
  * 过程（Process）：原始镜像、EL/KG 中间层、embedding 索引、训练日志、加速器状态，位于 `cache/` 和 `runs/<run_id>/logs/`。
  * 结果（Result）：可复用权重与指标，位于 `runs/<run_id>/artifacts/`（`best/`、`last/` 和指标 JSON、预测 JSONL）。
- **缓存与容错**：大胆使用 `cache/` 与 checkpoint；组件缺失（如 NE/NE2C 未实现）时优雅退化到文本-only，不阻塞端到端。
- **Windows 友好**：避免强制多进程；`num_workers` 默认为 0；路径用 `pathlib`；支持断点续训 `resume_from`。

# 约定 / Conventions
- 数据加载：通过 `kan.data.loaders.loader_from_config(data_cfg)` 获取 loader，Dataset 只存记录；张量化交由 `kan.data.batcher.Batcher.collate()` 在 `collate_fn` 中完成。
- 组件装配：优先 `kan.utils.registry.HUB`，fallback 到 `kan.modules.*` 的 `build_*`。
- 指标：桥接到 `kan.utils.metrics.compute_classification_metrics`，并在多标签/回归模式下切换。
"""

import argparse
import copy
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ------------------------------
# 依赖检测
# ------------------------------
try:
    from accelerate import Accelerator, DistributedType
except Exception as e:  # pragma: no cover
    raise RuntimeError("缺少 accelerate，请先 `pip install accelerate`。") from e

try:
    from transformers import get_linear_schedule_with_warmup, set_seed as hf_set_seed
except Exception as e:  # pragma: no cover
    raise RuntimeError("缺少 transformers，请先安装。")

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("缺少 PyYAML，请 `pip install pyyaml`。") from e

# ------------------------------
# 日志（集中式，若不可用则回退）
# ------------------------------
try:
    from kan.utils.logging import configure_logging, log_context
    import logging

    LOGGER = logging.getLogger("kan.pipelines.train_accelerate")
except Exception:
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s | %(message)s"
    )
    LOGGER = logging.getLogger("kan.pipelines.train_accelerate")

    def configure_logging(*args, **kwargs):  # type: ignore
        pass

    from contextlib import contextmanager

    @contextmanager
    def log_context(**kwargs):  # type: ignore
        yield


# ------------------------------
# 配置合并与覆盖
# ------------------------------


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
# 复用 train_trainer 的轻量 Dataset/Collator/组合模型与装配函数
# ------------------------------
try:
    from kan.pipelines.train_trainer import (
        RecordsDataset,
        KANDataCollator,
        KANForNewsClassification,
        build_components,
    )
except Exception:
    # 兜底导入：若上游尚未可用，可在本文件临时提供极简实现或抛错
    raise RuntimeError(
        "需要 kan.pipelines.train_trainer 中的 RecordsDataset/KANDataCollator/KANForNewsClassification。"
    )

# ------------------------------
# 指标桥接
# ------------------------------


def make_compute_metrics(problem_type: str = "single_label_classification"):
    from kan.utils.metrics import compute_classification_metrics

    def _fn(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        if problem_type == "single_label_classification":
            y_score = torch.softmax(logits, dim=-1).cpu().numpy()
            y_pred = y_score.argmax(axis=-1)
            return compute_classification_metrics(
                labels.cpu().numpy(), y_pred=y_pred, y_score=y_score
            )
        elif problem_type == "multilabel_classification":
            y_score = torch.sigmoid(logits).cpu().numpy()
            y_pred = (y_score >= 0.5).astype("i4")
            return compute_classification_metrics(
                labels.cpu().numpy(), y_pred=y_pred, y_score=y_score
            )
        else:  # regression
            from sklearn.metrics import mean_squared_error

            preds = logits.squeeze(-1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            return {"mse": float(mean_squared_error(labels_np, preds))}

    return _fn


# ------------------------------
# 优化器 & 参数组
# ------------------------------


def build_optimizer(model: nn.Module, opt_cfg: Mapping[str, Any]):
    lr = float(opt_cfg.get("lr", 5e-5))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))  # type: ignore
    eps = float(opt_cfg.get("eps", 1e-8))

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(
            x in n
            for x in ["bias", "LayerNorm.weight", "layer_norm.weight", "ln.weight"]
        ):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


# ------------------------------
# 评估循环
# ------------------------------


def evaluate(
    accel: Accelerator,
    model: nn.Module,
    loader: DataLoader,
    compute_metrics,
    problem_type: str,
    num_labels: int,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            logits = out["logits"] if isinstance(out, Mapping) else out
            labels = batch.get("labels")
            # 收集到主进程，注意使用 gather_for_metrics 以避免重复样本
            logits = accel.gather_for_metrics(logits)
            if labels is not None:
                labels = accel.gather_for_metrics(labels)
            all_logits.append(logits.cpu())
            if labels is not None:
                all_labels.append(labels.cpu())
    logits_cat = (
        torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, num_labels)
    )
    labels_cat = (
        torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
    )
    metrics: Dict[str, float] = {}
    if labels_cat.numel() > 0:
        metrics = compute_metrics(logits_cat, labels_cat)
    model.train()
    return metrics, logits_cat, labels_cat


# ------------------------------
# 主流程
# ------------------------------


def run_from_configs(
    config_paths: Sequence[str], overrides: Sequence[str] = ()
) -> Path:
    # 1) 配置合并
    cfg: MutableMapping[str, Any] = {}
    for p in config_paths:
        cp = Path(p)
        assert cp.exists(), f"配置文件不存在：{cp}"
        _deep_update(cfg, _read_yaml(cp))
    _apply_overrides(cfg, list(overrides))

    # 2) 目录与日志
    now = time.strftime("%Y%m%d-%H%M%S")
    run_id = cfg.get("run_id") or f"{cfg.get('name', 'kan-acc')}-{now}"
    out_dir = Path(cfg.get("output_dir", "runs")) / run_id
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    try:
        configure_logging(log_dir=(out_dir / "logs"))
    except Exception:
        pass

    with log_context(run_id=str(run_id), stage="train", step=0):
        LOGGER.info("run_id=%s | 输出目录=%s", run_id, out_dir)
        (out_dir / "configs_merged.yaml").write_text(
            yaml.safe_dump(dict(cfg), allow_unicode=True), encoding="utf-8"
        )

        # 3) 随机性
        seed = int(cfg.get("seed", 42))
        try:
            from kan.utils.seed import set_seed

            set_seed(seed, deterministic=bool(cfg.get("deterministic", True)))
        except Exception:
            hf_set_seed(seed)

        # 4) Accelerator 初始化
        train_cfg = cfg.get("train", {})
        mixed_precision = "no"
        if bool(cfg.get("bf16", False)):
            mixed_precision = "bf16"
        elif bool(cfg.get("fp16", False)):
            mixed_precision = "fp16"
        grad_accum = int(train_cfg.get("grad_accum", 1))
        log_with = train_cfg.get("report_to", []) or None
        accel = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum,
            log_with=log_with,
        )
        if log_with:
            accel.init_trackers(
                project_name=cfg.get("name", "kan"), config={"seed": seed, **train_cfg}
            )

        device = accel.device
        LOGGER.info(
            "accelerator: %s | device=%s | mp=%s",
            accel.state.distributed_type,
            device,
            mixed_precision,
        )

        # 5) 数据
        from kan.data.loaders import loader_from_config  # type: ignore
        from kan.data.batcher import Batcher, BatcherConfig, TextConfig, EntityConfig, ContextConfig  # type: ignore

        data_cfg = cfg.get("data") or cfg
        loader = loader_from_config(data_cfg)
        train_records = loader.load_split(data_cfg.get("train_split", "train"))
        valid_split = data_cfg.get("validation_split", "validation")
        test_split = data_cfg.get("test_split", "test")

        batcher = Batcher(
            BatcherConfig(
                text=TextConfig(**(data_cfg.get("batcher", {}).get("text", {}))),
                entity=EntityConfig(**(data_cfg.get("batcher", {}).get("entity", {}))),
                context=ContextConfig(
                    **(data_cfg.get("batcher", {}).get("context", {}))
                ),
                device="cpu",
            )
        )
        batcher.build_vocabs(train_records)
        collator = KANDataCollator(batcher)

        ds_train = RecordsDataset(train_records)
        ds_valid = (
            RecordsDataset(loader.load_split(valid_split))
            if hasattr(loader, "has_split") and loader.has_split(valid_split)
            else None
        )
        ds_test = (
            RecordsDataset(loader.load_split(test_split))
            if hasattr(loader, "has_split") and loader.has_split(test_split)
            else None
        )

        num_workers = int(train_cfg.get("dataloader", {}).get("num_workers", 0))
        batch_size = int(train_cfg.get("batch_size", 8))
        eval_bs = int(train_cfg.get("eval_batch_size", batch_size))
        drop_last = bool(train_cfg.get("dataloader", {}).get("drop_last", False))
        pin_memory = bool(train_cfg.get("dataloader", {}).get("pin_memory", True))

        dl_train = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collator,
        )
        dl_valid = (
            DataLoader(
                ds_valid,
                batch_size=eval_bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                collate_fn=collator,
            )
            if ds_valid is not None
            else None
        )
        dl_test = (
            DataLoader(
                ds_test,
                batch_size=eval_bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                collate_fn=collator,
            )
            if ds_test is not None
            else None
        )

        # 6) 组件与模型
        te, ee, ce, ne, ne2c, head = build_components(cfg)
        model = KANForNewsClassification(
            text_encoder=te,
            entity_encoder=ee,
            context_encoder=ce,
            ne=ne,
            ne2c=ne2c,
            head=head,
            use_q=bool(cfg.get("head", {}).get("use_q", True)),
            use_r=bool(cfg.get("head", {}).get("use_r", True)),
            num_labels=int(cfg.get("head", {}).get("num_labels", 2)),
        )
        if bool(cfg.get("compile", False)) and hasattr(torch, "compile"):
            model = torch.compile(model)  # type: ignore

        # 7) 优化器与调度器
        optimizer = build_optimizer(model, train_cfg.get("optimizer", {}))

        # 8) 准备分布式对象
        to_prepare = [model, optimizer, dl_train]
        if dl_valid is not None:
            to_prepare.append(dl_valid)
        if dl_test is not None:
            to_prepare.append(dl_test)
        model, optimizer, dl_train, *rest = accel.prepare(*to_prepare)
        # prepare 后再构造 scheduler（因为 optimizer 可能被包裹）
        max_epochs = float(train_cfg.get("max_epochs", 3))
        total_update_steps = math.ceil(len(dl_train) / grad_accum) * int(max_epochs)
        warmup = train_cfg.get("lr_scheduler", {}).get("warmup_ratio", 0.0)
        warmup_steps = int(
            train_cfg.get("lr_scheduler", {}).get(
                "warmup_steps", total_update_steps * warmup
            )
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )
        if dl_valid is not None and dl_test is not None:
            dl_valid, dl_test = rest  # type: ignore
        elif dl_valid is not None:
            (dl_valid,) = rest  # type: ignore
        elif dl_test is not None:
            (dl_test,) = rest  # type: ignore

        # 9) 断点续训
        resume_from = train_cfg.get("resume_from")
        if resume_from:
            ckpt_dir = Path(resume_from)
            if ckpt_dir.exists():
                accel.load_state(ckpt_dir)
                LOGGER.info("从断点恢复：%s", ckpt_dir)

        # 10) 训练循环
        compute_metrics = make_compute_metrics(
            problem_type=cfg.get("head", {}).get(
                "problem_type", "single_label_classification"
            )
        )
        clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
        log_steps = int(train_cfg.get("logging", {}).get("every_n_steps", 50))
        eval_strategy = train_cfg.get("evaluation_strategy", "epoch")  # 或 "steps"
        eval_steps = int(train_cfg.get("eval_steps", log_steps))
        save_strategy = train_cfg.get("save_strategy", "epoch")
        save_total_limit = int(train_cfg.get("save_total_limit", 3))
        metric_for_best = train_cfg.get("metric_for_best", "f1")
        greater_is_better = bool(train_cfg.get("greater_is_better", True))

        artifacts_dir = out_dir / "artifacts"
        best_dir = artifacts_dir / "best"
        last_dir = artifacts_dir / "last"
        best_score = None
        global_step = 0

        scaler_msg = f"accum={grad_accum} max_epochs={max_epochs} total_updates={total_update_steps}"
        LOGGER.info("开始训练 | %s", scaler_msg)

        for epoch in range(int(max_epochs)):
            model.train()
            for step, batch in enumerate(dl_train):
                with accel.accumulate(model):
                    out = model(**batch)
                    loss = out["loss"] if isinstance(out, Mapping) else out
                    accel.backward(loss)
                    if clip_norm and clip_norm > 0:
                        accel.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # 日志
                if accel.is_local_main_process and (global_step % log_steps == 0):
                    cur_lr = scheduler.get_last_lr()[0]
                    LOGGER.info(
                        "epoch=%d step=%d/%d lr=%.3e loss=%.4f",
                        epoch,
                        step + 1,
                        len(dl_train),
                        cur_lr,
                        float(loss),
                    )
                if log_with:
                    accel.log(
                        {
                            "train/loss": float(loss),
                            "lr": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

                # 按步评估/保存
                if (
                    eval_strategy == "steps"
                    and dl_valid is not None
                    and global_step % eval_steps == 0
                ):
                    metrics, logits_cat, labels_cat = evaluate(
                        accel,
                        model,
                        dl_valid,
                        compute_metrics,
                        cfg.get("head", {}).get(
                            "problem_type", "single_label_classification"
                        ),
                        int(cfg.get("head", {}).get("num_labels", 2)),
                    )
                    if accel.is_main_process:
                        (
                            out_dir / f"eval_validation@step{global_step}.json"
                        ).write_text(
                            json.dumps(metrics, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    # 保存
                    if save_strategy == "steps" and accel.is_main_process:
                        ckpt_dir = artifacts_dir / f"ckpt-step{global_step}"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)
                        accel.save_state(ckpt_dir)
                        # best 选择
                        cur = metrics.get(metric_for_best)
                        if cur is not None and (
                            best_score is None
                            or (
                                cur > best_score
                                if greater_is_better
                                else cur < best_score
                            )
                        ):
                            best_score = cur
                            # 同步 best 目录
                            for p in best_dir.glob("*"):
                                if p.is_file():
                                    p.unlink()
                            accel.save_state(best_dir)

            # 每 epoch 评估
            if dl_valid is not None:
                metrics, logits_cat, labels_cat = evaluate(
                    accel,
                    model,
                    dl_valid,
                    compute_metrics,
                    cfg.get("head", {}).get(
                        "problem_type", "single_label_classification"
                    ),
                    int(cfg.get("head", {}).get("num_labels", 2)),
                )
                if accel.is_main_process:
                    (out_dir / f"eval_validation@epoch{epoch}.json").write_text(
                        json.dumps(metrics, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                if log_with:
                    accel.log(
                        {**{f"val/{k}": v for k, v in metrics.items()}, "epoch": epoch},
                        step=global_step,
                    )
                if accel.is_main_process:
                    cur = metrics.get(metric_for_best)
                    if cur is not None and (
                        best_score is None
                        or (cur > best_score if greater_is_better else cur < best_score)
                    ):
                        best_score = cur
                        best_dir.mkdir(parents=True, exist_ok=True)
                        for p in best_dir.glob("*"):
                            if p.is_file():
                                p.unlink()
                        accel.save_state(best_dir)

            # 每 epoch 保存 last
            if accel.is_main_process:
                last_dir.mkdir(parents=True, exist_ok=True)
                for p in last_dir.glob("*"):
                    if p.is_file():
                        p.unlink()
                accel.save_state(last_dir)

        # 11) 最终评估与预测写出
        def _save_preds(name: str, logits: torch.Tensor, labels: torch.Tensor):
            if logits.numel() == 0:
                return
            if (
                cfg.get("head", {}).get("problem_type", "single_label_classification")
                == "single_label_classification"
            ):
                y_score = torch.softmax(logits, dim=-1).tolist()
                y_pred = torch.tensor(y_score).argmax(dim=-1).tolist()
            elif cfg.get("head", {}).get("problem_type") == "multilabel_classification":
                y_score = torch.sigmoid(logits).tolist()
                y_pred = (torch.tensor(y_score) >= 0.5).int().tolist()
            else:
                y_score = logits.squeeze(-1).tolist()
                y_pred = y_score
            y_true = labels.tolist() if labels.numel() > 0 else []
            rows = []
            for i in range(len(y_pred)):
                rows.append(
                    {
                        "y_true": (int(y_true[i]) if y_true else None),
                        "y_pred": y_pred[i],
                        "y_score": y_score[i],
                    }
                )
            (out_dir / f"pred_{name}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                encoding="utf-8",
            )

        if dl_valid is not None:
            metrics, logits_cat, labels_cat = evaluate(
                accel,
                model,
                dl_valid,
                compute_metrics,
                cfg.get("head", {}).get("problem_type", "single_label_classification"),
                int(cfg.get("head", {}).get("num_labels", 2)),
            )
            if accel.is_main_process:
                (out_dir / f"eval_validation.json").write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                _save_preds("validation", logits_cat, labels_cat)

        if dl_test is not None:
            metrics, logits_cat, labels_cat = evaluate(
                accel,
                model,
                dl_test,
                compute_metrics,
                cfg.get("head", {}).get("problem_type", "single_label_classification"),
                int(cfg.get("head", {}).get("num_labels", 2)),
            )
            if accel.is_main_process:
                (out_dir / f"eval_test.json").write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                _save_preds("test", logits_cat, labels_cat)

        LOGGER.info("训练完成，输出位于：%s", out_dir)
        if log_with:
            accel.end_training()
        return out_dir


# ------------------------------
# CLI 入口
# ------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train KAN with HF Accelerate (custom loop)"
    )
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
        help="点号覆盖，如 head.num_labels=2 train.optimizer.lr=3e-5",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover
    args = build_argparser().parse_args(argv)
    run_from_configs(args.config, args.override)


if __name__ == "__main__":  # pragma: no cover
    main()
