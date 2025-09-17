# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kan/pipelines/train_trainer.py
@brief  Pipeline: 使用 🤗 Transformers `Trainer` 训练 KAN（可切换为仅文本 / 文本+知识）。
@date   2025-09-16

@zh
  本文件定义 **流程（pipeline）**，而非具体模块实现：
  - 从 `configs/` 搜集并合并超参数（YAML + CLI overrides）。
  - 组织数据加载（kan.data.loaders）→ 打包批次（kan.data.batcher）。
  - 构建编码器/注意力/Head（kan.modules.*）并封装为可训练的 nn.Module。
  - 用 🤗 `Trainer` 训练/评估，结果写入 `runs/<run_id>/...`，中间缓存写入 `cache/`。

@en
  This is a **pipeline**, not a concrete module implementation. It:
  - Merges configs from YAML files and CLI overrides.
  - Loads data via `kan.data.loaders` and batches via `kan.data.batcher`.
  - Builds encoders/attentions/head (kan.modules.*) and wraps them into a trainable nn.Module.
  - Trains/Evaluates using 🤗 `Trainer`, writing artifacts to `runs/<run_id>/...` and caches to `cache/`.
"""

import argparse
import copy
import dataclasses
import json
import os
import sys
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
from torch.utils.data import Dataset

try:
    from transformers import (
        AutoTokenizer,
        HfArgumentParser,
        PreTrainedTokenizerBase,
        Trainer,
        TrainingArguments,
        set_seed as hf_set_seed,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError("未安装 transformers，请在环境中安装后再运行。") from e

# ------------------------------
# 日志（集中式，若不可用则回退到标准 logging）
# ------------------------------
try:
    from kan.utils.logging import configure_logging, log_context
    import logging

    LOGGER = logging.getLogger("kan.pipelines.train_trainer")
except Exception:  # 兼容仓库尚未落位 `kan/utils/logging.py` 的情况
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s | %(message)s"
    )
    LOGGER = logging.getLogger("kan.pipelines.train_trainer")

    def configure_logging(*args, **kwargs):  # type: ignore
        pass

    from contextlib import contextmanager

    @contextmanager
    def log_context(**kwargs):  # type: ignore
        yield


# ------------------------------
# 工具：YAML 读取与深合并、点号覆盖
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
        # 试图将字面量转为 JSON（数字/布尔/数组/对象）
        try:
            val_conv = json.loads(val)
        except Exception:
            val_conv = val
        # 点号赋值
        cur: MutableMapping[str, Any] = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], MutableMapping):
                cur[p] = {}
            cur = cur[p]  # type: ignore
        cur[parts[-1]] = val_conv


# ------------------------------
# 数据加载与批处理（依赖 kan.data.*）
# ------------------------------


class RecordsDataset(Dataset):
    """按需从 `kan.data.loaders` 返回的 List[NewsRecord] 暴露给 Trainer。

    我们不在此处做张量化，而是在 Collator 中调用 batcher 统一打包，便于灵活裁剪 E/N/T。
    """

    def __init__(self, records: List[Mapping[str, Any]]):
        self.records = records

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.records)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:  # type: ignore[override]
        return self.records[idx]


class KANDataCollator:
    """调用 `kan.data.batcher.Batcher` 的 collate 接口在 **collate_fn** 阶段完成打包。
    这样 Dataset 仅承担“样本记录容器”，Collator 负责把一批 `NewsRecord` → model inputs。
    """

    def __init__(self, batcher: Any):
        self.batcher = batcher

    def __call__(self, examples: List[Mapping[str, Any]]) -> Mapping[str, Any]:
        return self.batcher.collate(examples)


# ------------------------------
# KAN 组合模型封装为 nn.Module 供 Trainer 使用
# ------------------------------


class KANForNewsClassification(nn.Module):
    """把三路编码器 + NE/NE2C + Head 组合为一个可训练模块。

    约定：
    - `text_*` 键来自 batcher 的 `text_tok.*` 或 `text_vec`；
    - `ent_*` / `ctx_*` 键来自 batcher 的 `ent_ids/ctx_ids/...`；
    - 返回字典包含 `loss`（若提供 labels）与 `logits`。
    """

    def __init__(
        self,
        text_encoder: Any,
        entity_encoder: Optional[Any],
        context_encoder: Optional[Any],
        ne: Optional[Any],
        ne2c: Optional[Any],
        head: Any,
        use_q: bool = True,
        use_r: bool = True,
        num_labels: int = 2,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.entity_encoder = entity_encoder
        self.context_encoder = context_encoder
        self.ne = ne
        self.ne2c = ne2c
        self.head = head
        self.use_q = use_q
        self.use_r = use_r
        self.num_labels = num_labels

    def forward(self, **batch) -> Mapping[str, torch.Tensor]:  # type: ignore[override]
        # 1) 文本编码 → p
        if "text_tok" in batch:
            te_out = self.text_encoder(
                **batch["text_tok"]
            )  # 期望：{sequence_output, pooled_output, attention_mask}
            p = te_out["pooled_output"]
            news_hidden = te_out.get("sequence_output")
            news_mask = te_out.get("attention_mask")
        elif "text_vec" in batch:  # 句向量后端（备用）
            p = batch["text_vec"]
            news_hidden, news_mask = None, None
        else:
            raise ValueError("batch 缺少 text_tok 或 text_vec")

        q_prime = None
        r_prime = None
        ents_mask = None
        ctxs_mask = None

        # 2) 实体/上下文编码（若可用）
        if self.entity_encoder is not None and ("ent_ids" in batch):
            ent_out = self.entity_encoder(
                entity_ids=batch["ent_ids"],
                entity_mask=batch.get("ent_mask"),
                context_ids=batch.get("ctx_ids"),
                context_mask=batch.get("ctx_mask"),
            )
            q_prime = ent_out.get("entities_last_hidden")
            ents_mask = ent_out.get("entities_mask") or batch.get("ent_mask")
            # 上下文可能来源于 entity_encoder 或独立的 context_encoder
            r_prime = ent_out.get("contexts_last_hidden")
            ctxs_mask = ent_out.get("contexts_mask") or batch.get("ctx_mask")

        if self.context_encoder is not None and ("context_input_ids" in batch):
            ctx_out = self.context_encoder(
                context_input_ids=batch["context_input_ids"],
                context_attention_mask=batch["context_attention_mask"],
                context_token_type_ids=batch.get("context_token_type_ids"),
                contexts_mask=batch.get("ctx_mask"),
            )
            r_prime = ctx_out.get("contexts_last_hidden")
            ctxs_mask = ctx_out.get("contexts_mask") or batch.get("ctx_mask")

        # 3) 知识注意力融合（可选择关闭）
        q = None
        r = None
        if self.use_q and self.ne is not None and (q_prime is not None):
            try:
                q_out = self.ne(
                    news={"last_hidden_state": news_hidden, "pooled_state": p},
                    entities={"last_hidden_state": q_prime},
                    news_mask=news_mask,
                    entity_mask=ents_mask,
                    return_weights=False,
                )
                q = q_out.get("pooled") if isinstance(q_out, dict) else q_out
            except NotImplementedError:
                # 签名存在但算法未实现时，优雅退化：跳过 q
                q = None
        if (
            self.use_r
            and self.ne2c is not None
            and (q_prime is not None)
            and (r_prime is not None)
        ):
            try:
                r_out = self.ne2c(
                    news={"last_hidden_state": news_hidden, "pooled_state": p},
                    entities={"last_hidden_state": q_prime},
                    contexts_last_hidden=r_prime,
                    news_mask=news_mask,
                    entity_mask=ents_mask,
                    contexts_mask=ctxs_mask,
                    return_weights=False,
                )
                r = r_out.get("pooled") if isinstance(r_out, dict) else r_out
            except NotImplementedError:
                r = None

        # 4) Head 分类
        head_out = self.head(
            p=p if p is not None else None, q=q, r=r, labels=batch.get("labels")
        )
        # 兼容：head 可能返回 {logits,(loss),...}
        loss = head_out.get("loss") if isinstance(head_out, Mapping) else None
        logits = head_out.get("logits") if isinstance(head_out, Mapping) else head_out
        return {"loss": loss, "logits": logits}


# ------------------------------
# 组件构建（Registry 优先，退化到简单构造）
# ------------------------------


def build_components(
    cfg: Mapping[str, Any],
) -> Tuple[Any, Optional[Any], Optional[Any], Optional[Any], Optional[Any], Any]:
    """从配置构建 text/entity/context encoders、NE/NE2C、Head。
    优先使用 `kan.utils.registry.HUB`，便于按 `type/name` 解耦；若不可用则尝试直接从模块导入默认构造器。
    """
    # try registry hub
    try:
        from kan.utils.registry import HUB, build_from_config

        TEXT = HUB.get_or_create("text_encoder")
        ENTITY = HUB.get_or_create("entity_encoder")
        CONTEXT = HUB.get_or_create("context_encoder")
        ATT = HUB.get_or_create("attention")
        HEAD = HUB.get_or_create("head")

        te = build_from_config(cfg.get("text_encoder", {}), TEXT)
        ee = (
            build_from_config(cfg.get("entity_encoder", {}), ENTITY)
            if cfg.get("entity_encoder")
            else None
        )
        ce = (
            build_from_config(cfg.get("context_encoder", {}), CONTEXT)
            if cfg.get("context_encoder")
            else None
        )
        ne = (
            build_from_config(cfg.get("ne", {"type": "ne"}), ATT)
            if cfg.get("ne")
            else None
        )
        ne2c = (
            build_from_config(cfg.get("ne2c", {"type": "ne2c"}), ATT)
            if cfg.get("ne2c")
            else None
        )
        hd = build_from_config(cfg.get("head", {}), HEAD)
        return te, ee, ce, ne, ne2c, hd
    except Exception as e:
        LOGGER.warning("Registry 构建失败，尝试直接导入模块：%s", e)
        # Fallback：直接 import 预设构造函数（要求代码已实现）
        from kan.modules.text_encoder import build_text_encoder
        from kan.modules.entity_encoder import build_entity_encoder
        from kan.modules.context_encoder import build_context_encoder
        from kan.modules.head import build_head

        te = build_text_encoder(cfg.get("text_encoder", {}))
        ee = (
            build_entity_encoder(cfg.get("entity_encoder", {}))
            if cfg.get("entity_encoder")
            else None
        )
        ce = (
            build_context_encoder(cfg.get("context_encoder", {}))
            if cfg.get("context_encoder")
            else None
        )
        ne = None
        ne2c = None
        try:
            from kan.modules.attention.ne import NEAttention  # type: ignore

            ne = NEAttention(**cfg.get("ne", {})) if cfg.get("ne") else None
        except Exception:
            pass
        try:
            from kan.modules.attention.ne2c import NE2CAttention  # type: ignore

            ne2c = NE2CAttention(**cfg.get("ne2c", {})) if cfg.get("ne2c") else None
        except Exception:
            pass
        hd = build_head(cfg.get("head", {}))
        return te, ee, ce, ne, ne2c, hd


# ------------------------------
# 指标：桥接到 kan.utils.metrics
# ------------------------------


def make_compute_metrics(problem_type: str = "single_label_classification"):
    from kan.utils.metrics import compute_classification_metrics

    def _fn(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)
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
            labels = labels.cpu().numpy()
            return {"mse": float(mean_squared_error(labels, preds))}

    return _fn


# ------------------------------
# 主入口：组装配置 → 数据 → 训练器 → 训练/评估/保存
# ------------------------------


def run_from_configs(
    config_paths: Sequence[str], overrides: Sequence[str] = ()
) -> Path:
    """合并配置并运行训练。返回本次 run 的输出目录。"""
    # 1) 合并配置
    cfg: MutableMapping[str, Any] = {}
    for p in config_paths:
        cp = Path(p)
        assert cp.exists(), f"配置文件不存在：{cp}"
        _deep_update(cfg, _read_yaml(cp))
    _apply_overrides(cfg, list(overrides))

    # 2) 输出目录与日志
    now = time.strftime("%Y%m%d-%H%M%S")
    run_id = cfg.get("run_id") or f"{cfg.get('name', 'kan')}-{now}"
    out_dir = Path(cfg.get("output_dir", "runs")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 配置集中式日志
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        configure_logging(log_dir=log_dir)
    except Exception:
        pass

    with log_context(run_id=str(run_id), stage="train", step=0):
        LOGGER.info("run_id=%s | 输出目录=%s", run_id, out_dir)
        (out_dir / "configs_merged.yaml").write_text(
            yaml.safe_dump(dict(cfg), allow_unicode=True), encoding="utf-8"
        )

        # 3) 随机性 & 设备
        try:
            from kan.utils.seed import set_seed

            seed = int(cfg.get("seed", 42))
            set_seed(seed, deterministic=bool(cfg.get("deterministic", True)))
        except Exception:
            seed = int(cfg.get("seed", 42))
            hf_set_seed(seed)
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda"
            else "cpu"
        )
        LOGGER.info("device=%s cuda_available=%s", device, torch.cuda.is_available())

        # 4) 数据：加载 → 词表/批处理 → Dataset/Collator
        from kan.data.loaders import loader_from_config, Dataset as NewsDataset  # type: ignore
        from kan.data.batcher import Batcher, BatcherConfig, TextConfig, EntityConfig, ContextConfig  # type: ignore

        data_cfg = cfg.get("data") or cfg  # 兼容：直接把 data 字段铺在根部
        loader = loader_from_config(data_cfg)
        # 预加载训练集以便构建词表
        train_records = loader.load_split(data_cfg.get("train_split", "train"))
        # 词表/批处理器（明确 CPU 上构建）
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
        # Dataset & Collator
        collator = KANDataCollator(batcher)
        ds_train = RecordsDataset(train_records)
        split_valid = data_cfg.get("validation_split", "validation")
        ds_valid = (
            RecordsDataset(loader.load_split(split_valid))
            if loader.has_split(split_valid)
            else None
        )
        split_test = data_cfg.get("test_split", "test")
        ds_test = (
            RecordsDataset(loader.load_split(split_test))
            if loader.has_split(split_test)
            else None
        )

        # 5) 构建组件并包成模型
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
        model.to(device)

        # 6) 训练参数（转为 TrainingArguments）
        targs_dict = cfg.get("train", {})
        # 必填项：输出目录
        targs = TrainingArguments(
            output_dir=str(out_dir / "artifacts"),
            overwrite_output_dir=True,
            do_train=True,
            do_eval=ds_valid is not None,
            evaluation_strategy=targs_dict.get("evaluation_strategy", "epoch"),
            per_device_train_batch_size=int(targs_dict.get("batch_size", 8)),
            per_device_eval_batch_size=int(
                targs_dict.get("eval_batch_size", targs_dict.get("batch_size", 8))
            ),
            learning_rate=float(targs_dict.get("optimizer", {}).get("lr", 5e-5)),
            weight_decay=float(
                targs_dict.get("optimizer", {}).get("weight_decay", 0.0)
            ),
            num_train_epochs=float(targs_dict.get("max_epochs", 3)),
            gradient_accumulation_steps=int(targs_dict.get("grad_accum", 1)),
            logging_steps=int(targs_dict.get("logging", {}).get("every_n_steps", 50)),
            save_strategy=targs_dict.get("save_strategy", "epoch"),
            save_total_limit=int(targs_dict.get("save_total_limit", 3)),
            fp16=bool(cfg.get("fp16", False)),
            bf16=bool(cfg.get("bf16", False)),
            seed=int(seed),
            dataloader_num_workers=int(
                targs_dict.get("dataloader", {}).get("num_workers", 0)
            ),  # Windows 友好
            report_to=targs_dict.get("report_to", []),
        )

        # 7) Trainer
        compute_metrics = make_compute_metrics(
            problem_type=cfg.get("head", {}).get(
                "problem_type", "single_label_classification"
            )
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=ds_train,
            eval_dataset=ds_valid,
            data_collator=collator,
            compute_metrics=compute_metrics if ds_valid is not None else None,
        )

        # 8) 训练
        train_result = trainer.train()
        (out_dir / "train_metrics.json").write_text(
            json.dumps(train_result.metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        trainer.save_model()  # 保存权重

        # 9) 评估（valid/test）与预测持久化
        def _eval_and_save(name: str, dataset: Optional[Dataset]):
            if dataset is None:
                return
            metrics = trainer.evaluate(dataset=dataset)
            (out_dir / f"eval_{name}.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            # 预测分数与标签
            preds = trainer.predict(dataset)
            logits = (
                preds.predictions
                if not isinstance(preds.predictions, (list, tuple))
                else preds.predictions[0]
            )
            y_score = torch.softmax(torch.tensor(logits), dim=-1).tolist()
            y_pred = torch.tensor(y_score).argmax(dim=-1).tolist()
            y_true = (
                preds.label_ids.tolist()
                if isinstance(preds.label_ids, (list, tuple))
                else preds.label_ids
            )
            rows = [
                {
                    "y_true": int(y_true[i]),
                    "y_pred": int(y_pred[i]),
                    "y_score": y_score[i],
                }
                for i in range(len(y_pred))
            ]
            (out_dir / f"pred_{name}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                encoding="utf-8",
            )

        _eval_and_save("validation", ds_valid)
        _eval_and_save("test", ds_test)

        LOGGER.info("训练完成，输出位于：%s", out_dir)
        return out_dir


# ------------------------------
# CLI 入口（供 scripts/ 调用，亦可直接运行）
# ------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train KAN with 🤗 Trainer (pipeline)")
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
        help="点号覆盖，如 head.num_labels=2 optimizer.lr=3e-5",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover
    args = build_argparser().parse_args(argv)
    run_from_configs(args.config, args.override)


if __name__ == "__main__":  # pragma: no cover
    main()
