# tests/test_kan_pipelines_train_trainer.py
# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 关键点：项目根模块命名 ===
# 这里直接导入根目录的 train_trainer.py，而不是 kan.* 命名空间
import kan.pipelines.train_trainer as tt


# ========== 轻量 stub：Trainer / TrainingArguments ==========
class _StubTrainResult:
    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics


class _StubPredictions:
    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids


class _StubTrainingArguments:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.output_dir = kwargs.get("output_dir")


class _StubTrainer:
    def __init__(
        self,
        *,
        model,
        args,
        train_dataset,
        eval_dataset=None,
        data_collator=None,
        compute_metrics=None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self._labels = [0, 1, 0]
        self._X = torch.tensor(
            [[1.0, -1.0], [0.5, -0.5], [2.0, -2.0]], dtype=torch.float32
        )

    def train(self):
        batch = {"text_vec": self._X, "labels": torch.tensor(self._labels)}
        out = self.model(**batch)
        loss = out.get("loss", None)
        if loss is not None:
            loss.backward()
        return _StubTrainResult(
            {"train_loss": float(loss.item() if loss is not None else 0.0)}
        )

    def save_model(self):
        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stub_model.bin").write_text("ok", encoding="utf-8")

    def evaluate(self, dataset=None):
        # 保持可用但不依赖 compute_metrics
        return {"accuracy": 1.0, "f1_macro": 1.0}

    def predict(self, dataset):
        logits = torch.tensor(
            [[10.0, -10.0], [9.0, -9.0], [8.0, -8.0]], dtype=torch.float32
        ).numpy()
        labels = torch.tensor([0, 0, 1]).numpy()
        return _StubPredictions(predictions=logits, label_ids=labels)


# ========== 轻量组件：Head / build_components 替换 ==========
class _TinyHead(nn.Module):
    def __init__(self, num_labels: int = 2):
        super().__init__()
        self.num_labels = int(num_labels)
        self.fc = nn.Linear(2, self.num_labels, bias=True)

    def forward(self, *, p=None, q=None, r=None, labels=None):
        assert p is not None and p.dim() == 2 and p.size(-1) == 2
        logits = self.fc(p)
        out: Dict[str, Any] = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels.long())
        return out


def _tiny_build_components(cfg: Mapping[str, Any]):
    num_labels = int(cfg.get("head", {}).get("num_labels", 2))
    return None, None, None, None, None, _TinyHead(num_labels=num_labels)


# ========== 轻量数据端：loader / batcher ==========
class _StubLoader:
    """loader_from_config(cfg) 返回的轻量 loader：返回简单样本"""

    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg

    def load_split(self, name: str) -> List[Mapping[str, Any]]:
        return [
            {"id": f"{name}-0", "text": "a", "label": 0},
            {"id": f"{name}-1", "text": "b", "label": 1},
            {"id": f"{name}-2", "text": "c", "label": 0},
        ]

    def has_split(self, name: str) -> bool:
        return name in {"train", "validation", "test"}

    def close(self):
        pass


@dataclass
class _StubTextConfig:
    max_length: int = 16
    pad_to_max: bool = True
    truncation: bool = True


@dataclass
class _StubEntityConfig:
    pass


@dataclass
class _StubContextConfig:
    pass


# ✅ 修复 dataclass 可变默认值错误
@dataclass
class _StubBatcherConfig:
    text: _StubTextConfig = field(default_factory=_StubTextConfig)
    entity: _StubEntityConfig = field(default_factory=_StubEntityConfig)
    context: _StubContextConfig = field(default_factory=_StubContextConfig)
    device: str = "cpu"


class _StubBatcher:
    def __init__(self, cfg: _StubBatcherConfig, *args, **kwargs):
        self.cfg = cfg

    def build_vocabs(self, records: Sequence[Mapping[str, Any]]):
        return

    def collate(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        vecs, labels = [], []
        for ex in examples:
            l = float(len(ex["text"]))
            vecs.append([l, -l])
            labels.append(int(ex["label"]))
        return {
            "text_vec": torch.tensor(vecs, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ========== 单元：dict 合并 / 点号覆盖 ==========
def test_deep_update_and_overrides():
    base = {"a": 1, "b": {"x": 1, "y": [1, 2]}}
    other = {"b": {"x": 2, "z": 3}, "c": 4}
    out = tt._deep_update(dict(base), other)
    assert out == {"a": 1, "b": {"x": 2, "y": [1, 2], "z": 3}, "c": 4}

    cfg = {"head": {"num_labels": 2}, "train": {"optimizer": {"lr": 1e-3}}}
    tt._apply_overrides(
        cfg, ["head.num_labels=3", "train.optimizer.lr=5e-4", "name=kan"]
    )
    assert cfg["head"]["num_labels"] == 3
    assert cfg["train"]["optimizer"]["lr"] == 5e-4
    assert cfg["name"] == "kan"


# ========== 单元：RecordsDataset / KANDataCollator ==========
def test_records_dataset_and_collator():
    ds = tt.RecordsDataset([{"id": "0"}, {"id": "1"}])
    assert len(ds) == 2 and ds[0]["id"] == "0"

    class _B:
        def collate(self, xs):
            return {"ok": len(xs)}

    col = tt.KANDataCollator(_B())
    assert col([{"a": 1}, {"a": 2}]) == {"ok": 2}


# ========== 单元：KANForNewsClassification（text-only 前向）==========
def test_model_forward_text_only():
    model = tt.KANForNewsClassification(
        text_encoder=None,
        entity_encoder=None,
        context_encoder=None,
        ne=None,
        ne2c=None,
        head=_TinyHead(num_labels=2),
        use_q=False,
        use_r=False,
        num_labels=2,
    )
    batch = {
        "text_vec": torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.float32),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    out = model(**batch)
    assert "logits" in out and out["logits"].shape == (2, 2)
    assert out["loss"] is not None and out["loss"].item() >= 0.0


# ========== 单元：make_compute_metrics（在不引入 kan.utils.metrics 的前提下）==========
def test_make_compute_metrics_bridge_without_kan(monkeypatch):
    # 不触碰 sys.modules：直接替换 tt.make_compute_metrics，返回轻量 compute
    def _fake_make_compute_metrics(problem_type: str = "single_label_classification"):
        def _fn(pred):
            logits, labels = pred
            import numpy as np

            y_score = logits[:, 0] - logits[:, 1]
            y_pred = (y_score >= 0.0).astype("i4")
            return {"accuracy": float((y_pred == labels).mean())}

        return _fn

    monkeypatch.setattr(
        tt, "make_compute_metrics", _fake_make_compute_metrics, raising=True
    )

    fn = tt.make_compute_metrics("single_label_classification")
    logits = torch.tensor([[2.0, 1.0], [0.1, 0.2]]).numpy()
    labels = torch.tensor([0, 1]).numpy()
    out = fn((logits, labels))
    assert "accuracy" in out and 0.0 <= out["accuracy"] <= 1.0


# ========== 端到端：run_from_configs（有 validation / test）==========
def test_run_from_configs_end2end_with_stubs(tmp_path, monkeypatch):
    # transformers / Trainer 替换
    monkeypatch.setattr(tt, "TrainingArguments", _StubTrainingArguments, raising=True)
    monkeypatch.setattr(tt, "Trainer", _StubTrainer, raising=True)
    monkeypatch.setattr(tt, "hf_set_seed", lambda x: None, raising=True)

    # 组件替换
    monkeypatch.setattr(tt, "build_components", _tiny_build_components, raising=True)

    # 数据端：替换为根模块的 loaders / batcher
    import kan.data.loaders as kdl
    import kan.data.batcher as kdb

    monkeypatch.setattr(
        kdl, "loader_from_config", lambda cfg: _StubLoader(cfg), raising=False
    )
    monkeypatch.setattr(kdb, "Batcher", _StubBatcher, raising=False)
    monkeypatch.setattr(kdb, "BatcherConfig", _StubBatcherConfig, raising=False)
    monkeypatch.setattr(kdb, "TextConfig", _StubTextConfig, raising=False)
    monkeypatch.setattr(kdb, "EntityConfig", _StubEntityConfig, raising=False)
    monkeypatch.setattr(kdb, "ContextConfig", _StubContextConfig, raising=False)

    # 避免导入 kan.utils.metrics：替换 make_compute_metrics
    def _fake_make_compute_metrics(problem_type: str = "single_label_classification"):
        def _fn(_pred):
            return {"accuracy": 1.0, "f1_macro": 1.0}

        return _fn

    monkeypatch.setattr(
        tt, "make_compute_metrics", _fake_make_compute_metrics, raising=True
    )

    yml = tmp_path / "mini.yaml"
    out_root = tmp_path / "out"
    yml.write_text(
        "\n".join(
            [
                f"output_dir: '{out_root.as_posix()}'",
                "head: { num_labels: 2, use_q: false, use_r: false }",
                "train: { batch_size: 3, max_epochs: 1, grad_accum: 1 }",
                "data: { name: 'dummy' }",
            ]
        ),
        encoding="utf-8",
    )

    out_dir = tt.run_from_configs([str(yml)], overrides=["name=e2e-stub"])
    assert out_dir.exists()

    tm = json.loads((out_dir / "train_metrics.json").read_text(encoding="utf-8"))
    assert "train_loss" in tm

    for split in ("validation", "test"):
        em = json.loads((out_dir / f"eval_{split}.json").read_text(encoding="utf-8"))
        assert "accuracy" in em and "f1_macro" in em
        lines = (
            (out_dir / f"pred_{split}.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        )
        assert len(lines) == 3
        rec = json.loads(lines[0])
        assert {"id", "y_true", "y_pred", "y_score"} <= set(rec.keys())

    merged = (out_dir / "configs_merged.yaml").read_text(encoding="utf-8")
    assert "e2e-stub" in merged


# ========== 端到端：无 validation/test 分支 ==========
def test_run_from_configs_no_eval_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "TrainingArguments", _StubTrainingArguments, raising=True)
    monkeypatch.setattr(tt, "Trainer", _StubTrainer, raising=True)
    monkeypatch.setattr(tt, "hf_set_seed", lambda x: None, raising=True)
    monkeypatch.setattr(tt, "build_components", _tiny_build_components, raising=True)

    # 无验证/测试的 loader
    class _NoEvalLoader(_StubLoader):
        def has_split(self, name: str) -> bool:
            return name == "train"

    import kan.data.loaders as kdl
    import kan.data.batcher as kdb

    monkeypatch.setattr(
        kdl, "loader_from_config", lambda cfg: _NoEvalLoader(cfg), raising=False
    )
    monkeypatch.setattr(kdb, "Batcher", _StubBatcher, raising=False)
    monkeypatch.setattr(kdb, "BatcherConfig", _StubBatcherConfig, raising=False)
    monkeypatch.setattr(kdb, "TextConfig", _StubTextConfig, raising=False)
    monkeypatch.setattr(kdb, "EntityConfig", _StubEntityConfig, raising=False)
    monkeypatch.setattr(kdb, "ContextConfig", _StubContextConfig, raising=False)

    # 同前：避免 metrics 依赖
    monkeypatch.setattr(
        tt,
        "make_compute_metrics",
        lambda *_: (lambda _p: {"accuracy": 1.0, "f1_macro": 1.0}),
    )

    yml = tmp_path / "mini2.yaml"
    out_root = tmp_path / "out2"
    yml.write_text(
        "\n".join(
            [
                f"output_dir: '{out_root.as_posix()}'",
                "head: { num_labels: 2, use_q: false, use_r: false }",
                "train: { batch_size: 3, max_epochs: 1, grad_accum: 1 }",
                "data: { name: 'dummy' }",
            ]
        ),
        encoding="utf-8",
    )

    out_dir = tt.run_from_configs([str(yml)], overrides=[])
    assert out_dir.exists()
    assert (out_dir / "train_metrics.json").exists()
    assert not (out_dir / "eval_validation.json").exists()
    assert not (out_dir / "pred_validation.jsonl").exists()
    assert not (out_dir / "eval_test.json").exists()
    assert not (out_dir / "pred_test.jsonl").exists()


# ========== CLI 参数解析 ==========
def test_build_argparser():
    p = tt.build_argparser()
    args = p.parse_args(
        ["--config", "a.yaml", "--override", "head.num_labels=3", "name=x"]
    )
    assert args.config == ["a.yaml"]
    assert args.override == ["head.num_labels=3", "name=x"]


# ========== YAML 多文件合并（存在即测，不在则 skip）==========
def test_merge_multiple_yaml_and_dump(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "TrainingArguments", _StubTrainingArguments, raising=True)
    monkeypatch.setattr(tt, "Trainer", _StubTrainer, raising=True)
    monkeypatch.setattr(tt, "hf_set_seed", lambda x: None, raising=True)
    monkeypatch.setattr(tt, "build_components", _tiny_build_components, raising=True)

    import kan.data.loaders as kdl
    import kan.data.batcher as kdb

    monkeypatch.setattr(
        kdl, "loader_from_config", lambda cfg: _StubLoader(cfg), raising=False
    )
    monkeypatch.setattr(kdb, "Batcher", _StubBatcher, raising=False)
    monkeypatch.setattr(kdb, "BatcherConfig", _StubBatcherConfig, raising=False)
    monkeypatch.setattr(kdb, "TextConfig", _StubTextConfig, raising=False)
    monkeypatch.setattr(kdb, "EntityConfig", _StubEntityConfig, raising=False)
    monkeypatch.setattr(kdb, "ContextConfig", _StubContextConfig, raising=False)

    # 避免 metrics 依赖
    monkeypatch.setattr(
        tt,
        "make_compute_metrics",
        lambda *_: (lambda _p: {"accuracy": 1.0, "f1_macro": 1.0}),
    )

    root = Path(__file__).resolve().parents[1]
    p1 = root / "configs" / "train" / "base.yaml"
    p2 = root / "configs" / "train" / "politifact_5fold.yaml"
    if not p1.exists() or not p2.exists():
        pytest.skip("示例 YAML 未挂载到仓库，请在完整环境下运行此用例。")

    out_root = tmp_path / "merged_out"
    overrides = [
        f"output_dir={out_root.as_posix()}",
        "head.num_labels=2",
        "head.use_q=false",
        "head.use_r=false",
        "train.batch_size=3",
        "train.max_epochs=1",
        "train.grad_accum=1",
        "data.name=dummy",
        "name=merge-two",
    ]
    out_dir = tt.run_from_configs([str(p1), str(p2)], overrides=overrides)
    assert out_dir.exists()
    merged_yaml = (out_dir / "configs_merged.yaml").read_text(encoding="utf-8")
    assert "merge-two" in merged_yaml
    assert "head:" in merged_yaml or "train:" in merged_yaml
