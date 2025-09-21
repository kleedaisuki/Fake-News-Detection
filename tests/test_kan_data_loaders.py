# /tests/test_kan_data_loaders.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import pandas as pd
import pytest

from kan.data.loaders import (
    DatasetConfig,
    FieldMap,
    JSONLinesLoader,
    loader_from_config,
    Dataset,
    build_loader,
    CSVLoader,
)

# ========= 配置区：按你的真实路径/文件名修改 ==============
# 假设四个 JSONL 放在同一目录，比如 /datasets/politifact/processed/
DATA_DIR = Path("/datasets/politifact/processed")  # ← 若路径不同，这里改掉
SPLITS = {
    "train": "HF.jsonl",
    "validation": "HR.jsonl",
    "test": "MF.jsonl",
    "misc": "MR.jsonl",  # 可选：当第四份不是三大 split，用自定义 split 名
}

# 你的 JSONL 里“文本/标签/ID”的列名可能不完全一致。
# 这里给出常见候选；测试会逐一尝试找到能跑通的 FieldMap。
FIELDS_CANDIDATES = [
    # 常见命名
    dict(
        id="id",
        text="text",
        label="label",
        entities="entities",
        contexts="contexts",
        split=None,
        fold=None,
        meta=["source", "note", "src"],
    ),
    # FakeNewsNet/PolitiFact 常见别名
    dict(
        id="news_id",
        text="content",
        label="label",
        entities="entities",
        contexts="contexts",
        split=None,
        fold=None,
        meta=["domain", "src"],
    ),
    # 最小保底（只要 text/label 存在就能跑；id 自动生成）
    dict(
        id=None,
        text="text",
        label="label",
        entities=None,
        contexts=None,
        split=None,
        fold=None,
        meta=None,
    ),
]
LABEL_MAP_DEFAULT = {"real": 0, "fake": 1, "true": 0, "false": 1}  # 兜底映射


def _build_loader_try_fields(fields):
    cfg = DatasetConfig(
        name="politifact",
        format="jsonl",
        path=str(DATA_DIR),
        splits=SPLITS,
        fields=FieldMap(**fields),
        label_map=LABEL_MAP_DEFAULT,
        lines=True,  # JSONL（逐行）
        encoding="utf-8",
    )
    return JSONLinesLoader(cfg)


def _probe_first_nonempty_split(loader: JSONLinesLoader):
    for sp, fn in SPLITS.items():
        p = DATA_DIR / fn
        if p.exists() and p.stat().st_size > 0:
            recs = loader.load_split(sp)
            if len(recs) > 0:
                return sp, recs
    return None, []


# ---------------------------------------------------------
# 真实数据通路（不假设具体 schema，自动选一个可用 FieldMap）
# ---------------------------------------------------------


def test_real_jsonl_dataset_smoke_and_contract(caplog):
    missing = [fn for fn in SPLITS.values() if not (DATA_DIR / fn).exists()]
    if len(missing) == len(SPLITS):
        pytest.skip(f"真实 JSONL 未就绪：{missing}")

    picked = None
    recs = []
    for fm in FIELDS_CANDIDATES:
        loader = _build_loader_try_fields(fm)
        sp, r = _probe_first_nonempty_split(loader)
        if r:
            picked = (fm, sp)
            recs = r
            break

    if not recs:
        pytest.skip(
            "未能用候选字段映射加载任何样本，请调整 FIELDS_CANDIDATES 以匹配真实字段。"
        )

    fm, sp = picked
    # —— 契约检查：NewsRecord v1 的关键属性存在且类型正确
    r0 = recs[0]
    assert isinstance(r0.id, str) and len(r0.id) > 0
    assert isinstance(r0.text, str) and r0.text.strip() != ""
    assert isinstance(r0.label, int)

    # 可选字段（若存在）也应被正确解析为目标类型
    if fm.get("entities"):
        assert isinstance(r0.entities, list)
    if fm.get("contexts"):
        assert isinstance(r0.contexts, dict)

    # Dataset 容器：缓存/复用
    ds = Dataset(_build_loader_try_fields(fm), preload=[sp])
    got = ds.get(sp)
    assert len(got) == len(recs)
    ds.close()


# ---------------------------------------------------------
# 合成数据通路（边界与回退：非法行/空文本/label 回退/字符串化 JSON）
# ---------------------------------------------------------


def test_jsonl_lines_edge_cases(tmp_path, caplog):
    # 写一个包含多种边界情况的 JSONL 文件
    good1 = {
        "id": "pf_001",
        "text": "  Barack Obama met Gary Jones in West Texas.  ",
        "label": "real",
        "entities": '["Q76","Q30"]',  # 字符串化 JSON
        "contexts": '{"Q76": ["Q4918","Q29552"]}',  # 字符串化 JSON
        "source": "pf",
        "note": "ok",
    }
    auto_id_row = {
        # 缺 id -> 应自动生成（基于 text blake2b + id_prefix）
        "text": "Gary Jones is a politician from Texas.",
        "label": "fake",
        "entities": "Q123",  # 单字符串 -> list["Q123"]
        "contexts": '{"Q123": ["Q1"]}',
        "source": "pf",
        "note": "auto_id",
    }
    unknown_label = {
        "text": "This line triggers label fallback.",
        "label": "unknown",  # 未映射 -> 警告 + 回退 0
        "source": "pf",
        "note": "fallback",
    }
    empty_text = {"id": "drop_me", "text": "   ", "label": "real"}  # 空文本 -> 丢弃

    f = tmp_path / "edge.jsonl"
    with f.open("w", encoding="utf-8") as w:
        w.write(json.dumps(good1, ensure_ascii=False) + "\n")
        w.write(json.dumps(auto_id_row, ensure_ascii=False) + "\n")
        w.write(json.dumps(unknown_label, ensure_ascii=False) + "\n")
        w.write(json.dumps(empty_text, ensure_ascii=False) + "\n")
        w.write("{this is not valid json}\n")  # 非法行 -> warning + skip

    cfg = DatasetConfig(
        name="pf-edge",
        format="jsonl",
        path=str(tmp_path),
        splits={"train": f.name},
        fields=FieldMap(
            id="id",
            text="text",
            label="label",
            entities="entities",
            contexts="contexts",
            split=None,
            fold=None,
            meta=["source", "note"],
        ),
        label_map={"real": 0, "fake": 1},
        id_prefix="pf_auto",
        lines=True,
    )
    loader = JSONLinesLoader(cfg)
    recs = loader.load_split("train")

    # 共有 5 行：1 空文本丢弃 + 1 非法行跳过 => 只剩 3 条
    assert len(recs) == 3

    # good1：字段/解析/元信息
    r0 = recs[0]
    assert r0.entities == ["Q76", "Q30"]
    assert r0.contexts == {"Q76": ["Q4918", "Q29552"]}
    assert r0.meta["dataset"] == "pf-edge"
    assert r0.meta["source"] == "pf" and r0.meta["note"] == "ok"
    assert r0.label == 0

    # auto_id：应被赋予自动 ID（带前缀）
    r1 = next(r for r in recs if r.meta.get("note") == "auto_id")
    assert r1.id.startswith("pf_auto_")
    assert r1.label == 1

    # unknown_label：回退为 0 且日志有 warning
    r2 = next(r for r in recs if r.meta.get("note") == "fallback")
    assert r2.label == 0
    assert any("Label not int-like; fallback to 0" in m for m in caplog.messages)

    # 非法 JSON 行：应被跳过并有 warning
    assert any("Skip invalid JSON line" in m for m in caplog.messages)


# ---------------------------------------------------------
# 其它快速回归：builder/可用 split/CSV 兜底（非必须，但提升覆盖率）
# ---------------------------------------------------------


def test_loader_from_config_builds_jsonl(tmp_path):
    f = tmp_path / "mini.jsonl"
    with f.open("w", encoding="utf-8") as w:
        w.write(json.dumps({"text": "hello", "label": "real"}) + "\n")
    cfg = {
        "name": "pf-mini",
        "format": "jsonl",
        "path": str(tmp_path),
        "splits": {"train": f.name},
        "fields": {"text": "text", "label": "label"},
        "label_map": {"real": 0, "fake": 1},
        "lines": True,
    }
    loader = loader_from_config(cfg)
    recs = loader.load_split("train")
    assert len(recs) == 1 and recs[0].label == 0


def test_available_splits_listing_jsonl(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_text(json.dumps({"text": "x", "label": 0}) + "\n", encoding="utf-8")
    cfg = DatasetConfig(
        name="pf",
        format="jsonl",
        path=str(tmp_path),
        splits={"train": f.name, "validation": f.name},
        fields=FieldMap(text="text", label="label"),
    )
    loader = JSONLinesLoader(cfg)
    assert set(loader.available_splits()) == {"train", "validation"}
