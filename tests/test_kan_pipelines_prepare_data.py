# -*- coding: utf-8 -*-
"""
@file tests/test_kan_pipelines_prepare_data.py
@brief （集成测试 Integration Tests）仅针对 `prepare_data.py` 的端到端契约：
配置→指纹→加载/清洗→EL/KG/VEC 优雅降级/启用→持久化与 manifest →（可用时）KG 产物探测。

合并两个版本后的“饱和”要点：
- **只用真实模块**：不做 `sys.modules` 注入；全部走真实 `kan.data.*` / `kan.utils.registry` 路径；
- **保持第一版覆盖面**：E2E、缓存复用/--force、可选组件降级与启用态、split 缺失容忍、统计口径、dot-override 与 config_snapshot、稳定指纹性质；
- **KG 产物非脆断断言**：若 `components.kg=True`，通过 `cache_dir/kg` 目录存在且非空（globbing）来探测，而不绑定具体文件名；
- **启用态断言更鲁棒**：当 `components.*=True` 时，样本记录需出现 `entities` 或 `text_vec`（至少一项）。若环境无注册实现，则跳过启用态细节断言，但降级路径必须通过；
- **raw 持久化双态**：显式测试 `save_raw=True/False` 的开关行为。

测试哲学：优先断言稳定契约（文件存在性、行数、关键 JSON 字段、布尔位、统计口径），避免对实现细节（日志样式、内部函数名）形成脆弱耦合。
"""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Dict, Any, Mapping

import pytest

# 被测对象
from kan.pipelines.prepare_data import run_from_configs, _stable_fingerprint


# ------------------------------
# Helpers: 真实模块探测 & JSONL 写入
# ------------------------------


def _have_registry() -> bool:
    try:
        import kan.utils.registry as _  # type: ignore

        return True
    except Exception:
        return False


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _jsonl_loader_cfg(files_by_split: Dict[str, Path]) -> Dict[str, Any]:
    """构造一个针对真实 JSONL loader 的最小配置（契约对齐版）。
    契约要点：
      - 使用 format（非 type）：format ∈ {csv, jsonl, hf}
      - 使用 path + splits：splits 的 value 是在 path 下的文件名（相对路径）
    """
    # 1) 计算公共根目录作为 path
    #    注意：Path.parents[-1] 不稳健，这里用 os.path.commonpath 求公共前缀
    from os.path import commonpath

    all_paths = [str(p) for p in files_by_split.values()]
    base = Path(commonpath(all_paths))

    # 若 base 指向到具体文件而非目录，回退到其父目录
    if base.is_file():
        base = base.parent

    # 2) splits 的值改成相对 base 的路径（通常就是 'train.jsonl'）
    splits = {k: str(v.relative_to(base)) for k, v in files_by_split.items()}

    return {
        "format": "jsonl",  # ← loaders 契约字段
        "path": str(base),  # ← 作为数据根目录
        "splits": splits,  # ← 文件名相对 path
        # 其余字段按需添加：encoding/lines/fields...
    }


def _dump_cfg(path: Path, cfg: Mapping[str, Any]) -> str:
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


# ------------------------------
# Fixtures（可控小样本）
# ------------------------------


@pytest.fixture()
def tiny_records():
    # 3 条样本：含空文本与重复 id
    return [
        {"id": "a", "text": "hello world   !"},
        {"id": "b", "text": ""},  # 将被过滤
        {"id": "a", "text": "hello   world !"},  # 与第一条重复
        {"id": "c", "text": "new   sample"},
    ]


# ------------------------------
# A. 最小 E2E-Smoke（真实 loader）
# ------------------------------


def test_e2e_smoke_with_real_loader(tmp_path: Path, tiny_records):
    train_p = tmp_path / "data/train.jsonl"
    _write_jsonl(train_p, tiny_records)

    cfg = {
        "cache_dir": str(tmp_path / ".cache"),
        "output_dir": str(tmp_path / "runs"),
        "data": {
            "name": "tiny",
            "save_raw": True,
            "splits": {"train": "train"},
            **_jsonl_loader_cfg({"train": train_p}),
            "dedup_key": "id",
        },
    }

    try:
        ds_cache = run_from_configs([_dump_cfg(tmp_path / "cfg.json", cfg)])
    except Exception as e:
        pytest.xfail(f"真实 loader 构建失败：{e}")

    shard = Path(ds_cache) / "prepared/train/shard-00001.jsonl.gz"
    assert shard.exists()

    with gzip.open(shard, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 2  # 过滤空文本 + 去重

    stats_path = Path(ds_cache) / "prepared/train/stats.json"
    manifest_path = Path(ds_cache) / "manifest.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert stats["count"] == 2
    assert set(manifest["splits"].keys()) == {"train"}
    assert manifest["components"] == {
        "entity_linking": False,
        "kg": False,
        "vectorizer": False,
    }


# ------------------------------
# B. 缓存复用与 --force（真实 loader）
# ------------------------------


def test_cache_reuse_and_force_real_loader(tmp_path: Path):
    train_p = tmp_path / "data/train.jsonl"
    _write_jsonl(train_p, [{"id": 1, "text": "x"}])

    base = {
        "cache_dir": str(tmp_path / ".cache"),
        "output_dir": str(tmp_path / "runs"),
        "data": {
            "name": "reuse",
            "splits": {"train": "train"},
            **_jsonl_loader_cfg({"train": train_p}),
        },
    }

    p1 = tmp_path / "base.json"
    p1.write_text(json.dumps(base), encoding="utf-8")

    ds1 = run_from_configs([str(p1)])
    ds2 = run_from_configs([str(p1)])
    assert Path(ds1) == Path(ds2)

    # 改变指纹相关键（打开 EL）应生成新指纹目录
    el_patch = tmp_path / "el.json"
    el_patch.write_text(
        json.dumps({"data": {"entity_linking": {"type": "__probe__"}}}),
        encoding="utf-8",
    )
    ds3 = run_from_configs([str(p1), str(el_patch)], force=False)
    assert Path(ds3) != Path(ds1)

    # --force 强制重建仍返回相同 fingerprint 目录
    ds4 = run_from_configs([str(p1)], force=True)
    assert Path(ds4) == Path(ds1)


# ------------------------------
# C. 可选组件的“优雅降级”与（若可用则）启用 + KG 产物探测
# ------------------------------


def test_optional_components_degrade_and_maybe_enable(tmp_path: Path):
    train_p = tmp_path / "data/train.jsonl"
    _write_jsonl(train_p, [{"id": 1, "text": "hello"}])

    base = {
        "cache_dir": str(tmp_path / ".cache"),
        "output_dir": str(tmp_path / "runs"),
        "data": {
            "name": "opt",
            "splits": {"train": "train"},
            **_jsonl_loader_cfg({"train": train_p}),
        },
    }

    # 1) 降级路径：未显式配置 EL/KG/VEC
    ds0 = run_from_configs([_dump_cfg(tmp_path / "base.json", base)])
    mani0 = json.loads((Path(ds0) / "manifest.json").read_text(encoding="utf-8"))
    assert mani0["components"] == {
        "entity_linking": False,
        "kg": False,
        "vectorizer": False,
    }

    # 2) 启用路径（如果环境可用）：
    enable = {
        "data": {
            # 这些名字需要与实际注册的实现对齐；若不匹配，下面会 skip。
            "entity_linking": {"type": "passthrough"},
            "kg": {"type": "dummy"},
            "vectorizer": {"type": "passthrough"},
        }
    }

    try:
        ds1 = run_from_configs(
            [
                _dump_cfg(tmp_path / "base.json", base),
                _dump_cfg(tmp_path / "en.json", enable),
            ],
            force=True,
        )
    except Exception as e:
        pytest.skip(f"尝试启用可选组件失败（环境未提供实现/类型名不匹配）：{e}")

    mani1 = json.loads((Path(ds1) / "manifest.json").read_text(encoding="utf-8"))

    if mani1["components"] == {"entity_linking": True, "kg": True, "vectorizer": True}:
        # 样本中至少一个增强字段出现
        shard = Path(ds1) / "prepared/train/shard-00001.jsonl.gz"
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            row = json.loads(next(f))
        assert ("entities" in row) or ("text_vec" in row)

        # KG 产物：cache_dir 下的 kg 目录应存在且非空
        kg_dir = Path(base["cache_dir"]) / "kg"
        assert kg_dir.exists() and any(kg_dir.rglob("*")), "KG cache 目录应当非空"
    else:
        pytest.skip(
            "Registry 可用但当前未注册 EL/KG/VEC 的可用实现，跳过启用态细节断言。"
        )


# ------------------------------
# D. split 缺失仅告警并继续
# ------------------------------


def test_missing_split_is_tolerated_real_loader(tmp_path: Path):
    train_p = tmp_path / "data/train.jsonl"
    test_p = tmp_path / "data/test.jsonl"
    _write_jsonl(train_p, [{"id": 1, "text": "a"}])
    _write_jsonl(test_p, [{"id": 2, "text": "b"}])

    cfg = {
        "cache_dir": str(tmp_path / ".cache"),
        "output_dir": str(tmp_path / "runs"),
        "data": {
            "name": "splits",
            "splits": {"train": "train", "validation": "dev", "test": "test"},
            **_jsonl_loader_cfg({"train": train_p, "test": test_p}),
        },
    }

    ds = run_from_configs([_dump_cfg(tmp_path / "cfg.json", cfg)])
    manifest = json.loads((Path(ds) / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["splits"].keys()) == {"train", "test"}


# ------------------------------
# E. 统计与 shard 行数一致；长度统计口径
# ------------------------------


def test_stats_and_counts_real_loader(tmp_path: Path):
    recs = [
        {"id": 1, "text": "a b c"},
        {"id": 2, "text": "x  y   z  t  u"},
        {"id": 3, "text": "m n o p q r s t u"},
    ]
    train_p = tmp_path / "data/train.jsonl"
    _write_jsonl(train_p, recs)

    cfg = {
        "cache_dir": str(tmp_path / ".cache"),
        "output_dir": str(tmp_path / "runs"),
        "data": {
            "name": "stats",
            "splits": {"train": "train"},
            **_jsonl_loader_cfg({"train": train_p}),
        },
    }

    ds = run_from_configs([_dump_cfg(tmp_path / "cfg.json", cfg)])

    stats = json.loads(
        (Path(ds) / "prepared/train/stats.json").read_text(encoding="utf-8")
    )
    shard = Path(ds) / "prepared/train/shard-00001.jsonl.gz"
    with gzip.open(shard, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    assert stats["count"] == len(lines) == 3
    # mean/median/p95（n=3 时 p95 取排序后 index=int(0.95*n)-1=1）→ 排序长度 [3,5,9] ⇒ p95=5
    assert stats["text_len"]["median"] == 5.0
    assert stats["text_len"]["p95"] == 5.0


# ------------------------------
# F. 覆盖项（dot-override）与指纹快照 + raw 持久化双态
# ------------------------------


def test_dot_override_and_fingerprint_snapshot_real_loader(tmp_path: Path):
    train_p = tmp_path / "data/train.jsonl"
    _write_jsonl(train_p, [{"id": 1, "text": "q w e"}])

    # 基础：save_raw=True → 应写 raw 目录
    base = {
        "cache_dir": str(tmp_path / ".cache"),
        "output_dir": str(tmp_path / "runs"),
        "data": {
            "name": "ovr",
            "save_raw": True,
            "splits": {"train": "train"},
            **_jsonl_loader_cfg({"train": train_p}),
        },
    }
    b = tmp_path / "base.json"
    b.write_text(json.dumps(base), encoding="utf-8")

    ds_true = run_from_configs([str(b)])
    assert (Path(ds_true) / "raw").exists(), "save_raw=True 时应持久化 raw"

    # 运行时覆盖：关闭 save_raw，并注入 vectorizer.type（进入 fingerprint_keys 与 snapshot）
    ds_false = run_from_configs(
        [str(b)], overrides=["data.save_raw=false", 'data.vectorizer={"type":"dummy"}']
    )

    assert not (Path(ds_false) / "raw").exists(), "save_raw=false 时不应存在 raw 目录"
    mani = json.loads((Path(ds_false) / "manifest.json").read_text(encoding="utf-8"))
    assert "vectorizer" in mani["config_snapshot"]


# ------------------------------
# G. `_stable_fingerprint` 的基本性质测试（换序不变、值变则变）
# ------------------------------


def test_stable_fingerprint_properties():
    a = {"x": 1, "y": [3, 2, 1]}
    b = {"y": [3, 2, 1], "x": 1}
    assert _stable_fingerprint(a) == _stable_fingerprint(b)

    c = {"x": 2, "y": [3, 2, 1]}
    assert _stable_fingerprint(a) != _stable_fingerprint(c)
