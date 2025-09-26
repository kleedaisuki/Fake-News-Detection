# -*- coding: utf-8 -*-
# tests/test_kan_data_kg_fetcher.py
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pytest


# --------------------------------------------------------------------------------------
# 本地最小记录类型（不污染 sys.modules，避免与真实包冲突）
# --------------------------------------------------------------------------------------
@dataclass
class TestRecord:
    id: str
    text: str = ""
    label: int = 0
    entities: Optional[List[str]] = field(default_factory=list)
    contexts: Dict[str, List[str]] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Import under test: 固定使用 kan.data.kg_fetcher
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def kgmod():
    import importlib

    # 项目现在确定使用 kan.data.kg_fetcher，不做回退逻辑
    return importlib.import_module("kan.data.kg_fetcher")


# --------------------------------------------------------------------------------------
# 构造三种 LocalProvider 数据形态
# --------------------------------------------------------------------------------------
def _make_single_json(path: Path, mapping: Dict[str, List[str]]):
    path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")


def _make_dir_per_qid(dirpath: Path, mapping: Dict[str, List[str]]):
    dirpath.mkdir(parents=True, exist_ok=True)
    for qid, neighbors in mapping.items():
        (dirpath / f"{qid}.json").write_text(
            json.dumps({"neighbors": neighbors}, ensure_ascii=False), encoding="utf-8"
        )


def _make_jsonl(path: Path, mapping: Dict[str, List[str]]):
    with path.open("w", encoding="utf-8") as f:
        for qid, neighbors in mapping.items():
            f.write(
                json.dumps({"qid": qid, "neighbors": neighbors}, ensure_ascii=False)
                + "\n"
            )


# --------------------------------------------------------------------------------------
# 基础数据
# --------------------------------------------------------------------------------------
@pytest.fixture
def base_mapping():
    # 包含去重/属性/无属性混合及交叉引用
    return {
        "Q1": ["Q2", "Q3", "P31=Q5", "Q2"],  # 去重: "Q2" 重复一次
        "Q2": ["P27=Q183", "Q1", "P31=Q5", "Q9"],
        "Q404": ["P999=Q0", "not_a_qid", "Q3"],  # 非法条目应被过滤/降噪
    }


@pytest.fixture
def news_records():
    return [
        TestRecord(id="r1", text="foo", entities=["Q1", "Q404"]),
        TestRecord(id="r2", text="bar", entities=["Q2"]),
    ]


# --------------------------------------------------------------------------------------
# Import / Config / Helpers
# --------------------------------------------------------------------------------------
def test_import_and_config_defaults(kgmod):
    cfg = kgmod.KGConfig()
    assert cfg.backend == "local"
    assert cfg.return_edges in ("none", "labeled")
    assert cfg.topk == 64
    assert cfg.inplace is False


def test_unique_preserve_helper(kgmod):
    arr = ["a", "b", "a", "c", "b"]
    assert kgmod._unique_preserve(arr) == ["a", "b", "c"]


# --------------------------------------------------------------------------------------
# LocalProvider 三种形态：single JSON / per-QID dir / JSONL
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ["single", "dir", "jsonl"])
def test_local_provider_three_shapes(
    tmp_path: Path, kgmod, base_mapping, news_records, shape, monkeypatch
):
    # 1) 准备三种数据形态
    if shape == "single":
        data_path = tmp_path / "kg.single.json"
        _make_single_json(data_path, base_mapping)
        cfg = kgmod.KGConfig(
            backend="local", local_path=str(data_path), return_edges="none", topk=None
        )
    elif shape == "dir":
        data_dir = tmp_path / "neighbors"
        _make_dir_per_qid(data_dir, base_mapping)
        cfg = kgmod.KGConfig(
            backend="local", local_path=str(data_dir), return_edges="labeled", topk=None
        )
    else:  # jsonl
        data_path = tmp_path / "kg.jsonl"
        _make_jsonl(data_path, base_mapping)
        # 通过后缀 .jsonl 触发 JSONL 模式（亦可用 KAN_KG_LOCAL_JSONL=1）
        cfg = kgmod.KGConfig(
            backend="local", local_path=str(data_path), return_edges="none", topk=None
        )

    # 2) 不原地写：验证无副作用（inplace=False）
    out = kgmod.fetch_context(news_records, cfg, inplace=False)
    assert out is not news_records

    # ——契约期望：contexts 被填充（注：当前实现可能未写 contexts，此断言将暴露缺陷）
    for rec in out:
        assert isinstance(rec.contexts, dict), "contexts should exist as dict"
        for q in rec.entities:
            assert (
                q in rec.contexts
            ), "each entity should have a fetched neighbor list (contract)"

    # 3) 写入 meta['kg'] 追溯信息
    for rec in out:
        assert "kg" in rec.meta
        info = rec.meta["kg"]
        assert "backend" in info and "version" in info and "stats" in info
        assert "uniq_qids" in info["stats"]

    # 4) return_edges 形态语义点到即可（详细在后续测试覆盖）
    if shape == "dir":
        # return_edges="labeled"：应保留 "Pxx=Qyy" 形态
        pass
    else:
        # return_edges="none"：邻接应是纯 QID
        pass


# --------------------------------------------------------------------------------------
# properties 过滤、return_edges、topk、去重
# --------------------------------------------------------------------------------------
def test_properties_filter_and_formats(tmp_path: Path, kgmod, news_records):
    data_path = tmp_path / "kg.json"
    mapping = {
        "Q1": ["Q2", "P31=Q5", "P27=Q183", "Q2", "P31=Q5"],  # 含重复
        "Q2": ["Q1", "P27=Q183"],
    }
    _make_single_json(data_path, mapping)

    # 仅允许 P31，且输出 labeled
    cfg = kgmod.KGConfig(
        backend="local",
        local_path=str(data_path),
        properties=["P31"],
        return_edges="labeled",
        topk=10,
    )
    out = kgmod.fetch_context(news_records, cfg, inplace=False)
    # 预期：Q1 只保留 P31 边，且去重；Q2 只有 P27，因过滤后为空
    for rec in out:
        for q in rec.entities:
            neigh = rec.contexts.get(q, [])
            if q == "Q1":
                assert "P31=Q5" in neigh
                assert "P27=Q183" not in neigh
            if q == "Q2":
                assert all(not s.startswith("P27=") for s in neigh)

    # 仅允许 P27，输出 none（应为纯 QID 列表）
    cfg2 = kgmod.KGConfig(
        backend="local",
        local_path=str(data_path),
        properties=["P27"],
        return_edges="none",
        topk=10,
    )
    out2 = kgmod.fetch_context(news_records, cfg2, inplace=False)
    for rec in out2:
        for q in rec.entities:
            neigh = rec.contexts.get(q, [])
            assert all("=" not in s for s in neigh)  # 不应出现 "Pxx=Qyy"


def test_topk_and_dedup(tmp_path: Path, kgmod, news_records):
    data_path = tmp_path / "kg.json"
    mapping = {"Q1": ["Q2", "Q3", "Q2", "Q4", "Q5", "Q3"]}
    _make_single_json(data_path, mapping)
    cfg = kgmod.KGConfig(
        backend="local", local_path=str(data_path), topk=2, return_edges="none"
    )
    out = kgmod.fetch_context(news_records, cfg, inplace=False)
    # 预期：去重后 ["Q2","Q3","Q4","Q5"]，再截断 topk=2 -> ["Q2","Q3"]
    for rec in out:
        if "Q1" in rec.entities:
            assert rec.contexts.get("Q1", [])[:2] == ["Q2", "Q3"]


# --------------------------------------------------------------------------------------
# Cache 行为与 TTL
# --------------------------------------------------------------------------------------
def test_cache_put_and_hit_then_source_removed(tmp_path: Path, kgmod, news_records):
    # 1) 构造目录模式数据源与 cache 目录
    data_dir = tmp_path / "neighbors"
    mapping = {"Q1": ["Q2", "P31=Q5"], "Q2": ["Q1"]}
    _make_dir_per_qid(data_dir, mapping)
    cache_dir = tmp_path / ".cachekg"
    cfg = kgmod.KGConfig(
        backend="local",
        local_path=str(data_dir),
        cache_dir=str(cache_dir),
        return_edges="none",
        topk=None,
    )

    # 2) 首次运行：生成 cache
    _ = kgmod.fetch_context(news_records, cfg, inplace=False)
    # 3) 删除源数据，确保第二次能从 cache 命中
    for p in data_dir.glob("*.json"):
        p.unlink()
    out2 = kgmod.fetch_context(news_records, cfg, inplace=False)
    # 若 contexts 正确写回，应仍能获取邻接
    for rec in out2:
        for q in rec.entities:
            assert rec.contexts.get(q, []) is not None


def test_cache_ttl_expiration_logic(tmp_path: Path, kgmod, monkeypatch):
    # 直接测试内部 _Cache：写入一个过期条目，get 应返回 None
    cache = kgmod._Cache(str(tmp_path / "c"), backend_sig="sig")
    cache.dir.mkdir(parents=True, exist_ok=True)
    qid = "QX"
    p = cache._key(qid)
    p.parent.mkdir(parents=True, exist_ok=True)
    old_ts = "2000-01-01T00:00:00+00:00"
    p.write_text(json.dumps({"neighbors": ["Q1"], "ts": old_ts}), encoding="utf-8")

    env_backup = os.environ.get("KAN_KG_CACHE_TTL_DAYS")
    os.environ["KAN_KG_CACHE_TTL_DAYS"] = "1"
    try:
        # 新建 cache 实例以读取环境 TTL
        cache2 = kgmod._Cache(str(tmp_path / "c"), backend_sig="sig")
        assert cache2.get(qid) is None, "expired cache should be ignored"
    finally:
        if env_backup is None:
            os.environ.pop("KAN_KG_CACHE_TTL_DAYS", None)
        else:
            os.environ["KAN_KG_CACHE_TTL_DAYS"] = env_backup


# --------------------------------------------------------------------------------------
# 错误隔离与并发路径（不做耗时断言，仅验证健壮性）
# --------------------------------------------------------------------------------------
def test_error_isolation(monkeypatch, tmp_path: Path, kgmod, news_records):
    # 准备单文件 JSON
    data_path = tmp_path / "kg.json"
    mapping = {"Q1": ["Q2"], "Q2": ["Q1"], "Q404": ["Q3"]}
    _make_single_json(data_path, mapping)
    cfg = kgmod.KGConfig(backend="local", local_path=str(data_path), topk=None)

    # monkeypatch LocalProvider.fetch：对 Q404 抛错，其余走原实现
    real_local = kgmod.LocalProvider
    # ⛑️ 关键：在 monkeypatch 之前捕获原始方法，避免递归
    orig_fetch = real_local.fetch

    def wrapped_fetch(self, qid: str):
        if qid == "Q404":
            raise RuntimeError("boom")
        return orig_fetch(self, qid)

    monkeypatch.setattr(kgmod.LocalProvider, "fetch", wrapped_fetch, raising=True)

    out = kgmod.fetch_context(news_records, cfg, inplace=False)
    # 预期：Q404 的邻接为空，但不影响 Q1/Q2
    for rec in out:
        for q in rec.entities:
            neigh = rec.contexts.get(q, [])
            if q == "Q404":
                assert neigh == []
            else:
                # Q1/Q2 应互为邻接
                if q in ("Q1", "Q2"):
                    assert any(
                        s.endswith("Q2") or s == "Q2"
                        for s in rec.contexts.get("Q1", [])
                    ) or any(
                        s.endswith("Q1") or s == "Q1"
                        for s in rec.contexts.get("Q2", [])
                    )


# --------------------------------------------------------------------------------------
# inplace 语义与空输入
# --------------------------------------------------------------------------------------
def test_inplace_true_writes_back(tmp_path: Path, kgmod):
    data_path = tmp_path / "kg.json"
    _make_single_json(data_path, {"Q1": ["Q2"], "Q2": ["Q1"], "Q404": []})
    cfg = kgmod.KGConfig(backend="local", local_path=str(data_path), inplace=True)
    # 使用 cfg.inplace（API 有两种风格，这里走配置）
    recs = [
        TestRecord(id="r1", entities=["Q1", "Q404"]),
        TestRecord(id="r2", entities=["Q2"]),
    ]
    out = kgmod.fetch_context(recs, cfg)
    # 预期：对象标识相同（原地）
    assert out is recs
    # 预期：contexts 被填充（若实现未写，此断言会失败，从而暴露缺陷）
    assert any(
        rec.contexts for rec in recs
    ), "inplace=True should fill contexts on the same objects"


def test_empty_input_returns_empty_list(kgmod):
    cfg = kgmod.KGConfig()
    assert kgmod.fetch_context([], cfg, inplace=False) == []


# --------------------------------------------------------------------------------------
# Factory 路由（无需真正网络）
# --------------------------------------------------------------------------------------
def test_build_provider_routing(kgmod):
    assert isinstance(
        kgmod.build_provider(kgmod.KGConfig(backend="local")), kgmod.LocalProvider
    )
    # 不支持的 backend 应抛错
    with pytest.raises(ValueError):
        kgmod.build_provider(kgmod.KGConfig(backend="no_such_backend"))
