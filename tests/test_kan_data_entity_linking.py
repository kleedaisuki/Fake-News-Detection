# -*- coding: utf-8 -*-
"""
tests/test_kan_data_entity_linking.py

@brief 测试 KAN 实体链接（Entity Linking, EL）流水线的端到端行为与边界条件。
@brief (EN) Saturation tests for the entity-linking pipeline.

设计要点 (Design points)
- 覆盖空输入、inplace 语义、缓存命中、大小写与多词匹配、阈值过滤、去重、追溯元信息与偏移。
- 不依赖外部服务；仅使用 DummyLinker 与 monkeypatch 注入的 TestLinker。
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List

import pytest

from kan.data.entity_linking import (
    ELConfig,
    link_records,
    build_linker,
    DummyLinker,
    EntityMention,
)
from kan.data.loaders import NewsRecord


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def lexicon_file(tmp_path: Path) -> Path:
    """构造一个小型词表（JSON）。大小写混合、多词条目、包含标点邻接场景。"""
    data = {
        "Barack Obama": "Q76",
        "Angela Merkel": "Q567",
        "United States": "Q30",
        "OpenAI": "Q24286590",
        # 特意包含大小写差异条目，用于 case_sensitive 测试
        "San Francisco": "Q62",
    }
    p = tmp_path / "lex.en.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture
def records() -> List[NewsRecord]:
    """两条样本：一条有人名/国家，一条无实体。"""
    return [
        NewsRecord(
            id="n1",
            text="Barack Obama met Angela Merkel in the United States. OpenAI rocks!",
            label=0,
            entities=[],
            contexts={},
            meta={},
        ),
        NewsRecord(
            id="n2",
            text="No entity here.",
            label=0,
            entities=[],
            contexts={},
            meta={},
        ),
    ]


# -----------------------------------------------------------------------------
# 基础功能：实体回填 + 追溯元信息
# -----------------------------------------------------------------------------


def test_link_records_populates_entities_and_meta(records, lexicon_file, tmp_path):
    cfg = ELConfig(
        backend="dummy",
        lexicon_path=str(lexicon_file),
        cache_dir=str(tmp_path / "el_cache"),
        case_sensitive=False,
        max_surface_len=3,
        inplace=False,
    )
    out = link_records(records, cfg)

    # 源对象不变（deepcopy）
    assert records[0].entities == []
    assert isinstance(out, list) and len(out) == 2

    # 第 1 条应匹配出 Obama/Merkel/United States/OpenAI
    ents = set(out[0].entities)
    assert ents == {"Q76", "Q567", "Q30", "Q24286590"}

    # 第 2 条无实体
    assert out[1].entities == []

    # 追溯元信息存在，且不破坏已有 meta
    el_meta = out[0].meta.get("el")
    assert isinstance(el_meta, dict)
    for k in ["backend", "version", "language", "threshold", "time", "mentions"]:
        assert k in el_meta

    # mentions 内容与偏移基本正确（抽查 OpenAI 与 United States）
    text = out[0].text
    m_openai = [m for m in el_meta["mentions"] if m["surface"] == "OpenAI"]
    assert m_openai and text[m_openai[0]["start"] : m_openai[0]["end"]] == "OpenAI"

    m_us = [m for m in el_meta["mentions"] if m["surface"] == "United States"]
    assert m_us and text[m_us[0]["start"] : m_us[0]["end"]] == "United States"


# -----------------------------------------------------------------------------
# inplace 语义：True 原地写入；False 深拷贝
# -----------------------------------------------------------------------------


def test_inplace_true_mutates_input(records, lexicon_file, tmp_path):
    cfg = ELConfig(
        backend="dummy",
        lexicon_path=str(lexicon_file),
        cache_dir=str(tmp_path / "el_cache"),
        inplace=True,
    )
    out = link_records(records, cfg, inplace=None)  # 使用 cfg.inplace
    assert out is records  # 同一列表对象
    assert set(records[0].entities) == {"Q76", "Q567", "Q30", "Q24286590"}


# -----------------------------------------------------------------------------
# 空输入处理
# -----------------------------------------------------------------------------


def test_empty_records_returns_empty_list(lexicon_file, tmp_path):
    cfg = ELConfig(
        backend="dummy",
        lexicon_path=str(lexicon_file),
        cache_dir=str(tmp_path / "el_cache"),
    )
    out = link_records([], cfg)
    assert out == []


# -----------------------------------------------------------------------------
# 大小写敏感性
# -----------------------------------------------------------------------------


def test_case_sensitivity(records, lexicon_file, tmp_path):
    # 对大小写不敏感：应匹配 "openai"（文本中是 "OpenAI"；这里换小写造句）
    recs = [
        NewsRecord(
            id="a1",
            text="openai is cool.",
            label=0,
            entities=[],
            contexts={},
            meta={},
        )
    ]
    cfg_insensitive = ELConfig(
        backend="dummy",
        lexicon_path=str(lexicon_file),
        cache_dir=str(tmp_path / "el_cache"),
        case_sensitive=False,
    )
    out = link_records(recs, cfg_insensitive)
    assert out[0].entities == ["Q24286590"]

    # 对大小写敏感：不匹配
    cfg_sensitive = ELConfig(
        backend="dummy",
        lexicon_path=str(lexicon_file),
        cache_dir=str(tmp_path / "el_cache2"),
        case_sensitive=True,
    )
    out2 = link_records(recs, cfg_sensitive)
    assert out2[0].entities == []


# -----------------------------------------------------------------------------
# 多词窗口长度限制（max_surface_len）
# -----------------------------------------------------------------------------


def test_max_surface_len_controls_multiword_match(tmp_path, lexicon_file):
    recs = [
        NewsRecord(
            id="b1",
            text="He was born in the United States of America.",
            label=0,
            entities=[],
            contexts={},
            meta={},
        )
    ]
    # 窗口=1：无法匹配 "United States"
    cfg_len1 = ELConfig(
        backend="dummy",
        lexicon_path=str(lexicon_file),
        cache_dir=str(tmp_path / "el_cache_len1"),
        max_surface_len=1,
    )
    out1 = link_records(recs, cfg_len1)
    assert out1[0].entities == []

    # 窗口=2：可以匹配
    cfg_len2 = ELConfig(
        backend="dummy",
        lexicon_path=str(lexicon_file),
        cache_dir=str(tmp_path / "el_cache_len2"),
        max_surface_len=2,
    )
    out2 = link_records(recs, cfg_len2)
    assert out2[0].entities == ["Q30"]


# -----------------------------------------------------------------------------
# 去重与重叠匹配
# -----------------------------------------------------------------------------


def test_deduplicate_overlapping_mentions(tmp_path):
    # 构造一个包含重复表面的词表与文本
    lex = {"OpenAI": "Q24286590", "OpenAI rocks": "Q24286590"}
    p = tmp_path / "lex_dup.json"
    p.write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")

    recs = [
        NewsRecord(
            id="c1",
            text="OpenAI rocks, OpenAI rocks!!!",
            label=0,
            entities=[],
            contexts={},
            meta={},
        )
    ]
    cfg = ELConfig(
        backend="dummy",
        lexicon_path=str(p),
        cache_dir=str(tmp_path / "el_cache_dup"),
        max_surface_len=2,
    )
    out = link_records(recs, cfg)
    # 实体去重：只有一个 QID
    assert out[0].entities == ["Q24286590"]

    # mentions 至少包含一个较长的 "OpenAI rocks" span
    spans = {(m["start"], m["end"], m["qid"]) for m in out[0].meta["el"]["mentions"]}
    text = recs[0].text
    assert any(text[s:e] == "OpenAI rocks" for (s, e, _) in spans)


# -----------------------------------------------------------------------------
# 缓存行为：命中后不再调用后端
# -----------------------------------------------------------------------------


def test_cache_hit_skips_linker_call(monkeypatch, lexicon_file, tmp_path, records):
    cache_dir = tmp_path / "el_cache_main"
    cfg = ELConfig(
        backend="dummy",
        lexicon_path=str(lexicon_file),
        cache_dir=str(cache_dir),
    )

    # 第一次：生成缓存
    _ = link_records(records, cfg)

    # 第二次：强制让 DummyLinker.link 抛错；若命中缓存则不会触发
    def boom(self, text: str):
        raise RuntimeError("Should not be called when cache hits!")

    monkeypatch.setattr(DummyLinker, "link", boom, raising=True)
    out = link_records(records, cfg)
    assert isinstance(out, list) and len(out) == 2  # 未抛错 → 命中缓存成功


# -----------------------------------------------------------------------------
# 关闭缓存（cache_dir=None）仍应正常工作
# -----------------------------------------------------------------------------


def test_no_cache_mode_works(monkeypatch, lexicon_file, tmp_path):
    recs = [
        NewsRecord(
            id="d1",
            text="Barack Obama met Angela Merkel.",
            label=0,
            entities=[],
            contexts={},
            meta={},
        )
    ]
    cfg = ELConfig(
        backend="dummy",
        lexicon_path=str(lexicon_file),
        cache_dir=None,  # 关闭缓存
    )

    calls = {"n": 0}

    def count_calls(self, text: str):
        calls["n"] += 1
        return [
            EntityMention(
                surface="Barack Obama", start=0, end=12, qid="Q76", score=1.0
            ),
            EntityMention(
                surface="Angela Merkel", start=18, end=31, qid="Q567", score=1.0
            ),
        ]

    monkeypatch.setattr(DummyLinker, "link", count_calls, raising=True)
    _ = link_records(recs, cfg)
    _ = link_records(recs, cfg)
    # 无缓存 → 两次都调用
    assert calls["n"] == 2


# -----------------------------------------------------------------------------
# 阈值过滤：通过注入低分后端验证（不修改生产码）
# -----------------------------------------------------------------------------


class LowScoreLinker(DummyLinker):
    """返回低分与高分混合的 mentions，用于测试 threshold 行为。"""

    name = "low_score_dummy"
    version = "x"

    def link(self, text: str):
        # 固定一高一低
        return [
            EntityMention(
                surface="OpenAI",
                start=text.find("OpenAI"),
                end=text.find("OpenAI") + 6,
                qid="Q24286590",
                score=0.4,
            ),
            EntityMention(
                surface="Obama",
                start=text.find("Obama"),
                end=text.find("Obama") + 5,
                qid="Q76",
                score=0.9,
            ),
        ]


def test_threshold_filters_low_scores(monkeypatch, tmp_path):
    recs = [
        NewsRecord(
            id="e1",
            text="OpenAI vs Obama",
            label=0,
            entities=[],
            contexts={},
            meta={},
        )
    ]
    cfg = ELConfig(
        backend="dummy",  # 名字随意，反正要被 monkeypatch 的工厂替换
        lexicon_path=None,
        cache_dir=str(tmp_path / "el_cache_thr"),
        threshold=0.5,
    )

    # 将工厂替换为返回 LowScoreLinker
    def fake_build_linker(cfg):
        return LowScoreLinker(cfg)

    monkeypatch.setattr(
        "kan.data.entity_linking.build_linker", fake_build_linker, raising=True
    )
    out = link_records(recs, cfg)
    # 低分 OpenAI 被过滤，仅剩 Obama
    assert out[0].entities == ["Q76"]


# -----------------------------------------------------------------------------
# 工厂异常：未知后端
# -----------------------------------------------------------------------------


def test_build_linker_unsupported_backend_raises():
    with pytest.raises(ValueError):
        _ = build_linker(ELConfig(backend="__unknown_backend__"))


# -----------------------------------------------------------------------------
# meta 合并：原有 meta 保持不丢失
# -----------------------------------------------------------------------------


def test_meta_merge_preserves_existing_fields(tmp_path, lexicon_file):
    recs = [
        NewsRecord(
            id="m1",
            text="Barack Obama says hi.",
            label=0,
            entities=[],
            contexts={},
            meta={"source": "unit-test", "split": "train"},
        )
    ]
    cfg = ELConfig(
        backend="dummy",
        lexicon_path=str(lexicon_file),
        cache_dir=str(tmp_path / "el_cache_meta"),
    )
    out = link_records(recs, cfg)
    meta = out[0].meta
    assert meta.get("source") == "unit-test"
    assert "el" in meta and isinstance(meta["el"], dict)


# -----------------------------------------------------------------------------
# mentions 偏移在包含标点的情况下也应一致
# -----------------------------------------------------------------------------


def test_offsets_respect_punctuation(tmp_path):
    lex = {"San Francisco": "Q62"}
    p = tmp_path / "lex_sf.json"
    p.write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")

    recs = [
        NewsRecord(
            id="o1",
            text="He moved to San Francisco, CA.",
            label=0,
            entities=[],
            contexts={},
            meta={},
        )
    ]
    cfg = ELConfig(
        backend="dummy",
        lexicon_path=str(p),
        cache_dir=str(tmp_path / "el_cache_punc"),
        max_surface_len=2,
    )
    out = link_records(recs, cfg)
    assert out[0].entities == ["Q62"]
    m = out[0].meta["el"]["mentions"][0]
    text = recs[0].text
    assert text[m["start"] : m["end"]] == "San Francisco"
