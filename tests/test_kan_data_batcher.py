# tests/test_kan_data_batcher.py
# -*- coding: utf-8 -*-
"""
测试目标（Test Goals）
- 覆盖 Text / Entity / Context 三分支的张量形状、掩码(mask)、截断(truncation)与 PAD/UNK/NONE 语义正确性。
- 覆盖可选依赖（transformers/tokenizer 与 vectorizer）开启/关闭时的行为，不依赖外网下载。
- 验证词表(EntityVocab/PropertyVocab)构造、取值(get)、长度(len)与保留位（PAD/UNK/NONE）语义。
- 验证 _parse_neighbor 的等价类与无效输入的“宽容丢弃（forgiving drop）”策略。
"""

import types
import pytest
import torch

from kan.data.batcher import (
    Batcher,
    BatcherConfig,
    TextConfig,
    EntityConfig,
    ContextConfig,
    EntityVocab,
    PropertyVocab,
    _parse_neighbor,
)
from kan.data.loaders import NewsRecord


# ---------------------------------------------------------------------------
# Test doubles: Dummy tokenizer & vectorizer (无外部依赖、可控输出)
# ---------------------------------------------------------------------------


class DummyTokenizer:
    """最简 HuggingFace 形状兼容分词器（shape-compatible tokenizer）."""

    def __init__(self, vocab_size: int = 30522):
        self.vocab_size = vocab_size

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        B = len(texts)
        # 固定生成 [B, max_length] 的张量，值域模拟词表范围
        input_ids = torch.randint(
            low=5, high=10, size=(B, max_length), dtype=torch.long
        )
        attention_mask = torch.ones((B, max_length), dtype=torch.long)
        token_type_ids = torch.zeros((B, max_length), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,  # 始终提供，Batcher 会根据配置决定是否带出
        }


class DummyVectorizer:
    """最简文本向量器（text vectorizer），输出 [B, D] 的 FloatTensor."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode_texts(self, texts):
        B = len(texts)
        # 生成确定性但非零向量，避免全零导致误判
        base = torch.arange(self.dim, dtype=torch.float32).unsqueeze(0)
        return base.repeat(B, 1) + 0.1


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def make_record(
    idx: int, text: str = None, ents=None, ctx=None, label: int = 0
) -> NewsRecord:
    """构造最小合法 NewsRecord。"""
    return NewsRecord(
        id=f"r{idx}",
        text=text if text is not None else f"text-{idx}",
        label=label,
        entities=ents if ents is not None else [],
        contexts=ctx if ctx is not None else {},
        meta={},
    )


# ---------------------------------------------------------------------------
# Unit tests: _parse_neighbor（解析器等价类/边界）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inp,exp",
    [
        ("Q1", ("Q1", None)),
        ("P31=Q5", ("Q5", "P31")),
        ("P31=foo", (None, None)),
        ("foo=Q5", (None, None)),
        ("", (None, None)),
        (None, (None, None)),
        (123, (None, None)),
    ],
)
def test_parse_neighbor_equivalence(inp, exp):
    assert _parse_neighbor(inp) == exp


# ---------------------------------------------------------------------------
# Unit tests: Vocab 基本语义
# ---------------------------------------------------------------------------


def test_entity_vocab_add_get_and_reserved():
    ev = EntityVocab()
    assert ev.PAD == 0 and ev.UNK == 1
    # 未登录 -> UNK
    assert ev.get("Q404") == 1
    # 插入后可检索
    i = ev.add("Q42")
    assert i == ev.get("Q42")
    # 空值/None -> UNK
    assert ev.get(None) == 1
    # 长度随插入单调递增
    assert len(ev) >= 3


def test_property_vocab_add_get_and_reserved():
    pv = PropertyVocab()
    assert pv.PAD == 0 and pv.NONE == 1
    # None -> NONE
    assert pv.get(None) == 1
    i = pv.add("P31")
    assert i == pv.get("P31")
    assert len(pv) >= 3


# ---------------------------------------------------------------------------
# build_vocabs 语义：三来源覆盖（entities / contexts 键 / contexts 值）
# ---------------------------------------------------------------------------


def test_build_vocabs_with_properties_and_without():
    # 数据：实体、带属性与不带属性的邻居
    recs = [
        make_record(1, ents=["Q1", "Q2"], ctx={"Q1": ["Q3", "P10=Q4"]}),
        make_record(2, ents=["Q2"], ctx={"Q2": ["P20=Q5"]}),
    ]

    # keep_properties=True：应收集到 P10/P20
    cfg_true = BatcherConfig(
        text=TextConfig(tokenizer_backend=None),
        context=ContextConfig(max_neighbors=8, keep_properties=True),
    )
    b_true = Batcher(cfg_true)
    b_true.build_vocabs(recs)
    assert b_true.entity_vocab.get("Q1") != 1
    assert b_true.entity_vocab.get("Q5") != 1
    assert b_true.property_vocab.get("P10") != 1
    assert b_true.property_vocab.get("P20") != 1

    # keep_properties=False：不应收集属性
    cfg_false = BatcherConfig(
        text=TextConfig(tokenizer_backend=None),
        context=ContextConfig(max_neighbors=8, keep_properties=False),
    )
    b_false = Batcher(cfg_false)
    b_false.build_vocabs(recs)
    # entity vocab 仍然应收集
    assert b_false.entity_vocab.get("Q5") != 1
    # property vocab 仅有 PAD/NONE
    assert len(b_false.property_vocab) == 2  # "<PAD>", "<NONE>"


# ---------------------------------------------------------------------------
# collate：最小用例（无文本分支），实体/上下文张量形状、mask、截断策略
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E,N", [(3, 2), (1, 1)])
def test_collate_minimal_entities_contexts(E, N):
    cfg = BatcherConfig(
        text=TextConfig(tokenizer_backend=None),  # 关闭文本分支
        entity=EntityConfig(max_entities=E),
        context=ContextConfig(max_neighbors=N, keep_properties=True),
        device="cpu",
    )
    b = Batcher(cfg)

    # 先用训练集构词表
    train = [
        make_record(1, ents=["Q1", "Q2", "Q3"], ctx={"Q1": ["Q10", "P1=Q11"]}),
        make_record(2, ents=["Q2"], ctx={"Q2": ["P2=Q12"]}),
    ]
    b.build_vocabs(train)

    # 测试批：包含未知实体（应回退 UNK=1）与过长截断
    test = [
        make_record(
            10, ents=["Q1", "QX", "Q3", "Q999"], ctx={"Q1": ["Q10", "P1=Q11", "Q13"]}
        ),
        make_record(11, ents=[], ctx={}),
    ]
    out = b.collate(test)

    # ent_ids/ent_mask 形状
    assert out["ent_ids"].shape == (2, E)
    assert out["ent_mask"].shape == (2, E)
    # 第一个样本：至多保留 E 个实体；未知 QX -> UNK=1；mask 对应有效位
    ids0 = out["ent_ids"][0]
    mask0 = out["ent_mask"][0]
    assert mask0.sum().item() == min(E, 3)  # 原始有效实体数为 3（Q1,QX,Q3），截断到 E
    # 上下文分支形状/掩码
    assert out["ctx_ids"].shape == (2, E, N)
    assert out["ctx_mask"].shape == (2, E, N)
    if cfg.context.keep_properties:
        assert out["ctx_prop"].shape == (2, E, N)
        # 有属性与无属性位置应区分：NONE(=1) 与 PAD(=0)
        # 若某实体没有邻居，则该 [N] 槽位应全 PAD/NONE，并且 mask 为 False
        assert out["ctx_mask"][1].sum().item() == 0  # 第二个样本无上下文


# ---------------------------------------------------------------------------
# collate：keep_properties 开/关 对 ctx_prop 键存在性的影响
# ---------------------------------------------------------------------------


def test_collate_properties_on_off_key_presence():
    rec = make_record(1, ents=["Q1"], ctx={"Q1": ["Q2", "P3=Q4"]})

    # on
    b1 = Batcher(
        BatcherConfig(
            text=TextConfig(tokenizer_backend=None),
            context=ContextConfig(max_neighbors=4, keep_properties=True),
        )
    )
    b1.build_vocabs([rec])
    out1 = b1.collate([rec])
    assert "ctx_prop" in out1
    assert out1["ctx_prop"].dtype == torch.long

    # off
    b2 = Batcher(
        BatcherConfig(
            text=TextConfig(tokenizer_backend=None),
            context=ContextConfig(max_neighbors=4, keep_properties=False),
        )
    )
    b2.build_vocabs([rec])
    out2 = b2.collate([rec])
    assert "ctx_prop" not in out2


# ---------------------------------------------------------------------------
# 文本分支：DummyTokenizer（含 token_type_ids）与 return_token_type_ids 分支
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("return_tti", [False, True])
def test_collate_text_tokenizer_branch(return_tti):
    dummy_tok = DummyTokenizer()
    cfg = BatcherConfig(
        text=TextConfig(
            tokenizer_backend=None,  # 禁止自动构造 HF；直接使用 dummy
            tokenizer_name=None,
            max_length=32,
            pad_to_max=True,
            truncation=True,
            return_token_type_ids=return_tti,
            vectorizer=None,
        ),
        entity=EntityConfig(max_entities=2),
        context=ContextConfig(max_neighbors=2, keep_properties=True),
        device="cpu",
    )
    b = Batcher(cfg, tokenizer=dummy_tok)
    recs = [make_record(1), make_record(2)]
    b.build_vocabs(recs)
    out = b.collate(recs)

    assert "text_tok" in out
    t = out["text_tok"]
    assert t["input_ids"].shape == (2, 32)
    assert t["attention_mask"].shape == (2, 32)
    if return_tti:
        assert "token_type_ids" in t and t["token_type_ids"].shape == (2, 32)
    else:
        assert "token_type_ids" not in t


# ---------------------------------------------------------------------------
# 文本分支：DummyVectorizer（Sentence Embedding 风格）
# ---------------------------------------------------------------------------


def test_collate_text_vectorizer_branch():
    dummy_vec = DummyVectorizer(dim=24)
    cfg = BatcherConfig(
        text=TextConfig(
            tokenizer_backend=None, vectorizer=types.SimpleNamespace()
        ),  # 占位
        entity=EntityConfig(max_entities=1),
        context=ContextConfig(max_neighbors=1, keep_properties=False),
        device="cpu",
    )
    # 直接注入实例，避免 build_vectorizer
    b = Batcher(cfg, vectorizer=dummy_vec)
    recs = [make_record(1), make_record(2), make_record(3)]
    out = b.collate(recs)
    assert "text_vec" in out
    assert out["text_vec"].shape == (3, 24)
    assert out["text_vec"].dtype == torch.float32


# ---------------------------------------------------------------------------
# 文本分支：Tokenizer + Vectorizer 同时开启
# ---------------------------------------------------------------------------


def test_collate_both_text_branches():
    dummy_tok = DummyTokenizer()
    dummy_vec = DummyVectorizer(dim=8)
    cfg = BatcherConfig(
        text=TextConfig(
            tokenizer_backend=None,  # 不自动拉取 HF
            tokenizer_name=None,
            max_length=16,
            pad_to_max=True,
            truncation=True,
            return_token_type_ids=False,
            vectorizer=types.SimpleNamespace(),
        ),
        entity=EntityConfig(max_entities=2),
        context=ContextConfig(max_neighbors=2, keep_properties=True),
        device="cpu",
    )
    b = Batcher(cfg, tokenizer=dummy_tok, vectorizer=dummy_vec)
    recs = [make_record(1, ents=["Q1"], ctx={"Q1": ["Q2"]})]
    b.build_vocabs(recs)
    out = b.collate(recs)
    assert "text_tok" in out and "text_vec" in out
    assert out["text_tok"]["input_ids"].shape == (1, 16)
    assert out["text_vec"].shape == (1, 8)


# ---------------------------------------------------------------------------
# 设备/数据类型一致性（device/dtype consistency）
# ---------------------------------------------------------------------------


def test_device_and_dtype_consistency_cpu():
    cfg = BatcherConfig(
        text=TextConfig(tokenizer_backend=None),
        entity=EntityConfig(max_entities=3),
        context=ContextConfig(max_neighbors=2, keep_properties=True),
        device="cpu",
    )
    b = Batcher(cfg)
    recs = [make_record(1, ents=["Q1"], ctx={"Q1": ["Q2"]})]
    b.build_vocabs(recs)
    out = b.collate(recs)
    # dtypes
    assert out["ent_ids"].dtype == torch.long
    assert out["ent_mask"].dtype == torch.bool
    assert out["ctx_ids"].dtype == torch.long
    assert out["ctx_mask"].dtype == torch.bool
    if "ctx_prop" in out:
        assert out["ctx_prop"].dtype == torch.long
    # device
    for k, v in out.items():
        if isinstance(v, dict):
            for t in v.values():
                assert t.device.type == "cpu"
        elif torch.is_tensor(v):
            assert v.device.type == "cpu"


# ---------------------------------------------------------------------------
# 确定性（determinism）：相同输入多次 collate 结果一致（对 dummy vec/tok 也固定化）
# ---------------------------------------------------------------------------


def test_determinism():
    dummy_tok = DummyTokenizer()
    dummy_vec = DummyVectorizer(dim=12)
    cfg = BatcherConfig(
        text=TextConfig(
            tokenizer_backend=None, max_length=8, vectorizer=types.SimpleNamespace()
        ),
        entity=EntityConfig(max_entities=2),
        context=ContextConfig(max_neighbors=2, keep_properties=True),
        device="cpu",
    )
    b = Batcher(cfg, tokenizer=dummy_tok, vectorizer=dummy_vec)
    recs = [
        make_record(1, ents=["Q1", "Q2"], ctx={"Q1": ["Q3", "P1=Q4"]}),
        make_record(2, ents=["Q2"], ctx={"Q2": ["Q5"]}),
    ]
    b.build_vocabs(recs)
    out1 = b.collate(recs)
    out2 = b.collate(recs)
    # 文本向量确定性
    assert torch.allclose(out1["text_vec"], out2["text_vec"])
    # 实体/上下文张量确定性
    assert torch.equal(out1["ent_ids"], out2["ent_ids"])
    assert torch.equal(out1["ent_mask"], out2["ent_mask"])
    assert torch.equal(out1["ctx_ids"], out2["ctx_ids"])
    assert torch.equal(out1["ctx_mask"], out2["ctx_mask"])
    if "ctx_prop" in out1:
        assert torch.equal(out1["ctx_prop"], out2["ctx_prop"])
