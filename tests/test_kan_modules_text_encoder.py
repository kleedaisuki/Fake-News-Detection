# tests/test_kan_modules_text_encoder.py
# -*- coding: utf-8 -*-
"""
饱和式单元测试：kan.modules.text_encoder
- 全面覆盖 pooling/编码/冻结参数/工厂函数/批处理 等分支
- 使用 Fake AutoModel/AutoTokenizer，避免网络与大模型依赖
"""

import types
import pytest
import torch
from torch import nn, Tensor

# 导入待测模块
from kan.modules.text_encoder import (
    TextEncoderConfig,
    HFTextEncoder,
    build_text_encoder,
)

# ------------------------------
# Fakes for transformers backend
# ------------------------------


class _DummyConfig:
    def __init__(self, hidden_size=4):
        self.hidden_size = hidden_size


class _DummyModel(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.config = _DummyConfig(hidden_size=hidden_size)
        # 新增：一个用不到的线性层，仅用于提供参数
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, return_dict=True
    ):
        B, L = input_ids.shape
        H = self.config.hidden_size
        base = (
            torch.arange(L, dtype=torch.float32)
            .view(1, L, 1)
            .expand(B, L, H)
            .contiguous()
        )
        # 注意：不使用 self.dense，保持可预期输出
        return types.SimpleNamespace(last_hidden_state=base)


class _DummyTokenizer:
    """模拟 HF tokenizer：
    - 当 padding='max_length' 时，pad 到给定 max_length
    - 当 padding=False 时，pad 到 batch 内最长（HF 在 return_tensors='pt' 下的行为）
    - token 长度规则：len(text) + (2 if add_special_tokens else 0)，再截断到 max_length
    """

    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(
        self,
        texts,
        padding=False,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_tensors="pt",
    ):
        assert return_tensors == "pt"
        lengths = []
        base_extra = 2 if add_special_tokens else 0
        for t in texts:
            L = len(t) + base_extra
            if truncation:
                L = min(L, max_length)
            lengths.append(L)

        if padding == "max_length":
            tgt = max_length
        else:
            tgt = max(lengths)

        input_ids = []
        attention_mask = []
        token_type_ids = []

        for L in lengths:
            ids = [1] * L + [0] * (tgt - L)  # 1 for real token, 0 for pad
            am = [1] * L + [0] * (tgt - L)
            tt = [0] * tgt
            input_ids.append(ids)
            attention_mask.append(am)
            token_type_ids.append(tt)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }


# ------------------------------
# Pytest fixtures
# ------------------------------


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """强制 _auto_device 返回 CPU，避免 CI/GPU 差异。"""
    monkeypatch.setattr(HFTextEncoder, "_auto_device", lambda self: torch.device("cpu"))
    yield


@pytest.fixture
def patched_transformers(monkeypatch):
    """把 AutoTokenizer/AutoModel.from_pretrained 指向我们的 Fakes。"""
    # 打补丁到模块命名空间（kan.modules.text_encoder）使用处
    import kan.modules.text_encoder as m

    class _AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _DummyModel(hidden_size=4)

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _DummyTokenizer()

    monkeypatch.setattr(m, "AutoModel", _AutoModel, raising=True)
    monkeypatch.setattr(m, "AutoTokenizer", _AutoTokenizer, raising=True)
    yield


# ------------------------------
# Helper for expected pooling
# ------------------------------


def _expected_mean_from_mask(attn: Tensor) -> float:
    # 非 padding 位置的 l 索引均匀平均；mask 的第 l 位=1 则计入
    idx = torch.arange(attn.shape[-1], dtype=torch.float32)
    val = (idx * attn.to(torch.float32)).sum().item()
    denom = max(1.0, attn.sum().item())
    return val / denom


def _expected_max_from_mask(attn: Tensor) -> float:
    idx = torch.arange(attn.shape[-1], dtype=torch.float32)
    if attn.sum().item() <= 0:
        return float("-inf")
    return idx[attn.bool()].max().item()


# ------------------------------
# Tests
# ------------------------------


def test_from_pretrained_and_forward_cls(patched_transformers):
    cfg = TextEncoderConfig(
        model_name_or_path="dummy/bert",
        max_length=8,
        pooling="cls",  # 取 l=0
        trainable=False,
        pad_to_max_length=True,
    )
    enc = build_text_encoder(cfg)
    batch = enc.batch_encode(["ab", "cdef"], device="cpu")
    out = enc(**batch)

    assert out["sequence_output"].shape == (2, 8, 4)
    assert out["pooled_output"].shape == (2, 4)
    # cls pooling ⇒ 第 0 个 token 的值为 0
    assert torch.allclose(out["pooled_output"], torch.zeros(2, 4))

    # 参数是否冻结
    assert all(p.requires_grad is False for p in enc.model.parameters())


def test_trainable_true_sets_requires_grad(patched_transformers):
    cfg = TextEncoderConfig(pooling="cls", trainable=True)
    enc = build_text_encoder(cfg)
    assert any(p.requires_grad for p in enc.model.parameters())
    assert all(p.requires_grad is True for p in enc.model.parameters())


@pytest.mark.parametrize("pooling", ["mean", "max"])
def test_pooling_mean_max_values(patched_transformers, pooling):
    cfg = TextEncoderConfig(max_length=10, pooling=pooling, pad_to_max_length=False)
    enc = build_text_encoder(cfg)
    batch = enc.batch_encode(["aaaa", "bbb"], device="cpu")  # 可变长→按 batch 最长对齐
    out = enc(**batch)
    attn = out["attention_mask"]  # [B, L]

    if pooling == "mean":
        exp0 = _expected_mean_from_mask(attn[0])
        exp1 = _expected_mean_from_mask(attn[1])
    else:
        exp0 = _expected_max_from_mask(attn[0])
        exp1 = _expected_max_from_mask(attn[1])

    # 每个隐藏维都应为同一个标量（我们的 dummy 模型把 l 复制到所有 H 维）
    assert torch.allclose(out["pooled_output"][0], torch.full((4,), float(exp0)))
    assert torch.allclose(out["pooled_output"][1], torch.full((4,), float(exp1)))


def test_mean_or_max_requires_attention_mask_error(patched_transformers):
    cfg = TextEncoderConfig(pooling="mean")
    enc = build_text_encoder(cfg)

    # 构造一个没有 attention_mask 的 batch
    batch = {
        "input_ids": torch.ones(2, 5, dtype=torch.long),
        # "attention_mask": 缺失
    }
    with pytest.raises(ValueError):
        enc(**batch)


def test_unknown_pooling_raises(patched_transformers, monkeypatch):
    # 通过直接改 cfg.pooling 绕过类型检查（pytest 环境不做 mypy）
    cfg = TextEncoderConfig()
    object.__setattr__(cfg, "pooling", "UNKNOWN")  # type: ignore

    enc = build_text_encoder(cfg)
    batch = enc.batch_encode(["hi"], device="cpu")
    with pytest.raises(ValueError):
        enc(**batch)


def test_batch_encode_padding_modes(patched_transformers):
    # pad_to_max_length=True → 长度固定为 max_length
    cfg = TextEncoderConfig(max_length=12, pad_to_max_length=True)
    enc = build_text_encoder(cfg)
    batch = enc.batch_encode(["short", "looooooong text"], device="cpu")
    assert batch["input_ids"].shape[1] == 12

    # pad_to_max_length=False → pad 到 batch 内最长（且受 truncation 限制）
    cfg2 = TextEncoderConfig(max_length=10, pad_to_max_length=False)
    enc2 = build_text_encoder(cfg2)
    batch2 = enc2.batch_encode(["a", "abcdefghiJK"], device="cpu")  # 第二个会被截到 10
    assert batch2["input_ids"].shape[1] == 10


def test_forward_output_contract_includes_attention_mask(patched_transformers):
    cfg = TextEncoderConfig()
    enc = build_text_encoder(cfg)
    out = enc(**enc.batch_encode(["x", "y"], device="cpu"))
    assert set(out.keys()) == {"sequence_output", "pooled_output", "attention_mask"}


def test_encode_end_to_end_batched_progress(patched_transformers, capsys):
    cfg = TextEncoderConfig(pooling="cls")
    enc = build_text_encoder(cfg)
    texts = ["t{}".format(i) for i in range(5)]
    embs = enc.encode(texts, device="cpu", batch_size=2, progress=True)
    assert embs.shape == (5, enc.hidden_size)

    # 简单检查有进度输出（不做严格匹配）
    captured = capsys.readouterr().out
    assert "progress" in captured


def test_build_text_encoder_delegates_to_from_pretrained(
    monkeypatch, patched_transformers
):
    calls = {}

    def fake_from_pretrained(cls, cfg, trust_remote_code=False):
        calls["cfg"] = cfg
        # 直接返回一个真实的 HFTextEncoder（依赖我们已 patch 的 Auto*）
        tok = _DummyTokenizer()
        mdl = _DummyModel()
        return HFTextEncoder(model=mdl, tokenizer=tok, cfg=cfg)

    monkeypatch.setattr(
        HFTextEncoder, "from_pretrained", classmethod(fake_from_pretrained)
    )
    cfg = TextEncoderConfig(model_name_or_path="any")
    enc = build_text_encoder(cfg)
    assert calls["cfg"] is cfg
    assert isinstance(enc, HFTextEncoder)
