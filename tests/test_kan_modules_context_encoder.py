# tests/test_kan_modules_context_encoder.py
# -*- coding: utf-8 -*-
import pytest
import torch
from torch import nn, Tensor

# 被测模块
from kan.modules.context_encoder import (
    ContextEncoder,
    ContextEncoderConfig,
    build_context_encoder,
)

# 需要 BaseTextEncoder 接口定义以构造 Dummy
from kan.modules.text_encoder import BaseTextEncoder


# -------------------------------
# Utilities: Deterministic Dummy
# -------------------------------
class DummyTextEncoder(BaseTextEncoder):
    """
    一个可注入的极简文本编码器（不依赖 transformers/HF）。
    约定：
      - 输入：input_ids [N, Lt], attention_mask [N, Lt]
      - 输出：sequence_output [N, Lt, H] 与 pooled_output [N, H]
      - pooled_output = repeat(sum(input_ids * attention_mask), H次)
      - sequence_output 用于契约完整性（不是被 ContextEncoder 使用）
    """

    def __init__(self, hidden_size: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        # 造一个参数，用于测试 requires_grad 冻结逻辑
        self.dummy_weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, **batch):  # type: ignore[override]
        input_ids: Tensor = batch["input_ids"]
        attention_mask: Tensor = batch["attention_mask"]
        assert input_ids.dim() == 2 and attention_mask.shape == input_ids.shape

        # 标量：sum(token_id * mask) -> [N]
        scalars = (input_ids * attention_mask).sum(dim=1).to(dtype=torch.float32)

        # pooled_output: [N, H]，每列相同，便于后续 mean/max 验证
        pooled = scalars.unsqueeze(1).repeat(1, self.hidden_size)

        # sequence_output: [N, Lt, H]（不被 ContextEncoder 使用，仅契约完整）
        seq = torch.zeros(
            input_ids.size(0), input_ids.size(1), self.hidden_size, dtype=torch.float32
        )
        # 将每个时间步的“嵌入”设为 scalars，简化起见
        seq[:] = pooled.unsqueeze(1)

        return {
            "sequence_output": seq,
            "pooled_output": pooled,
            "attention_mask": attention_mask,
        }


def _align_valid_mask_shape(
    valid_ctx_mask: torch.Tensor, raw_ctx: torch.Tensor
) -> torch.Tensor:
    """
    将 valid_ctx_mask 的轴顺序对齐到 raw_ctx（期望 [B,E,Lc]）。
    若本身一致，直接返回；若是 [B,Lc,E]，做 permute；否则抛出友好错误。
    """
    if valid_ctx_mask.shape == raw_ctx.shape:
        return valid_ctx_mask
    # 常见错位：把 E 与 Lc 颠倒了
    if valid_ctx_mask.shape == (raw_ctx.shape[0], raw_ctx.shape[2], raw_ctx.shape[1]):
        return valid_ctx_mask.permute(0, 2, 1).contiguous()
    raise AssertionError(
        f"valid_ctx_mask shape {valid_ctx_mask.shape} not aligned with raw_ctx {raw_ctx.shape}"
    )


def _expected_ctx_be_from_raw(
    raw_ctx: torch.Tensor, attn_mask: torch.Tensor, inner_mode: str
) -> torch.Tensor:
    """
    raw_ctx: [B,E,Lc] （每条上下文已变成一个可验证标量；我们在 Dummy 里让 H 个维度相同，取其中一维）
    attn_mask: [B,E,Lc,Lt]（来自 batch）
    """
    # 仅把“有 token 的上下文”计为有效
    valid = attn_mask.sum(dim=-1) > 0  # [B,?,?]
    # 对齐轴到 [B,E,Lc]
    valid = _align_valid_mask_shape(valid, raw_ctx)

    if inner_mode == "mean":
        valid_f = valid.to(dtype=raw_ctx.dtype)
        denom = valid_f.sum(dim=2).clamp(min=1e-6).unsqueeze(-1)  # [B,E,1]
        # 为了广播安全，这里把 raw_ctx 显式扩一维再 squeeze 回来（避免奇怪的 stride 触发）
        num = (raw_ctx * valid_f).sum(dim=2, keepdim=True)  # [B,E,1]
        return (num / denom).squeeze(-1)  # [B,E]
    elif inner_mode == "max":
        very_small = torch.finfo(raw_ctx.dtype).min
        masked = raw_ctx.masked_fill(~valid, very_small)
        return masked.max(dim=2).values  # [B,E]
    else:
        raise ValueError(f"Unknown inner_mode: {inner_mode}")


# -------------------------------
# Fixtures
# -------------------------------
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_encoder():
    enc = DummyTextEncoder(hidden_size=7)
    return enc


def _mk_batch(B=2, E=3, Lc=2, Lt=4, device=torch.device("cpu")):
    """
    构造一个可控批次：
      - 第 0 个样本、第 0 个实体、第 0 条上下文为全 pad
      - 其余条目有不同的 token 值，便于 mean/max 可验证
    """
    # 形状：[B, E, Lc, Lt]
    ids = torch.zeros((B, E, Lc, Lt), dtype=torch.long, device=device)
    attn = torch.zeros_like(ids)

    # helper：写入一条上下文
    def set_ctx(b, e, j, toks):
        L = min(Lt, len(toks))
        ids[b, e, j, :L] = torch.tensor(toks[:L], dtype=torch.long, device=device)
        attn[b, e, j, :L] = 1

    # 构造一些确定性的 token
    # b=0
    # e=0: j=0 -> 全 pad, j=1 -> tokens [1,2,3,0]
    set_ctx(0, 0, 1, [1, 2, 3, 0])  # sum=6
    # e=1: j=0 -> [2,2,0,0], j=1 -> [5,0,0,0]
    set_ctx(0, 1, 0, [2, 2])
    set_ctx(0, 1, 1, [5])
    # e=2: j=0 -> [7,1,0,0], j=1 -> [1,1,1,1]
    set_ctx(0, 2, 0, [7, 1])
    set_ctx(0, 2, 1, [1, 1, 1, 1])  # sum=4

    # b=1
    # e=0: j=0 -> [3,0,0,0], j=1 -> [4,0,0,0]
    set_ctx(1, 0, 0, [3])
    set_ctx(1, 0, 1, [4])
    # e=1: j=0 -> 全 pad, j=1 -> [10,0,0,0]
    set_ctx(1, 1, 1, [10])
    # e=2: j=0 -> [1,2,3,4], j=1 -> 全 pad
    set_ctx(1, 2, 0, [1, 2, 3, 4])  # sum=10

    return {
        "context_input_ids": ids,
        "context_attention_mask": attn,
        # 不传 contexts_mask，触发由 attention_mask 推断
    }


# -------------------------------
# Tests: Construction & Freezing
# -------------------------------
def test_freeze_text_encoder_true_freezes_params(dummy_encoder, device):
    cfg = ContextEncoderConfig(
        text_encoder=None,
        freeze_text_encoder=True,
        inner_pooling="mean",
        entity_pooling="mean",
    )
    enc = ContextEncoder(cfg, text_encoder_module=dummy_encoder).to(device)
    # 冻结应已生效
    assert all(not p.requires_grad for p in enc.text_encoder.parameters())


def test_freeze_text_encoder_false_keeps_trainable(dummy_encoder, device):
    # 先将 Dummy 的参数置为可训练
    for p in dummy_encoder.parameters():
        p.requires_grad = True

    cfg = ContextEncoderConfig(
        text_encoder=None,
        freeze_text_encoder=False,
        inner_pooling="mean",
        entity_pooling="mean",
    )
    enc = ContextEncoder(cfg, text_encoder_module=dummy_encoder).to(device)
    assert any(p.requires_grad for p in enc.text_encoder.parameters())


def test_build_context_encoder_keeps_device(dummy_encoder, device, monkeypatch):
    # 用 build_context_encoder 走一遍（主要覆盖 device 分支与 logger）
    cfg = ContextEncoderConfig(text_encoder=None, device=str(device))

    # 注入构造函数，让工厂使用我们已有的 dummy
    def _ctor(cfg, text_encoder_module=None):
        return ContextEncoder(cfg, text_encoder_module=dummy_encoder)

    monkeypatch.setattr("kan.modules.context_encoder.ContextEncoder", ContextEncoder)
    # 直接构建
    enc = build_context_encoder(cfg, text_encoder_module=dummy_encoder)
    assert isinstance(enc, ContextEncoder)
    assert next(enc.parameters()).device.type == device.type


# -------------------------------
# Tests: Forward core (mean/max)
# -------------------------------
@pytest.mark.parametrize("inner_mode", ["mean", "max"])
@pytest.mark.parametrize("entity_mode", ["mean", "max"])
def test_forward_shapes_masks_and_pooling(
    inner_mode, entity_mode, dummy_encoder, device
):
    cfg = ContextEncoderConfig(
        text_encoder=None,
        freeze_text_encoder=True,
        inner_pooling=inner_mode,
        entity_pooling=entity_mode,
        return_raw_contexts=True,
    )
    enc = ContextEncoder(cfg, text_encoder_module=dummy_encoder).to(device)
    enc.eval()  # 关闭 dropout 影响

    batch = _mk_batch(device=device)
    out = enc(**batch)

    # 形状断言
    B, E, Lc, Lt = batch["context_input_ids"].shape
    H = dummy_encoder.hidden_size
    assert out["contexts_last_hidden"].shape == (B, E, H)
    assert out["contexts_pooled"].shape == (B, H)
    assert out["contexts_mask"].shape == (B, E)
    assert out["raw_contexts_last_hidden"].shape == (B, E, Lc, H)

    # 掩码推断：只要某实体至少有一条上下文非空 -> 1
    # b=0: e0 有一条有效 ->1, e1 有两条 ->1, e2 有两条 ->1
    # b=1: e0 两条->1, e1 一条->1, e2 一条->1
    expect_mask = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long, device=device)
    assert torch.equal(out["contexts_mask"], expect_mask)

    # 数值验证（按 Dummy 规则）
    # 为了便于验证，我们只取 H 的第一维（其它维度相同）
    ctx_be = out["contexts_last_hidden"][..., 0]  # [B, E]
    raw_ctx = out["raw_contexts_last_hidden"][..., 0]  # [B, E, Lc]

    # b=0:
    #  e=0: sums=[0,6] -> mean=3, max=6
    #  e=1: sums=[4,5] -> mean=4.5, max=5
    #  e=2: sums=[8,4] -> mean=6,   max=8
    sums_b0 = torch.tensor([[0, 6], [4, 5], [8, 4]], dtype=torch.float32, device=device)
    expect_b0_mean = sums_b0.mean(dim=1)
    expect_b0_max = sums_b0.max(dim=1).values

    # b=1:
    #  e=0: [3,4] -> mean=3.5, max=4
    #  e=1: [0,10] -> mean=5,  max=10
    #  e=2: [10,0] -> mean=5,  max=10
    sums_b1 = torch.tensor(
        [[3, 4], [0, 10], [10, 0]], dtype=torch.float32, device=device
    )
    expect_b1_mean = sums_b1.mean(dim=1)
    expect_b1_max = sums_b1.max(dim=1).values

    expect_ctx_be = _expected_ctx_be_from_raw(
        raw_ctx, batch["context_attention_mask"], inner_mode
    )
    assert torch.allclose(ctx_be, expect_ctx_be, atol=1e-6)

    # 跨实体聚合（到 [B, H]），由于每维相同，只需验证第一维
    pooled = out["contexts_pooled"][..., 0]  # [B]
    if entity_mode == "mean":
        expect_pooled_b0 = expect_ctx_be[0].mean()
        expect_pooled_b1 = expect_ctx_be[1].mean()
    else:  # max
        expect_pooled_b0 = expect_ctx_be[0].max()
        expect_pooled_b1 = expect_ctx_be[1].max()

    assert torch.allclose(
        pooled, torch.tensor([expect_pooled_b0, expect_pooled_b1], device=device)
    )


# -------------------------------
# Tests: All invalid contexts
# -------------------------------
def test_forward_all_padding_returns_zeros(dummy_encoder, device):
    cfg = ContextEncoderConfig(text_encoder=None, freeze_text_encoder=True)
    enc = ContextEncoder(cfg, text_encoder_module=dummy_encoder).to(device)
    enc.eval()

    B, E, Lc, Lt, H = 2, 2, 3, 4, dummy_encoder.hidden_size
    ids = torch.zeros((B, E, Lc, Lt), dtype=torch.long, device=device)
    attn = torch.zeros_like(ids)
    # 显式 contexts_mask 全 0
    ctxmask = torch.zeros((B, E, Lc), dtype=torch.long, device=device)
    out = enc(
        context_input_ids=ids,
        context_attention_mask=attn,
        contexts_mask=ctxmask,
    )

    assert out["contexts_last_hidden"].shape == (B, E, H)
    assert out["contexts_pooled"].shape == (B, H)
    assert out["raw_contexts_last_hidden"] is None  # 未开启 return_raw_contexts
    assert torch.count_nonzero(out["contexts_last_hidden"]) == 0
    assert torch.count_nonzero(out["contexts_pooled"]) == 0
    assert torch.equal(
        out["contexts_mask"], torch.zeros((B, E), dtype=torch.long, device=device)
    )


# -------------------------------
# Tests: batch_encode_contexts guard
# -------------------------------
def test_batch_encode_contexts_raises_on_non_hf_encoder(dummy_encoder, device):
    cfg = ContextEncoderConfig(text_encoder=None)
    enc = ContextEncoder(cfg, text_encoder_module=dummy_encoder).to(device)
    # 由于内部不是 HFTextEncoder，应抛出 RuntimeError（契约声明）
    with torch.no_grad():
        with pytest.raises(RuntimeError):
            enc.batch_encode_contexts([[["a"]]])


# -------------------------------
# Tests: Only valid entries encoded
# -------------------------------
def test_only_valid_contexts_are_encoded(dummy_encoder, device, monkeypatch):
    """
    验证：forward 只会对有效上下文构建子批并编码。
    我们用一个计数包装器，统计 DummyTextEncoder 的调用批大小是否等于有效条目数。
    """
    cfg = ContextEncoderConfig(text_encoder=None)
    enc = ContextEncoder(cfg, text_encoder_module=dummy_encoder).to(device)
    enc.eval()

    batch = _mk_batch(device=device)
    ids = batch["context_input_ids"]
    attn = batch["context_attention_mask"]
    # 有效上下文数 = attention_mask 在 Lt 维度 sum > 0 的条目数
    valid = (attn.sum(dim=-1) > 0).sum().item()

    call_sizes = []

    orig_forward = dummy_encoder.forward

    def wrap_forward(**kw):
        call_sizes.append(kw["input_ids"].shape[0])
        return orig_forward(**kw)

    monkeypatch.setattr(dummy_encoder, "forward", wrap_forward)
    _ = enc(**batch)

    # 只应被调用一次，且批大小等于有效条目总数
    assert len(call_sizes) == 1
    assert call_sizes[0] == valid
