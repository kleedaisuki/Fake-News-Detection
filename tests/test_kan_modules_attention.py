# tests/test_kan_modules_attention.py
# -*- coding: utf-8 -*-
"""
@brief(中) KAN 注意力模块 NE / NE2C 的饱和式单元测试，覆盖正常路径与所有关键异常路径。
@brief(EN) Saturated unit tests for KAN attention modules NE / NE2C: normal paths and failure paths.

设计要点：
- 形状契约 / 掩码契约 / 权重形状 / all-padding 回退 / 池化稳定性 / 梯度可达性。
- 不依赖真实 EncoderOutput 类，使用 SimpleNamespace 模拟 last_hidden_state / pooled_state 字段。
"""

import types
import pytest
import torch

# ---- 待测模块导入 ------------------------------------------------------------
from kan.modules.attention.ne import NEAttention, NEAttentionOutput
from kan.modules.attention.ne2c import NE2CAttention, NE2CAttentionOutput


# ---- 小工具：构造 EncoderOutput-like -----------------------------------------
def mk_enc(last_hidden_state: torch.Tensor, pooled_state: torch.Tensor | None = None):
    """构造符合接口字段的对象；pooled_state 缺省为 tokens 的均值。"""
    if pooled_state is None:
        pooled_state = last_hidden_state.mean(dim=1)
    ns = types.SimpleNamespace()
    ns.last_hidden_state = last_hidden_state
    ns.pooled_state = pooled_state
    return ns


def assert_no_nan(*tensors):
    for t in tensors:
        assert torch.isfinite(t).all(), "Tensor has NaN/Inf"


# ---- 公共固定参数 ------------------------------------------------------------
SEED = 20251005
torch.manual_seed(SEED)
DEV = torch.device("cpu")


# =============================================================================
#                                NEAttention
# =============================================================================


@pytest.mark.parametrize(
    "B,Lt,Le,D,H",
    [
        (2, 5, 3, 16, 4),  # 小维度
        (3, 7, 9, 32, 8),  # 稍大
    ],
)
def test_ne_shapes_and_basic_forward(B, Lt, Le, D, H):
    news = torch.randn(B, Lt, D, device=DEV)
    ents = torch.randn(B, Le, D, device=DEV)
    m_news = torch.ones(B, Lt, dtype=torch.long, device=DEV)
    m_ent = torch.ones(B, Le, dtype=torch.long, device=DEV)

    mod = NEAttention(d_model=D, n_heads=H, dropout=0.0, use_bias=True).to(DEV)
    out: NEAttentionOutput = mod(
        news=mk_enc(news),
        entities=mk_enc(ents),
        news_mask=m_news,
        entity_mask=m_ent,
        return_weights=True,
    )

    assert isinstance(out, NEAttentionOutput)
    assert out.fused_states.shape == (B, Lt, D)
    assert out.pooled.shape == (B, D)
    assert out.attn_weights is not None and out.attn_weights.shape == (B, H, Lt, Le)
    assert_no_nan(out.fused_states, out.pooled, out.attn_weights)

    # 性质：当 news_mask 全 1 时，pooled 应为 fused 的按 L 维均值
    fused_mean = out.fused_states.mean(dim=1)
    torch.testing.assert_close(out.pooled, fused_mean, rtol=1e-5, atol=1e-6)


def test_ne_news_mask_all_zero_fallback_mean():
    B, Lt, Le, D, H = 2, 6, 4, 16, 4
    news = torch.randn(B, Lt, D)
    ents = torch.randn(B, Le, D)
    m_news = torch.zeros(B, Lt, dtype=torch.long)  # 全 0
    m_ent = torch.ones(B, Le, dtype=torch.long)

    mod = NEAttention(d_model=D, n_heads=H, dropout=0.0, use_bias=True)
    out = mod(
        news=mk_enc(news),
        entities=mk_enc(ents),
        news_mask=m_news,
        entity_mask=m_ent,
        return_weights=False,
    )
    # 池化应退化为 fused.mean(dim=1)
    fused_mean = out.fused_states.mean(dim=1)
    torch.testing.assert_close(out.pooled, fused_mean, rtol=1e-5, atol=1e-6)


def test_ne_entity_mask_all_padding_row_triggers_fallback_and_no_nan():
    B, Lt, Le, D, H = 3, 5, 4, 16, 4
    news = torch.randn(B, Lt, D)
    ents = torch.randn(B, Le, D)
    m_news = torch.ones(B, Lt, dtype=torch.long)

    # 构造：第 2 个样本实体全 padding，其余正常
    m_ent = torch.ones(B, Le, dtype=torch.long)
    m_ent[1].zero_()

    mod = NEAttention(d_model=D, n_heads=H, dropout=0.0, use_bias=True)
    out = mod(
        news=mk_enc(news),
        entities=mk_enc(ents),
        news_mask=m_news,
        entity_mask=m_ent,
        return_weights=True,
    )
    assert_no_nan(out.fused_states, out.pooled)
    # 第 0 / 2 样本的权重应有形状；第 1 个样本由于降级，不应产生 NaN
    assert torch.isfinite(out.attn_weights[1]).all()


def test_ne_attention_respects_key_mask_when_only_one_valid():
    B, Lt, Le, D, H = 1, 3, 5, 16, 4
    news = torch.randn(B, Lt, D)
    ents = torch.randn(B, Le, D)
    m_news = torch.ones(B, Lt, dtype=torch.long)

    # 仅保留实体 idx=2 有效
    m_ent = torch.zeros(B, Le, dtype=torch.long)
    m_ent[0, 2] = 1

    mod = NEAttention(d_model=D, n_heads=H, dropout=0.0, use_bias=True)
    out = mod(
        news=mk_enc(news),
        entities=mk_enc(ents),
        news_mask=m_news,
        entity_mask=m_ent,
        return_weights=True,
    )
    w = out.attn_weights  # [1,H,Lt,Le]
    # 有效位置为 idx=2，应获得接近 1 的权重总和
    mass_on_valid = (w[..., 2].sum() / (H * Lt)).detach().item()
    assert (
        mass_on_valid > 0.99
    ), f"Expected ~1.0 mass on the only valid entity, got {mass_on_valid}"


def test_ne_backward_gradients_flow():
    B, Lt, Le, D, H = 2, 4, 3, 8, 2
    news = torch.randn(B, Lt, D, requires_grad=True)
    ents = torch.randn(B, Le, D, requires_grad=True)
    m_news = torch.ones(B, Lt, dtype=torch.long)
    m_ent = torch.ones(B, Le, dtype=torch.long)

    mod = NEAttention(d_model=D, n_heads=H, dropout=0.0, use_bias=True)
    out = mod(
        news=mk_enc(news), entities=mk_enc(ents), news_mask=m_news, entity_mask=m_ent
    )
    loss = out.pooled.pow(2).mean()
    loss.backward()
    assert news.grad is not None and torch.isfinite(news.grad).all()
    assert ents.grad is not None and torch.isfinite(ents.grad).all()


# ---- 异常路径（严格验证） -----------------------------------------------------


def test_ne_raises_on_d_model_mismatch():
    B, Lt, Le, D, H = 2, 5, 4, 16, 4
    news = torch.randn(B, Lt, D)
    ents = torch.randn(B, Le, D + 1)  # 故意错维
    mod = NEAttention(d_model=D, n_heads=H)
    with pytest.raises(ValueError):
        _ = mod(news=mk_enc(news), entities=mk_enc(ents))


def test_ne_raises_on_mask_shape_mismatch_and_batch_mismatch():
    B, Lt, Le, D, H = 2, 5, 4, 16, 4
    news = torch.randn(B, Lt, D)
    ents = torch.randn(B + 1, Le, D)  # batch 不匹配
    mod = NEAttention(d_model=D, n_heads=H)
    with pytest.raises(ValueError):
        _ = mod(news=mk_enc(news), entities=mk_enc(ents))

    ents = torch.randn(B, Le, D)
    m_news_bad = torch.ones(B, Lt + 1, dtype=torch.long)
    with pytest.raises(ValueError):
        _ = mod(news=mk_enc(news), entities=mk_enc(ents), news_mask=m_news_bad)

    m_ent_bad = torch.ones(B, Le + 2, dtype=torch.long)
    with pytest.raises(ValueError):
        _ = mod(news=mk_enc(news), entities=mk_enc(ents), entity_mask=m_ent_bad)


# =============================================================================
#                               NE2CAttention
# =============================================================================


@pytest.mark.parametrize(
    "B,Lt,Le,Lc,D,H",
    [
        (2, 5, 3, 4, 16, 4),
        (1, 7, 5, 6, 32, 8),
    ],
)
def test_ne2c_shapes_masks_weights(B, Lt, Le, Lc, D, H):
    news = torch.randn(B, Lt, D)
    ents = torch.randn(B, Le, D)
    ctx = torch.randn(B, Le, Lc, D)

    m_news = torch.ones(B, Lt, dtype=torch.long)
    m_ent = torch.ones(B, Le, dtype=torch.long)
    m_ctx = torch.ones(B, Le, Lc, dtype=torch.long)

    mod = NE2CAttention(d_model=D, n_heads=H, dropout=0.0, use_bias=True)
    out: NE2CAttentionOutput = mod(
        news=mk_enc(news),
        entities=mk_enc(ents),
        contexts_last_hidden=ctx,
        news_mask=m_news,
        entity_mask=m_ent,
        contexts_mask=m_ctx,
        return_weights=True,
    )

    assert isinstance(out, NE2CAttentionOutput)
    assert out.fused_states.shape == (B, Lt, D)
    assert out.pooled.shape == (B, D)
    assert out.weights is not None
    assert out.weights["ne"].shape == (B, H, Lt, Le)
    assert out.weights["e2c"].shape == (B, H, 1, Le)
    assert_no_nan(out.fused_states, out.pooled, out.weights["ne"], out.weights["e2c"])

    # 池化不变量：当 news_mask 全 1 时，pooled 来源的 token 池化应与 fused.mean 接近（NE2C里还有残差混合，不做等式断言）
    fused_mean = out.fused_states.mean(dim=1)
    # 仅检查有限值与形状，由于 NE2C 的 pooled 进一步融合了 r_doc 与 ent_pooled
    assert_no_nan(fused_mean)


def test_ne2c_random_masks_no_nan_with_at_least_one_entity_and_context():
    """
    @note(中/EN)
      由于 NE2C 的 _attend 在“实体全被 mask”时 softmax([-inf,...]) 会产生 NaN，
      本测试确保每个 batch 至少一个实体有效；且对每个有效实体，至少一个上下文有效。
      This mirrors realistic masking while avoiding undefined all-masked rows.
    """
    B, Lt, Le, Lc, D, H = 3, 6, 4, 5, 16, 4
    news = torch.randn(B, Lt, D)
    ents = torch.randn(B, Le, D)
    ctx = torch.randn(B, Le, Lc, D)

    # 随机掩码
    m_news = torch.randint(0, 2, (B, Lt), dtype=torch.long)
    m_ent = torch.randint(0, 2, (B, Le), dtype=torch.long)
    m_ctx = torch.randint(0, 2, (B, Le, Lc), dtype=torch.long)

    # 修正：每个 batch 至少 1 个实体有效
    for b in range(B):
        if m_ent[b].sum().item() == 0:
            m_ent[b, torch.randint(0, Le, (1,))] = 1

    # 修正：对每个有效实体，至少 1 个上下文有效
    for b in range(B):
        for e in range(Le):
            if m_ent[b, e].item() == 1 and m_ctx[b, e].sum().item() == 0:
                m_ctx[b, e, torch.randint(0, Lc, (1,))] = 1

    mod = NE2CAttention(d_model=D, n_heads=H, dropout=0.0, use_bias=True)
    out = mod(
        news=mk_enc(news),
        entities=mk_enc(ents),
        contexts_last_hidden=ctx,
        news_mask=m_news,
        entity_mask=m_ent,
        contexts_mask=m_ctx,
        return_weights=True,
    )
    assert out.weights is not None and "ne" in out.weights and "e2c" in out.weights
    assert_no_nan(out.fused_states, out.pooled, out.weights["ne"], out.weights["e2c"])


def test_ne2c_attention_respects_entity_mask_when_only_one_valid():
    B, Lt, Le, Lc, D, H = 1, 4, 6, 3, 16, 4
    news = torch.randn(B, Lt, D)
    ents = torch.randn(B, Le, D)
    ctx = torch.randn(B, Le, Lc, D)

    m_news = torch.ones(B, Lt, dtype=torch.long)
    m_ent = torch.zeros(B, Le, dtype=torch.long)
    m_ent[0, 4] = 1  # 仅实体 idx=4 有效
    m_ctx = torch.ones(B, Le, Lc, dtype=torch.long)  # 上下文全有效

    mod = NE2CAttention(d_model=D, n_heads=H, dropout=0.0, use_bias=True)
    out = mod(
        news=mk_enc(news),
        entities=mk_enc(ents),
        contexts_last_hidden=ctx,
        news_mask=m_news,
        entity_mask=m_ent,
        contexts_mask=m_ctx,
        return_weights=True,
    )
    # e2c 的权重形状 [1,H,1,Le]，质量应集中到 idx=4
    w = out.weights["e2c"]
    mass_on_valid = w[..., 4].sum() / (H * 1)
    assert (
        float(mass_on_valid) > 0.99
    ), f"E2C weight mass not concentrated: {mass_on_valid}"


def test_ne2c_backward_gradients_flow():
    B, Lt, Le, Lc, D, H = 2, 5, 3, 4, 8, 2
    news = torch.randn(B, Lt, D, requires_grad=True)
    ents = torch.randn(B, Le, D, requires_grad=True)
    ctx = torch.randn(B, Le, Lc, D, requires_grad=True)

    m_news = torch.ones(B, Lt, dtype=torch.long)
    m_ent = torch.ones(B, Le, dtype=torch.long)
    m_ctx = torch.ones(B, Le, Lc, dtype=torch.long)

    mod = NE2CAttention(d_model=D, n_heads=H, dropout=0.0, use_bias=True)
    out = mod(
        news=mk_enc(news),
        entities=mk_enc(ents),
        contexts_last_hidden=ctx,
        news_mask=m_news,
        entity_mask=m_ent,
        contexts_mask=m_ctx,
        return_weights=False,
    )
    loss = out.pooled.pow(2).mean()
    loss.backward()
    for t in (news, ents, ctx):
        assert t.grad is not None and torch.isfinite(t.grad).all()


# ---- 异常路径（严格验证） -----------------------------------------------------


def test_ne2c_raises_on_dimension_and_mask_mismatches():
    B, Lt, Le, Lc, D, H = 2, 5, 3, 4, 16, 4
    news = torch.randn(B, Lt, D)
    ents = torch.randn(B, Le, D + 1)  # d_model 不匹配
    ctx = torch.randn(B, Le, Lc, D)

    mod = NE2CAttention(d_model=D, n_heads=H, dropout=0.0, use_bias=True)
    with pytest.raises(ValueError):
        _ = mod(news=mk_enc(news), entities=mk_enc(ents), contexts_last_hidden=ctx)

    # batch 不匹配
    ents = torch.randn(B + 1, Le, D)
    with pytest.raises(ValueError):
        _ = mod(news=mk_enc(news), entities=mk_enc(ents), contexts_last_hidden=ctx)

    # E != Le
    ents = torch.randn(B, Le, D)
    ctx_badE = torch.randn(B, Le + 1, Lc, D)
    with pytest.raises(ValueError):
        _ = mod(news=mk_enc(news), entities=mk_enc(ents), contexts_last_hidden=ctx_badE)

    # 各类 mask 形状错误
    m_news_bad = torch.ones(B, Lt + 1, dtype=torch.long)
    with pytest.raises(ValueError):
        _ = mod(
            news=mk_enc(news),
            entities=mk_enc(ents),
            contexts_last_hidden=ctx,
            news_mask=m_news_bad,
        )

    m_ent_bad = torch.ones(B, Le + 2, dtype=torch.long)
    with pytest.raises(ValueError):
        _ = mod(
            news=mk_enc(news),
            entities=mk_enc(ents),
            contexts_last_hidden=ctx,
            entity_mask=m_ent_bad,
        )

    m_ctx_bad = torch.ones(B, Le, Lc + 1, dtype=torch.long)
    with pytest.raises(ValueError):
        _ = mod(
            news=mk_enc(news),
            entities=mk_enc(ents),
            contexts_last_hidden=ctx,
            contexts_mask=m_ctx_bad,
        )
