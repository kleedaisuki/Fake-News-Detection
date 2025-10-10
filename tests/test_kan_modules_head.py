# -*- coding: utf-8 -*-
"""
饱和式单元测试（Head） / Saturation Tests for Head
==================================================

这些测试覆盖 kan.modules.head 中 Head/HeadConfig/build_head 的主要功能与边界条件。
- 兼顾 concat/sum/mean 融合（fusion），含 proj_dim 的必要性
- 回归/单标签/多标签三类损失（loss）
- LazyLinear 的延迟绑定与 bias 行为
- LayerNorm 的延迟创建
- 分支选择 use_p/use_q/use_r 与错误输入
- 工厂函数 build_head

测试依赖：pytest, torch
"""
from __future__ import annotations

import math
import types
import pytest
import torch
from torch import nn

# 被测模块
from kan.modules.head import Head, HeadConfig, build_head


# ---------------------------- 工具函数 / Helpers ----------------------------


def _devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


def _mk_inputs(B=4, Hp=8, Hq=6, Hr=10, device=torch.device("cpu")):
    torch.manual_seed(42)
    p = torch.randn(B, Hp, device=device)
    q = torch.randn(B, Hq, device=device)
    r = torch.randn(B, Hr, device=device)
    return p, q, r


# -------------------------------- 基础形状 ---------------------------------
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("hidden_sizes", [[], [32], [64, 32]])
@pytest.mark.parametrize("fusion", ["concat"])  # 其他融合在专门用例测试
def test_forward_shapes_concat(fusion, hidden_sizes, device):
    B, Hp, Hq, Hr, C = 3, 5, 7, 9, 4
    p, q, r = _mk_inputs(B, Hp, Hq, Hr, device)

    cfg = HeadConfig(
        num_labels=C,
        fusion=fusion,
        proj_dim=None,
        hidden_sizes=hidden_sizes,
        dropout=0.0,
        layernorm=False,
        bias=True,
    )
    head = build_head(cfg).to(device)
    head.eval()

    out = head(p=p, q=q, r=r)
    assert out["logits"].shape == (B, C)
    assert out["fused"].shape[-1] == Hp + Hq + Hr


# ------------------------------- 分支选择与错误 ------------------------------
@pytest.mark.parametrize("device", _devices())
def test_missing_branch_raises(device):
    B, Hp = 2, 6
    p, _, _ = _mk_inputs(B, Hp, 4, 4, device)

    # use_p=True 但不传 p
    cfg = HeadConfig(use_p=True, use_q=False, use_r=False, dropout=0.0, layernorm=False)
    head = Head(cfg).to(device)
    head.eval()
    with pytest.raises(ValueError):
        _ = head()

    # use_q=True 但不传 q
    cfg = HeadConfig(use_p=False, use_q=True, use_r=False, dropout=0.0, layernorm=False)
    head = Head(cfg).to(device)
    head.eval()
    with pytest.raises(ValueError):
        _ = head(p=p)

    # use_r=True 但不传 r
    cfg = HeadConfig(use_p=False, use_q=False, use_r=True, dropout=0.0, layernorm=False)
    head = Head(cfg).to(device)
    head.eval()
    with pytest.raises(ValueError):
        _ = head(p=p)

    # 三个都关闭时报错
    cfg = HeadConfig(
        use_p=False, use_q=False, use_r=False, dropout=0.0, layernorm=False
    )
    head = Head(cfg).to(device)
    head.eval()
    with pytest.raises(ValueError):
        _ = head()


# ------------------------------ sum/mean 融合逻辑 -----------------------------
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("fusion", ["sum", "mean"])
def test_sum_mean_need_equal_dims_or_proj(fusion, device):
    B, Hp, Hq, Hr, C = 2, 8, 6, 10, 3
    p, q, r = _mk_inputs(B, Hp, Hq, Hr, device)

    # 不投影时维度不同，应报错
    cfg = HeadConfig(
        num_labels=C,
        fusion=fusion,
        proj_dim=None,
        hidden_sizes=[16],
        dropout=0.0,
        layernorm=False,
    )
    head = Head(cfg).to(device)
    head.eval()
    with pytest.raises(ValueError):
        _ = head(p=p, q=q, r=r)

    # 设置统一投影后应当通过，且 fused 等于逐分支投影后的 sum/mean
    cfg = HeadConfig(
        num_labels=C,
        fusion=fusion,
        proj_dim=12,
        hidden_sizes=[16],
        dropout=0.0,
        layernorm=False,
    )
    head = Head(cfg).to(device)
    head.eval()

    out = head(p=p, q=q, r=r)
    fused = out["fused"]

    # 由于 eval 且 dropout=0 且无 LN，可直接对比数值
    z_p = head.proj_p(p)
    z_q = head.proj_q(q)
    z_r = head.proj_r(r)
    if fusion == "sum":
        expected = z_p + z_q + z_r
    else:
        expected = (z_p + z_q + z_r) / 3.0
    assert torch.allclose(fused, expected, atol=1e-6)


# ------------------------------ LayerNorm 延迟创建 ----------------------------
@pytest.mark.parametrize("device", _devices())
def test_layernorm_is_created_on_first_forward(device):
    p, q, r = _mk_inputs(device=device)
    cfg = HeadConfig(layernorm=True, dropout=0.0)
    head = Head(cfg).to(device)
    head.eval()

    assert getattr(head, "_post_fuse_ln") is None
    _ = head(p=p, q=q, r=r)
    assert head._post_fuse_ln is not None
    # 作为子模块注册名应为 post_fuse_ln
    assert isinstance(dict(head.named_modules()).get("post_fuse_ln"), nn.LayerNorm)


# ------------------------------ hidden_sizes 边界 ----------------------------
@pytest.mark.parametrize("device", _devices())
def test_hidden_sizes_empty_means_direct_linear(device):
    p, q, r = _mk_inputs(device=device)
    cfg = HeadConfig(hidden_sizes=[], dropout=0.0, layernorm=False)
    head = Head(cfg).to(device)
    head.eval()

    _ = head(p=p, q=q, r=r)
    # 不应存在激活层
    acts = [m for m in head.mlp.modules() if isinstance(m, (nn.GELU, nn.ReLU))]
    assert len(acts) == 0


# ------------------------------ 三种任务的损失 -------------------------------
@pytest.mark.parametrize("device", _devices())
def test_loss_single_label_ce_with_smoothing_and_weights(device):
    B, C = 5, 3
    p, q, r = _mk_inputs(B=B, device=device)
    labels = torch.randint(0, C, (B,), device=device)

    cfg = HeadConfig(
        num_labels=C,
        problem_type="single_label_classification",
        label_smoothing=0.1,
        class_weights=[1.0, 2.0, 3.0],
        dropout=0.0,
        layernorm=False,
    )
    head = Head(cfg).to(device)
    head.train()

    out = head(p=p, q=q, r=r, labels=labels)
    assert out["loss"].ndim == 0 and out["loss"].requires_grad
    # 权重应已绑定到与 logits 相同的 device/dtype
    assert head._ce_weight is not None
    assert head._ce_weight.device == out["logits"].device
    assert head._ce_weight.dtype == out["logits"].dtype


@pytest.mark.parametrize("device", _devices())
def test_loss_multi_label_bce(device):
    B, C = 4, 5
    p, q, r = _mk_inputs(B=B, device=device)
    labels = torch.randint(0, 2, (B, C), device=device).float()

    cfg = HeadConfig(
        num_labels=C,
        problem_type="multi_label_classification",
        dropout=0.0,
        layernorm=False,
    )
    head = Head(cfg).to(device)
    head.train()

    out = head(p=p, q=q, r=r, labels=labels)
    assert out["logits"].shape == (B, C)
    assert out["loss"].ndim == 0


@pytest.mark.parametrize("device", _devices())
def test_loss_regression_numlabels1_squeeze(device):
    B = 6
    p, q, r = _mk_inputs(B=B, device=device)
    labels = torch.randn(B, device=device)

    cfg = HeadConfig(
        num_labels=1,
        problem_type="regression",
        dropout=0.0,
        layernorm=False,
    )
    head = Head(cfg).to(device)
    head.train()

    out = head(p=p, q=q, r=r, labels=labels)
    assert out["logits"].shape == (B, 1)
    assert out["loss"].ndim == 0


# ------------------------------ bias 标志与 Lazy 绑定 -------------------------
@pytest.mark.parametrize("device", _devices())
def test_bias_flag_false_applies_to_all_linear_layers(device):
    p, q, r = _mk_inputs(device=device)
    cfg = HeadConfig(bias=False, dropout=0.0, layernorm=False)
    head = Head(cfg).to(device)
    head.eval()

    _ = head(p=p, q=q, r=r)  # 触发 LazyLinear 绑定

    linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
    assert len(linears) > 0
    assert all((m.bias is None) or (m.bias is False) for m in linears)


@pytest.mark.parametrize("device", _devices())
def test_grad_flow(device):
    B, C = 4, 2
    p, q, r = _mk_inputs(B=B, device=device)
    labels = torch.randint(0, C, (B,), device=device)

    cfg = HeadConfig(num_labels=C, dropout=0.0, layernorm=False)
    head = Head(cfg).to(device)
    head.train()

    out = head(p=p, q=q, r=r, labels=labels)
    out["loss"].backward()

    # 至少有一层权重拿到梯度
    got_grad = any((p.grad is not None) for p in head.parameters())
    assert got_grad


# ------------------------------ 仅启用指定分支 -------------------------------
@pytest.mark.parametrize("device", _devices())
def test_use_only_p_branch(device):
    B, Hp = 3, 7
    p, _, _ = _mk_inputs(B=B, Hp=Hp, Hq=1, Hr=1, device=device)

    cfg = HeadConfig(
        use_p=True,
        use_q=False,
        use_r=False,
        proj_dim=None,
        dropout=0.0,
        layernorm=False,
    )
    head = Head(cfg).to(device)
    head.eval()

    out = head(p=p)
    # 未投影、无 LN、无 dropout，fused 应与 p 数值相同
    assert out["fused"].shape[-1] == Hp
    assert torch.allclose(out["fused"], p, atol=1e-6)


# ------------------------------ 无需投影时的 Identity ------------------------
def test_proj_identity_when_proj_dim_none():
    cfg = HeadConfig(proj_dim=None)
    head = Head(cfg)
    assert isinstance(head.proj_p, nn.Identity)
    assert isinstance(head.proj_q, nn.Identity)
    assert isinstance(head.proj_r, nn.Identity)


# ------------------------------ 工厂函数 ------------------------------------
def test_build_head_returns_head():
    head = build_head(HeadConfig())
    assert isinstance(head, Head)


# ------------------------------ 错误分支 ------------------------------------
def test_invalid_activation_raises():
    with pytest.raises(ValueError):
        _ = HeadConfig(activation="swish")  # 构建没报错，但模块中会检查
        # 触发 _activation_layer 时抛异常
        Head(HeadConfig(activation="swish"))


def test_invalid_fusion_raises_on_forward():
    p, q, r = _mk_inputs()
    cfg = HeadConfig(fusion="bad", dropout=0.0, layernorm=False)
    head = Head(cfg)
    with pytest.raises(ValueError):
        _ = head(p=p, q=q, r=r)


# ------------------------------ 推断时 dropout 关闭 --------------------------
@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("fusion", ["sum", "mean"])  # 复用上一测试逻辑
def test_eval_mode_disables_dropout_in_fused(fusion, device):
    B = 2
    p, q, r = _mk_inputs(B=B, device=device)

    cfg = HeadConfig(fusion=fusion, proj_dim=16, dropout=0.75, layernorm=False)
    head = Head(cfg).to(device)
    head.eval()  # 关闭 dropout

    out1 = head(p=p, q=q, r=r)
    out2 = head(p=p, q=q, r=r)
    # eval 模式下 fused 应可复现
    assert torch.allclose(out1["fused"], out2["fused"], atol=1e-6)


# ------------------------------ 数值稳定性（基本） ---------------------------
@pytest.mark.parametrize("device", _devices())
def test_forward_is_finite(device):
    p, q, r = _mk_inputs(device=device)
    cfg = HeadConfig(dropout=0.0)
    head = Head(cfg).to(device)
    head.eval()

    out = head(p=p, q=q, r=r)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["fused"]).all()
