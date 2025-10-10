# -*- coding: utf-8 -*-
"""
饱和式单元测试（EntityEncoder）/ Saturation tests for EntityEncoder
===================================================================
目标（Goal）
-----------
- 覆盖配置加载、设备放置、前向形状/掩码、聚合策略（mean/max）、
  Transformer 层行为（包含自动 head 调整）、预训练权重加载（.pt/.npy/不匹配异常）、
  训练态/冻结态、上下文缺省/部分缺省、错误分支（非法 pooling）。

注意（Notes）
------------
- 为了避免 CUDA/Dropout 带来的非确定性，测试统一在 CPU + eval() 下进行。
- 仅使用 PyTorch/pytest/临时文件 API，不依赖外部资源。
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import numpy as np
import pytest
import torch

# 项目内导入（由用户保证模块路径可用）
from kan.modules.entity_encoder import (
    EntityEncoder,
    EntityEncoderConfig,
    build_entity_encoder,
)


# ------------------------ Fixtures ------------------------
@pytest.fixture(autouse=True)
def _fix_seed():
    torch.manual_seed(42)


@pytest.fixture()
def device_cpu():
    return torch.device("cpu")


@pytest.fixture()
def small_cfg(device_cpu):
    return EntityEncoderConfig(
        vocab_size=128,
        embedding_dim=32,
        padding_idx=0,
        unk_idx=0,
        trainable=False,
        xformer_layers=1,
        xformer_heads=4,
        dropout=0.0,  # 便于数值可复现
        entity_pooling="mean",
        context_inner_pooling="mean",
        device=str(device_cpu),
    )


def _mk_batch(B=2, E=5, Lc=7, *, device=torch.device("cpu"), vmax: int = 120):
    # indices in [1, vmax-1]; keep 0 as padding explicitly
    entity_ids = torch.randint(low=1, high=vmax, size=(B, E), device=device)
    entity_ids[:, -1] = 0  # 保留一列 padding
    context_ids = torch.randint(low=1, high=vmax, size=(B, E, Lc), device=device)
    context_ids[:, :, -2:] = 0  # 每个实体末尾两个邻居 padding
    return {
        "entity_ids": entity_ids,
        "context_ids": context_ids,
    }


# ------------------------ Core shape & mask ------------------------
@pytest.mark.parametrize("inner_mode", ["mean", "max"])  # Lc 聚合
@pytest.mark.parametrize("entity_mode", ["mean", "max"])  # E 聚合
@pytest.mark.parametrize("with_context", [False, True])
@torch.no_grad()
def test_forward_shapes_masks_and_pooling(
    inner_mode, entity_mode, with_context, small_cfg, device_cpu
):
    cfg = small_cfg
    cfg = EntityEncoderConfig(
        **{
            **cfg.__dict__,
            "context_inner_pooling": inner_mode,
            "entity_pooling": entity_mode,
        }
    )
    enc = build_entity_encoder(cfg)
    enc.eval()

    batch = _mk_batch(device=device_cpu)
    if not with_context:
        batch.pop("context_ids")

    out = enc(**batch)

    B, E = batch["entity_ids"].shape
    D = cfg.embedding_dim

    # 形状断言
    assert out["entities_last_hidden"].shape == (B, E, D)
    assert out["entities_pooled"].shape == (B, D)
    assert out["entities_mask"].shape == (B, E)
    assert out["contexts_last_hidden"].shape == (B, E, D)
    assert out["contexts_pooled"].shape == (B, D)
    assert out["contexts_mask"].shape == (B, E)

    # 掩码应与 padding 对齐
    expect_ent_mask = (batch["entity_ids"] != cfg.padding_idx).long()
    assert torch.equal(out["entities_mask"], expect_ent_mask)

    # 无上下文时：contexts_last_hidden 应为零，mask 为 0；pooling 在 mean 下为零，在 max 下为 dtype 最小
    if not with_context:
        assert torch.allclose(
            out["contexts_last_hidden"], torch.zeros_like(out["contexts_last_hidden"])
        )
        assert torch.equal(out["contexts_mask"], torch.zeros_like(out["contexts_mask"]))
        if entity_mode == "mean":
            assert torch.allclose(
                out["contexts_pooled"], torch.zeros_like(out["contexts_pooled"])
            )
        else:  # max
            very_small = torch.finfo(out["contexts_pooled"].dtype).min
            assert torch.all(out["contexts_pooled"] == very_small)


# ------------------------ Encoders & heads ------------------------
@torch.no_grad()
def test_transformer_layers_zero_builds_identity_like(device_cpu):
    cfg = EntityEncoderConfig(
        vocab_size=64,
        embedding_dim=16,
        xformer_layers=0,
        xformer_heads=4,
        dropout=0.0,
        device=str(device_cpu),
    )
    enc = build_entity_encoder(cfg)
    enc.eval()
    assert enc.ent_encoder is None
    assert enc.ctx_encoder is None

    batch = _mk_batch(B=1, E=3, Lc=4, device=device_cpu, vmax=cfg.vocab_size)
    out = enc(**batch)
    # 能顺利前向并给出合理形状
    assert tuple(out["entities_last_hidden"].shape) == (1, 3, 16)


@torch.no_grad()
def test_auto_adjust_heads_when_not_divisible(device_cpu):
    # 96 不能整除 7，但能被 8 整除，应自动调整为 8
    cfg = EntityEncoderConfig(
        vocab_size=64,
        embedding_dim=96,
        xformer_layers=1,
        xformer_heads=7,
        dropout=0.0,
        device=str(device_cpu),
    )
    enc = build_entity_encoder(cfg)
    # 访问 TransformerEncoderLayer 的多头数
    layer0 = enc.ent_encoder.layers[0]
    assert getattr(layer0.self_attn, "num_heads", None) == 8


# ------------------------ Trainable / requires_grad ------------------------
@pytest.mark.parametrize("trainable", [False, True])
@torch.no_grad()
def test_trainable_flag_sets_requires_grad(trainable, device_cpu):
    cfg = EntityEncoderConfig(
        vocab_size=32,
        embedding_dim=8,
        trainable=trainable,
        xformer_layers=0,
        device=str(device_cpu),
    )
    enc = build_entity_encoder(cfg)
    assert enc.emb.weight.requires_grad is bool(trainable)


# ------------------------ Pretrained loading (.pt tensor / dict / .npy) ------------------------
@torch.no_grad()
def test_load_pretrained_tensor_pt(tmp_path: Path, device_cpu):
    vocab, dim = 50, 12
    weights = torch.randn(vocab, dim)
    path = tmp_path / "ent.pt"
    torch.save(weights, path)

    cfg = EntityEncoderConfig(
        vocab_size=vocab,
        embedding_dim=dim,
        embeddings_path=str(path),
        xformer_layers=0,
        device=str(device_cpu),
    )
    enc = build_entity_encoder(cfg)
    assert torch.allclose(enc.emb.weight.detach().cpu(), weights, atol=0, rtol=0)


@torch.no_grad()
def test_load_pretrained_state_dict_like(tmp_path: Path, device_cpu):
    vocab, dim = 40, 10
    weights = torch.randn(vocab, dim)
    path = tmp_path / "ent_sd.pt"
    torch.save({"weight": weights}, path)

    cfg = EntityEncoderConfig(
        vocab_size=vocab,
        embedding_dim=dim,
        embeddings_path=str(path),
        xformer_layers=0,
        device=str(device_cpu),
    )
    enc = build_entity_encoder(cfg)
    assert torch.allclose(enc.emb.weight.detach().cpu(), weights)


@torch.no_grad()
def test_load_pretrained_npy(tmp_path: Path, device_cpu):
    vocab, dim = 30, 6
    weights = np.random.randn(vocab, dim).astype("float32")
    path = tmp_path / "ent.npy"
    np.save(path, weights)

    cfg = EntityEncoderConfig(
        vocab_size=vocab,
        embedding_dim=dim,
        embeddings_path=str(path),
        xformer_layers=0,
        device=str(device_cpu),
    )
    enc = build_entity_encoder(cfg)
    assert np.allclose(enc.emb.weight.detach().cpu().numpy(), weights)


def test_load_pretrained_shape_mismatch_raises(tmp_path: Path, device_cpu):
    vocab, dim = 20, 5
    bad = torch.randn(vocab + 1, dim)
    path = tmp_path / "bad.pt"
    torch.save(bad, path)

    cfg = EntityEncoderConfig(
        vocab_size=vocab,
        embedding_dim=dim,
        embeddings_path=str(path),
        xformer_layers=0,
        device=str(device_cpu),
    )
    with pytest.raises(ValueError):
        _ = build_entity_encoder(cfg)


# ------------------------ Context semantics ------------------------
@torch.no_grad()
def test_return_raw_contexts_flag(device_cpu):
    cfg = EntityEncoderConfig(
        vocab_size=64,
        embedding_dim=16,
        return_raw_contexts=True,
        xformer_layers=0,
        device=str(device_cpu),
    )
    enc = build_entity_encoder(cfg)
    enc.eval()

    B, E, Lc = 2, 4, 3
    batch = _mk_batch(B=B, E=E, Lc=Lc, device=device_cpu, vmax=cfg.vocab_size)
    out = enc(**batch)
    assert "raw_contexts_last_hidden" in out
    assert out["raw_contexts_last_hidden"].shape == (B, E, Lc, cfg.embedding_dim)


@torch.no_grad()
def test_context_mask_propagation(device_cpu):
    cfg = EntityEncoderConfig(
        vocab_size=64,
        embedding_dim=16,
        xformer_layers=1,
        device=str(device_cpu),
    )
    enc = build_entity_encoder(cfg)
    enc.eval()

    B, E, Lc = 1, 5, 4
    entity_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device_cpu)
    # 构造上下文：第 2、5 个实体完全 padding，其余至少一个有效
    context_ids = torch.tensor(
        [
            [
                [10, 0, 0, 0],  # e1 -> 有效 1
                [0, 0, 0, 0],  # e2 -> 全 0
                [11, 12, 0, 0],  # e3 -> 有效 2
                [13, 14, 15, 0],  # e4 -> 有效 3
                [0, 0, 0, 0],  # e5 -> 全 0
            ]
        ],
        device=device_cpu,
    )

    out = enc(entity_ids=entity_ids, context_ids=context_ids)
    ctx_mask = out["contexts_mask"][0]
    # 仅 e2 与 e5 无上下文
    expect = torch.tensor([1, 0, 1, 1, 0], device=device_cpu)
    assert torch.equal(ctx_mask, expect)


# ------------------------ Error branches ------------------------
@torch.no_grad()
def test_invalid_entity_pooling_raises(device_cpu):
    cfg = EntityEncoderConfig(
        vocab_size=32,
        embedding_dim=8,
        entity_pooling="oops",  # 非法
        xformer_layers=0,
        device=str(device_cpu),
    )
    enc = build_entity_encoder(cfg)
    with pytest.raises(ValueError):
        _ = enc(entity_ids=torch.ones(1, 2, dtype=torch.long))


@torch.no_grad()
def test_invalid_context_inner_pooling_raises(device_cpu):
    cfg = EntityEncoderConfig(
        vocab_size=32,
        embedding_dim=8,
        context_inner_pooling="boom",  # 非法
        xformer_layers=0,
        device=str(device_cpu),
    )
    enc = build_entity_encoder(cfg)
    with pytest.raises(ValueError):
        _ = enc(
            entity_ids=torch.ones(1, 2, dtype=torch.long),
            context_ids=torch.ones(1, 2, 3, dtype=torch.long),
        )


# ------------------------ Device placement ------------------------
@torch.no_grad()
def test_build_on_cpu_device(device_cpu):
    cfg = EntityEncoderConfig(
        vocab_size=16,
        embedding_dim=4,
        xformer_layers=0,
        device=str(device_cpu),
    )
    enc = build_entity_encoder(cfg)
    # 任取一个参数验证在 CPU
    p = next(enc.parameters())
    assert p.device.type == "cpu"
