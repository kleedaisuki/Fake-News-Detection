# -*- coding: utf-8 -*-
import types
import torch
import pytest

# 导入被测模块
import kan.data.vectorizer as V


def test_config_defaults_and_post_init():
    cfg = V.VectorizerConfig()
    assert cfg.backend == "hf"
    assert cfg.hf_tokenizer_kwargs == {}
    assert cfg.hf_model_kwargs == {}


def test_l2_normalize_idempotent_on_unit_vectors():
    x = torch.tensor([[3.0, 4.0]], dtype=torch.float32)  # norm=5
    y = V._l2_normalize(x)
    assert torch.allclose(
        torch.linalg.norm(y, ord=2, dim=-1), torch.tensor([1.0]), atol=1e-6
    )


def test_to_dtype_variants():
    assert V._to_dtype("float32") == torch.float32
    assert V._to_dtype("bf16") == torch.bfloat16
    assert V._to_dtype("half") == torch.float16


def test_pick_device_cpu_when_no_cuda_mps(monkeypatch):
    monkeypatch.setattr(
        torch, "cuda", types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setattr(
        getattr(torch, "backends"),
        "mps",
        types.SimpleNamespace(is_available=lambda: False),
    )
    assert V._pick_device(None) == "cpu"


def test_vec_cache_roundtrip(tmp_path):
    cfg = V.VectorizerConfig(cache_dir=str(tmp_path))
    base = V.BaseVectorizer(cfg)
    cache = base.cache
    text = "hello world"
    # 尚未缓存
    assert cache.get(text) is None
    # 写入并读取
    v = torch.randn(8)
    cache.put(text, v)
    v2 = cache.get(text)
    assert v2 is not None and torch.allclose(v2, v.cpu(), atol=0)
    # 路径由 backend_sig + text 决定
    p = cache._path(text)
    assert p.exists()


class DummyTok:
    def __init__(self, max_len=512):
        self.max_len = max_len

    def __call__(
        self, texts, padding=True, truncation=True, max_length=None, return_tensors="pt"
    ):
        # 构造最小编码：input_ids 全1，attention_mask 用长度模拟
        maxL = min(max(len(t.split()), 1) for t in texts)
        T = min(max_length or self.max_len, max(max(len(t.split()), 1) for t in texts))
        B = len(texts)
        return {
            "input_ids": torch.ones(B, T, dtype=torch.long),
            "attention_mask": torch.ones(B, T, dtype=torch.long),
        }


class DummyOut:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class DummyModel:
    def __init__(self, H=8, device="cpu"):
        self.H = H
        self._device = device
        self._dtype = torch.float32

    def to(self, *args, **kwargs):
        # 支持 dtype/device 迁移链式调用
        if "dtype" in kwargs:
            self._dtype = kwargs["dtype"]
        if len(args) == 1 and isinstance(args[0], str):
            self._device = args[0]
        return self

    def eval(self):
        return self

    def __call__(self, **enc):
        B, T = enc["input_ids"].shape
        # 生成形状 [B,T,H] 的恒定值，便于校验 mean/cls pooling
        hs = (
            torch.arange(self.H, dtype=self._dtype)
            .repeat(B * T, 1)
            .reshape(B, T, self.H)
            .to(self._device)
        )
        return DummyOut(hs)


def _monkeypatch_hf(monkeypatch):
    # 替换 transformers.AutoTokenizer/AutoModel

    monkeypatch.setattr(
        V,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTok()),
    )
    monkeypatch.setattr(
        V,
        "AutoModel",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyModel()),
    )


def test_hf_mean_pooling_and_normalize_cpu(tmp_path, monkeypatch):
    _monkeypatch_hf(monkeypatch)
    cfg = V.VectorizerConfig(
        backend="hf",
        model_name="dummy-hf",
        pooling="mean",
        cache_dir=str(tmp_path),
        normalize=True,
        dtype="float32",
        device="cpu",
    )
    vec = V.HFVectorizer(cfg)
    X = vec.encode_texts(["a b c", "d e"])  # [2,D]
    assert X.device.type == "cpu"
    assert X.shape[0] == 2 and X.ndim == 2
    # L2 归一化应为单位范数
    nrm = torch.linalg.norm(X, ord=2, dim=-1)
    assert torch.allclose(nrm, torch.ones_like(nrm), atol=1e-6)


def test_hf_cls_pooling_and_cache_hit(tmp_path, monkeypatch):
    _monkeypatch_hf(monkeypatch)
    cfg = V.VectorizerConfig(
        backend="hf",
        model_name="dummy-hf",
        pooling="cls",
        cache_dir=str(tmp_path),
        normalize=False,
    )
    vec = V.HFVectorizer(cfg)
    # 第一次：走模型并写缓存
    X1 = vec.encode_texts(["repeat", "repeat"])
    # 第二次：应全命中缓存，不再调用模型；我们通过修改模型输出来检测
    # 用新的 DummyModel，让输出不同；如果 hit 缓存，结果不变
    monkeypatch.setattr(
        V,
        "AutoModel",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyModel(H=16)),
    )
    vec2 = V.HFVectorizer(cfg)  # 重新实例化，但缓存签名相同
    X2 = vec2.encode_texts(["repeat", "repeat"])
    assert torch.allclose(X1, X2)


class DummyST:
    def __init__(self, name, device="cpu"):
        self.name = name
        self.device = device
        self.calls = 0

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=False,
        show_progress_bar=False,
        device=None,
    ):
        self.calls += 1
        B = len(texts)
        D = 8
        out = torch.arange(D).repeat(B, 1).to(device or self.device).float()
        if normalize_embeddings:
            out = out / torch.linalg.norm(out, ord=2, dim=-1, keepdim=True)
        return out


def _monkeypatch_st(monkeypatch):
    monkeypatch.setattr(V, "SentenceTransformer", DummyST)


def test_st_encode_and_normalize_and_stack(tmp_path, monkeypatch):
    _monkeypatch_st(monkeypatch)
    cfg = V.VectorizerConfig(
        backend="st", model_name="dummy-st", cache_dir=str(tmp_path), normalize=True
    )
    vec = V.STVectorizer(cfg)
    X = vec.encode_texts(["x", "y", "z"])
    assert X.shape == (3, 8) and X.device.type == "cpu"
    # 单条 encode
    v = vec.encode("solo")
    assert v.shape == (8,)
    # 单测空输入
    empty = vec.encode_texts([])
    assert empty.shape == (0, 0)


def test_st_cache_merging_order_and_hit(tmp_path, monkeypatch):
    _monkeypatch_st(monkeypatch)
    cfg = V.VectorizerConfig(
        backend="st", model_name="dummy-st", cache_dir=str(tmp_path), normalize=False
    )
    vec = V.STVectorizer(cfg)
    # 先缓存部分元素（混合命中）
    first = vec.encode_texts(["a", "b"])
    # 第二次混合：["b","a","c"] -> b/a 命中缓存，c 新算；顺序需与输入一致
    second = vec.encode_texts(["b", "a", "c"])
    assert second.shape[0] == 3
    # b 与 a 应与 first 对应行相等
    assert torch.allclose(second[0], first[1])
    assert torch.allclose(second[1], first[0])


def test_build_vectorizer_fallback_and_unknown_backend(monkeypatch):
    # 去掉 HUB，以便走 fallback
    monkeypatch.setattr(V, "HUB", None, raising=False)
    # mock 依赖
    monkeypatch.setattr(
        V,
        "AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: object()),
        raising=False,
    )
    monkeypatch.setattr(
        V,
        "AutoModel",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: object()),
        raising=False,
    )
    monkeypatch.setattr(
        V, "SentenceTransformer", lambda *a, **k: object(), raising=False
    )

    cfg_hf = V.VectorizerConfig(backend="hf")
    obj_hf = V.build_vectorizer(cfg_hf)
    assert isinstance(obj_hf, V.HFVectorizer)

    cfg_st = V.VectorizerConfig(backend="st")
    obj_st = V.build_vectorizer(cfg_st)
    assert isinstance(obj_st, V.STVectorizer)

    with pytest.raises(ValueError):
        V.build_vectorizer(V.VectorizerConfig(backend="word2vec"))  # 目前未实现


def test_registry_take_precedence(monkeypatch):
    # 构造简易注册表：当 backend="hf" 时返回自定义类
    class DummyReg:
        def __init__(self):
            self.map = {"hf": DummyVec}

        def get_or_create(self, name):
            return self

        def get(self, key):
            return self.map.get(key)

        def register(self, *a, **k):
            def deco(x):
                self.map[k.get("alias", [a[0]])[0]] = x
                return x

            return deco

    class DummyVec(V.BaseVectorizer):
        name = "dummy"
        version = "1"

        def encode_texts(self, texts):
            import torch

            return torch.zeros(len(texts), 4)

    monkeypatch.setattr(V, "HUB", DummyReg(), raising=False)
    out = V.build_vectorizer(V.VectorizerConfig(backend="hf"))
    assert isinstance(out, DummyVec)
