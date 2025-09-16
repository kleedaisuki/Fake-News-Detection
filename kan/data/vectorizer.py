# -*- coding: utf-8 -*-
from __future__ import annotations
"""
@file   kan/modules/vectorizer.py
@brief  Pluggable text vectorizer with swap-friendly backends (HF / SentenceTransformers).
@date   2025-09-16

@zh
  面向配置（config-driven）的文本向量化器，支持在 **不改用户态** 的前提下自由切换嵌入模型：
  - 后端：HuggingFace Transformers（AutoModel）/ SentenceTransformers；OpenAI 等留作占位
  - 特性：批处理、池化策略（CLS/mean）、设备与 dtype 管理、L2 归一化、可选按文本缓存
  - 设计：注册表 `vectorizer` 命名空间 + 统一 `BaseVectorizer` 接口 + `VectorizerConfig`

@en
  Config-driven vectorizer with swap-friendly backends (HuggingFace/SentenceTransformers).
  - Features: batching, pooling (CLS/mean), device & dtype mgmt, L2 normalization, optional per-text cache
  - Registry namespace: `vectorizer`. Stable user API regardless of backend choice.

@contract
  class BaseVectorizer:
      def encode_texts(self, texts: list[str]) -> "Tensor":  # CPU tensor by default
      def encode(self, text: str) -> "Tensor":                # shape [D]

  def build_vectorizer(cfg: VectorizerConfig) -> BaseVectorizer

@notes
  - 返回默认是 **CPU torch.Tensor**，便于跨进程/落盘；你可手动 `.to(device)` 参与下游计算。
  - 训练时若要直接在 GPU 上向量化，建议在上层 batcher 控制，把输入张量移到目标设备。
  - Windows 友好：不依赖 POSIX-only 特性；路径用 `pathlib.Path`；缓存文件是独立条目。
"""

from dataclasses import dataclass, asdict
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import logging
import math

LOGGER = logging.getLogger("kan.modules.vectorizer")

# Optional deps guarded --------------------------------------------------------
try:
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover - allow import without torch at doc build
    torch = None  # type: ignore
    Tensor = Any  # type: ignore

try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass
class VectorizerConfig:
    """Vectorizer configuration (align with configs/model/vectorizer.yaml).

    @fields
      backend: 后端（'hf' | 'sentence_transformers' | 'openai'(占位) | ...）
      model_name: 模型名（如 'sentence-transformers/all-MiniLM-L6-v2' 或任意 HF 编码模型）
      pooling: 池化策略（'mean' | 'cls'）
      max_length: 截断长度（token 级）
      batch_size: 批大小
      device: 设备（'cpu' | 'cuda' | 'mps' | 具体索引，如 'cuda:0'）。None 时自动推断
      dtype: 张量精度（'float32' | 'bfloat16' | 'float16'）；仅在支持设备上生效
      normalize: 是否做 L2 归一化
      cache_dir: 文本级缓存根目录（None 关闭缓存）
      trust_remote_code: HF 可执行权（谨慎使用）
      local_files_only: 仅本地权重（离线环境）

      # backend specific
      hf_tokenizer_kwargs: 传入 AutoTokenizer 的附加参数
      hf_model_kwargs:     传入 AutoModel 的附加参数
    """

    backend: str = "hf"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    pooling: str = "mean"
    max_length: int = 512
    batch_size: int = 32
    device: Optional[str] = None
    dtype: str = "float32"
    normalize: bool = True
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    local_files_only: bool = False

    hf_tokenizer_kwargs: Dict[str, Any] = None  # type: ignore[assignment]
    hf_model_kwargs: Dict[str, Any] = None      # type: ignore[assignment]

    def __post_init__(self) -> None:  # defaults for dict fields
        if self.hf_tokenizer_kwargs is None:
            self.hf_tokenizer_kwargs = {}
        if self.hf_model_kwargs is None:
            self.hf_model_kwargs = {}


# ----------------------------------------------------------------------------
# Cache (per-text, content addressed by backend signature + text)
# ----------------------------------------------------------------------------

class _VecCache:
    def __init__(self, dir_: Optional[str], backend_sig: str) -> None:
        self.dir = Path(dir_) if dir_ else None
        self.sig = backend_sig
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, text: str) -> Path:
        h = blake2b((self.sig + "\n" + text).encode("utf-8"), digest_size=16).hexdigest()
        assert self.dir is not None
        return self.dir / h[:2] / f"{h}.pt"

    def get(self, text: str) -> Optional[Tensor]:
        if not (self.dir and torch is not None):
            return None
        p = self._path(text)
        if not (p.parent.exists() and p.exists()):
            return None
        try:
            v = torch.load(p, map_location="cpu")  # type: ignore[arg-type]
            if isinstance(v, (list, tuple)):
                v = torch.tensor(v)
            return v
        except Exception:
            return None

    def put(self, text: str, vec: Tensor) -> None:
        if not (self.dir and torch is not None):
            return
        p = self._path(text)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(vec.cpu(), p)  # type: ignore[arg-type]
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Base interface
# ----------------------------------------------------------------------------

class BaseVectorizer:
    name: str = "base"
    version: str = "0"

    def __init__(self, cfg: VectorizerConfig) -> None:
        self.cfg = cfg
        # Build backend signature for cache & reproducibility
        self.backend_sig = json.dumps({
            "backend": self.__class__.__name__,
            "version": getattr(self, "version", "0"),
            "model": cfg.model_name,
            "pooling": cfg.pooling,
            "max_length": cfg.max_length,
            "normalize": cfg.normalize,
            "dtype": cfg.dtype,
        }, sort_keys=True)
        self.cache = _VecCache(cfg.cache_dir, self.backend_sig)

    # Public API --------------------------------------------------------------
    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        raise NotImplementedError

    def encode(self, text: str) -> Tensor:
        vs = self.encode_texts([text])
        return vs[0]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _pick_device(spec: Optional[str]) -> str:
    if spec:
        return spec
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def _to_dtype(dtype: str):
    if torch is None:
        return None
    d = dtype.lower()
    if d in ("fp32", "float32", "f32"):
        return torch.float32
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _l2_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    if torch is None:
        return x
    n = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / n


# ----------------------------------------------------------------------------
# HF backend
# ----------------------------------------------------------------------------

class HFVectorizer(BaseVectorizer):
    name = "hf"
    version = "1"

    def __init__(self, cfg: VectorizerConfig) -> None:
        if torch is None or AutoTokenizer is None or AutoModel is None:
            raise RuntimeError("HF backend requires 'torch' and 'transformers'.")
        super().__init__(cfg)
        self.device = _pick_device(cfg.device)
        self.torch_dtype = _to_dtype(cfg.dtype)
        tok_kwargs = dict(trust_remote_code=cfg.trust_remote_code, local_files_only=cfg.local_files_only)
        tok_kwargs.update(cfg.hf_tokenizer_kwargs)
        mdl_kwargs = dict(trust_remote_code=cfg.trust_remote_code, local_files_only=cfg.local_files_only)
        mdl_kwargs.update(cfg.hf_model_kwargs)
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name, **tok_kwargs)
        self.mdl = AutoModel.from_pretrained(cfg.model_name, **mdl_kwargs)
        if self.torch_dtype is not None:
            try:
                self.mdl = self.mdl.to(dtype=self.torch_dtype)
            except Exception:
                pass
        self.mdl = self.mdl.to(self.device)
        self.mdl.eval()
        LOGGER.info("HFVectorizer ready: model=%s, device=%s, dtype=%s, pooling=%s", cfg.model_name, self.device, cfg.dtype, cfg.pooling)

    @torch.no_grad()  # type: ignore[misc]
    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        if not texts:
            return torch.empty((0, 0))  # type: ignore[return-value]
        bs = max(1, int(self.cfg.batch_size))
        all_vecs: List[Tensor] = []
        for i in range(0, len(texts), bs):
            batch = list(texts[i:i+bs])
            # cache probe first
            cached_flags = [False] * len(batch)
            cached_vecs: List[Optional[Tensor]] = [None] * len(batch)
            for j, t in enumerate(batch):
                v = self.cache.get(t)
                if v is not None:
                    cached_flags[j] = True
                    cached_vecs[j] = v
            to_run = [batch[j] for j, f in enumerate(cached_flags) if not f]
            out_vecs: List[Tensor] = []
            if to_run:
                enc = self.tok(to_run, padding=True, truncation=True, max_length=self.cfg.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.mdl(**enc)
                # hidden states: [B, T, H]
                hs: Tensor = out.last_hidden_state  # type: ignore[assignment]
                if self.cfg.pooling == "cls":
                    vec = hs[:, 0, :]
                else:  # mean pooling with mask
                    mask: Tensor = enc["attention_mask"].unsqueeze(-1)  # [B,T,1]
                    sum_h = (hs * mask).sum(dim=1)
                    len_h = mask.sum(dim=1).clamp_min(1)
                    vec = sum_h / len_h
                if self.cfg.normalize:
                    vec = _l2_normalize(vec)
                out_vecs = vec.detach().to("cpu")  # keep CPU for portability
            # merge cached and fresh results back in order
            merged: List[Tensor] = []
            k = 0
            for j in range(len(batch)):
                if cached_flags[j]:
                    v = cached_vecs[j]
                    assert v is not None
                    merged.append(v)
                else:
                    merged.append(out_vecs[k])
                    # write cache per item
                    self.cache.put(batch[j], out_vecs[k])
                    k += 1
            all_vecs.extend(merged)
        return torch.stack(all_vecs, dim=0)


# ----------------------------------------------------------------------------
# SentenceTransformers backend
# ----------------------------------------------------------------------------

class STVectorizer(BaseVectorizer):
    name = "sentence_transformers"
    version = "1"

    def __init__(self, cfg: VectorizerConfig) -> None:
        if torch is None or SentenceTransformer is None:
            raise RuntimeError("SentenceTransformers backend requires 'torch' and 'sentence-transformers'.")
        super().__init__(cfg)
        self.device = _pick_device(cfg.device)
        self.mdl = SentenceTransformer(cfg.model_name, device=self.device)
        LOGGER.info("STVectorizer ready: model=%s, device=%s, pooling=%s", cfg.model_name, self.device, cfg.pooling)

    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        if not texts:
            return torch.empty((0, 0))  # type: ignore[return-value]
        # probe cache
        cached_flags = [False] * len(texts)
        cached_vecs: List[Optional[Tensor]] = [None] * len(texts)
        to_run_idx: List[int] = []
        to_run_texts: List[str] = []
        for i, t in enumerate(texts):
            v = self.cache.get(t)
            if v is not None:
                cached_flags[i] = True
                cached_vecs[i] = v
            else:
                to_run_idx.append(i)
                to_run_texts.append(t)
        out_vecs: List[Tensor] = []
        if to_run_texts:
            arr = self.mdl.encode(
                to_run_texts,
                batch_size=max(1, int(self.cfg.batch_size)),
                convert_to_tensor=True,
                normalize_embeddings=self.cfg.normalize,
                show_progress_bar=False,
                device=self.device,
                )
            # SentenceTransformers returns [N, D] tensor on device
            out_vecs = arr.detach().to("cpu").split(1, dim=0)  # list of [1,D]
            out_vecs = [v.squeeze(0) for v in out_vecs]
        # merge
        merged: List[Tensor] = [None] * len(texts)  # type: ignore[list-item]
        k = 0
        for i in range(len(texts)):
            if cached_flags[i]:
                merged[i] = cached_vecs[i]  # type: ignore[assignment]
            else:
                merged[i] = out_vecs[k]
                self.cache.put(texts[i], out_vecs[k])
                k += 1
        return torch.stack(merged, dim=0)


# ----------------------------------------------------------------------------
# Registry & factory
# ----------------------------------------------------------------------------

try:
    from kan.utils.registry import HUB
    _VEC_REG = HUB.get_or_create("vectorizer")
except Exception:  # pragma: no cover
    class _DummyReg:
        def register(self, *_a, **_k):
            def deco(x):
                return x
            return deco
        def get(self, *_a, **_k):
            return None
    _VEC_REG = _DummyReg()  # type: ignore

@_VEC_REG.register("hf", alias=["transformers"])  # type: ignore[attr-defined]
class _HFReg(HFVectorizer):
    pass

@_VEC_REG.register("sentence_transformers", alias=["st", "sentence-transformers"])  # type: ignore[attr-defined]
class _STReg(STVectorizer):
    pass


def build_vectorizer(cfg: VectorizerConfig) -> BaseVectorizer:
    key = (cfg.backend or "hf").lower()
    # Prefer registry
    try:
        from kan.utils.registry import HUB
        reg = HUB.get_or_create("vectorizer")
        klass = reg.get(key)
        if klass is None:
            raise KeyError(key)
        return klass(cfg)
    except Exception:
        # Fallback
        if key in ("hf", "transformers"):
            return HFVectorizer(cfg)
        if key in ("sentence_transformers", "st", "sentence-transformers"):
            return STVectorizer(cfg)
        raise ValueError(f"Unsupported vectorizer backend: {cfg.backend}")


# ----------------------------------------------------------------------------
# Doxygen examples (bilingual)
# ----------------------------------------------------------------------------

__doc_examples__ = r"""
/**
 * @zh 用法（HF 后端）：
 * ```python
 * from kan.modules.vectorizer import VectorizerConfig, build_vectorizer
 * cfg = VectorizerConfig(backend='hf', model_name='sentence-transformers/all-MiniLM-L6-v2',
 *                        pooling='mean', batch_size=64, device=None, dtype='float16', normalize=True,
 *                        cache_dir='.cache/vec')
 * vec = build_vectorizer(cfg)
 * X = vec.encode_texts(["Hello world", "Knowledge-aware attention network"])  # -> torch.Tensor [2, D] on CPU
 * ```
 *
 * @en Usage (SentenceTransformers backend):
 * ```python
 * cfg = VectorizerConfig(backend='st', model_name='all-mpnet-base-v2', batch_size=64, cache_dir='.cache/vec')
 * vec = build_vectorizer(cfg)
 * z = vec.encode("A single sentence")  # -> [D]
 * ```
 */
"""
