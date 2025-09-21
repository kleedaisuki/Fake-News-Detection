# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kan/pipelines/entity_linking.py
@brief  Pluggable entity linking pipeline stage for KAN (knowledge-aware attention).
@date   2025-09-16

@zh
  面向配置（config-driven）的实体链接（Entity Linking, EL）流水线：
  - 输入：`List[NewsRecord]`（来自 kan/data/loaders.py 的稳定 Schema）
  - 输出：回填 `entities` 字段（Wikidata QID 列表），并在 `meta['el']` 记录追溯信息
  - 设计：后端可插拔（TagMe、spaCy/Wikipedia、内置 Dummy 词表等），统一注册表与缓存层

@en
  Config-driven entity linking (EL) pipeline:
  - Input: `List[NewsRecord]` (stable schema from kan/data/loaders.py)
  - Output: populate `entities` (list of Wikidata QIDs) and write tracing info into `meta['el']`
  - Backend: pluggable (TagMe, spaCy/Wikipedia, built-in Dummy gazetteer, etc.) with unified registry & cache.

@contract
  def link_records(records: List[NewsRecord], cfg: ELConfig) -> List[NewsRecord]
    * Must not mutate input in-place unless `inplace=True` was explicitly requested.
    * Must deduplicate entities per record; order is not guaranteed.
    * Must be Windows-friendly (no hard-coded POSIX paths).
    * Must be reproducible: caching keyed by (backend, version, lang, text-hash, threshold, lexicon-hash...).

@notes
  - Knowledge graph neighbor fetching (context expansion) 不在本模块，交由 `kan/pipelines/kg_context.py`。
  - 远端服务（如 TagMe）默认关闭；DummyLinker 提供零依赖可运行路径，便于本地/CI。
  - 统一日志命名空间：`kan.pipelines.entity_linking`。
"""

from dataclasses import dataclass, asdict, field
from hashlib import blake2b
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from copy import deepcopy
from datetime import datetime, timezone

# Stable NewsRecord schema
from kan.data.loaders import NewsRecord

# Registry hub (optional at import time)
try:
    from kan.utils.registry import HUB

    _EL_REG = HUB.get_or_create("entity_linker")
except Exception:  # pragma: no cover - registry may be unavailable at import time

    class _DummyReg:
        def register(self, *_a, **_k):
            def deco(x):
                return x

            return deco

        def get(self, *_a, **_k):
            return None

    _EL_REG = _DummyReg()  # type: ignore

LOGGER = logging.getLogger("kan.pipelines.entity_linking")

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityMention:
    """A single entity mention (for traceability).

    @zh
      - surface: 文本片段
      - start, end: 在原文中的字符偏移（半开区间）
      - qid: 选定的 Wikidata QID（若无可用则 None）
      - score: 置信度（0~1，后端定义）
    """

    surface: str
    start: int
    end: int
    qid: Optional[str]
    score: float = 1.0


@dataclass
class ELConfig:
    """Entity Linking config (align with configs/pipelines/entity_linking.yaml).

    @fields
      backend: 后端标识（dummy / tagme / wikipedia / custom...）
      language: 文本语种，影响后端与词表
      threshold: 分数阈值，小于该值的候选将被丢弃
      cache_dir: 缓存目录（JSON，每条文本一文件）
      inplace: 是否原地回填；默认 False（返回深拷贝）
      # backend-specific
      lexicon_path: DummyLinker 的词表（JSON: {"name": "Q42", ...}）
      case_sensitive: 词表是否区分大小写（默认 False）
      max_surface_len: 单次匹配的最大表面长度（词表匹配上限，默认 5 个词）
      api: 任意后端参数袋（如 TagMe token、endpoint 等）
    """

    backend: str = "dummy"
    language: str = "en"
    threshold: float = 0.0
    cache_dir: Optional[str] = ".cache/el"
    inplace: bool = False

    # Dummy backend options
    lexicon_path: Optional[str] = None
    case_sensitive: bool = False
    max_surface_len: int = 5

    # Extensible bag for other backends
    api: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Cache store (content-addressed by config+text)
# -----------------------------------------------------------------------------


class _Cache:
    def __init__(self, dir_: Optional[str], backend_sig: str) -> None:
        self.dir = Path(dir_) if dir_ else None
        self.backend_sig = backend_sig
        if self.dir:
            (self.dir).mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> Path:
        h = blake2b(
            (self.backend_sig + "\n" + text).encode("utf-8"), digest_size=16
        ).hexdigest()
        assert self.dir is not None
        return self.dir / f"{h[:2]}" / f"{h}.json"

    def get(self, text: str) -> Optional[List[EntityMention]]:
        if not self.dir:
            return None
        p = self._key(text)
        if not p.parent.exists():
            return None
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return [EntityMention(**m) for m in data.get("mentions", [])]
        except Exception:
            return None

    def put(self, text: str, mentions: List[EntityMention]) -> None:
        if not self.dir:
            return
        p = self._key(text)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(
                {"mentions": [asdict(m) for m in mentions]}, f, ensure_ascii=False
            )


# -----------------------------------------------------------------------------
# Base linker interface
# -----------------------------------------------------------------------------


class BaseLinker:
    name: str = "base"
    version: str = "0"

    def __init__(self, cfg: ELConfig) -> None:
        self.cfg = cfg

    def link(self, text: str) -> List[EntityMention]:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Dummy gazetteer-based linker (zero dependency, CI-friendly)
# -----------------------------------------------------------------------------


@_EL_REG.register("dummy", alias=["lex", "gazetteer"])  # type: ignore[attr-defined]
class DummyLinker(BaseLinker):
    name = "dummy"
    version = "1"

    def __init__(self, cfg: ELConfig) -> None:
        super().__init__(cfg)
        self.lex: Dict[str, str] = {}
        if cfg.lexicon_path:
            p = Path(cfg.lexicon_path)
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    self.lex = json.load(f)
        # Normalize keys according to case setting
        if not cfg.case_sensitive:
            self.lex = {k.lower(): v for k, v in self.lex.items()}
        LOGGER.info(
            "DummyLinker loaded lexicon: %d entries (case_sensitive=%s)",
            len(self.lex),
            cfg.case_sensitive,
        )

    def link(self, text: str) -> List[EntityMention]:
        if not text:
            return []
        # Simple longest-match, whitespace tokenized windows up to max_surface_len
        tokens = re.findall(r"\w+|\S", text)
        # Build index to reconstruct char offsets
        offsets: List[Tuple[str, int, int]] = []  # (tok, start, end)
        pos = 0
        for tok in tokens:
            start = text.find(tok, pos)
            if start == -1:
                start = pos
            end = start + len(tok)
            offsets.append((tok, start, end))
            pos = end

        mentions: List[EntityMention] = []
        n = len(tokens)
        for i in range(n):
            max_j = min(n, i + self.cfg.max_surface_len)
            surface = ""
            s_start = offsets[i][1]
            for j in range(i, max_j):
                # Rebuild surface by original text slice to respect punctuation/spaces
                s_end = offsets[j][2]
                surface = text[s_start:s_end]
                key = surface if self.cfg.case_sensitive else surface.lower()
                qid = self.lex.get(key)
                if qid:
                    mentions.append(
                        EntityMention(
                            surface=surface,
                            start=s_start,
                            end=s_end,
                            qid=qid,
                            score=1.0,
                        )
                    )
        # Deduplicate overlapping identical spans keeping first
        uniq = {}
        for m in mentions:
            uniq[(m.start, m.end, m.qid)] = m
        return list(uniq.values())


# -----------------------------------------------------------------------------
# (Optional) Stubs for real-world backends (left unimplemented by default)
# -----------------------------------------------------------------------------


@_EL_REG.register("tagme")  # type: ignore[attr-defined]
class TagMeLinker(BaseLinker):
    name = "tagme"
    version = "0"

    def link(self, text: str) -> List[EntityMention]:  # pragma: no cover - placeholder
        raise NotImplementedError(
            "TagMeLinker requires external API; implement in your environment."
        )


@_EL_REG.register("wikipedia")  # type: ignore[attr-defined]
class WikipediaLinker(BaseLinker):
    name = "wikipedia"
    version = "0"

    def link(self, text: str) -> List[EntityMention]:  # pragma: no cover - placeholder
        raise NotImplementedError(
            "WikipediaLinker requires local index or API; implement in your environment."
        )


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def build_linker(cfg: ELConfig) -> BaseLinker:
    key = (cfg.backend or "dummy").lower()
    # Prefer registry lookup
    try:
        from kan.utils.registry import HUB

        reg = HUB.get_or_create("entity_linker")
        klass = reg.get(key)
        if klass is None:
            raise KeyError(key)
        return klass(cfg)
    except Exception:
        # local fallback
        if key in ("dummy", "lex", "gazetteer"):
            return DummyLinker(cfg)
        if key == "tagme":
            return TagMeLinker(cfg)
        if key == "wikipedia":
            return WikipediaLinker(cfg)
        raise ValueError(f"Unsupported entity linker backend: {cfg.backend}")


# -----------------------------------------------------------------------------
# Public pipeline function
# -----------------------------------------------------------------------------


def link_records(
    records: List[NewsRecord], cfg: ELConfig, *, inplace: Optional[bool] = None
) -> List[NewsRecord]:
    """Link entities for a batch of NewsRecord and return updated records.

    @params
      records: 待链接的样本列表
      cfg:     EL 配置
      inplace: 是否原地修改（覆盖 cfg.inplace）
    """
    if not records:
        return [] if not cfg.inplace else records

    inplace = cfg.inplace if inplace is None else inplace
    out = records if inplace else deepcopy(records)

    linker = build_linker(cfg)

    # Compose backend signature for cache keying
    backend_sig = json.dumps(
        {
            "backend": linker.__class__.__name__,
            "version": getattr(linker, "version", "0"),
            "language": cfg.language,
            "threshold": cfg.threshold,
            "lex_hash": _hash_file(cfg.lexicon_path) if cfg.lexicon_path else None,
            "case_sensitive": cfg.case_sensitive,
            "max_surface_len": cfg.max_surface_len,
        },
        sort_keys=True,
    )

    cache = _Cache(cfg.cache_dir, backend_sig)

    num_cache_hit = 0
    num_linked = 0

    for rec in out:
        text = rec.text or ""
        mentions = cache.get(text)
        if mentions is None:
            mentions = linker.link(text)
            # apply threshold filtering
            mentions = [m for m in mentions if m.score >= cfg.threshold and m.qid]
            cache.put(text, mentions)
        else:
            num_cache_hit += 1

        # dedupe entities
        qids = sorted({m.qid for m in mentions if m.qid})  # type: ignore[arg-type]
        if qids:
            num_linked += 1
        rec.entities = list(qids)
        # Traceability
        rec.meta = dict(rec.meta or {})
        rec.meta.setdefault("el", {})
        rec.meta["el"].update(
            {
                "backend": linker.__class__.__name__,
                "version": getattr(linker, "version", "0"),
                "language": cfg.language,
                "threshold": cfg.threshold,
                "time": _now_iso(),
                "mentions": [asdict(m) for m in mentions],
            }
        )

    LOGGER.info(
        "EL done: records=%d, linked=%d, cache_hit=%d, backend=%s/%s",
        len(out),
        num_linked,
        num_cache_hit,
        linker.__class__.__name__,
        getattr(linker, "version", "0"),
    )
    return out


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    try:
        data = Path(p).read_bytes()
        return blake2b(data, digest_size=12).hexdigest()
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Doxygen examples (bilingual)
# -----------------------------------------------------------------------------

__doc_examples__ = r"""
/**
 * @zh 用法（最小可运行 Dummy 后端）：
 * ```python
 * from kan.data.loaders import NewsRecord
 * from kan.pipelines.entity_linking import ELConfig, link_records
 * records = [NewsRecord(id="x1", text="Barack Obama met Angela Merkel.", label=1, entities=[], contexts={}, meta={})]
 * cfg = ELConfig(backend="dummy", lexicon_path="./lexicon.en.json", case_sensitive=False)
 * out = link_records(records, cfg, inplace=False)
 * print(out[0].entities)  # ["Q76", "Q567"]  ← 取决于你的词表
 * print(out[0].meta["el"]["mentions"])   # 追溯命中片段与偏移
 * ```
 *
 * @en Usage (cache & inplace):
 * ```python
 * cfg = ELConfig(backend="dummy", cache_dir=".cache/el")
 * out = link_records(records, cfg, inplace=True)  # modify in place
 * ```
 */
"""
