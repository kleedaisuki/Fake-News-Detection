# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kan/data/loaders.py
@brief  Dataset loaders with stable schema for KAN (Knowledge-aware Attention Network).
@date   2025-09-16

@zh
  统一的新闻数据加载层：输出稳定的 `NewsRecord` 架构，并通过配置/注册表适配不同数据源
  （CSV/JSONL/HuggingFace Datasets）。强调**向后兼容**、**Windows 友好**、**可扩展**。

@en
  Unified dataset loading layer that normalizes various sources (CSV/JSONL/HF Datasets)
  into a stable `NewsRecord` schema. Designed for backward compatibility, Windows
  friendliness, and extensibility via registry.

@contract
  * Stable schema (v1):
      NewsRecord {
        id: str,
        text: str,
        label: int,                 # 0 real / 1 fake (binary by default)
        entities: list[str],        # Optional: Wikidata QIDs
        contexts: dict[str, list[str]],  # Optional: {entity -> one-hop neighbors}
        meta: dict,                 # Free-form meta for traceability
      }
  * Splits: {"train", "validation", "test"} + custom (e.g., "fold0", ...)
  * Resilience: missing optional fields default to [] / {}

@notes
  - Entity linking & KG fetching are **not** performed here; they belong to pipelines.
  - This module focuses on IO normalization and light validation only.
"""

from dataclasses import dataclass, field
from enum import Enum
from hashlib import blake2b
import json
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import pandas as pd

try:
    import datasets as hfd
except Exception:  # pragma: no cover - optional dependency
    hfd = None  # type: ignore

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger("kan.data.loaders")

# -----------------------------------------------------------------------------
# Stable schema (v1)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class NewsRecord:
    """\
    @zh 新闻样本的稳定数据架构（Schema v1）。
    @en Stable sample schema for news (Schema v1).

    @field id: 样本唯一 ID（若源数据缺失则由 text 哈希生成） / Unique sample ID.
    @field text: 原文文本 / Original text.
    @field label: 整型标签（默认二分类：0=real, 1=fake）/ Integer label.
    @field entities: Wikidata QIDs（可选）/ Optional Wikidata entity IDs.
    @field contexts: 上下文字典（可选）/ Optional mapping entity->neighbors.
    @field meta: 附加元信息（数据来源、原始字段、fold、split 等）/ Free-form metadata.
    """

    id: str
    text: str
    label: int
    entities: List[str]
    contexts: Dict[str, List[str]]
    meta: Dict[str, Any]


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# -----------------------------------------------------------------------------
# Config & Field mapping
# -----------------------------------------------------------------------------


@dataclass
class FieldMap:
    """\
    @zh 将源数据字段映射到稳定 Schema 的映射表。
    @en Mapping from raw-source columns to the stable schema fields.

    仅 `id/text/label` 是硬需求，其他字段可缺省。
    """

    id: Optional[str] = None
    text: Optional[str] = None
    label: Optional[str] = None
    # Optional fields in source
    entities: Optional[str] = None  # column that holds List[str]
    contexts: Optional[str] = None  # column that holds Dict[str, List[str]]
    split: Optional[str] = None  # column that holds split names
    fold: Optional[str] = None  # column that holds fold indices
    meta: Optional[Union[str, List[str]]] = None  # column(s) to pack into meta


@dataclass
class DatasetConfig:
    """\
    @zh 数据集加载配置；与 `configs/data/*.yaml` 对齐。
    @en Dataset loading config; aligned with configs/data/*.yaml.
    """

    name: str
    format: str  # "csv" | "jsonl" | "hf"
    # Source location
    path: Optional[Union[str, Path]] = None  # file path or directory
    hf_name: Optional[str] = None  # huggingface dataset name
    hf_config: Optional[str] = None  # huggingface config/subset
    # Split specification
    splits: Mapping[str, Union[str, Path]] | None = None  # {split: file_or_hf_split}
    # CSV/JSONL options
    delimiter: str = ","
    encoding: str = "utf-8"
    lines: bool = True  # for json
    # Field mapping & transforms
    fields: FieldMap = field(default_factory=FieldMap)
    label_map: Optional[Mapping[Any, int]] = None  # map raw label -> {0,1,...}
    # Misc
    id_prefix: Optional[str] = None
    drop_duplicates_on_text: bool = False
    lowercase_text: bool = False
    strip_text: bool = True


# -----------------------------------------------------------------------------
# Registry (plug-in style)
# -----------------------------------------------------------------------------
try:
    from kan.utils.registry import HUB

    _DATASET_REG = HUB.get_or_create("dataset")
except Exception:  # pragma: no cover - registry optional at import time

    class _DummyReg:
        def register(self, *_a, **_k):
            def deco(x):
                return x

            return deco

    _DATASET_REG = _DummyReg()  # type: ignore


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _ensure_id(raw_id: Optional[Any], text: str, prefix: Optional[str]) -> str:
    if raw_id is not None and str(raw_id).strip():
        return str(raw_id)
    h = blake2b(text.encode("utf-8"), digest_size=10).hexdigest()
    return f"{prefix or 'auto'}_{h}"


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str):
        # try to parse JSON-like list
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("(") and s.endswith(")")
        ):
            try:
                j = json.loads(s.replace("(", "[").replace(")", "]"))
                return [str(i) for i in j]
            except Exception:
                return [s]
        return [s]
    return [str(x)]


def _as_dict_list(x: Any) -> Dict[str, List[str]]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return {str(k): _as_list(v) for k, v in x.items()}
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                j = json.loads(s)
                if isinstance(j, dict):
                    return {str(k): _as_list(v) for k, v in j.items()}
            except Exception:
                pass
    return {}


def _normalize_text(text: str, *, lower: bool, strip: bool) -> str:
    if text is None:
        return ""
    if strip:
        text = text.strip()
    if lower:
        text = text.lower()
    return text


# -----------------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------------


class BaseLoader:
    """\
    @zh 数据集加载器抽象基类：负责把任意来源转为 `NewsRecord` 列表。
    @en Abstract base loader that converts arbitrary sources into `NewsRecord` list.
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        self.name = cfg.name
        LOGGER.info("DatasetLoader init: name=%s, format=%s", cfg.name, cfg.format)

    # --- Public API ---------------------------------------------------------
    def load_split(self, split: str) -> List[NewsRecord]:
        """\
        @zh 加载指定划分，返回标准化样本列表；若未找到，返回空列表。
        @en Load the given split and return normalized samples; return empty if not found.
        """
        raise NotImplementedError

    def available_splits(self) -> List[str]:  # pragma: no cover - trivial
        s = list(self.cfg.splits.keys()) if self.cfg.splits else []
        if not s and isinstance(self, HFDatasetLoader):
            # HF loader can introspect
            try:
                ds = hfd.load_dataset(self.cfg.hf_name, self.cfg.hf_config)
                s = list(ds.keys())
            except Exception:
                pass
        return s

    def close(self) -> None:  # pragma: no cover - nothing to close by default
        LOGGER.debug("DatasetLoader close: name=%s", self.name)

    # --- Normalization ------------------------------------------------------
    def _normalize_row(self, row: Mapping[str, Any]) -> Optional[NewsRecord]:
        f = self.cfg.fields
        # Required fields: text, label
        raw_text = row.get(f.text or "text")
        raw_label = row.get(f.label or "label")

        if raw_text is None or (isinstance(raw_text, float) and pd.isna(raw_text)):
            return None  # drop invalid rows silently

        text = _normalize_text(
            str(raw_text), lower=self.cfg.lowercase_text, strip=self.cfg.strip_text
        )
        
        # 归一化后仍为空（或纯空白）→ 丢弃该行
        if text.strip() == "":
            return None

        # id
        raw_id = row.get(f.id or "id")
        rid = _ensure_id(raw_id, text, self.cfg.id_prefix)

        # label mapping
        label = raw_label
        if self.cfg.label_map is not None:
            try:
                label = self.cfg.label_map[raw_label]
            except Exception:
                # try str/raw fallbacks
                label = self.cfg.label_map.get(
                    str(raw_label),
                    self.cfg.label_map.get(
                        int(raw_label) if str(raw_label).isdigit() else raw_label,
                        raw_label,
                    ),
                )
        try:
            ilabel = int(label)
        except Exception:
            LOGGER.warning(
                "Label not int-like; fallback to 0: id=%s, raw=%r", rid, raw_label
            )
            ilabel = 0

        # optional fields
        ents: List[str] = []
        ctxs: Dict[str, List[str]] = {}
        if f.entities:
            ents = _as_list(row.get(f.entities))
        if f.contexts:
            ctxs = _as_dict_list(row.get(f.contexts))

        # meta pack
        meta: Dict[str, Any] = {
            "dataset": self.cfg.name,
        }
        if f.split and (sv := row.get(f.split)) is not None:
            meta["split"] = str(sv)
        if f.fold and (fv := row.get(f.fold)) is not None:
            try:
                meta["fold"] = int(fv)
            except Exception:
                meta["fold"] = str(fv)
        # pack selected meta cols
        if f.meta:
            if isinstance(f.meta, str):
                meta[f.meta] = row.get(f.meta)
            else:
                for k in f.meta:
                    meta[k] = row.get(k)

        return NewsRecord(
            id=rid, text=text, label=ilabel, entities=ents, contexts=ctxs, meta=meta
        )


# -----------------------------------------------------------------------------
# CSV/JSONL loader
# -----------------------------------------------------------------------------


@_DATASET_REG.register("csv", alias=["CSV"])  # type: ignore[attr-defined]
class CSVLoader(BaseLoader):
    """\
    @zh 从 CSV 文件加载；支持每个 split 对应一个独立文件。
    @en Load from CSV files; each split maps to a separate file.
    """

    def load_split(self, split: str) -> List[NewsRecord]:
        if not self.cfg.splits or split not in self.cfg.splits:
            LOGGER.warning(
                "Split '%s' not configured for dataset '%s'", split, self.name
            )
            return []
        file = Path(self.cfg.splits[split])
        if not file.is_absolute() and self.cfg.path:
            file = Path(self.cfg.path) / file
        if not file.exists():
            LOGGER.error("CSV file not found: %s", file)
            return []
        LOGGER.info("Loading CSV: %s (split=%s)", file, split)
        df = pd.read_csv(
            file,
            encoding=self.cfg.encoding,
            sep=self.cfg.delimiter,
            dtype=str,
            keep_default_na=False,
        )
        if (
            self.cfg.drop_duplicates_on_text
            and (self.cfg.fields.text or "text") in df.columns
        ):
            df = df.drop_duplicates(
                subset=[self.cfg.fields.text or "text"]
            ).reset_index(drop=True)
        recs: List[NewsRecord] = []
        for _, row in df.iterrows():
            r = self._normalize_row(row)
            if r is not None:
                recs.append(r)
        LOGGER.info("Loaded %d records from %s", len(recs), file.name)
        return recs


@_DATASET_REG.register("jsonl", alias=["json", "JSONL", "JSON"])  # type: ignore[attr-defined]
class JSONLinesLoader(BaseLoader):
    """\
    @zh 从 JSON 或 JSON Lines 文件加载。
    @en Load from JSON or JSON Lines files.
    """

    def load_split(self, split: str) -> List[NewsRecord]:
        if not self.cfg.splits or split not in self.cfg.splits:
            LOGGER.warning(
                "Split '%s' not configured for dataset '%s'", split, self.name
            )
            return []
        file = Path(self.cfg.splits[split])
        if not file.is_absolute() and self.cfg.path:
            file = Path(self.cfg.path) / file
        if not file.exists():
            LOGGER.error("JSON file not found: %s", file)
            return []
        LOGGER.info(
            "Loading JSON%s: %s (split=%s)", "L" if self.cfg.lines else "", file, split
        )

        rows: List[Mapping[str, Any]]
        if self.cfg.lines:
            rows = []
            with file.open("r", encoding=self.cfg.encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        LOGGER.warning("Skip invalid JSON line in %s", file.name)
        else:
            with file.open("r", encoding=self.cfg.encoding) as f:
                data = json.load(f)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict) and "data" in data:
                rows = list(data["data"])  # common pattern
            else:
                LOGGER.error("Unsupported JSON structure in %s", file.name)
                return []

        recs: List[NewsRecord] = []
        for row in rows:
            r = self._normalize_row(row)
            if r is not None:
                recs.append(r)
        LOGGER.info("Loaded %d records from %s", len(recs), file.name)
        return recs


# -----------------------------------------------------------------------------
# HuggingFace Datasets loader
# -----------------------------------------------------------------------------


@_DATASET_REG.register("hf", alias=["huggingface", "datasets"])  # type: ignore[attr-defined]
class HFDatasetLoader(BaseLoader):
    """\
    @zh 从 HuggingFace Datasets 加载；要求安装 `datasets` 包。
    @en Load from HuggingFace Datasets; requires `datasets` package.
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__(cfg)
        if hfd is None:
            raise RuntimeError(
                "'datasets' is not installed; please `pip install datasets`."
            )

    def load_split(self, split: str) -> List[NewsRecord]:
        ds_split = None
        if self.cfg.splits and split in self.cfg.splits:
            ds_split = str(self.cfg.splits[split])
        LOGGER.info(
            "Loading HF dataset: name=%s, config=%s, split=%s",
            self.cfg.hf_name,
            self.cfg.hf_config,
            ds_split or split,
        )
        ds = hfd.load_dataset(
            self.cfg.hf_name, self.cfg.hf_config, split=ds_split or split
        )
        recs: List[NewsRecord] = []
        for row in ds:
            r = self._normalize_row(row)
            if r is not None:
                recs.append(r)
        LOGGER.info("Loaded %d records from HF[%s]", len(recs), split)
        return recs


# -----------------------------------------------------------------------------
# Builder helpers
# -----------------------------------------------------------------------------


def build_loader(cfg: DatasetConfig) -> BaseLoader:
    """\
    @zh 通过注册表构建加载器：cfg.format ∈ {csv,jsonl,hf}。
    @en Build loader via registry using cfg.format.
    """
    key = (cfg.format or "").lower()
    # Prefer registry (if available)
    try:
        from kan.utils.registry import HUB

        reg = HUB.get_or_create("dataset")
        klass = reg.get(key)  # type: ignore[attr-defined]
        if klass is None:
            raise KeyError(key)
        return klass(cfg)
    except Exception:
        # Fallback to local mapping
        if key == "csv":
            return CSVLoader(cfg)
        if key in ("jsonl", "json", "jsonlines"):
            return JSONLinesLoader(cfg)
        if key in ("hf", "huggingface", "datasets"):
            return HFDatasetLoader(cfg)
        raise ValueError(f"Unsupported dataset format: {cfg.format}")


# -----------------------------------------------------------------------------
# Convenience: tiny in-memory Dataset wrapper
# -----------------------------------------------------------------------------


class Dataset:
    """\
    @zh 简单的数据集容器：按 split 缓存 `NewsRecord` 列表，便于下游 batcher 使用。
    @en Simple in-memory container of `NewsRecord` lists per split.
    """

    def __init__(
        self, loader: BaseLoader, *, preload: Optional[Sequence[str]] = None
    ) -> None:
        self.loader = loader
        self._data: Dict[str, List[NewsRecord]] = {}
        if preload:
            for sp in preload:
                self._data[sp] = loader.load_split(sp)

    def get(self, split: str) -> List[NewsRecord]:
        if split not in self._data:
            self._data[split] = self.loader.load_split(split)
        return self._data[split]

    def __len__(self) -> int:  # pragma: no cover - aggregate length
        return sum(len(v) for v in self._data.values())

    def close(self) -> None:
        self.loader.close()


# -----------------------------------------------------------------------------
# Example factory from a dict-like config (e.g., loaded YAML)
# -----------------------------------------------------------------------------


def loader_from_config(cfg_dict: Mapping[str, Any]) -> BaseLoader:
    """\
    @zh 从 dict 配置构建加载器（与 YAML 对齐）。
    @en Build loader from a dict-like config (aligns with YAML files).
    
    Example:
    {
      "name": "politifact",
      "format": "csv",
      "path": "data/politifact",
      "splits": {"train": "train.csv", "validation": "val.csv", "test": "test.csv"},
      "fields": {"id": "id", "text": "content", "label": "label"},
      "label_map": {"real": 0, "fake": 1},
      "id_prefix": "pf",
      "drop_duplicates_on_text": true
    }
    """
    fields = cfg_dict.get("fields") or {}
    fm = FieldMap(**fields)
    cfg = DatasetConfig(
        name=cfg_dict.get("name", "dataset"),
        format=cfg_dict.get("format", "csv"),
        path=cfg_dict.get("path"),
        hf_name=cfg_dict.get("hf_name"),
        hf_config=cfg_dict.get("hf_config"),
        splits=cfg_dict.get("splits"),
        delimiter=cfg_dict.get("delimiter", ","),
        encoding=cfg_dict.get("encoding", "utf-8"),
        lines=cfg_dict.get("lines", True),
        fields=fm,
        label_map=cfg_dict.get("label_map"),
        id_prefix=cfg_dict.get("id_prefix"),
        drop_duplicates_on_text=cfg_dict.get("drop_duplicates_on_text", False),
        lowercase_text=cfg_dict.get("lowercase_text", False),
        strip_text=cfg_dict.get("strip_text", True),
    )
    return build_loader(cfg)


# -----------------------------------------------------------------------------
# Doxygen examples (bilingual) for maintainers
# -----------------------------------------------------------------------------

__doc_examples__ = r"""
/**
 * @zh 用法示例（CSV）：
 * ```python
 * from kan.data.loaders import loader_from_config, Dataset
 * from kan.utils.logging import configure_logging
 * configure_logging()
 * cfg = {
 *   "name": "gossipcop",
 *   "format": "csv",
 *   "path": "./data/gossipcop",
 *   "splits": {"train": "train.csv", "validation": "dev.csv", "test": "test.csv"},
 *   "fields": {"id": "nid", "text": "text", "label": "label"},
 *   "label_map": {"true": 0, "false": 1},
 *   "id_prefix": "gc",
 * }
 * loader = loader_from_config(cfg)
 * ds = Dataset(loader, preload=["train"])  # 可选预加载
 * train_records = ds.get("train")          # -> List[NewsRecord]
 * print(train_records[0].text[:80])
 * ds.close()
 * ```
 *
 * @en Usage (HF Datasets):
 * ```python
 * from kan.data.loaders import DatasetConfig, build_loader
 * cfg = DatasetConfig(name="pheme", format="hf", hf_name="pheme", hf_config=None,
 *                     fields=FieldMap(text="text", label="label"),
 *                     label_map={"nonrumor": 0, "rumor": 1})
 * loader = build_loader(cfg)
 * records = loader.load_split("train")
 * ```
 */
"""
