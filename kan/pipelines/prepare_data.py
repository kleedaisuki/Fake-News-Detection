# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kan/pipelines/prepare_data.py
@brief  Pipeline: 准备数据（下载/清洗/实体链接/知识抓取/向量化/词表）→ 产出可复用的数据缓存与清单。
@date   2025-09-16

设计要点（与 pipelines/train_trainer.py 一致）：
- 这是**流程编排**，不在此处实现具体算法；实际工作委托给 `kan.data.*` 与 `kan.modules.*`。
- 超参数从 `configs/` 合并，并支持点号覆盖；将最终配置快照与指纹写入产物目录。
- 明确区分 **过程** 与 **结果**：
  * 过程（Process Cache）：`cache/` 下的原始镜像、EL/KG 中间层、embedding 索引、持久化的 vocab 等。
  * 结果（Reusable Dataset）：`cache/datasets/<name>/<fingerprint>/` 下按 split 分片的 `*.jsonl.gz` 与 `manifest.json`。
- 可重入 / 可复用：若检测到同一数据指纹且 `--no-force`，则直接复用；否则重新构建。
- Windows 友好：尽量避免多进程；文件路径用 `pathlib.Path`；符号链接不是硬要求。
"""

import argparse
import copy
import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import torch

# ------------------------------
# 日志（集中式，若不可用则回退到标准 logging）
# ------------------------------
try:
    from kan.utils.logging import configure_logging, log_context
    import logging

    LOGGER = logging.getLogger("kan.pipelines.prepare_data")
except Exception:  # 兼容仓库尚未落位 `kan/utils/logging.py` 的情况
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s | %(message)s"
    )
    LOGGER = logging.getLogger("kan.pipelines.prepare_data")

    def configure_logging(*args, **kwargs):  # type: ignore
        pass

    from contextlib import contextmanager

    @contextmanager
    def log_context(**kwargs):  # type: ignore
        yield


# ------------------------------
# YAML 合并与覆盖
# ------------------------------
try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("缺少 PyYAML，请 `pip install pyyaml`。") from e


def _deep_update(
    base: MutableMapping[str, Any], other: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for k, v in other.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            _deep_update(base[k], v)  # type: ignore
        else:
            base[k] = copy.deepcopy(v)
    return base


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_overrides(cfg: MutableMapping[str, Any], overrides: Sequence[str]) -> None:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"覆盖项格式错误：{ov}，应为 key.subkey=VALUE")
        key, val = ov.split("=", 1)
        try:
            val_conv = json.loads(val)
        except Exception:
            val_conv = val
        cur: MutableMapping[str, Any] = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], MutableMapping):
                cur[p] = {}
            cur = cur[p]  # type: ignore
        cur[parts[-1]] = val_conv


# ------------------------------
# Utils
# ------------------------------


def _stable_fingerprint(obj: Any) -> str:
    """对 dict/list/标量做稳定 JSON 序列化后哈希，作为数据配置指纹。"""
    b = json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha1(b).hexdigest()[:16]


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split())


# ------------------------------
# 具体步骤的桥接：委托 kan.data.* 模块
# ------------------------------


def _build_loader(data_cfg: Mapping[str, Any]):
    from kan.data.loaders import loader_from_config  # type: ignore

    return loader_from_config(data_cfg)


def _build_entity_linker(el_cfg: Optional[Mapping[str, Any]]):
    if not el_cfg:
        return None
    try:
        from kan.utils.registry import HUB, build_from_config

        EL = HUB.get_or_create("entity_linker")
        return build_from_config(el_cfg, EL)
    except Exception:
        try:
            from kan.data.entity_linking import build_entity_linker  # type: ignore

            return build_entity_linker(el_cfg)
        except Exception:
            LOGGER.warning("未找到 entity_linker，实现缺失，将跳过 EL。")
            return None


def _build_kg_fetcher(kg_cfg: Optional[Mapping[str, Any]]):
    if not kg_cfg:
        return None
    try:
        from kan.utils.registry import HUB, build_from_config

        KG = HUB.get_or_create("kg_fetcher")
        return build_from_config(kg_cfg, KG)
    except Exception:
        try:
            from kan.data.kg_fetcher import build_kg_fetcher  # type: ignore

            return build_kg_fetcher(kg_cfg)
        except Exception:
            LOGGER.warning("未找到 kg_fetcher，实现缺失，将跳过 KG 抓取。")
            return None


def _build_vectorizer(vec_cfg: Optional[Mapping[str, Any]]):
    if not vec_cfg:
        return None
    try:
        from kan.utils.registry import HUB, build_from_config

        VEC = HUB.get_or_create("vectorizer")
        return build_from_config(vec_cfg, VEC)
    except Exception:
        try:
            from kan.data.vectorizer import build_vectorizer  # type: ignore

            return build_vectorizer(vec_cfg)
        except Exception:
            LOGGER.warning("未找到 vectorizer，实现缺失，将跳过向量化。")
            return None


# ------------------------------
# 统计信息（轻量）
# ------------------------------


def _compute_stats(records: List[Mapping[str, Any]]) -> Mapping[str, Any]:
    n = len(records)
    if n == 0:
        return {"count": 0}
    lens = []
    ent_counts = []
    for r in records:
        text = r.get("text") or r.get("content") or ""
        lens.append(len(_normalize_text(text).split()))
        ents = r.get("entities") or []
        ent_counts.append(len(ents))
    import statistics as st

    return {
        "count": n,
        "text_len": {
            "mean": float(st.mean(lens)),
            "median": float(st.median(lens)),
            "p95": float(sorted(lens)[int(0.95 * n) - 1]),
        },
        "entities": {
            "mean": float(st.mean(ent_counts)),
            "median": float(st.median(ent_counts)),
        },
    }


# ------------------------------
# 主流程：准备数据
# ------------------------------


def run_from_configs(
    config_paths: Sequence[str], overrides: Sequence[str] = (), force: bool = False
) -> Path:
    """合并配置→准备数据→返回**可复用数据缓存**目录。"""
    # 1) 合并配置
    cfg: MutableMapping[str, Any] = {}
    for p in config_paths:
        cp = Path(p)
        assert cp.exists(), f"配置文件不存在：{cp}"
        _deep_update(cfg, _read_yaml(cp))
    _apply_overrides(cfg, list(overrides))

    data_cfg = cfg.get("data") or cfg
    name = data_cfg.get("name", "dataset")
    # 指纹仅由“影响数据内容”的字段组成，避免把纯训练参数搀进去
    fp_keys = data_cfg.get("fingerprint_keys") or [
        "source",
        "splits",
        "preprocess",
        "entity_linking",
        "kg",
        "vectorizer",
        "filters",
        "normalize",
    ]
    content_for_fp = {k: data_cfg.get(k) for k in fp_keys}
    fingerprint = _stable_fingerprint(content_for_fp)

    # 2) 目录：cache（复用）与 runs（记录本次执行）
    cache_root = Path(cfg.get("cache_dir", "cache"))
    ds_cache = cache_root / "datasets" / name / fingerprint
    _ensure_dir(ds_cache)

    runs_root = Path(cfg.get("output_dir", "runs"))
    run_id = cfg.get("run_id") or f"prep-{name}-{fingerprint}"
    out_dir = _ensure_dir(runs_root / run_id)
    logs_dir = _ensure_dir(out_dir / "logs")
    try:
        configure_logging(log_dir=logs_dir)
    except Exception:
        pass

    with log_context(run_id=str(run_id), stage="prepare", step=0):
        LOGGER.info("dataset=%s fingerprint=%s cache=%s", name, fingerprint, ds_cache)
        # 保存合并配置快照
        (out_dir / "configs_merged.yaml").write_text(
            yaml.safe_dump(dict(cfg), allow_unicode=True), encoding="utf-8"
        )
        (out_dir / "fingerprint.json").write_text(
            json.dumps({"fingerprint": fingerprint}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest_path = ds_cache / "manifest.json"
        if manifest_path.exists() and not force:
            LOGGER.info("检测到已存在的数据缓存，直接复用：%s", ds_cache)
            return ds_cache

        # 3) 加载原始数据
        loader = _build_loader(data_cfg)
        available_splits = data_cfg.get(
            "splits", {"train": "train", "validation": "validation", "test": "test"}
        )

        raw_dir = _ensure_dir(ds_cache / "raw")
        pre_dir = _ensure_dir(ds_cache / "prepared")
        meta_dir = _ensure_dir(ds_cache / "meta")

        split_records: Dict[str, List[Mapping[str, Any]]] = {}
        for split_alias, split_name in available_splits.items():
            LOGGER.info("加载 split=%s(%s)", split_alias, split_name)
            if hasattr(loader, "has_split") and not loader.has_split(split_name):
                LOGGER.warning("数据集缺少 split=%s，跳过。", split_name)
                continue
            recs = loader.load_split(split_name)
            # 可选：保存原始镜像，便于可溯源
            if bool(data_cfg.get("save_raw", True)):
                _write_jsonl_gz(raw_dir / f"{split_alias}.jsonl.gz", recs)
            split_records[split_alias] = recs

        # 4) 轻量清洗 / 过滤 / 归一化
        #    仅做最小必要工作（去重、空文本过滤、标准字段化），严肃处理交给上游模块
        def _dedup(records: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
            seen = set()
            out = []
            key = data_cfg.get("dedup_key") or "id"
            by_text = key == "__text__"
            for r in records:
                k = (
                    _normalize_text(r.get("text") or r.get("content") or "")
                    if by_text
                    else (r.get(key) or _normalize_text(r.get("text") or ""))
                )
                if not k or k in seen:
                    continue
                seen.add(k)
                out.append(r)
            return out

        for split, recs in list(split_records.items()):
            before = len(recs)
            recs = [
                r
                for r in recs
                if _normalize_text(r.get("text") or r.get("content") or "")
            ]
            recs = _dedup(recs)
            LOGGER.info("split=%s 清洗：%d → %d", split, before, len(recs))
            split_records[split] = recs

        # 5) 实体链接（EL）
        el = _build_entity_linker(data_cfg.get("entity_linking"))
        if el is not None:
            for split, recs in split_records.items():
                LOGGER.info("实体链接：split=%s, n=%d", split, len(recs))
                recs = el.link(
                    recs
                )  # 约定返回附加 `entities`: List[{'id':..., 'surface':..., 'offset':...}]
                split_records[split] = recs

        # 6) 知识抓取（KG fetch）
        kg = _build_kg_fetcher(data_cfg.get("kg"))
        if kg is not None:
            # 收集所有实体 id 以批量抓取
            all_ent_ids: List[str] = []
            for recs in split_records.values():
                for r in recs:
                    for e in r.get("entities") or []:
                        eid = e.get("id") or e.get("kb_id") or e.get("wiki_id")
                        if eid:
                            all_ent_ids.append(str(eid))
            uniq_ids = sorted(set(all_ent_ids))
            LOGGER.info("KG 抓取实体总数=%d（去重后）", len(uniq_ids))
            kg_cache_dir = _ensure_dir(cache_root / "kg")
            kg.fetch(uniq_ids, out_dir=kg_cache_dir)  # 由实现负责去重/速率限制/持久化

        # 7) 向量化（文本/实体/上下文）
        vec = _build_vectorizer(data_cfg.get("vectorizer"))
        if vec is not None:
            for split, recs in split_records.items():
                LOGGER.info("向量化：split=%s, n=%d", split, len(recs))
                recs = vec.transform(
                    recs
                )  # 约定附加诸如 'text_vec'/'ent_vecs'/'ctx_vecs'
                split_records[split] = recs

        # 8) 词表与 batcher 配置（供下游复用）
        try:
            from kan.data.batcher import Batcher, BatcherConfig, TextConfig, EntityConfig, ContextConfig  # type: ignore

            batcher = Batcher(
                BatcherConfig(
                    text=TextConfig(**(data_cfg.get("batcher", {}).get("text", {}))),
                    entity=EntityConfig(
                        **(data_cfg.get("batcher", {}).get("entity", {}))
                    ),
                    context=ContextConfig(
                        **(data_cfg.get("batcher", {}).get("context", {}))
                    ),
                    device="cpu",
                )
            )
            # 使用 train split 构建词表（必要时也能支持 union 策略）
            if split_records.get("train"):
                batcher.build_vocabs(split_records["train"])  # 内部可选择是否持久化
            # 持久化词表到 ds_cache
            voc_dir = _ensure_dir(meta_dir / "vocabs")
            if hasattr(batcher, "save_vocabs"):
                batcher.save_vocabs(voc_dir)
        except Exception as e:
            LOGGER.warning("Batcher/vocab 跳过（未实现或异常）：%s", e)

        # 9) 写出 prepared 分片与统计
        manifest: Dict[str, Any] = {
            "name": name,
            "fingerprint": fingerprint,
            "splits": {},
            "paths": {"root": str(ds_cache)},
            "components": {
                "entity_linking": bool(el is not None),
                "kg": bool(kg is not None),
                "vectorizer": bool(vec is not None),
            },
            "config_snapshot": content_for_fp,
        }

        for split, recs in split_records.items():
            split_dir = _ensure_dir(pre_dir / split)
            shard_path = split_dir / "shard-00001.jsonl.gz"
            n = _write_jsonl_gz(shard_path, recs)
            stats = _compute_stats(recs)
            _write_json(split_dir / "stats.json", stats)
            manifest["splits"][split] = {
                "count": n,
                "shards": [str(shard_path)],
                "stats": stats,
            }
            LOGGER.info("写出 split=%s 条目=%d → %s", split, n, shard_path)

        _write_json(manifest_path, manifest)
        LOGGER.info("数据准备完成：%s", ds_cache)
        return ds_cache


# ------------------------------
# CLI 入口
# ------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare dataset pipeline: load→clean→EL→KG→vectorize→vocab→manifest"
    )
    p.add_argument(
        "-c",
        "--config",
        nargs="+",
        required=True,
        help="YAML 配置文件列表，后者覆盖前者",
    )
    p.add_argument(
        "-o",
        "--override",
        nargs="*",
        default=[],
        help="点号覆盖，如 data.save_raw=false vectorizer.type=...",
    )
    p.add_argument("--force", action="store_true", help="若数据缓存已存在也强制重建")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover
    args = build_argparser().parse_args(argv)
    run_from_configs(args.config, args.override, force=args.force)


if __name__ == "__main__":  # pragma: no cover
    main()
