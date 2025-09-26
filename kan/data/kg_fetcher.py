# -*- coding: utf-8 -*-
from __future__ import annotations

"""
@file   kan/pipelines/kg_fetcher.py
@brief  Knowledge-graph context fetcher for KAN: fetch 1-hop neighbors for entities.
@date   2025-09-16

@zh
  以配置为中心（config-driven）的知识图谱上下文拉取器：
  - 输入：`List[NewsRecord]`（其中 `entities` 已被实体链接阶段填充为 Wikidata QID 列表）
  - 输出：回填 `contexts: Dict[str, List[str]]` 映射（entity QID -> 邻接列表），并在 `meta['kg']` 记录追溯信息
  - 设计：后端可插拔（本地 JSON 词典 / Wikidata REST / SPARQL），统一注册表、缓存、限速与容错

@en
  Config-driven KG context fetcher:
  - Input: `List[NewsRecord]` with `entities` populated by the EL stage (Wikidata QIDs)
  - Output: fill `contexts: Dict[str, List[str]]` mapping (entity QID -> neighbors) and
    write tracing info into `meta['kg']` for reproducibility.
  - Backends: pluggable (local JSON, Wikidata REST, SPARQL). Unified registry/cache.

@contract
  def fetch_context(records: List[NewsRecord], cfg: KGConfig) -> List[NewsRecord]
    * Must not mutate input unless `inplace=True` is requested (default False).
    * Must deduplicate neighbors; apply top-K truncation if configured.
    * Caching unit is **per QID** (content-addressed by backend signature + QID).
    * Windows-friendly (no POSIX-only assumptions) & deterministic.

@notes
  - 本模块仅负责**邻接获取**（一跳为主）。更复杂的子图裁剪/路径搜索可在上游 pipeline 扩展。
  - 远端网络后端默认不启用；LocalProvider 提供零依赖可运行路径，利于 CI。
  - 统一日志命名空间：`kan.pipelines.kg_fetcher`。
"""

from dataclasses import dataclass, asdict, field
from hashlib import blake2b
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import time
import os

# Stable NewsRecord schema
from kan.data.loaders import NewsRecord

# Optional deps guarded
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional
    requests = None  # type: ignore

# Registry hub (optional at import time)
try:
    from kan.utils.registry import HUB

    _KG_REG = HUB.get_or_create("kg_fetcher")
except Exception:  # pragma: no cover - registry may be unavailable at import time

    class _DummyReg:
        def register(self, *_a, **_k):
            def deco(x):
                return x

            return deco

        def get(self, *_a, **_k):
            return None

    _KG_REG = _DummyReg()  # type: ignore

LOGGER = logging.getLogger("kan.pipelines.kg_fetcher")

_ENV_TTL_DAYS = "KAN_KG_CACHE_TTL_DAYS"  # 默认 7
_ENV_MAX_WORKERS = "KAN_KG_MAX_WORKERS"  # 默认 min(32, cpu*5)
_ENV_USE_JSONL_INDEX = "KAN_KG_LOCAL_JSONL"  # "1" 时把 local_path 当作 JSONL

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class KGConfig:
    """Knowledge Graph fetcher config (align with configs/pipelines/kg_fetcher.yaml).

    @fields
      backend: 后端标识（local / wikidata_rest / sparql / custom）
      cache_dir: 缓存目录（按 QID 切分的 JSON 文件）
      topk: 每个 QID 保留的邻居上限（None 表示不限）
      return_edges: 输出类型（'none' 仅返回邻居 QID；'labeled' 返回"Pxxxx=Qyyyy"）
      properties: 仅保留这些属性的边（如 ["P31","P279","P361","P17","P27","P106","P463"]）；None 表示不过滤
      direction: 'out' | 'in' | 'both'
      rate_limit: 每秒请求数上限（远端后端用；<=0 表示不节流）
      timeout: 单请求超时（秒）
      inplace: 是否原地回填

      # backend-specific
      local_path: 本地 JSON 词典（两种形态之一）：
                  1) 单文件：{"Q76": ["Q...","P31=Q5", ...], ...}
                  2) 目录：  每个 QID 一个 JSON：{"neighbors": ["Q..." 或 "P..=Q..", ...]}
      wikidata_endpoint: REST 端点（例如 https://www.wikidata.org/wiki/Special:EntityData/ 变体）
      sparql_endpoint:   SPARQL 端点（例如 https://query.wikidata.org/sparql）
    """

    backend: str = "local"
    cache_dir: Optional[str] = ".cache/kg"
    topk: Optional[int] = 64
    return_edges: str = "none"  # or "labeled"
    properties: Optional[List[str]] = None
    direction: str = "both"
    rate_limit: float = 0.0
    timeout: float = 15.0
    inplace: bool = False

    # backend specific
    local_path: Optional[str] = None
    wikidata_endpoint: Optional[str] = None
    sparql_endpoint: Optional[str] = None


# -----------------------------------------------------------------------------
# Cache (per QID)
# -----------------------------------------------------------------------------


class _Cache:
    def __init__(self, dir_: Optional[str], backend_sig: str) -> None:
        self.dir = Path(dir_) if dir_ else None
        self.backend_sig = backend_sig
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
        try:
            self.ttl_days = int(os.environ.get(_ENV_TTL_DAYS, "7"))
        except Exception:
            self.ttl_days = 7

    def _key(self, qid: str) -> Path:
        # Stable shard by backend signature and QID
        h = blake2b(
            (self.backend_sig + "\n" + qid).encode("utf-8"), digest_size=12
        ).hexdigest()
        assert self.dir is not None
        return self.dir / h[:2] / f"{qid}.json"

    def get(self, qid: str) -> Optional[List[str]]:
        if not self.dir:
            return None
        p = self._key(qid)
        if not p.parent.exists() or not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            # backward compatible: old cache may not have "ts"
            ts = data.get("ts")
            if ts:
                from datetime import datetime, timezone, timedelta

                try:
                    dt = datetime.fromisoformat(ts)
                    if (datetime.now(timezone.utc) - dt).days > self.ttl_days:
                        return None  # expired
                except Exception:
                    pass
            v = data.get("neighbors", [])
            if isinstance(v, list):
                return [str(x) for x in v]
        except Exception:
            return None
        return None

    def put(self, qid: str, neighbors: List[str]) -> None:
        if not self.dir:
            return
        p = self._key(qid)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"neighbors": neighbors, "ts": _now_iso()}
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# -----------------------------------------------------------------------------
# Provider interface
# -----------------------------------------------------------------------------


class BaseProvider:
    name: str = "base"
    version: str = "0"

    def __init__(self, cfg: KGConfig) -> None:
        self.cfg = cfg

    def fetch(self, qid: str) -> List[Tuple[str, Optional[str]]]:
        """Fetch neighbors for a given QID.
        Returns list of (neighbor_qid, property) where property may be None.
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Local provider (zero dependency, CI friendly)
# -----------------------------------------------------------------------------


@_KG_REG.register("local", alias=["file", "json"])  # type: ignore[attr-defined]
class LocalProvider(BaseProvider):
    name = "local"
    version = "1"

    def __init__(self, cfg: KGConfig) -> None:
        super().__init__(cfg)
        self.index: Dict[str, List[str]] = {}  # single-file JSON cache
        self._jsonl_offsets: Dict[str, int] = {}  # qid -> byte offset (JSONL)
        self._jsonl_fp: Optional[io.BufferedReader] = None
        self._lru: Dict[str, List[str]] = {}
        self._lru_cap = 4096  # tiny hotset
        path = Path(cfg.local_path) if cfg.local_path else None
        if path and path.exists():
            import io

            # Prefer JSONL when env enabled or file endswith .jsonl
            use_jsonl = (
                os.environ.get(_ENV_USE_JSONL_INDEX, "0") == "1"
                or path.suffix.lower() == ".jsonl"
            )
            if use_jsonl:
                # build a lightweight offset index: one pass, store qid -> offset
                self._jsonl_fp = open(path, "rb")
                pos = 0
                for line in self._jsonl_fp:
                    try:
                        obj = json.loads(line)
                        qid = str(obj.get("qid", ""))
                        if qid.startswith("Q"):
                            self._jsonl_offsets[qid] = pos
                    except Exception:
                        pass
                    pos = self._jsonl_fp.tell()
            else:
                # original single-file JSON (may be huge)
                try:
                    self.index = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    self.index = {}
        else:
            # directory mode: defer IO to fetch()
            self.index = {}
        LOGGER.info("LocalProvider ready: path=%s", str(path) if path else None)

    def fetch(self, qid: str) -> List[Tuple[str, Optional[str]]]:
        # LRU hot cache for directory/JSONL modes
        if qid in self._lru:
            return [(x.split("=",1)[-1] if "=" in x else x,
                     x.split("=",1)[0] if "=" in x else None) for x in self._lru[qid]]
        raw: List[str] = []
        if self.index:
            raw = self.index.get(qid, [])
        elif self._jsonl_offsets:
            # random access JSONL
            try:
                off = self._jsonl_offsets.get(qid)
                if off is not None and self._jsonl_fp:
                    self._jsonl_fp.seek(off)
                    obj = json.loads(self._jsonl_fp.readline())
                    raw = obj.get("neighbors", [])
            except Exception:
                raw = []
        else:
            # directory mode: {dir}/{qid}.json with {"neighbors": [ ... ]}
            if not self.cfg.local_path:
                return []
            p = Path(self.cfg.local_path) / f"{qid}.json"
            if not p.exists():
                return []
            try:
                raw = json.loads(p.read_text(encoding="utf-8")).get("neighbors", [])
            except Exception:
                return []
        out: List[Tuple[str, Optional[str]]] = []
        for x in raw:
            if isinstance(x, str) and "=" in x:
                prop, val = x.split("=", 1)
                if val and val.startswith("Q"):
                    out.append((val, prop))
            elif isinstance(x, str) and x.startswith("Q"):
                out.append((x, None))
        # put into tiny LRU for hot reuse
        try:
            if raw:
                if len(self._lru) >= self._lru_cap:
                    self._lru.pop(next(iter(self._lru)))
                self._lru[qid] = raw
        except Exception:
            pass
        return out


# -----------------------------------------------------------------------------
# Wikidata REST provider (placeholder; optional network)
# -----------------------------------------------------------------------------


@_KG_REG.register("wikidata_rest")  # type: ignore[attr-defined]
class WikidataRESTProvider(BaseProvider):
    name = "wikidata_rest"
    version = "0"

    def fetch(
        self, qid: str
    ) -> List[Tuple[str, Optional[str]]]:  # pragma: no cover - network placeholder
        if requests is None:
            raise RuntimeError("'requests' is not available. Please install requests.")
        if not self.cfg.wikidata_endpoint:
            # Default to JSON summary endpoint style if not provided
            base = "https://www.wikidata.org/wiki/Special:EntityData/"
        else:
            base = self.cfg.wikidata_endpoint.rstrip("/") + "/"
        url = f"{base}{qid}.json"
        r = requests.get(url, timeout=self.cfg.timeout)
        r.raise_for_status()
        data = r.json()
        # parse statements: entities[qid]['claims'][Pxxxx] -> list[mainsnak.datavalue.value.id]
        try:
            claims = data["entities"][qid]["claims"]
        except Exception:
            return []
        out: List[Tuple[str, Optional[str]]] = []
        for prop, arr in claims.items():
            if self.cfg.properties and prop not in self.cfg.properties:
                continue
            for c in arr:
                try:
                    v = c["mainsnak"]["datavalue"]["value"]
                    if isinstance(v, dict) and "id" in v and v["id"].startswith("Q"):
                        out.append((v["id"], prop))
                except Exception:
                    continue
        return out


# -----------------------------------------------------------------------------
# SPARQL provider (placeholder; optional network)
# -----------------------------------------------------------------------------


@_KG_REG.register("sparql")  # type: ignore[attr-defined]
class SPARQLProvider(BaseProvider):
    name = "sparql"
    version = "0"

    def fetch(
        self, qid: str
    ) -> List[Tuple[str, Optional[str]]]:  # pragma: no cover - network placeholder
        if requests is None:
            raise RuntimeError("'requests' is not available. Please install requests.")
        if not self.cfg.sparql_endpoint:
            endpoint = "https://query.wikidata.org/sparql"
        else:
            endpoint = self.cfg.sparql_endpoint
        # Simplified 1-hop both-direction query; throttle by cfg.rate_limit in caller
        query = f"""
        SELECT ?p ?q WHERE {{
          {{ wd:{qid} ?p ?qnode . BIND(STR(AFTER(STR(?qnode), 'entity/')) AS ?q) }}
          UNION
          {{ ?qnode ?p wd:{qid} . BIND(STR(AFTER(STR(?qnode), 'entity/')) AS ?q) }}
          FILTER(STRSTARTS(?q, 'Q'))
        }} LIMIT 2000
        """
        r = requests.get(
            endpoint,
            params={"query": query, "format": "json"},
            timeout=self.cfg.timeout,
            headers={"User-Agent": "KAN-KGFetcher/0.1"},
        )
        r.raise_for_status()
        data = r.json()
        out: List[Tuple[str, Optional[str]]] = []
        for b in data.get("results", {}).get("bindings", []):
            p = b.get("p", {}).get("value", "")
            q = b.get("q", {}).get("value", "")
            # normalize property id from url
            if "property/" in p:
                prop = "P" + p.split("property/")[-1]
            else:
                prop = None
            if q.startswith("Q"):
                out.append((q, prop))
        return out


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


def build_provider(cfg: KGConfig) -> BaseProvider:
    key = (cfg.backend or "local").lower()
    try:
        from kan.utils.registry import HUB

        reg = HUB.get_or_create("kg_fetcher")
        klass = reg.get(key)
        if klass is None:
            raise KeyError(key)
        return klass(cfg)
    except Exception:
        if key in ("local", "file", "json"):
            return LocalProvider(cfg)
        if key == "wikidata_rest":
            return WikidataRESTProvider(cfg)
        if key == "sparql":
            return SPARQLProvider(cfg)
        raise ValueError(f"Unsupported KG backend: {cfg.backend}")


# -----------------------------------------------------------------------------
# Public pipeline function
# -----------------------------------------------------------------------------


def fetch_context(
    records: List[NewsRecord], cfg: KGConfig, *, inplace: Optional[bool] = None
) -> List[NewsRecord]:
    """Fetch 1-hop neighbors for entities and return updated records.

    @params
      records: 输入样本列表（需要 rec.entities 已填充为 QID 列表）
      cfg:     KG 配置
      inplace: 是否原地修改（覆盖 cfg.inplace）
    """
    if not records:
        return [] if not cfg.inplace else records

    inplace = cfg.inplace if inplace is None else inplace
    out = records if inplace else deepcopy(records)

    provider = build_provider(cfg)

    backend_sig = json.dumps(
        {
            "backend": provider.__class__.__name__,
            "version": getattr(provider, "version", "0"),
            "return_edges": cfg.return_edges,
            "properties": tuple(cfg.properties) if cfg.properties else None,
            "direction": cfg.direction,
            "topk": cfg.topk,
        },
        sort_keys=True,
    )

    cache = _Cache(cfg.cache_dir, backend_sig)

    # Rate limit (approx) & concurrency
    last_req_ts = 0.0
    rps = cfg.rate_limit if cfg.rate_limit and cfg.rate_limit > 0 else None
    try:
       import os
       max_workers = int(os.environ.get(_ENV_MAX_WORKERS, "0")) or max(4, min(32, (os.cpu_count() or 8)*5))
    except Exception:
       max_workers = 16

    total_q = 0
    cache_hit = 0

    # Pre-collect unique QIDs across records to amortize fetching
    qids: List[str] = []
    for rec in out:
        for q in rec.entities or []:
            if q and q.startswith("Q"):
                qids.append(q)
    uniq_qids = sorted(set(qids))

    qid2neighbors: Dict[str, List[str]] = {}
    errors: List[Dict[str, str]] = []

    import threading, time
    lock = threading.Lock()
    # naive rate limiter: serialize "outbound" by sleeping per rps
    rate_lock = threading.Lock()

    def _work(qid: str) -> None:
        nonlocal cache_hit, total_q, last_req_ts
        with lock:
            total_q += 1
        # cache
        cached = cache.get(qid)
        if cached is not None:
            with lock:
                qid2neighbors[qid] = cached
                cache_hit += 1
            return
        # throttle (approx)
        if rps:
            with rate_lock:
                now = time.time()
                min_interval = 1.0 / rps
                sleep_t = last_req_ts + min_interval - now
                if sleep_t > 0:
                    time.sleep(sleep_t)
                last_req_ts = time.time()
        # fetch + isolate errors
        try:
            pairs = provider.fetch(qid)
            if cfg.properties:
                pairs = [(n,p) for (n,p) in pairs if (p in cfg.properties) or (p is None and cfg.return_edges=="none")]
            neighbors = [f"{p}={n}" if (cfg.return_edges=="labeled" and p) else n for (n,p) in pairs]
            neighbors = _unique_preserve(neighbors)
            if cfg.topk is not None and cfg.topk > 0:
                neighbors = neighbors[: cfg.topk]
            cache.put(qid, neighbors)
            with lock:
                qid2neighbors[qid] = neighbors
        except Exception as e:
            LOGGER.warning("KG fetch failed for %s: %s", qid, e)
            with lock:
                qid2neighbors[qid] = []
                errors.append({"qid": qid, "error": str(e)})

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_work, q) for q in uniq_qids]
        for _ in as_completed(futures):
            pass  # join

        # traceability
        rec.meta = dict(rec.meta or {})
        rec.meta.setdefault("kg", {})
        rec.meta["kg"].update(
            {
                "backend": provider.__class__.__name__,
                "version": getattr(provider, "version", "0"),
                "time": _now_iso(),
                "qids": list(rec.entities or []),
                "stats": {
                    "uniq_qids": len(uniq_qids),
                    "total_q": total_q,
                    "cache_hit": cache_hit,
                },
                "return_edges": cfg.return_edges,
                "topk": cfg.topk,
                "properties": list(cfg.properties) if cfg.properties else None,
            }
        )

    LOGGER.info(
        "KG fetch done: records=%d, uniq_qids=%d, cache_hit=%d, backend=%s/%s",
        len(out),
        len(uniq_qids),
        cache_hit,
        provider.__class__.__name__,
        getattr(provider, "version", "0"),
    )
    return out


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _unique_preserve(arr: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in arr:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------------
# Doxygen examples (bilingual)
# -----------------------------------------------------------------------------

__doc_examples__ = r"""
/**
 * @zh 用法（Local provider, 单文件索引）：
 * ```python
 * from kan.data.loaders import NewsRecord
 * from kan.pipelines.kg_fetcher import KGConfig, fetch_context
 * # local JSON 形如：{"Q76": ["Q30","P31=Q5"], "Q567": ["Q183","P27=Q183"]}
 * cfg = KGConfig(backend="local", local_path="./kg.local.json", return_edges="labeled", topk=32)
 * recs = [NewsRecord(id="x1", text="...", label=1, entities=["Q76","Q567"], contexts={}, meta={})]
 * out = fetch_context(recs, cfg, inplace=False)
 * print(out[0].contexts["Q76"])  # -> ["Q30","P31=Q5", ...]  取决于你的词典
 * ```
 *
 * @en Usage (cache & per-QID files):
 * ```python
 * cfg = KGConfig(backend="local", local_path="./neighbors/", cache_dir=".cache/kg")
 * out = fetch_context(recs, cfg, inplace=True)  # modify in place
 * ```
 */
"""
