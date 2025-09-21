# -*- coding: utf-8 -*-
"""
@file   scripts/datasets/acquire_datasets.py
@brief  YAML 驱动的数据集获取与规范化落地（兼容旧 CLI）/ YAML-driven dataset acquisition (backward compatible)

[CN]
- 从 ./scripts/datasets/configs/ 读取 base.yaml + <name>.yaml，进行字段级合并；
- 通过 Source Driver 注册表分派到 HF/URL 驱动；
- 落地目录保持 ./datasets/<name>/{raw,extracted,processed}/；
- 生成增强版 dataset_card.json（含 spec_snapshot/spec_hash/driver/command_line）。

[EN]
- Load base.yaml + <name>.yaml from ./scripts/datasets/configs/; merge fields;
- Dispatch via Source Driver registry (HF/URL provided);
- Preserve output layout ./datasets/<name>/{raw,extracted,processed}/;
- Emit enhanced dataset_card.json with spec_snapshot/spec_hash/driver/command_line.

兼容性（Compatibility）：
- 保留子命令 list/show/download/verify；仍支持 --spec-json/--spec-file 覆盖（最高优先级）。

依赖（Deps）：PyYAML, datasets, pandas, pyarrow, requests, tqdm
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import typing as t
from pathlib import Path

# ------------------------------
# Logging：优先使用项目 logging.py，其次回落到 std logging
# ------------------------------
try:
    import importlib

    _proj_logging = importlib.import_module("logging")
    get_logger = getattr(_proj_logging, "get_logger", None)
    if get_logger is None:
        init_logging = getattr(_proj_logging, "init_logging", None)
        if callable(init_logging):
            init_logging()
        import logging as _std

        _std.basicConfig(level=_std.INFO, format="[%(levelname)s] %(message)s")
        logger = _std.getLogger("acquire_datasets")
    else:
        logger = get_logger("acquire_datasets")
except Exception:
    import logging as _std

    _std.basicConfig(level=_std.INFO, format="[%(levelname)s] %(message)s")
    logger = _std.getLogger("acquire_datasets")

# ------------------------------
# Optional deps (lazy)
# ------------------------------
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # 延迟到使用处报错

try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401
except Exception:
    pq = None

try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    from datasets import load_dataset, DatasetDict  # type: ignore
except Exception:
    load_dataset = None
    DatasetDict = None

from tqdm import tqdm  # type: ignore

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# ------------------------------
# Dataclasses
# ------------------------------
@dataclasses.dataclass
class DatasetSpec:
    """@brief 数据集规格（Dataset specification）/ CN then EN

    @param name[str]  数据集名（目录名）/ dataset name used as folder
    @param source[str] 源类型：hf|url / source kind: hf|url
    @param hf_id[str|None]  当 source=hf 时的 HF 数据集 id / HF id when source==hf
    @param subset[str|None]  HF 子集 / HF subset
    @param urls[list[str]|None] URL 源的直链 / direct URLs when source==url
    @param checksums[dict[str,str]|None] 文件名->sha256 / filename to sha256
    @param format_hint[str|None] parquet|csv|jsonl 建议导出格式 / preferred export
    @param notes[str|None]  备注 / notes
    @param license[str|None]  许可信息 / license string
    @param provenance[dict|None] 来源元信息（主页/论文/revision）/ provenance dict
    @param tags[list[str]|None] 标签 / tags
    """

    name: str
    source: str
    hf_id: t.Optional[str] = None
    subset: t.Optional[str] = None
    urls: t.Optional[t.List[str]] = None
    checksums: t.Optional[t.Dict[str, str]] = None
    format_hint: t.Optional[str] = None
    notes: t.Optional[str] = None
    license: t.Optional[str] = None
    provenance: t.Optional[dict] = None
    tags: t.Optional[t.List[str]] = None


# ------------------------------
# Paths & helpers
# ------------------------------


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[2] / "datasets"


def _default_configs_root() -> Path:
    # ./scripts/datasets/configs/
    return Path(__file__).resolve().parents[0] / "configs"


def ensure_dirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_json(obj: t.Any, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("缺少 PyYAML，请安装 pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if v is None:
            # 空值不覆盖非空，跳过
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ------------------------------
# Validation（轻量 MVP）
# ------------------------------


def _validate_spec(d: dict, strict: bool = False) -> None:
    required = ["name", "source"]
    for k in required:
        if k not in d or d[k] in (None, ""):
            raise SystemExit(f"spec 缺少必填字段: {k}")
    src = d["source"].lower()
    if src not in {"hf", "url"}:
        raise SystemExit(f"未知 source: {src}")
    if src == "hf":
        if strict and not d.get("hf_id"):
            raise SystemExit("source=hf 需要 hf_id（strict 模式）")
    if src == "url":
        if strict and not d.get("urls"):
            raise SystemExit("source=url 需要 urls（strict 模式）")


# ------------------------------
# Driver registry & implementations
# ------------------------------


class AcquisitionError(RuntimeError):
    pass


def _layout(out_root: Path, name: str) -> tuple[Path, Path, Path, Path]:
    base = out_root / name
    raw_dir = base / "raw"
    extracted_dir = base / "extracted"
    processed_dir = base / "processed"
    for d in (raw_dir, extracted_dir, processed_dir):
        ensure_dirs(d)
    return base, raw_dir, extracted_dir, processed_dir


# ---- Safe extraction to prevent Zip Slip ----
import zipfile, tarfile


def _safe_extract_zip(zf: zipfile.ZipFile, target: Path) -> None:
    for m in zf.infolist():
        # Normalize and ensure within target
        dest = (target / m.filename).resolve()
        if not str(dest).startswith(str(target.resolve())):
            raise AcquisitionError(f"非法解压路径: {m.filename}")
    zf.extractall(target)


def _safe_extract_tar(tf: tarfile.TarFile, target: Path) -> None:
    for m in tf.getmembers():
        dest = (target / m.name).resolve()
        if not str(dest).startswith(str(target.resolve())):
            raise AcquisitionError(f"非法解压路径: {m.name}")
    tf.extractall(target)


# ---- Drivers ----


def driver_hf(spec: DatasetSpec, out_root: Path, fmt: str) -> None:
    """@brief HuggingFace 驱动 / HF driver.
    @param spec  规格 / spec
    @param out_root  输出根目录 / output root
    @param fmt  导出格式 parquet|csv|jsonl / export format
    """
    if load_dataset is None:
        raise AcquisitionError("HuggingFace datasets 未安装: pip install datasets")
    if pd is None and fmt in {"parquet", "csv"}:
        raise AcquisitionError("pandas 未安装: pip install pandas")
    assert spec.hf_id, f"{spec.name}: 缺少 hf_id"

    ds = load_dataset(spec.hf_id, spec.subset)  # type: ignore
    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
    else:
        splits = ["train"]
    logger.info("[%s] HF 拉取完成，splits=%s", spec.name, splits)

    base, _, _, processed_dir = _layout(out_root, spec.name)

    exported: list[str] = []
    for split in splits:
        split_ds = ds[split] if isinstance(ds, DatasetDict) else ds
        out_path = processed_dir / f"{split}.{fmt}"
        if out_path.exists():
            logger.info("跳过已存在: %s", out_path)
            continue
        if fmt == "parquet":
            df = split_ds.to_pandas()  # type: ignore  # TODO: 可改为流式导出
            df.to_parquet(out_path, index=False)
        elif fmt == "csv":
            df = split_ds.to_pandas()  # type: ignore
            df.to_csv(out_path, index=False)
        elif fmt == "jsonl":
            split_ds.to_json(out_path.as_posix(), orient="records", lines=True, force_ascii=False)  # type: ignore
        else:
            raise ValueError(f"不支持的导出格式: {fmt}")
        exported.append(out_path.name)
        logger.info("导出完成: %s", out_path)

    # dataset_card.json（增强）
    card = _make_card(
        name=spec.name,
        source="hf",
        extra={
            "hf_id": spec.hf_id,
            "subset": spec.subset,
            "exported": exported,
            "format": fmt,
        },
    )
    save_json(card, base / "dataset_card.json")
    logger.info("[%s] dataset_card.json 写入完成", spec.name)


def driver_url(spec: DatasetSpec, out_root: Path) -> None:
    """@brief URL 驱动 / URL driver.
    @param spec  规格 / spec
    @param out_root  输出根目录 / output root
    """
    if requests is None:
        raise AcquisitionError("requests 未安装: pip install requests")
    assert spec.urls, f"{spec.name}: 缺少 urls 配置"

    base, raw_dir, extracted_dir, _ = _layout(out_root, spec.name)
    downloaded: list[Path] = []

    for url in spec.urls:
        fn = url.split("/")[-1].split("?")[0]
        dst = raw_dir / fn
        if dst.exists():
            logger.info("跳过已存在: %s", dst)
            downloaded.append(dst)
            continue
        logger.info("下载 %s -> %s", url, dst)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dst, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        downloaded.append(dst)

    # optional verify
    if spec.checksums:
        for p in downloaded:
            fn = p.name
            if fn in spec.checksums:
                calc = sha256_of_file(p)
                expect = spec.checksums[fn].lower()
                if calc.lower() != expect:
                    raise AcquisitionError(f"校验失败: {fn} sha256 {calc} != {expect}")
                logger.info("校验通过: %s", fn)

    # try extract with sandboxing
    for p in downloaded:
        try:
            if zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as zf:
                    _safe_extract_zip(zf, extracted_dir)
                    logger.info("解压 zip: %s", p.name)
            elif tarfile.is_tarfile(p):
                with tarfile.open(p) as tf:
                    _safe_extract_tar(tf, extracted_dir)
                    logger.info("解压 tar: %s", p.name)
        except Exception as e:
            logger.warning("解压失败(忽略): %s: %s", p.name, e)

    card = _make_card(
        name=spec.name,
        source="url",
        extra={
            "downloaded_files": [p.name for p in downloaded],
            "checksums": spec.checksums or {},
        },
    )
    save_json(card, base / "dataset_card.json")
    logger.info("[%s] dataset_card.json 写入完成", spec.name)


# ------------------------------
# Card & context helpers
# ------------------------------

_MERGED_SPEC_SNAPSHOT: dict | None = None  # 最近一次合并后的快照


def _current_cli() -> list[str]:
    return ["python", *sys.argv]


def _make_card(name: str, source: str, extra: dict) -> dict:
    global _MERGED_SPEC_SNAPSHOT
    snap = _MERGED_SPEC_SNAPSHOT or {}
    snap_json = json.dumps(snap, ensure_ascii=False, sort_keys=True)
    return {
        "name": name,
        "source": source,
        "created_at": int(time.time()),
        "tool": Path(__file__).name,
        "version": 1,
        "spec_snapshot": snap,
        "spec_hash": sha256_text(snap_json),
        "driver": source,
        "command_line": _current_cli(),
        **extra,
    }


# ------------------------------
# Config loading & resolving
# ------------------------------


def _load_from_configs_dir(configs_root: Path, name: str) -> dict:
    base = configs_root / "base.yaml"
    if not base.exists():
        logger.warning("缺少 base.yaml：%s（将使用空默认）", base)
        merged: dict = {}
    else:
        merged = _load_yaml(base)

    ds = configs_root / f"{name}.yaml"
    if not ds.exists():
        raise SystemExit(f"未找到配置文件: {ds}")
    merged = _deep_merge(merged, _load_yaml(ds))
    return merged


def _inject_overrides(
    spec: dict, spec_file: Path | None, spec_json: str | None
) -> dict:
    # 支持现有 --spec-file/--spec-json 覆盖
    out = dict(spec)
    if spec_file:
        text = spec_file.read_text(encoding="utf-8")
        if spec_file.suffix in {".json", ".jsonl", ".ndjson"}:
            lines = [line for line in text.splitlines() if line.strip()]
            if len(lines) == 1:
                obj = json.loads(text)
                if isinstance(obj, dict) and "name" in obj:
                    obj = [obj]
            else:
                obj = [json.loads(line) for line in lines]
            for it in obj:
                if it.get("name") == out.get("name"):
                    out = _deep_merge(out, it)
        elif spec_file.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("缺少 PyYAML 解析 --spec-file")
            data = yaml.safe_load(text)
            if isinstance(data, dict):
                if data.get("name") == out.get("name"):
                    out = _deep_merge(out, data)
            elif isinstance(data, list):
                for it in data:
                    if it.get("name") == out.get("name"):
                        out = _deep_merge(out, it)
        else:
            raise SystemExit("spec-file 只支持 .json/.jsonl/.ndjson/.yaml/.yml")
    if spec_json:
        try:
            obj = json.loads(spec_json)
            if isinstance(obj, dict):
                if obj.get("name") == out.get("name"):
                    out = _deep_merge(out, obj)
            elif isinstance(obj, list):
                for it in obj:
                    if it.get("name") == out.get("name"):
                        out = _deep_merge(out, it)
        except Exception as e:
            raise SystemExit(f"解析 --spec-json 失败: {e}")
    return out


def resolve_specs(
    names: list[str],
    configs_root: Path,
    spec_file: Path | None,
    spec_json: str | None,
    *,
    strict: bool,
) -> dict[str, DatasetSpec]:
    """@brief 解析名字到 DatasetSpec（YAML 合并后）/ Resolve names to DatasetSpec from YAML files."""
    specs: dict[str, DatasetSpec] = {}
    for n in names:
        merged = _load_from_configs_dir(configs_root, n)
        merged = _inject_overrides(merged, spec_file, spec_json)
        _validate_spec(merged, strict=strict)
        specs[n] = DatasetSpec(
            **{
                k: merged.get(k)
                for k in dataclasses.asdict(DatasetSpec(name=n, source="hf")).keys()
            }
        )
    return specs


# ------------------------------
# CLI commands
# ------------------------------


def cmd_list(args: argparse.Namespace) -> None:
    root: Path = Path(args.configs_root).resolve()
    if not root.exists():
        raise SystemExit(f"配置目录不存在: {root}")
    files = [p for p in root.glob("*.yaml") if p.name != "base.yaml"]
    if not files:
        print("(空) 在 configs 目录下放置 <name>.yaml 即可被发现")
        return
    print("可用数据集（来自 configs）：")
    for p in sorted(files):
        try:
            d = _load_yaml(p)
            nm = d.get("name", p.stem)
            src = d.get("source", "?")
            hint = d.get("notes", "")
            print(f" - {nm:12s} [source={src}]  {hint}")
        except Exception as e:
            print(f" - {p.stem:12s} [解析失败: {e}]")


def cmd_show(args: argparse.Namespace) -> None:
    global _MERGED_SPEC_SNAPSHOT
    spec_dict = _load_from_configs_dir(Path(args.configs_root), args.name)
    spec_dict = _inject_overrides(spec_dict, args.spec_file, args.spec_json)
    _validate_spec(spec_dict, strict=args.strict)
    _MERGED_SPEC_SNAPSHOT = spec_dict
    print(json.dumps(spec_dict, ensure_ascii=False, indent=2))


def cmd_download(args: argparse.Namespace) -> None:
    global _MERGED_SPEC_SNAPSHOT
    out_root = Path(args.output_root).resolve()
    ensure_dirs(out_root)

    specs = resolve_specs(
        args.names,
        Path(args.configs_root),
        args.spec_file,
        args.spec_json,
        strict=args.strict,
    )

    # driver registry
    drivers: dict[str, t.Callable[..., None]] = {
        "hf": lambda spec: driver_hf(
            spec, out_root, (args.format or spec.format_hint or "parquet").lower()
        ),
        "url": lambda spec: driver_url(spec, out_root),
    }

    for name, spec in specs.items():
        base = out_root / name
        if base.exists() and args.force:
            logger.warning("--force: 移除已存在目录 %s", base)
            import shutil

            shutil.rmtree(base)
        logger.info("开始获取: %s (source=%s)", name, spec.source)
        # 记录合并后的快照
        merged = _load_from_configs_dir(Path(args.configs_root), name)
        merged = _inject_overrides(merged, args.spec_file, args.spec_json)
        _MERGED_SPEC_SNAPSHOT = merged
        src = spec.source.lower()
        if src not in drivers:
            raise SystemExit(f"未知的 source: {src}")
        drivers[src](spec)  # 调用具体驱动
        logger.info("完成: %s -> %s", name, out_root / name)


def cmd_verify(args: argparse.Namespace) -> None:
    spec_dict = _load_from_configs_dir(Path(args.configs_root), args.name)
    spec_dict = _inject_overrides(spec_dict, args.spec_file, args.spec_json)
    _validate_spec(spec_dict, strict=args.strict)
    spec = DatasetSpec(
        **{
            k: spec_dict.get(k)
            for k in dataclasses.asdict(
                DatasetSpec(name=args.name, source=spec_dict["source"])
            ).keys()
        }
    )
    if spec.source != "url" or not spec.checksums:
        print("注意: verify 仅针对 source=url 且提供 checksums 的场景。")
        return
    out_root = Path(args.output_root).resolve()
    base = out_root / spec.name / "raw"
    ok = True
    for fn, expect in spec.checksums.items():
        p = base / fn
        if not p.exists():
            print(f"缺失文件: {fn}")
            ok = False
            continue
        calc = sha256_of_file(p)
        if calc.lower() != expect.lower():
            print(f"校验失败: {fn}: {calc} != {expect}")
            ok = False
        else:
            print(f"OK: {fn}")
    sys.exit(0 if ok else 2)


# ------------------------------
# Argparse
# ------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Acquire and normalize datasets via YAML configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--configs-root",
        default=str(_default_configs_root()),
        help="配置目录（含 base.yaml 与 <name>.yaml）",
    )
    p.add_argument(
        "--output-root", default=str(_default_output_root()), help="数据集根目录"
    )
    p.add_argument(
        "--spec-file",
        type=Path,
        default=None,
        help="额外数据集规格文件(.json/.jsonl/.yaml)",
    )
    p.add_argument(
        "--spec-json",
        type=str,
        default=None,
        help="直接通过 JSON 字符串注入规格(单个或数组)",
    )
    p.add_argument(
        "--strict", action="store_true", help="严格校验必填字段（production/CI 推荐）"
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="列出可用数据集（来自 configs 目录）")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("show", help="展示某个数据集合并后的配置快照")
    sp.add_argument("name", help="数据集名")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("download", help="下载并规范化指定数据集")
    sp.add_argument("names", nargs="+", help="一个或多个数据集名")
    sp.add_argument(
        "--format",
        default=None,
        choices=["parquet", "csv", "jsonl"],
        help="HF 数据集导出格式（覆盖 format_hint）",
    )
    sp.add_argument("--force", action="store_true", help="如已存在则强制覆盖目录")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("verify", help="对 url 源进行校验(sha256)")
    sp.add_argument("name", help="数据集名")
    sp.set_defaults(func=cmd_verify)

    return p


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_argparser()
    args = parser.parse_args(argv)
    args.func(args)  # type: ignore


if __name__ == "__main__":
    main()
