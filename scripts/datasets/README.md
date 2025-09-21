# Datasets Acquisition Toolkit

本目录 (`./scripts/datasets/`) 提供统一的数据集获取与落地工具，配合 YAML 配置驱动，保证团队实验的 **可复现性**、**可扩展性** 与 **安全性**。

---

## 📂 目录结构

```
./scripts/datasets/
  ├─ acquire_datasets.py   # CLI 主入口 (YAML 驱动)
  ├─ configs/              # 数据集规格配置目录
  │    ├─ base.yaml        # 全局缺省配置
  │    └─ <name>.yaml     # 单个数据集规格
```

下载后的数据统一落在仓库根目录下：

```
./datasets/
  └─ <name>/
      ├─ raw/         # 原始下载物 (zip/tar/json/csv/...)
      ├─ extracted/   # 解压后的文件树
      ├─ processed/   # 规范化后的文件 (parquet/csv/jsonl)
      └─ dataset_card.json  # 元信息与追溯快照
```

---

## 🚀 使用方法

### 1. 列出可用数据集

```bash
python scripts/datasets/acquire_datasets.py list
```

> 从 `configs/` 目录读取所有 `<name>.yaml` 配置。

### 2. 查看某个数据集配置

```bash
python scripts/datasets/acquire_datasets.py show liar
```

> 打印合并后的配置 (base.yaml + liar.yaml + CLI 覆盖)。

### 3. 下载数据集

```bash
python scripts/datasets/acquire_datasets.py download liar --format parquet
```

* 默认导出格式来自 `base.yaml` 的 `format_hint`。
* 支持 `--force` 覆盖已存在目录。

### 4. 校验 URL 源

```bash
python scripts/datasets/acquire_datasets.py verify politifact
```

> 对 `source=url` 且提供了 `checksums` 的数据集进行 sha256 校验。

---

## 📝 配置规范 (YAML Schema)

所有配置文件最终会与 `base.yaml` 合并，字段级覆盖：

* 空值 **不会覆盖** 非空值。
* 优先级：`base.yaml` < `<name>.yaml` < CLI `--spec-file/--spec-json`。

### 必填字段

* `name`: 数据集名 (目录名)
* `source`: 数据源类型 (`hf` | `url`)

### 可选字段

* `hf_id`: HuggingFace 数据集 ID
* `subset`: HuggingFace 子集名
* `urls`: URL 列表 (当 `source=url` 时必填)
* `checksums`: 文件名 -> sha256 校验值
* `format_hint`: parquet | csv | jsonl
* `license`: 许可信息
* `provenance`: 来源 (homepage/paper/revision)
* `tags`: 标签列表
* `notes`: 备注

### 示例

```yaml
# configs/liar.yaml
schema_version: 1
name: liar
source: hf
hf_id: liar
license: CC BY-NC 4.0
provenance:
  homepage: https://huggingface.co/datasets/liar
  paper: https://arxiv.org/abs/1705.00648
tags: [fake-news, claim-verification]
notes: "LIAR via HuggingFace"
```

---

## 🔒 安全与追溯

* 解压采用 **Zip Slip 防护**，保证文件仅释放到 `extracted/` 子目录。
* `dataset_card.json` 内包含：

  * `spec_snapshot` (合并后的配置快照)
  * `spec_hash` (快照哈希)
  * `driver` (使用的驱动)
  * `command_line` (运行命令)
  * `created_at` 时间戳

---

## 📦 依赖

详见仓库根目录下的 `requirements.txt`：

* [datasets](https://github.com/huggingface/datasets)
* pandas, pyarrow
* requests, tqdm
* PyYAML

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 🐾 日志

* 工具会优先调用项目内的集中式日志模块 `kan/utils/logging.py`，注入 `run_id/stage/step` 上下文，输出到控制台与轮转日志文件。
* 如果该模块不可用，则回退标准库 logging。

---

## 📌 小结

* 配置与流程彻底解耦，新增数据集只需在 `configs/` 下添加 `<name>.yaml`。
* 兼容旧 CLI 参数 (`--spec-json/--spec-file`)。
* 安全可追溯，适合科研与生产复现。
