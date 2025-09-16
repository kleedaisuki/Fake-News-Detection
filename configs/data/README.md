# configs/data/ 使用说明（不含具体参数）

本目录定义 **数据侧** 的可配置项：原始路径、清洗与分词、拆分策略、实体链接（EL）、知识图谱抓取（KG）、向量化与批处理等。其输出需满足 `kan.interfaces` 的数据契约，供 `kan/modules/*` 与注意力算子使用。

## 文件清单（示例）

* `common.yaml`：全局默认（路径、清洗、分词、拆分、EL/KG、向量化、批处理、日志）。其他数据集 YAML 按需覆盖。
* `politifact.yaml`：政治事实核查新闻；推荐**分层采样**拆分。
* `gossipcop.yaml`：娱乐新闻辟谣；规模更大，适度收紧文本长度与 EL 噪声阈值。
* `pheme.yaml`：推特谣言数据；默认 **root-only** 使用源推文，可切换 `conversation=thread` 聚合整棵会话树。

> 以上 YAML 不强制数值，仅给出键位与语义；训练/评估时由 `scripts/` 读取具体值。

## 关键键位与契约

* `paths.*`：I/O 与缓存。`cache_dir` 下的 `el/`、`kg/`、`vectorizer/` 建议按数据集分目录，便于复用与清理。
* `fields.*`：最少包含 `id_col,text_col,label_col`。若进行时间拆分或会话树建模，可提供 `timestamp_col, root_id, conversation_id` 等。
* `preprocess.*`：HTML 清洗、大小写、URL 去除、去重、`tokenizer` 类型与截断策略。
* `split.*`：`random/stratified/temporal` 三种；`group_by_id` 可确保同一新闻或会话不跨集合。
* `el.*`：EL 提供方与阈值、候选数、缓存与限流；本项目默认 **离线优先**，在线服务需遵守速率限制。
* `kg.*`：KG 提供方（默认 Wikidata）、一跳邻居规模、关系过滤、重试与缓存。
* `vectorizer.*`：词/子词向量方案（如 GloVe 100d），确保与模型隐藏维 `D` 对齐。
* `batcher.*`：批大小、长度桶、sortish 采样等；需与 `kan.utils.metrics` 的 `ignore_index/pad_id` 对齐。
* `logging.*`：建议落地 JSONL 审计便于调试与回溯（如每步清洗的 diff/长度、EL 命中率、KG 命中率等）。

## 典型用法

```bash
# 训练前：预处理 + 实体链接 + KG 抓取 + 向量化
python scripts/preprocess.py \
  --config configs/data/politifact.yaml \
  --save-to data/processed/politifact

# 训练：
python scripts/train.py \
  --data-config configs/data/politifact.yaml \
  --model-config configs/model/transformer_text.yaml \
  --fusion-config configs/fusion/ne2c.yaml \
  --train-config configs/train/base.yaml
```

## 不变量（We don't break userspace）

1. 更换 `tokenizer.type` 或 `vectorizer.type` 不改变导出的 **张量字段名**（如 `input_ids`, `lengths`）。
2. `EL/KG` 的缓存键依赖 `dataset_name + text_hash`；只有当清洗策略变化时才会失效。
3. `batcher` 的填充 ID（`pad_id`）需与模型嵌入 `padding_idx` 一致。

## 常见坑位

* **维度不匹配**：更换向量化维度时需同步调整模型 `D` 或开启投影层。
* **在线服务限流**：Wikidata SPARQL 有速率限制；务必设置 `rate_limit` 与 `retries/backoff`，并本地缓存。
* **标签映射**：`class_map` 要么在数据侧统一，要么在 head 读取时统一，避免训练/评估标签不一致。
