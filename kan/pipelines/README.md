# Pipelines 设计总览（KAN）

> **目标**：把“数据→训练→评估”的工程链路做成可复用、可追溯、可扩展的 **流程（pipeline）**。本目录下的脚本 **只做编排（orchestration）**，不实现模型或算法细节，把所有“重活”交给 `kan.data.*` 与 `kan.modules.*`。

* 本 README 覆盖 4 个核心文件：

  * `prepare_data.py`：数据准备流水线（下载/清洗/EL/KG/向量化/词表/清单）。
  * `train_trainer.py`：基于 🤗 Transformers **Trainer** 的训练流水线（高效基线）。
  * `train_accelerate.py`：基于 **Accelerate** 的自定义训练循环（灵活可控）。
  * `evaluate.py`：聚合评估（单 run / 多 run / k-fold），导出指标 & 混淆矩阵 & 曲线数据。

---

## 0. 设计哲学（Design Philosophy）

**Pipeline ≠ Module**：

* Pipeline 负责 **装配、调度、日志、落盘**；
* 模型结构（encoders/attention/head）、数据张量化（tokenization/collate）、指标计算等 **在模块侧实现**：`kan.modules.*`, `kan.data.*`, `kan.utils.*`。

**配置即契约（Config-as-Contract）**：

* 所有超参数来自 `configs/`，支持：

  1. 多个 YAML 顺序合并（后者覆盖前者）；
  2. `--override key.subkey=value` 点号覆盖（值支持 JSON 字面量：数值/布尔/数组/对象）；
* 最终合并快照写入 `runs/<run_id>/configs_merged.yaml`，确保可复现。

**过程 vs 结果（Process vs Result）**：

* 过程（**Process**）：`cache/` 与 `runs/<run_id>/logs/`，包含 raw 镜像、EL/KG 中间层、embedding 索引、词表、训练日志、状态等。
* 结果（**Result**）：`runs/<run_id>/artifacts/`（权重、检查点、指标、预测）与 `runs/<eval_run_id>/reports/`（评估汇总）。

**无畏缓存（Cache Aggressively）**：

* 数据准备、KG 抓取、向量化、词表都可缓存；
* 训练/评估也生成可复用产物，避免重复工作。

**优雅降级（Graceful Degradation）**：

* 某些组件（如 `NE/NE2C` 注意力）如果尚未实现，只要函数签名对齐，pipeline 会自动退化为 “文本-only” 路径，不阻塞端到端。

**Windows 友好**：

* `num_workers` 默认 0；路径统一用 `pathlib`；避免强制多进程与符号链接；支持断点续训。

---

## 1. 目录与产物（Artifacts & Layout）

```
./
├─ cache/                               # 过程缓存（可复用）
│  ├─ datasets/<name>/<fingerprint>/
│  │  ├─ raw/                           # 原始镜像（*.jsonl.gz，可选）
│  │  ├─ prepared/<split>/              # 清洗/增强后的分片
│  │  │  ├─ shard-00001.jsonl.gz
│  │  │  └─ stats.json
│  │  ├─ meta/vocabs/                   # 词表（batcher 持久化）
│  │  └─ manifest.json                  # 可消费清单（见 §2.5）
│  └─ kg/                               # 知识抓取缓存（由 kg_fetcher 实现）
├─ runs/
│  ├─ <run_id>/
│  │  ├─ configs_merged.yaml            # 最终配置快照（可复现）
│  │  ├─ logs/                          # 训练/准备/评估日志
│  │  ├─ artifacts/                     # 训练权重与断点
│  │  │  ├─ best/                       # 最优（按 metric_for_best）
│  │  │  └─ last/                       # 最近一次
│  │  ├─ eval_*.json                    # 训练阶段评估指标（Trainer 版）
│  │  ├─ pred_<split>.jsonl             # 预测结果（给 evaluate 聚合）
│  │  └─ ...
│  └─ <eval_run_id>/
│     ├─ configs_merged.yaml
│     ├─ logs/
│     └─ reports/
│        ├─ metrics_<split>.json        # micro & macro 汇总
│        ├─ confusion_matrix_<split>.csv
│        └─ curves/<split>/{roc,pr}_class<i>.csv
└─ ...
```

---

## 2. 数据准备：`prepare_data.py`

### 2.1 流程

1. **合并配置**：多 YAML + overrides → `data_cfg`；
2. **数据指纹**：仅采集“影响数据内容”的键（默认：`source/splits/preprocess/entity_linking/kg/vectorizer/filters/normalize`）→ `fingerprint`；
3. **加载原始数据**：`kan.data.loaders.loader_from_config(data_cfg)` → `load_split(name)` 返回 `List[NewsRecord]`；
4. **轻量清洗**：去空文本、按 `dedup_key` 去重（可设为 `__text__`）；
5. **实体链接（EL）**：`entity_linker.link(records)` 补充 `entities: [{id,surface,offset,...}]`；
6. **知识抓取（KG）**：对全体实体 id 去重后批量抓取至 `cache/kg`（由实现负责速率限制/去重/持久化）；
7. **向量化（Vectorizer）**：补充 `text_vec/ent_vecs/ctx_vecs` 等；
8. **词表**：`kan.data.batcher` 以 train split 构建并保存词表到 `meta/vocabs/`；
9. **写出**：每个 split 写 `prepared/<split>/shard-00001.jsonl.gz` 与 `stats.json`；整体验证清单写 `manifest.json`。

> **Note**：EL/KG/向量化优先通过 `kan.utils.registry.HUB` 构建，找不到则回退到 `kan.data.*.build_*`，找不到实现就**跳过**。

### 2.2 输入记录（NewsRecord）约定

最小字段：

```json
{
  "id": "unique-id",
  "text": "news content..."  // 或 content
}
```

EL 后：

```json
{
  "entities": [{"id": "Q42", "surface": "Douglas Adams", "offset": 123}, ...]
}
```

向量化后（可选）：

```json
{
  "text_vec": [0.1, 0.2, ...],
  "ent_vecs": [[...], ...],
  "ctx_vecs": [[...], ...]
}
```

### 2.3 `manifest.json`

```json
{
  "name": "gossipcop",
  "fingerprint": "a1b2c3d4e5f6a7b8",
  "paths": {"root": "cache/datasets/gossipcop/a1b2..."},
  "splits": {
    "train": {
      "count": 12345,
      "shards": ["prepared/train/shard-00001.jsonl.gz"],
      "stats": {"count":12345, "text_len": {"mean": 62.3, "median": 55, "p95": 141}}
    },
    "validation": {...}
  },
  "components": {"entity_linking": true, "kg": true, "vectorizer": false},
  "config_snapshot": { ... }            // 仅用于重建指纹的关键子配置
}
```

### 2.4 配置键一览（常用）

* `data.name`: 数据集名（用于路径）；
* `data.splits`: `{alias: actual_split_name}`；
* `data.dedup_key`: 去重键，或 `__text__`；
* `data.entity_linking`: 实体链接组件的构造参数；
* `data.kg`: 知识抓取组件的构造参数；
* `data.vectorizer`: 向量化组件的构造参数；
* `data.batcher.text/entity/context`: 词表/截断等（仅用于构建 batcher 的配置）；
* `cache_dir`, `output_dir`, `run_id`；
* `force`（CLI）：存在缓存也强制重建。

### 2.5 典型命令

```bash
python -m kan.pipelines.prepare_data \
  -c configs/data/gossipcop.yaml configs/data/common.yaml \
  -o data.save_raw=true data.dedup_key=__text__ vectorizer.type=sentencetransformers
```

---

## 3. 训练（Trainer 版）：`train_trainer.py`

### 3.1 流程

1. 合并配置；准备 `runs/<run_id>/artifacts/` 与 `logs/`；
2. 加载数据：`kan.data.loaders` → `RecordsDataset`（仅存记录）；
3. 构建 `Batcher` 并 `build_vocabs(train)`；以 **DataCollator** 调 `batcher.collate()` 进行张量化；
4. 组件装配：优先 `kan.utils.registry.HUB`，否则回退 `kan.modules.*` 的 `build_*`；
5. 封装组合模型 `KANForNewsClassification(p,q,r)`：

   * 文本：`text_encoder` → `p`
   * 实体：`entity_encoder` → `q'`，经 `NE` → `q`
   * 上下文：`context_encoder`/`entity_encoder` → `r'`，经 `NE2C` → `r`
   * `head(p,q,r)` → `loss, logits`
6. 构造 `TrainingArguments` 与 🤗 `Trainer`；
7. `train()` 并持久化：`train_metrics.json`、`save_model()`；
8. `evaluate()` / `predict()` → `eval_*.json`、`pred_*.jsonl`。

### 3.2 常用配置键（片段）

* `train.batch_size`, `train.eval_batch_size`, `train.grad_accum`, `train.max_epochs`；
* `train.optimizer.{lr,weight_decay}`；
* `train.evaluation_strategy`=`epoch|steps`, `train.logging.every_n_steps`, `train.save_strategy`, `train.save_total_limit`；
* 顶层：`fp16|bf16|seed|deterministic|compile`；
* `head.{num_labels,problem_type,use_q,use_r}`；
* `data.*` 与 `batcher.*` 同上。

---

## 4. 训练（Accelerate 版）：`train_accelerate.py`

### 4.1 何时选 Accelerate？

* 需要自定义训练循环（复杂采样器、异步负载、梯度逻辑定制）；
* 需要灵活的 checkpoint/评估频率控制；
* 希望结合 `accelerate.init_trackers` 使用自定义日志后端。

### 4.2 流程差异点

* 使用 `Accelerator(mixed_precision, gradient_accumulation_steps, log_with)`；
* `accelerator.prepare(model, optimizer, dataloader, ...)`；
* 自管 `optimizer` 与 `get_linear_schedule_with_warmup`；
* 训练时用 `with accelerator.accumulate(model): ...`；
* 断点续训：`accelerator.save_state(dir)` / `accelerator.load_state(dir)`；
* best/last checkpoint 策略：按 `metric_for_best` 维护 `artifacts/best/` 与 `artifacts/last/`；
* 评估：`accelerator.gather_for_metrics()` 汇总张量，防止重复样本。

### 4.3 关键配置键

与 Trainer 基本一致，另外增加：

* `train.lr_scheduler.{warmup_ratio,warmup_steps}`
* `train.metric_for_best`, `train.greater_is_better`
* `train.resume_from`

---

## 5. 评估聚合：`evaluate.py`

### 5.1 输入契约

* 评估读取 **训练 run** 输出的 `pred_<split>.jsonl(.gz)`；单条结构：

  * **单标签**：`{"y_true": int, "y_score": [p0, p1, ...], "y_pred": 可选}`
  * **多标签**：`{"y_true": [0/1,...], "y_score": [prob,...]}`
  * **回归**：`{"y_true": float, "y_score": float}`

### 5.2 汇总逻辑

* **micro**：把多个 run 的样本全部拼接后 **重新计算**指标；
* **macro**：先算每个 run 的指标，再做 **均值/标准差**；
* 单标签额外输出：

  * `confusion_matrix_<split>.csv`（K×K）
  * `curves/<split>/{roc,pr}_class<i>.csv`（一对多曲线点）

### 5.3 典型命令

```bash
python -m kan.pipelines.evaluate \
  -c configs/eval/kfold.yaml \
  -o eval.inputs_glob="runs/kan-*-fold*/" eval.splits='["validation","test"]' \
     eval.problem_type=single_label_classification eval.num_labels=2
```

---

## 6. 前后端契约（Contracts）

### 6.1 Batcher 输入/输出

* **输入**：`List[NewsRecord]`（prepare 阶段规范的记录）；
* **Collate 输出（示例）**：

```python
{
  "text_tok": {"input_ids": Long[BS, L], "attention_mask": Long[BS, L]},
  "ent_ids": Long[BS, E], "ent_mask": Bool[BS, E],
  "ctx_ids": Long[BS, C], "ctx_mask": Bool[BS, C],
  "labels": Long[BS]
}
```

### 6.2 模块装配（Registry）

* 优先通过 `kan.utils.registry.HUB`：命名空间 `text_encoder/entity_encoder/context_encoder/attention/head/entity_linker/kg_fetcher/vectorizer`；
* 配置中使用 `{ "type": "<impl-name>", ... }` 选择实现；
* 若无 HUB 条目，回退调用模块侧 `build_*` 入口函数。

### 6.3 组合模型 `KANForNewsClassification`

* `text_encoder → p`（新闻句向量/池化）；
* `entity_encoder → q'`；`NE(q' ⟶ news)` → `q`；
* `context_encoder 或 entity_encoder → r'`；`NE2C(r' ⟶ news, entities)` → `r`；
* `head(p,q,r, labels?) → {loss, logits}`（`use_q/use_r` 可开关）。

---

## 7. 复现性与命名约定

* `run_id` 默认：`{name}-{YYYYmmdd-HHMMSS}`；可手动指定确保可控对齐；
* 每次运行都固化 `configs_merged.yaml`；
* 数据缓存按 `fingerprint` 分区，**内容相同即命中同一缓存**；
* 评估 run 单独生成 `<eval_run_id>`，与训练 run 解耦。

---

## 8. 常见操作与排错（Cookbook）

**Q: 显存不够？**
A: 调小 `train.batch_size` 或启用 `train.grad_accum`；考虑 `bf16/fp16`；如使用 `Accelerate`，可再尝试 `torch.compile(true)` 以及分阶段冻结参数。

**Q: 想按步评估/保存？**
A: 在 `Trainer` 设 `train.evaluation_strategy=steps` + `train.eval_steps` + `train.save_strategy=steps`；在 `Accelerate` 版同理有对应开关。

**Q: 如何断点续训？**
A: Accelerate 版设置 `train.resume_from=path/to/ckpt`; Trainer 版用 `Trainer.train(resume_from_checkpoint=...)`（可在配置中做桥接）。

**Q: k-fold 如何汇总？**
A: 训练时把每折的 run 放进 `runs/`，评估时 `eval.inputs=[...]` 或 `eval.inputs_glob="runs/kan-*-fold*/"`，`evaluate.py` 会生成 macro/micro。

**Q: Windows 卡死/慢？**
A: 设 `train.dataloader.num_workers=0`；路径避免中文或超级长路径；使用 PowerShell 时注意引号转义。

---

## 9. 扩展指南（Extensibility）

**新增一个文本编码器**

1. 在 `kan.modules.text_encoder` 中实现构造器 `build_text_encoder(cfg)`；
2. 在 `kan.utils.registry.HUB` 注册：`HUB.register("text_encoder", "my_encoder", build_text_encoder)`；
3. 在配置里：`text_encoder: {type: my_encoder, ...}`。

**替换实体链接器/向量器/KG 抓取器** 亦同理（命名空间见 §6.2）。

**新增 Head**：实现 `build_head(cfg)`，约定参数 `p,q,r,labels?`，返回 `{logits, loss?}`；在配置中指向新的 `type`。

---

## 10. 脚本层（scripts/）

建议提供以下最小脚本，便于 CI 与新人上手：

* `scripts/prepare.py`：包装 `run_from_configs([...])` 调用 `prepare_data`；
* `scripts/train.py`：可切换 `trainer|accelerate`；
* `scripts/eval.py`：聚合评估与产物打包（把 reports 导出到团队共享盘）。

脚本中应：

* 解析命令行以组装 **有序的** `configs/` 列表（从“通用 → 任务 → 数据集 → 折次/实验”）；
* 打印最终 `run_dir` 或 `reports_dir`；
* 对异常进行降级提示（缺少 NE/NE2C 时仅发 warning）。

---

## 11. 质量保障（QA Checklist）

* [ ] `runs/<run_id>/configs_merged.yaml` 存在且可读；
* [ ] `cache/datasets/<name>/<fingerprint>/manifest.json` 完整且路径可用；
* [ ] `pred_<split>.jsonl` 存在，可被 `evaluate.py` 读取；
* [ ] 训练日志（loss、lr）曲线单调/合理，无异常 NaN；
* [ ] `artifacts/best/` 与 `artifacts/last/` 状态一致，能 `resume_from`；
* [ ] macro 与 micro 指标变化方向一致（多 run 场景）；
* [ ] Windows 下执行不需管理员权限。

---

## 12. 快速开始（One‑pager）

```bash
# 1) 准备数据（可复用缓存）
python -m kan.pipelines.prepare_data -c configs/data/gossipcop.yaml

# 2) 训练（任选其一）
python -m kan.pipelines.train_trainer -c <cfgs...> -o train.batch_size=16
# 或
python -m kan.pipelines.train_accelerate -c <cfgs...> -o bf16=true train.grad_accum=2

# 3) 评估（多 run 聚合可用 glob）
python -m kan.pipelines.evaluate -c configs/eval/kfold.yaml -o eval.inputs_glob="runs/kan-*/"
```

---

## 13. 术语表（Glossary）

* **Orchestration（编排）**：将多个已有组件按顺序/依赖关系组织执行，并处理其产物与日志。
* **Manifest（清单）**：描述数据集可消费形态（分片路径、样本数、统计、关键配置指纹）的 JSON 文件。
* **Micro vs. Macro**：前者在样本层拼接后统一计算，后者在 run 层先算后平均。
* **Graceful Degradation（优雅降级）**：当部分模块不可用时，不让流程崩溃，退化成功能子集继续产出可用结果。

---

> **工程信条**：“We don't break userspace.”
>
> * Pipeline 改动必须保证已有 `configs/` 能跑通；
> * 新增参数要提供默认值或向后兼容的合并逻辑；
> * 任何影响复现性的改动，需要 bump 数据 `fingerprint_keys` 或更新 `manifest` 以免误复用旧缓存。
