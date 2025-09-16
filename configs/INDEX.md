# configs/INDEX.md — 参数索引与用法导航

> 目的：提供“**参数 → 配置文件**”的反向索引（带链接），帮助快速定位该改哪里；并统一说明 `configs/` 的加载/覆盖规则与最佳实践。四份子目录文档： [data/README](data/README.md) · [model/README](model/README.md) · [fusion/README](fusion/README.md) · [train/README](train/README.md)

---

## 1) 速查表：参数 → 文件（带链接）

> 说明：下表按 **数据 → 模型 → 融合 → 训练** 的流水线顺序组织；列出的键名为**常用键路径**（YAML 路径），不同实现可能有同义键（已在“别名”列标注）。

### A. 数据层（Data）

| 参数/键路径                           | 作用                                    | 去哪里改                                         | 别名/备注                       |
| -------------------------------- | ------------------------------------- | -------------------------------------------- | --------------------------- |
| `name`                           | 数据集名（打印/日志用）                          | [configs/data/\*.yaml](data/)                | —                           |
| `format`                         | 源格式：`csv`/`jsonl`/`hf`                | [configs/data/\*.yaml](data/)                | —                           |
| `path`                           | 数据根目录                                 | [configs/data/\*.yaml](data/)                | HF 源可空                      |
| `splits.{train,validation,test}` | 各切分文件或 HF split 名                     | [configs/data/\*.yaml](data/)                | `dev`≈`validation`          |
| `fields.id`                      | 源列 → 统一 `id`                          | [configs/data/\*.yaml](data/)                | 缺失将由文本哈希生成                  |
| `fields.text`                    | 源列 → 统一 `text`                        | [configs/data/\*.yaml](data/)                | 必填                          |
| `fields.label`                   | 源列 → 统一 `label`                       | [configs/data/\*.yaml](data/)                | 必填（用 `label_map` 归一）        |
| `fields.entities`                | 源列 → `entities: List[QID]`            | [configs/data/\*.yaml](data/)                | 可选；字符串化 JSON 也可             |
| `fields.contexts`                | 源列 → `contexts: Dict[QID, List[...]]` | [configs/data/\*.yaml](data/)                | 可选；支持 `P=Q` 形态              |
| `label_map`                      | 统一标签到 int（`0/1/...`）                  | [configs/data/\*.yaml](data/)                | 例：`{true:0,false:1}`        |
| `id_prefix`                      | 为缺失 ID 自动生成的前缀                        | [configs/data/\*.yaml](data/)                | 例：`pf_`                     |
| `drop_duplicates_on_text`        | 文本去重                                  | [configs/data/\*.yaml](data/)                | 默认 `false`                  |
| `lowercase_text` / `strip_text`  | 文本归一化                                 | [configs/data/\*.yaml](data/)                | —                           |
| `batcher.entity.max_entities`    | 每样本实体上限 `E`                           | [configs/data/common.yaml](data/) 或各数据集 yaml | 若未拆分 `common.yaml`，则写到对应数据集 |
| `batcher.context.max_neighbors`  | 每实体邻居上限 `N`                           | 同上                                           | —                           |
| `batcher.text.tokenizer_name`    | HF 分词器                                | 同上                                           | `bert-base-uncased` 等       |
| `batcher.text.max_length`        | 文本截断长度 `T`                            | 同上                                           | —                           |

> 更多细节见：[data/README](data/README.md)（字段映射、可复现性与缓存策略、词表约定）。

---

### B. 模型层（Model Encoders & Head）

| 参数/键路径                             | 作用                    | 去哪里改                                                              | 别名/备注             |                                                             |                  |
| ---------------------------------- | --------------------- | ----------------------------------------------------------------- | ----------------- | ----------------------------------------------------------- | ---------------- |
| `text_encoder.model_name_or_path`  | 文本编码器底座               | [model/transformer\_text.yaml](model/transformer_text.yaml)       | HF 模型名或本地路径       |                                                             |                  |
| `text_encoder.max_length`          | 分词最大长度                | [model/transformer\_text.yaml](model/transformer_text.yaml)       | 与 batcher.text 对齐 |                                                             |                  |
| `text_encoder.pooling`             | \`cls                 | mean                                                              | max\`             | [model/transformer\_text.yaml](model/transformer_text.yaml) | —                |
| `text_encoder.trainable`           | 是否微调                  | [model/transformer\_text.yaml](model/transformer_text.yaml)       | —                 |                                                             |                  |
| `entity_encoder.vocab_size`        | 实体词表大小                | [model/transformer\_entity.yaml](model/transformer_entity.yaml)   | 由 batcher 词表决定    |                                                             |                  |
| `entity_encoder.embedding_dim`     | 实体嵌入维度 `D`            | [model/transformer\_entity.yaml](model/transformer_entity.yaml)   | = 各路 `d_model`    |                                                             |                  |
| `entity_encoder.xformer_layers`    | 编码层数                  | [model/transformer\_entity.yaml](model/transformer_entity.yaml)   | 论文推荐 1 层          |                                                             |                  |
| `entity_encoder.xformer_heads`     | 注意力头数 `h`             | [model/transformer\_entity.yaml](model/transformer_entity.yaml)   | 论文常用 4            |                                                             |                  |
| `entity_encoder.ffn_dim`           | FFN 维度 `d_ff`         | [model/transformer\_entity.yaml](model/transformer_entity.yaml)   | 论文常用 2048         |                                                             |                  |
| `entity_encoder.dropout`           | 丢弃率                   | [model/transformer\_entity.yaml](model/transformer_entity.yaml)   | 论文常用 0.5          |                                                             |                  |
| `context_encoder.*`                | 与 `entity_encoder` 同构 | [model/transformer\_context.yaml](model/transformer_context.yaml) | 文本上下文或 ID 上下文版本   |                                                             |                  |
| `head.num_labels`                  | 类别数                   | [model/head.yaml](model/head.yaml)                                | 二分类取 2            |                                                             |                  |
| `head.fusion`                      | \`concat              | sum                                                               | mean\`            | [model/head.yaml](model/head.yaml)                          | 与 `proj_dim` 一起用 |
| `head.proj_dim`                    | p/q/r 对齐维度            | [model/head.yaml](model/head.yaml)                                | sum/mean 时必需      |                                                             |                  |
| `head.hidden_sizes`                | MLP 隐层结构              | [model/head.yaml](model/head.yaml)                                | 例：`[256]`         |                                                             |                  |
| `head.dropout` / `head.activation` | 正则与激活                 | [model/head.yaml](model/head.yaml)                                | —                 |                                                             |                  |
| `head.label_smoothing`             | 标签平滑                  | [model/head.yaml](model/head.yaml)                                | 可选                |                                                             |                  |
| `head.class_weights`               | 类别权重                  | [model/head.yaml](model/head.yaml)                                | 处理类不平衡            |                                                             |                  |

> 设计/契约详见：[model/README](model/README.md)（三路编码器输出形状、命名与日志规范、Head 融合策略）。

---

### C. 知识融合层（Fusion: NE / NE2C）

| 参数/键路径                          | 作用          | 去哪里改                                 | 备注                |                                      |              |
| ------------------------------- | ----------- | ------------------------------------ | ----------------- | ------------------------------------ | ------------ |
| `ne.d_model` / `ne.n_heads`     | NE 注意力维度/头数 | [fusion/ne.yaml](fusion/ne.yaml)     | 与编码器 `d_model` 对齐 |                                      |              |
| `ne.dropout`                    | 注意力 dropout | [fusion/ne.yaml](fusion/ne.yaml)     | —                 |                                      |              |
| `ne.return_weights`             | 是否回传权重      | [fusion/ne.yaml](fusion/ne.yaml)     | 仅分析/可视化时开启        |                                      |              |
| `ne2c.d_model` / `ne2c.n_heads` | NE2C 维度/头数  | [fusion/ne2c.yaml](fusion/ne2c.yaml) | —                 |                                      |              |
| `ne2c.dropout`                  | 注意力 dropout | [fusion/ne2c.yaml](fusion/ne2c.yaml) | —                 |                                      |              |
| `ne2c.context_pooling`          | \`attn      | mean                                 | cls\`             | [fusion/ne2c.yaml](fusion/ne2c.yaml) | NE2C 上下文聚合策略 |
| `ne2c.return_weights`           | 是否回传权重      | [fusion/ne2c.yaml](fusion/ne2c.yaml) | —                 |                                      |              |

> 细节/张量契约见：[fusion/README](fusion/README.md)（NE/NE2C 的 Q/K/V 角色与日志约定）。

---

### D. 训练/评估（Train）

| 参数/键路径                            | 作用              | 去哪里改                               | 别名/备注             |         |
| --------------------------------- | --------------- | ---------------------------------- | ----------------- | ------- |
| `seed` / `deterministic`          | 复现性             | [train/base.yaml](train/base.yaml) | 见 utils/seed 策略   |         |
| `device`                          | 训练设备            | [train/base.yaml](train/base.yaml) | `cuda`/`cpu`      |         |
| `bf16` / `fp16`                   | 混合精度            | [train/base.yaml](train/base.yaml) | 二选一               |         |
| `compile`                         | `torch.compile` | [train/base.yaml](train/base.yaml) | 提速选项              |         |
| `optimizer.type`                  | 优化器             | [train/base.yaml](train/base.yaml) | 例：`adamw`         |         |
| `optimizer.lr` / `weight_decay`   | 学习率/权重衰减        | [train/base.yaml](train/base.yaml) | —                 |         |
| `scheduler.type` / `warmup_ratio` | 学习率计划           | [train/base.yaml](train/base.yaml) | 例：`linear`        |         |
| `batch_size` / `grad_accum`       | 批大小/梯度累积        | [train/base.yaml](train/base.yaml) | 显存友好              |         |
| `max_epochs` / `max_steps`        | 训练轮数/步数         | [train/base.yaml](train/base.yaml) | 二选一优先 `max_steps` |         |
| `grad_clip_norm`                  | 梯度裁剪            | [train/base.yaml](train/base.yaml) | —                 |         |
| `eval.strategy` / `eval.steps`    | 评估策略            | [train/base.yaml](train/base.yaml) | \`epoch           | steps\` |
| `early_stopping.patience`         | 提前停止            | [train/base.yaml](train/base.yaml) | —                 |         |
| `logging.every_n_steps`           | 日志频率            | [train/base.yaml](train/base.yaml) | 搭配集中式日志           |         |
| `output_dir`                      | 输出目录（权重/日志）     | [train/base.yaml](train/base.yaml) | —                 |         |
| `dataloader.num_workers`          | DataLoader 线程   | [train/base.yaml](train/base.yaml) | Windows 友好设置      |         |
| `folds.k` / `folds.seed`          | 交叉验证折数          | [train/\*\_5fold.yaml](train/)     | 数据集特定文件           |         |

> 训练约定与脚手架详见：[train/README](train/README.md)（调用入口、覆盖顺序、日志与指标聚合）。

---

## 2) `configs/` 的加载与覆盖规则（建议）

1. **分层合并**（由脚本或 `OmegaConf`/`Hydra`/自研 loader 完成）：

   * 基线：`data/*.yaml` × `model/*.yaml` × `fusion/*.yaml` × `train/base.yaml`
   * 数据集特定：再叠加 `train/{dataset}_5fold.yaml`（或自定义 `train/*.yaml`）。
2. **覆盖优先级**：命令行/环境变量 > `train/{dataset}_*.yaml` > `train/base.yaml` > `model/*.yaml`/`fusion/*.yaml` > `data/*.yaml`。
3. **键路径风格**：统一小写、用点分层（如 `optimizer.lr`）；数组用 `[]` 字面量。
4. **跨文件一致性检查**：

   * `text_encoder.max_length` 与 `batcher.text.max_length` 必须一致；
   * 三路 `d_model`（文本/实体/上下文）与 `fusion.ne{,2c}.d_model` 必须一致；
   * `head.num_labels` 与数据标签空间一致。
5. **论文对齐的缺省值**（建议写入各自 YAML 的默认）：**单层 Transformer**、**多头数 `h=4`**、**FFN 维度 `2048`**、**dropout=0.5**。

---

## 3) 常见任务：我想要…改哪里？（导航）

* **缩短输入文本**（显存不足）：改 `batcher.text.max_length` 与 `text_encoder.max_length` → [data/common.yaml](data/) & [model/transformer\_text.yaml](model/transformer_text.yaml)。
* **减少实体/邻居数**（加速）：改 `batcher.entity.max_entities` 与 `batcher.context.max_neighbors` → [data/common.yaml](data/)。
* **只用文本，不用知识**：将 [fusion/ne.yaml](fusion/ne.yaml) 与 [fusion/ne2c.yaml](fusion/ne2c.yaml) 置空或在 [model/head.yaml](model/head.yaml) 中 `use_q=false,use_r=false`（若支持）。
* **开启注意力权重可视化**：在 [fusion/ne\*.yaml](fusion/) 打开 `return_weights=true`（训练时谨慎，显存↑）。
* **做 5 折复现**：改 [train/{dataset}\_5fold.yaml](train/) 的 `folds.k` 与随机种；运行脚本会汇总到 `runs/*/fold_summary.json`。

---

## 4) 最小合并示例（伪命令）

```bash
# 以 GossipCop 为例
# 1) 基本数据 + 模型 + 融合 + 通用训练
BASE=(configs/data/gossipcop.yaml \
      configs/model/transformer_text.yaml \
      configs/model/transformer_entity.yaml \
      configs/model/transformer_context.yaml \
      configs/fusion/ne.yaml \
      configs/fusion/ne2c.yaml \
      configs/model/head.yaml \
      configs/train/base.yaml)

# 2) 叠加数据集特定训练参数
EXTRA=configs/train/gossipcop_5fold.yaml

python scripts/run.py train ${BASE[@]} ${EXTRA} \
  optimizer.lr=3e-5 batch_size=16 compile=true
```

> 你的项目脚本若采用 `--key=value` 覆盖，注意数组/字典的转义规则；Windows PowerShell 下请使用反引号 `` ` `` 断行。

---

## 5) 术语对照（含一次英文）

* 实体链接（Entity Linking, **EL**）
* 知识图谱（Knowledge Graph, **KG**）
* 一跳邻居（1‑hop neighbors）
* 注意力（Attention）/ 多头注意力（Multi‑Head Attention）
* 池化（Pooling）
* 交叉验证（Cross‑Validation）

---

## 6) 变更记录

* v0.1 — 初版：建立“参数 → 文件”的反向索引；补充加载/覆盖规则与一致性检查清单。
