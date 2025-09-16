# configs/model/ 使用说明（不含具体参数）

本目录定义 **模型层组件的可配置超参**，由 `kan.utils.registry` 在运行时构建组件实例。它与 `configs/fusion/*.yaml`（注意力算子）与 `configs/train/*.yaml`（训练过程）解耦。

## 目录内文件

- `transformer_text.yaml`：**文本编码器**（Text Encoder）
  - 作用：将新闻文本转换为序列表征 `p`（`[B,L,D]`）与池化向量（`[B,D]`）。
  - 约定：若使用自研 `vanilla_transformer_text`，输入依赖词表与位置编码；若使用 HF 模型，则由分词器提供 `input_ids/attention_mask`。
  - 关键注意：与实体/上下文编码器在注意力层前的隐藏维度 **`D` 必须一致**。

- `transformer_entity.yaml`：**实体编码器**（Entity Encoder）
  - 作用：将每条样本的实体 ID 序列编码为序列表征 `q'` 与池化向量；可同时处理每实体的上下文 ID（用于 NE2C 的 `V`）。
  - 关键注意：`vocab_size` 由 `kan.data.batcher` 的词表**预热**后注入；聚合策略（对 `E`、对 `Lc`）可在此切换。

- `transformer_context.yaml`：**上下文编码器**（Context Encoder）
  - 作用：对“实体的一跳邻居”形成的上下文序列编码为 `r'` 与池化向量；与实体编码器同构，可选择**共享嵌入表**。
  - 关键注意：`share_with_entity_encoder=true` 时，需在构建阶段将实体编码器实例传入或在 Registry 中做权重绑定。

- `head.yaml`：**预测头**（Prediction Head）
  - 作用：将三路向量 `p/q/r` 融合（`concat|sum|mean` 等）并输出 logits/损失。
  - 关键注意：若三路维度不一致，需打开 `proj_dim` 做线性对齐；二分类任务配置 `problem_type=single_label_classification`。

## 典型接线（伪代码）

```python
from kan.utils.registry import HUB, build_from_config

text_enc = build_from_config(load_yaml('configs/model/transformer_text.yaml'), HUB.get('text_encoder'))
entity_enc = build_from_config(load_yaml('configs/model/transformer_entity.yaml'), HUB.get('entity_encoder'))
ctx_enc    = build_from_config(load_yaml('configs/model/transformer_context.yaml'), HUB.get('context_encoder'))
head      = build_from_config(load_yaml('configs/model/head.yaml'), HUB.get('head'))
```

## 契约与不变量（MECE）

1. **维度对齐**：注意力层（NE/NE2C）前的隐藏维度统一为 `D`。
2. **批形状稳定**：编码器输出 `last_hidden_state:[B,L,D]`、`pooled_state:[B,D]`；掩码 `mask:[B,L]`；与 `kan.interfaces` 一致。
3. **可复现性**：随机种子由 `configs/train/*.yaml` 控制；本目录仅描述**模型结构**。
4. **权重共享**：`context` 可与 `entity` 共享嵌入；共享策略应在实现中通过名称或句柄绑定，确保 **We don’t break userspace**。

## 常见坑位

- **D 不一致**：若改用 HF 文本编码器（如 `bert-tiny`，`D=128`），请同步把 `entity/context` 的 `embedding_dim` 改到相同值，或在注意力前显式投影。
- **词表未注入**：`vocab_size:null` 需要在 `batcher.build_vocabs()` 后由构建器填入；否则应抛出显式错误而非静默回退。
- **上下文维度**：`E`（实体数）与 `Lc`（每实体上下文数）由数据侧裁剪；模型侧不改变这两个维度的语义。

## 版本策略

- 小版本允许新增键，保留默认值；删除/重命名键视为破坏性变更，需在 `CHANGELOG` 标明迁移。
