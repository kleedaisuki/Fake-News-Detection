# `kan/modules` 设计与使用手册

> **目标（Goal）**：把 KAN（Knowledge-aware Attention Network for Fake News Detection）的编码与融合层拆分为可插拔模块，稳定契约（Contract），统一日志（Logging），便于扩展与复现。

---

## 0. 快速导览（TL;DR）

* 文本编码器（Text Encoder） → `text_encoder.py`

  * 输入：`input_ids/attention_mask/(token_type_ids)`
  * 输出：`sequence_output[B,L,H]`、`pooled_output[B,H]`、`attention_mask[B,L]`
  * Pooling：`cls|mean|max`
* 实体编码器（Entity Encoder, ID-based） → `entity_encoder.py`

  * 输入：`entity_ids[B,E]`、（可选）`context_ids[B,E,Lc]`
  * 输出：`entities_last_hidden[B,E,D]`、`contexts_last_hidden[B,E,D]`、及各自的 `*_pooled[B,D]` 与 `*_mask[B,E]`
* 文本上下文编码器（Context Encoder, text-based） → `context_encoder.py`

  * 输入：`context_input_ids[B,E,Lc,Lt]` 及对应 `attention_mask`
  * 过程：展平 `(B*E*Lc,Lt)` → 文本编码器 → 还原 `(B,E,Lc,D)` → Lc 聚合
  * 输出：`contexts_last_hidden[B,E,D]` 等
* 预测头（Prediction Head） → `head.py`

  * 输入：`p`（新闻向量）、`q`（实体注意力输出）、`r`（上下文注意力输出）
  * 融合：`concat|sum|mean`（支持分支投影 `proj_dim`）
  * 任务：`single_label_classification|multi_label_classification|regression`

> 命名日志（Namespaced Logger）：`kan.text_encoder`、`kan.entity_encoder`、`kan.context_encoder`、`kan.head`、`kan.attention.ne`、`kan.attention.ne2c`

---

## 1. 模块契约（Contracts）

### 1.1 Text Encoder（`text_encoder.py`）

* **Forward**：`forward(**batch) -> {sequence_output, pooled_output, attention_mask}`
* **输入键**：

  * `input_ids: Long[B,L]`
  * `attention_mask: Long[B,L]`
  * `token_type_ids: Long[B,L]`（可选）
* **输出键**：

  * `sequence_output: Float[B,L,H]`
  * `pooled_output: Float[B,H]`
  * `attention_mask: Long[B,L]`
* **配置（`TextEncoderConfig`）**：`model_name_or_path`、`max_length`、`pooling`（`cls|mean|max`）、`trainable`、`use_fast_tokenizer`、`pad_to_max_length`
* **便捷方法**：`batch_encode(texts)`；`encode(texts)->[N,H]`

### 1.2 Entity Encoder（`entity_encoder.py`）

* **Forward（keyword-only）**：

  ```py
  forward(
    *, entity_ids[B,E], entity_mask[B,E]=None,
    context_ids[B,E,Lc]=None, context_mask[B,E,Lc]=None,
  ) -> {
    entities_last_hidden[B,E,D], entities_pooled[B,D], entities_mask[B,E],
    contexts_last_hidden[B,E,D], contexts_pooled[B,D], contexts_mask[B,E],
    (raw_contexts_last_hidden[B,E,Lc,D])
  }
  ```

* **配置（`EntityEncoderConfig`）**：

  * 嵌入：`vocab_size`、`embedding_dim(D)`、`padding_idx/unk_idx`、`embeddings_path`（.pt/.bin/.npy）、`trainable`
  * 编码：`xformer_layers/heads/ffn_dim`（`0` 则跳过编码层）、`dropout`
  * 聚合：`entity_pooling(mean|max)`、`context_inner_pooling(mean|max)`
  * 其他：`return_raw_contexts`、`device`
* **要点**：ID 上下文先在 Lc 维聚合（mean/max），再对 E 维过 Transformer（若启用）。

### 1.3 Context Encoder（`context_encoder.py`）

* **Forward（keyword-only）**：

  ```py
  forward(
    *,
    context_input_ids[B,E,Lc,Lt], context_attention_mask[B,E,Lc,Lt],
    context_token_type_ids[B,E,Lc,Lt]=None,
    contexts_mask[B,E,Lc]=None,
  ) -> {
    contexts_last_hidden[B,E,D], contexts_pooled[B,D], contexts_mask[B,E],
    (raw_contexts_last_hidden[B,E,Lc,D])
  }
  ```

* **配置（`ContextEncoderConfig`）**：文本编码器 `TextEncoderConfig`（或外部注入模块）、`freeze_text_encoder`、`inner_pooling`、`entity_pooling`、`return_raw_contexts`、`device`
* **要点**：展平 `(B*E*Lc,Lt)` 编码 → 复原 → Lc 聚合（mean/max）→ E 聚合；仅计算有效上下文。

### 1.4 Prediction Head（`head.py`）

* **Forward**：`forward(p[ B,Hp ]?, q[ B,Hq ]?, r[ B,Hr ]?, labels=?) -> {logits, (loss), fused}`
* **配置（`HeadConfig`）**：`num_labels`、`problem_type`、`use_p/q/r`、`fusion`、`proj_dim`、`hidden_sizes`、`activation`、`dropout`、`layernorm`、`label_smoothing`、`class_weights`、`bias`
* **要点**：

  * 首层与分支投影使用 `nn.LazyLinear`，首个 forward 自动定型。
  * `fusion=sum/mean` 需 `proj_dim` 或手动保证同维。
  * 支持三种任务的损失：CE / BCEWithLogits / MSE。

---

## 2. 注意力模块（签名草案，`ne.py` / `ne2c.py`）

> 这些在 `interfaces/` 的契约里定义；此处给出推荐签名，保持与上游编码器对齐。

### 2.1 NE（News → Entities）

```py
forward(
  *,
  p[B,Hp],
  entities_last_hidden[B,E,D],
  entities_mask[B,E],
) -> {
  q[B,D],                 # 聚合后的实体注意力结果
  attn_weights[B,E],      # 可选：注意力权重，便于可解释
}
```

### 2.2 NE2C（News → Entities & Contexts）

```py
forward(
  *,
  p[B,Hp],
  entities_last_hidden[B,E,D], entities_mask[B,E],
  contexts_last_hidden[B,E,D], contexts_mask[B,E],
) -> {
  r[B,D],                 # 聚合后的上下文注意力结果
  attn_weights[B,E],      # 可选
}
```

> **对齐 KAN**：NE 用 `K=entities_last_hidden(q')`；NE2C 用 `K=q'`、`V=contexts_last_hidden(r')`；`Q=p`。

---

## 3. 数据流（Pipeline）

```text
Text → text_encoder → p[B,Hp]
Entities(ID) → entity_encoder → q'[B,E,D], r'[B,E,D]
Contexts(Text) → context_encoder → r'[B,E,D]  (可替换/并联)

NE(p, q')  → q[B,D]
NE2C(p, q', r') → r[B,D]

Head([p,q,r]) → logits / value
```

* **可替换**：ID 上下文 vs 文本上下文二选一或并联（ensemble/gating）。
* **消融（Ablation）**：`Head.use_p/q/r` 即开即用。

---

## 4. 日志（Logging）与配置（Configuration）

* 命名空间：`kan.*`；各模块使用 `getLogger("kan.<name>")`，不在模块内重复添加 Handler。
* 推荐入口统一配置：读取 `LOG_CFG`（YAML/JSON）→ `dictConfig`；或退回 `basicConfig`。
* 级别建议：

  * `INFO`：构建/形状/关键超参；
  * `DEBUG`：中间张量形状（过滤频率高的循环内日志）。

---

## 5. 设备与性能（Device & Performance）

* 设备自动选择：CUDA 可用则上 GPU，否则 CPU。Windows 友好（无 fork 依赖）。
* 速度要点：

  * `context_encoder` 只编码有效 `(B,E,Lc)` 条目；
  * `entity_encoder` 的 `xformer_layers=0` 可作快速基线；
  * Head 使用 `LazyLinear` 简化图构建。

---

## 6. 最小示例（Minimal Example）

```python
from kan.modules.text_encoder import TextEncoderConfig, build_text_encoder
from kan.modules.entity_encoder import EntityEncoderConfig, build_entity_encoder
from kan.modules.context_encoder import ContextEncoderConfig, build_context_encoder
from kan.modules.head import HeadConfig, build_head

# 1) 文本 p
te_cfg = TextEncoderConfig(model_name_or_path="prajjwal1/bert-tiny", max_length=128, pooling="mean")
text_enc = build_text_encoder(te_cfg)
news_batch = text_enc.batch_encode(["Breaking: ...", "Rumor: ..."], device="cuda")
po = text_enc(**news_batch)["pooled_output"]  # p: [B,Hp]

# 2) 实体与上下文（ID）
ee_cfg = EntityEncoderConfig(vocab_size=50000, embedding_dim=256, xformer_layers=1, xformer_heads=4)
entity_enc = build_entity_encoder(ee_cfg)
entity_ids = ...          # [B,E]
context_ids = ...         # [B,E,Lc]
ents = entity_enc(entity_ids=entity_ids, context_ids=context_ids)
q_prime = ents["entities_last_hidden"]   # [B,E,D]
r_prime = ents["contexts_last_hidden"]   # [B,E,D]

# 3) （可选）文本上下文
ce_cfg = ContextEncoderConfig(text_encoder=te_cfg)
ctx_enc = build_context_encoder(ce_cfg)
ctx_batch = ...  # 由 ctx_enc.batch_encode_contexts(...) 构建
ctx = ctx_enc(**ctx_batch)

# 4) 注意力与融合（伪代码，待接 NE/NE2C 实现）
q = NE(p=po, entities_last_hidden=q_prime, entities_mask=ents["entities_mask"])  # -> [B,D]
r = NE2C(p=po, entities_last_hidden=q_prime, entities_mask=ents["entities_mask"],
         contexts_last_hidden=r_prime, contexts_mask=ents["contexts_mask"])       # -> [B,D]

# 5) 头
head = build_head(HeadConfig(num_labels=2, fusion='concat', hidden_sizes=[256]))
out = head(p=po, q=q, r=r, labels=...)   # -> {logits,(loss),fused}
```

---

## 7. 兼容性（Compatibility）

* **We don't break userspace!**：新增字段有默认值；关键张量键名稳定；`state_dict` 前向兼容。
* 依赖：`torch`（必需）、`transformers`（当使用 `text_encoder`/`context_encoder`）。

---

## 8. 常见问答（FAQ）

* **Q：`fusion=sum/mean` 时报维度不一致？**
  A：为 `p/q/r` 各开 `proj_dim`（例如 256）或手动保证三者同维。
* **Q：如何只用实体，不用上下文？**
  A：只调用 NE 得到 `q`，Head 设 `use_r=False`。
* **Q：如何只用文本上下文？**
  A：不用 `entity_encoder` 的 `contexts_last_hidden`，改用 `context_encoder` 的 `contexts_last_hidden` 作 NE2C 的 V。
* **Q：如何做消融？**
  A：Head 的 `use_p/use_q/use_r`；或把某一路置零/冻结以验证贡献度。

---

## 9. 命名与风格（Conventions）

* 变量命名：`B` 批、`L` 词长、`E` 实体数、`Lc` 每实体上下文数、`Lt` 上下文 token 长、`H/Hp/D` 维度。
* 张量键名：`*_last_hidden`（序列/实体维度）、`*_pooled`（聚合）、`*_mask`（1 有效）。
* 日志级别：模块 INIT 用 `INFO`；循环中仅在 `DEBUG` 打形状。

---

## 10. 展望（Roadmap）

* `modules/attention/ne.py` / `ne2c.py`：签名按本 README，实现按 `interfaces/` 契约。
* `configs/`：提供 `bert-tiny` 与 `deberta-v3-small` 的最小可跑配置；CI 烟雾测试。
* 融合策略扩展：自适应门控（Gating）/ 特征选择（Feature Selection）。

> 有任何契约变更都必须在本 README 先行更新；
> 模块内严禁自建 Handler；遵循统一命名空间日志策略。
