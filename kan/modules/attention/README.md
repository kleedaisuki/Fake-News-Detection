# `kan.modules.attention` — 知识感知注意力（Knowledge-aware Attention）

> 版本：v0.1（与 `kan.interfaces` v0.1 对齐）
> 依赖：Python 3.11、PyTorch ≥ 2.1
> 术语首次出现附英文名：如“掩码（mask）”“池化（pooling）”“缩放点积注意力（scaled dot-product attention）”。

本目录提供两类可插拔注意力模块，用于把**文本编码**与**知识编码**进行跨模态融合：

* `NEAttention`：新闻→实体的跨注意力（cross-attention），输出**token-级融合**表征与文级池化；实现为 PyTorch `nn.MultiheadAttention` 的封装，强调**接口契约与健壮性**（1=valid 的 mask、极端样本回退等）。
* `NE2CAttention`：新闻→实体 & 实体上下文的**分层注意力（hierarchical attention）**，两级：N-E（token→entity）与 N-E2C（doc→context），再做门控融合与 LayerNorm，输出融合后的 token 序列与文级向量，可选返回 `{'ne':…, 'e2c':…}` 两张热力图。

两者均遵循 `kan.interfaces.encoders.EncoderOutput` 的张量约定：
`last_hidden_state: [B, L, D]`，`pooled_state: [B, D]`；mask 统一为 `[B, L]` 且 **1=有效**（valid）。

## 1. 快速上手（Registry 构建）

```python
from kan.utils.registry import HUB
from kan.interfaces.encoders import EncoderOutput
import torch

# 通过 registry 构建（attention 命名空间）
ATTN = HUB.get_or_create("attention")
ne = ATTN.build({"type": "ne", "d_model": 256, "n_heads": 4, "dropout": 0.1})
ne2c = ATTN.build({"type": "ne2c", "d_model": 256, "n_heads": 4, "dropout": 0.1})

# 伪造输入
B, Lt, Le, Lc, D = 2, 128, 32, 16, 256
news = EncoderOutput(last_hidden_state=torch.randn(B, Lt, D), pooled_state=torch.randn(B, D))
ents = EncoderOutput(last_hidden_state=torch.randn(B, Le, D), pooled_state=torch.randn(B, D))
ctx_last = torch.randn(B, Le, Lc, D)

news_mask = torch.ones(B, Lt, dtype=torch.long)     # 1=valid
ent_mask  = torch.ones(B, Le, dtype=torch.long)
ctx_mask  = torch.ones(B, Le, Lc, dtype=torch.long)

# N-E
out_ne = ne(news=news, entities=ents, news_mask=news_mask, entity_mask=ent_mask, return_weights=True)
print(out_ne.fused_states.shape, out_ne.pooled.shape, None if out_ne.attn_weights is None else out_ne.attn_weights.shape)
# -> [B,Lt,D], [B,D], [B,H,Lt,Le]

# N-E2C
out_ne2c = ne2c(
    news=news, entities=ents, contexts_last_hidden=ctx_last,
    news_mask=news_mask, entity_mask=ent_mask, contexts_mask=ctx_mask, return_weights=True
)
print(out_ne2c.fused_states.shape, out_ne2c.pooled.shape, {k: v.shape for k,v in out_ne2c.weights.items()})
# -> [B,Lt,D], [B,D], {'ne':[B,H,Lt,Le], 'e2c':[B,H,1,Le]}
```

> 说明：`NEAttention` 已在构造函数中内置 `nn.MultiheadAttention(batch_first=True)`；`NE2CAttention` 则自实现**缩放点积注意力**与多头拆合，便于二级注意力的定制与权重曝光。

---

## 2. API 规格（MECE）

### 2.1 `NEAttention`

* **输入**

  * `news: EncoderOutput` → `last_hidden_state [B,Lt,D]`
  * `entities: EncoderOutput` → `last_hidden_state [B,Le,D]`
  * `news_mask: Optional[Tensor[B,Lt]]`，`entity_mask: Optional[Tensor[B,Le]]`（**1=valid**）
  * `return_weights: bool=False`
* **计算**

  * `Q ← news.tokens`；`K,V ← entities.tokens`；多头注意力 + 恒等输出层
  * **key_padding_mask** 语义转换：接口侧 1=valid → PyTorch 侧 True=padding（取反）；当某样本 `entity_mask` 全 0（全 padding）时，**退化为“无 key mask”**，避免 `softmax(-inf)`→NaN。
  * 对 `fused_states` 在 `news_mask` 上做**掩码均值池化（masked mean pooling）**得 `pooled`。
* **输出**

  * `NEAttentionOutput(fused_states[B,Lt,D], pooled[B,D], attn_weights? [B,H,Lt,Le])`（可选）

**典型用途**：当你只想“让文本按需读取实体序列信息”而不引入上下文时，用它作为**轻量级**融合层。与 KAN 论文中的 N-E 注意力在角色上等价（Q from news；K/V from entities）。

---

### 2.2 `NE2CAttention`

* **输入**

  * 同上，外加 `contexts_last_hidden: Tensor[B,E,Lc,D]`（E==Le），`contexts_mask: Optional[Tensor[B,E,Lc]]`。
* **两级注意力**

  1. **N-E（token→entity）**
     `Q ← news.tokens`，`K,V ← entities.tokens` → `ne_out[B,Lt,D]` 与 `ne_w[B,H,Lt,Le]`。
  2. **N-E2C（doc→context）**
     先对每实体上下文序列 `contexts_last_hidden[B,E,Lc,D]` 以 `contexts_mask` 作**掩码均值**→ `[B,E,D]`；
     `Q ← news.pooled[1,D]`（升维为 Lq=1），`K ← entities.tokens（等效实体级聚合）`，`V ← per-entity context pooled` → `r_doc[B,D]` 与 `e2c_w[B,H,1,Le]`。

     > 这里遵循论文角色：N-E2C 用**实体“活性”**给上下文聚合加权（K 来自实体，V 来自上下文），Q 保持来自新闻。
* **融合**

  * 小**门控（gating）**避免外部知识“淹没”文本：对 `ne_out`、`r_doc` 各自 sigmoid 门控后残差注入；再 `LayerNorm`。随后在 token 级做 `masked mean` 得到 `pooled`，并与 `r_doc`、实体摘要混合后经 `pooler` 线性层。
* **输出**

  * `NE2CAttentionOutput(fused_states[B,Lt,D], pooled[B,D], weights?={'ne':[B,H,Lt,Le],'e2c':[B,H,1,Le]})`

**与论文一致性**：N-E 与 N-E2C 的 Q/K/V 角色与 KAN 原文完全同构；实现细节（如上下文按邻居**均值**）亦与论文“用一跳邻居均值作为实体上下文向量”的做法对齐。

---

## 3. 掩码（mask）与形状校验

* **统一语义**：所有 mask 的 **1 表示有效（valid）**；内部在需要位置进行取反以满足底层算子语义（例如 PyTorch 的 `key_padding_mask` 要求 True=padding）。
* **严格校验**：模块会检查 `B/L/D` 与 mask 形状是否一致，否则抛 `ValueError` 并记录错误日志。
* **退化与兜底**：

  * `NEAttention`：若某样本 `entity_mask` 全 0，降级为**不传 key mask**，避免 NaN；
  * `masked_mean`：当分母为 0（全无效）时退化为**普通 mean**，确保数值稳定。

---

## 4. 可观测性与调试

* **权重可视化**：`return_weights=True` 时，N-E 返回 `[B,H,Lt,Le]`，N-E2C 返回 `[B,H,1,Le]`；可直接做 token/实体热力图。
* **日志**：在 DEBUG 级别打印输入形状、mask 密度（density），并对缺失 mask、极端全 padding、形状不符给出清晰告警。配合 `kan.utils.logging.configure_logging()` 食用更香。

---

## 5. 与论文 KAN 的关系（原理对照）

* **N-E**：Q 来自新闻语义，K/V 来自实体编码，对实体分配**重要性权重（importance weights）**；
* **N-E2C**：Q 仍来自新闻，K 来自实体，V 来自“实体的一跳邻居均值”（即实体上下文），以实体“活性”指导上下文聚合；
* **分类头（外部）**：论文将 `p,q,r` 拼接入分类器；本项目把融合/池化职责前移到模块内，保证下游 head 更简单可换。

## 6. 常见问题（FAQ）

* **Q：为什么权重形状在 N-E2C 是 `[B,H,1,Le]`？**
  **A**：N-E2C 的 `Q` 取文级 `pooled_state`（长度 1），语义是“整篇文章对每个实体上下文的权重”，与论文设计一致。
* **Q：mask 为什么 1=valid 而不是 1=pad？**
  **A**：与 `kan.interfaces` 的统一契约保持一致，避免跨模块歧义；在内部转换以适配底层算子语义。
* **Q：能否输出每头权重而不是平均？**
  **A**：`NEAttention` 默认返回**每头**权重（`average_attn_weights=False`）；`NE2CAttention` 自实现的多头拆合，天然提供 per-head 热力图。

---

## 7. 版本与扩展位

* **当前版本**：v0.1（与接口 v0.1 对齐，**不破坏向后兼容**）；
* **扩展位**：

  * `NE2CAttention.context_pooling`: 预留策略枚举（当前实现等效为 masked-mean，可扩展为 attention/Max/GRU）；
  * 增加实体/关系类型感知（relation-aware）K/V 投影；
  * 在 `weights` 中返回**实体级**与**上下文级**双尺度归一化权重，便于可解释性分析。
