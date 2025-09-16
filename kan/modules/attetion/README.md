# kan.modules.attention — NE / NE2C 注意力模块设计规范 & 日志约定

> 日期：2025-09-16
> 运行环境：Windows · Python 3.11 · PyTorch 2.8（CUDA 12.8）
> 依赖基座：`kan/utils/logging.py`（集中式日志）、`kan/interfaces/encoders.py`（统一编码器契约）

## 0. 范围与不做什么（Scope / Non-Goals）

* **范围**：本 README 约束 `kan/modules/attention/` 目录中 **NE**（News→Entities）与 **NE2C**（News→Entities→Contexts）两类注意力模块的：

  * 数学定义（简述），
  * **张量契约（Tensor Contracts）**，
  * **日志（logging）** 集成与规范，
  * 错误语义、性能建议、单测约定与最小使用样例。
* **不做**：本README不提供注意力的具体实现算法；当前仓库中的 `ne.py`、`ne2c.py` 仅包含**签名 + 日志 + 形状校验**，`forward()` 会抛出 `NotImplementedError`（便于先打通数据/日志链路与单测）。

---

## 1. 目录与文件映射（Directory ↔ Modules）

```
kan/modules/attention/
├─ __init__.py                 # 可选：导出 NEAttention / NE2CAttention
├─ ne.py                       # 新闻↔实体 注意力算子（签名+日志+校验）
└─ ne2c.py                     # 新闻↔实体↔上下文 分层注意力（签名+日志+校验）
```

* README 对应目录：**`kan.modules.attention.README.md`**（你现在看到的这份）

---

## 2. 数学定义（简述）

> 术语首次出现给出英文：注意力（Attention）、上下文（Context）、多头（Multi-Head）

* 基础公式（单头）

  $$
  \mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
  $$
* **NE（News→Entities）**

  * Query $Q = W_Q \cdot p$，其中 $p \in \mathbb{R}^{L_t\times D}$ 为**新闻编码器**序列表示（`news.last_hidden_state`）。
  * Key/Value $K,V = W_{K/V} \cdot q'$，其中 $q' \in \mathbb{R}^{L_e\times D}$ 为**实体编码器**序列表示（`entities.last_hidden_state`）。
* **NE2C（News→Entities→Contexts）**（两种常见策略，任选其一或变体）

  1. **分层**：先做 N↔E 得实体权重，再用实体权重在（实体）上下文 $r' \in \mathbb{R}^{E\times L_c\times D}$ 上聚合；
  2. **直连**：以新闻为 Q、实体为 K、上下文为 V，直接得到融合后的新闻表示。

> 工程要求：所有隐层维度 **D 必须一致**，由各编码器 `d_model` 统一对齐。

---

## 3. 张量契约（Tensor Contracts）

### 3.1 公共输入输出结构

* 统一批结构来自 `kan.interfaces.encoders.EncoderOutput`：

  * `last_hidden_state`: `[B, L, D]`
  * `pooled_state`: `[B, D]`（池化策略由具体编码器实现定义）
* 掩码（mask）：整型/布尔张量，`1` 表示有效 token。**形状必须严格匹配**下述契约。

### 3.2 NE（`ne.py`）

**输入**

* `news.last_hidden_state` `[B, Lt, D]`，`news_mask` `[B, Lt]`
* `entities.last_hidden_state` `[B, Le, D]`，`entity_mask` `[B, Le]`

**输出（`NEAttentionOutput`）**

* `fused_states` `[B, Lt, D]` — 融合后的新闻序列
* `pooled` `[B, D]` — 融合后的池化向量
* `attn_weights?` `[B, H, Lt, Le]` — 可选注意力权重（可视化/分析）

### 3.3 NE2C（`ne2c.py`）

**输入**

* `news.last_hidden_state` `[B, Lt, D]`，`news_mask` `[B, Lt]`
* `entities.last_hidden_state` `[B, Le, D]`，`entity_mask` `[B, Le]`
* `contexts_last_hidden` `[B, E, Lc, D]`（每个实体的上下文序列）
* `contexts_mask?` `[B, E, Lc]`

**输出（`NE2CAttentionOutput`）**

* `fused_states` `[B, Lt, D]`
* `pooled` `[B, D]`
* `weights?`：`dict`，含

  * `weights["ne"]` `[B, H, Lt, Le]`（新闻↔实体）
  * `weights["e2c"]` `[B, H, Le, Lc]` 或 `[B, H, E, Lc]`（实体↔上下文，视对齐策略而定）

> 备注：若 `E != Le`，**必须在进入 NE2C 前完成 E 与 Le 的对齐/填充**（collate 或上游对齐逻辑）。

---

## 4. API 签名与错误语义

### 4.1 构造参数（`__init__`）

* 共同参数：`d_model: int`、`n_heads: int`、`dropout: float=0.0`、`use_bias: bool=True`
* NE2C 额外：`context_pooling: str="attn"`（`"mean"|"cls"|"attn"` 等策略自定义）

### 4.2 前向（`forward`）签名（仅摘要）

```python
# NE
def forward(
    *, news: EncoderOutput, entities: EncoderOutput,
    news_mask: Optional[Tensor] = None, entity_mask: Optional[Tensor] = None,
    return_weights: bool = False,
) -> NEAttentionOutput: ...

# NE2C
def forward(
    *, news: EncoderOutput, entities: EncoderOutput, contexts_last_hidden: Tensor,
    news_mask: Optional[Tensor] = None, entity_mask: Optional[Tensor] = None, contexts_mask: Optional[Tensor] = None,
    return_weights: bool = False,
) -> NE2CAttentionOutput: ...
```

### 4.3 错误语义（MUST）

* **形状/维度不一致**：`ValueError`，错误消息必须包含**期望/实际形状**与 **D**。
* **算法未实现**：`NotImplementedError`（当前阶段保留签名与日志）。
* **掩码缺失**：不直接报错，但\*\*`WARNING` 日志\*\*提示“将视为全有效”。

---

## 5. 日志（Logging）约定

### 5.1 统一入口与上下文

* 初始化：在训练/评估脚本早期调用一次

  ```python
  from kan.utils.logging import configure_logging, log_context
  configure_logging()  # 或传入 configs/logging/*.yaml
  with log_context(run_id="2025A001", stage="train", step=0):
      ...
  ```

* 配置加载优先级：显式路径 > 环境变量 `KAN_LOG_CFG` > 内置默认
* Windows 文件轮转：`logs/kan-debug.log`，10MB×5 份

### 5.2 模块 Logger 名称（MUST）

* NE：`kan.modules.attention.ne.NEAttention`
* NE2C：`kan.modules.attention.ne2c.NE2CAttention`

### 5.3 日志级别与内容（SHOULD）

* `INFO @ __init__`：记录超参（`d_model/n_heads/dropout/(context_pooling)/use_bias`）
* `DEBUG @ forward`：

  * **形状**：所有输入张量的 `[B, L(… ), D]`
  * **dtype/device**：三路输入的 `dtype/device`
  * **掩码密度**：`valid_ratio = mask.sum()/mask.numel()`（四舍五入到 4 位小数）
* `WARNING @ forward`：任一 mask 为 `None`（提醒将默认所有 token 有效）
* `ERROR @ 校验`：形状/维度不匹配时，写清**期望 vs 实得**，随后抛 `ValueError`
* `INFO @ forward`：声明“被调用但实现未提供”（当前阶段）

> 低开销实践：统计（如密度）应处于 `if logger.isEnabledFor(DEBUG):` 分支内，避免热路径损耗。

---

## 6. 性能与工程建议（SHOULD/MAY）

* **attn\_weights**：仅在 `return_weights=True` 时计算/返回（节省显存/带宽）。
* **长上下文**（NE2C 的 `[B,E,Lc,D]`）：

  * 优先上游截断/筛选，必要时做**分块注意力**或**梯度检查点**（grad checkpointing）。
* **掩码 dtype**：优先使用 `bool` 或 `long(0/1)`，与 `torch.nn.functional.scaled_dot_product_attention` 兼容。
* **对齐策略**：若实体数 `Le` 与上下文实体数 `E` 不一致，**在 collate 层完成映射或 padding**，避免在注意力算子里耦合样本级对齐细节。
* **日志降噪**：将第三方 noisy logger（`urllib3`, `aiohttp`, `transformers.*`）提到 `WARNING`（`configure_logging` 已内建）。

---

## 7. 单元测试（pytest）要点

* **形状断言**：构造最小张量，断言 `ValueError` 的错误消息包含期望/实际形状与 D。
* **日志断言**：使用 `caplog` 检查 `DEBUG` 中是否出现 `mask density=`、设备与 dtype 字样；`WARNING` 在掩码缺失时出现。
* **幂等性**：重复 `__init__` 与 `forward`（在 `NotImplementedError` 前）不应产生额外副作用或句柄泄漏。
* **设备一致性**（如 GPU 存在）：三路输入 device 应一致，否则提前在上游 `.to(device)`。

---

## 8. 快速检查清单

* [ ] `d_model` 与编码器一致，`B` 对齐，mask 形状完全匹配
* [ ] `logger` 名称正确，`__init__` 打 `INFO`，`forward` 打 `DEBUG/WARN/ERROR`
* [ ] `return_weights` 分支下才计算/返回注意力权重
* [ ] 错误消息包含**期望/实得形状**与 **D**
* [ ] 新参数/行为均符合小版本兼容策略
