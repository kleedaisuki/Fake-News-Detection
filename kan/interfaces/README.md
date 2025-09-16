# KAN Interfaces — 设计规范与使用指南

> 版本：v0.1（2025-09-16）  
> 适用：Windows + Python 3.11 + PyTorch 2.8  

本目录定义 **KAN** 的“稳定契约层（stable contracts）”，将上游**实体链接（Entity Linking, EL）**、**知识图谱访问（Knowledge Graph, KG）**、与**编码器（Encoders）**解耦为可替换模块，确保：

- 研究可快速做**消融/替换**，工程可稳定**演进/扩展**；
- 训练/评估脚本仅依赖接口，不直接依赖具体实现或第三方库。

---

## 目录结构

kan/interfaces/
├─ linker.py # ILinker 协议 + Mention 数据结构 + 异常
├─ kg.py # IKG 协议 + KGNode/KGEdge + 异常
└─ encoders.py # ITextEncoder / IKnowledgeEncoder + Batch/Output

---

## 总体设计原则（MECE）

- **契约稳定**：接口一旦发布，**不破坏向后兼容**（We don’t break userspace）。
- **数据最小充分**：传入/传出仅包含下游所需的**最小必需字段**。
- **批处理友好**：所有主路径 API 支持**批处理（batch）**；形状清晰。
- **错误可控**：仅抛出**本模块定义**的异常类型，隐藏实现细节。
- **可观测性**：保留 `provider`/`extra` 字段用于调试与追踪。
- **Windows 亲和**：统一 `pathlib.Path`/`os.fspath`；避免 `fork` 语义依赖；提供 `close()`。

---

## 统一类型与张量形状

- **隐藏维度**：`D = encoder.d_model`（所有编码器须对齐相同 `D`）。  
- **文本/实体序列输出**（来自 `EncoderOutput`）：
  - `last_hidden_state`: `[B, L, D]`
  - `pooled_state`: `[B, D]`（具体池化策略由实现定义，如 `[CLS]` 或 mean-pool）
- **掩码**（mask）：`[B, L]`，`1` 表示有效（valid）。  
- **NE2C 上下文打包**（在 `modules/attention/ne2c.py` 使用）：
  - `contexts_last_hidden`: `[B, E, Lc, D]`（`E`：每样本实体数；`Lc`：每上下文长度）
  - `contexts_mask`: `[B, E, Lc]`（可选）

> 术语：
>
> - **batch**：批量（Batch）
> - **hidden state**：隐藏状态
> - **pooling**：池化
> - **context**：（实体）上下文

---

## 接口规范

### 1) 实体链接 `ILinker`（`linker.py`）

- **目的**：从原文中抽取**实体提及（mention）**并链接到唯一实体 ID（如 Wikidata QID）。  
- **核心方法**：
  - `set_cache_dir(cache_dir: Optional[Path]) -> None`：设置/禁用缓存；
  - `link(text: str, *, lang=None, max_mentions=None, score_threshold=None) -> List[Mention]`
  - `link_batch(texts: Sequence[str], ...) -> List[List[Mention]]`
  - `close() -> None`：释放资源（HTTP 会话、进程池等）。
- **数据结构**：
  - `Mention(text, start_char, end_char, entity_id?, score?, provider?, extra={})`
- **异常**：`LinkerError`（基类）、`RateLimitError`（触发速率限制时抛出）
- **必须/建议**：
  - **MUST**：同输入→同输出（函数式）；  
  - **MUST**：返回结果按 `score` 降序；  
  - **SHOULD**：`score_threshold` 在实现侧统一解释（0~1 或原始分）；  
  - **MAY**：静态词典/HTTP 缓存；缓存路径应可控。

### 2) 知识图谱 `IKG`（`kg.py`）

- **目的**：给定实体 ID，返回**一跳邻居**（one-hop neighbors）。  
- **核心方法**：
  - `one_hop(entity_id, *, max_neighbors=None, return_ids_only=True, relation_whitelist=None, relation_blacklist=None) -> List[EntityID] | List[KGEdge]`
  - `one_hop_many(entity_ids: Sequence[EntityID], ...) -> Dict[EntityID, List[...]]`
  - `close() -> None`
- **数据结构**：`KGNode`、`KGRelation`、`KGEdge(src, rel, dst, directed, extra)`  
- **异常**：`KGError`、`KGNotFound`
- **必须/建议**：
  - **MUST**：默认 `return_ids_only=True`（集合语义，方向忽略）；  
  - **SHOULD**：尊重 white/blacklist；实现可选择忽略但需记录 `extra["ignored_filters"]=True`；  
  - **MAY**：度裁剪（degree capping）、关系过滤。  

### 3) 编码器 `ITextEncoder` / `IKnowledgeEncoder`（`encoders.py`）

- **目的**：将**文本/实体/上下文**编码为统一维度 `D` 的序列表征与池化向量。  
- **核心方法**：
  - 只读属性：`d_model: int`，`device: torch.device`
  - `encode(batch: SequenceBatch, *, train: bool=False) -> EncoderOutput`
  - `to(device) -> Self`，`close()`
- **批结构**：`SequenceBatch(ids: Long[B,L], mask: Bool/Long[B,L], type_ids?: Long[B,L])`
- **必须/建议**：
  - **MUST**：`encode()` 不在接口层做并行/多进程约定（让实现自决）；  
  - **MUST**：`d_model` 相互一致；  
  - **SHOULD**：`pooled_state` 的策略在实现文档中明确；  
  - **MAY**：按需缓存 tokenizer/权重到 `set_cache_dir()` 所指路径（若实现支持）。

---

## 错误处理与可观测性

- 仅抛出 `interfaces` 自身异常类型（避免第三方异常“漏出”）。  
- 推荐在 `extra` 字段传回调试信息，比如：上游 `status_code`、缓存命中、裁剪统计。  
- 日志靠上层注入（logger 由上层管理），接口层保持“纯净”。

---

## 资源管理与 Windows 注意事项

- **MUST**：实现 `close()` 幂等可多次调用。  
- Windows：
  - 统一使用 `pathlib.Path`；避免硬编码反斜杠；  
  - 不在接口层绑定 `multiprocessing` 策略；  
  - 训练脚本退出前显式调用 `close()` 释放句柄/套接字。

---

## 版本兼容策略

- 小版本（x.y+1）：可新增字段/方法（仅向后兼容扩展）。  
- 大版本（x+1.0）：如需破坏式调整，**必须**在顶层文档与 `CHANGELOG` 明示迁移路径。

---

## 简单用例

```python
from kan.interfaces.linker import ILinker
from kan.interfaces.kg import IKG
from kan.interfaces.encoders import ITextEncoder, SequenceBatch

def build_batch(tokenizer, texts):
    tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return SequenceBatch(ids=tok["input_ids"], mask=tok["attention_mask"])

def demo(linker: ILinker, kg: IKG, text_encoder: ITextEncoder, texts: list[str]):
    linker.set_cache_dir(Path(".cache/el"))
    mentions_per_text = linker.link_batch(texts, lang="en", max_mentions=16, score_threshold=0.2)

    # 收集实体ID并做一跳邻居
    entity_ids = sorted({m.entity_id for ms in mentions_per_text for m in ms if m.entity_id})
    neighbors = kg.one_hop_many(entity_ids, max_neighbors=64)

    # 编码文本
    batch = build_batch(tokenizer=text_encoder.tokenizer, texts=texts)  # 取决于实现
    enc = text_encoder.encode(batch)
    print(enc.last_hidden_state.shape, enc.pooled_state.shape)  # [B,L,D], [B,D]

    linker.close(); kg.close(); text_encoder.close()

