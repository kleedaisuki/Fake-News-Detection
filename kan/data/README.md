# `kan.data` — 数据层设计与使用指南

> 版本：v0.1（2025-09-26）
> 适用环境：Windows/Linux/Mac，Python ≥3.10（推荐 3.11），PyTorch 2.8
> 契约：输出稳定 `NewsRecord` 架构，保证**向后兼容**（We don’t break userspace）

本目录提供 **KAN（Knowledge-aware Attention Network）** 的数据处理基座，涵盖从 **加载（Loaders）** → **实体链接（Entity Linking）** → **知识图谱上下文获取（KG Fetcher）** → **文本向量化（Vectorizer）** → **批处理（Batcher）** 的完整链路。

---

## 目录结构

```
kan/data/
├─ loaders.py        # 数据加载：CSV/JSONL/HF → NewsRecord Schema v1
├─ entity_linking.py # 实体链接：文本 → Wikidata QIDs
├─ kg_fetcher.py     # 知识图谱邻居获取：QID → 一跳邻居/属性
├─ vectorizer.py     # 文本向量化：HF/SentenceTransformers → Torch 向量
└─ batcher.py        # 批处理器：NewsRecord → 张量化（Text/Entity/Context）
```

---

## 1. `loaders.py` — 数据加载层

### 功能

* 将不同格式（CSV/JSONL/HuggingFace Datasets）统一转换为稳定 `NewsRecord` Schema。
* 保证向后兼容（Schema v1 固化）：

```python
NewsRecord {
  id: str, text: str, label: int,
  entities: list[str], contexts: dict[str, list[str]],
  meta: dict
}
```

### 特点

* **字段映射（FieldMap）**：配置驱动，将原始列名映射到标准字段。
* **多源支持**：`CSVLoader` / `JSONLinesLoader` / `HFDatasetLoader`。
* **鲁棒性**：缺失字段自动回退；文本归一化（大小写/空白）；标签映射。
* **注册表集成**：支持通过 `kan.utils.registry.HUB` 构建。

---

## 2. `entity_linking.py` — 实体链接

### 功能

* 将 `NewsRecord.text` 中的实体提及（mentions）映射到 Wikidata QID。
* 输出：更新 `entities: List[QID]`，并在 `meta['el']` 写入追溯信息。

### 特点

* **可插拔后端**：Dummy gazetteer / TagMe / Wikipedia / 自定义。
* **缓存**：基于配置与文本内容哈希，保证可复现与高效。
* **可追踪性**：记录 `EntityMention(surface, span, qid, score)`。
* **契约**：`link_records(records, cfg: ELConfig) -> List[NewsRecord]`

### 默认实现

* `DummyLinker`：零依赖词表匹配，CI 友好。
* `TagMeLinker` / `WikipediaLinker`：预留接口，需外部 API 或索引。

---

## 3. `kg_fetcher.py` — 知识图谱邻居获取

### 功能

* 给定实体 QID，获取其一跳邻居（entity contexts）。
* 输出：更新 `contexts: Dict[QID, List[str]]`，并在 `meta['kg']` 写入追溯。

### 特点

* **可插拔后端**：

  * `LocalProvider`：本地 JSON/目录/JSONL 索引。
  * `WikidataRESTProvider`：Wikidata REST API。
  * `SPARQLProvider`：SPARQL 查询接口。
* **缓存**：逐 QID 文件缓存，支持 TTL 与禁用选项。
* **契约**：`fetch_context(records, cfg: KGConfig) -> List[NewsRecord]`
* **鲁棒性**：限速、错误容忍、Windows 友好。

---

## 4. `vectorizer.py` — 文本向量化

### 功能

* 面向配置的文本向量器，支持 HuggingFace Transformers / SentenceTransformers。
* 输出：`torch.FloatTensor`，可直接参与下游计算。

### 特点

* **接口**：

  ```python
  class BaseVectorizer:
      def encode_texts(self, texts: list[str]) -> Tensor
      def encode(self, text: str) -> Tensor
  ```

* **特性**：批处理、池化策略（CLS/mean）、设备与 dtype 管理、L2 归一化。
* **缓存**：基于文本内容哈希，按需持久化到本地。
* **契约**：`build_vectorizer(cfg: VectorizerConfig) -> BaseVectorizer`

---

## 5. `batcher.py` — 批处理器

### 功能

* 将 `List[NewsRecord]` 整理为三路张量：

  * **Text branch**：HuggingFace 分词结果 (`input_ids/attention_mask`) 与/或向量化特征。
  * **Entity branch**：实体 ID 张量 `[B,E]` + 掩码。
  * **Context branch**：一跳邻居 `[B,E,N]` + 可选属性 ID + 掩码。

### 特点

* **配置**：`BatcherConfig(text, entity, context, device)`。
* **词表**：`EntityVocab` 与 `PropertyVocab`（双向映射，含 PAD/UNK）。
* **契约**：`collate(records) -> Dict[str, Tensor]`
* **截断策略**：超限实体/邻居会截断，保证接口不破坏。
* **兼容性**：仅依赖 `torch`，可选依赖 transformers/vectorizer。

---

## 6. 总体设计原则

* **契约稳定**：Schema 与张量键名固定，扩展仅新增。
* **批处理友好**：所有组件支持 batch 输入。
* **缓存优先**：EL/KG/Vectorizer 全面支持缓存，保障复现性。
* **Windows 友好**：无 POSIX 依赖，统一 `pathlib` 路径。
* **可扩展性**：统一 Registry 命名空间（`entity_linker`、`kg_fetcher`、`vectorizer`）。

---

## 7. 典型流程（Pipeline）

```python
from kan.data.loaders import build_loader
from kan.data.entity_linking import link_records, ELConfig
from kan.data.kg_fetcher import fetch_context, KGConfig
from kan.data.vectorizer import build_vectorizer, VectorizerConfig
from kan.data.batcher import Batcher, BatcherConfig

# 1. 加载数据
loader = build_loader(...)
records = loader.load_split("train")

# 2. 实体链接
records = link_records(records, ELConfig(backend="dummy", lexicon_path="lex.json"))

# 3. 知识图谱邻居
records = fetch_context(records, KGConfig(backend="local", local_path="kg.json"))

# 4. 文本向量化
vec = build_vectorizer(VectorizerConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# 5. 批处理
batcher = Batcher(BatcherConfig())
batch = batcher.collate(records[:8])
print(batch.keys())  # ids, text, text_tok, ent_ids, ctx_ids, ...
```
