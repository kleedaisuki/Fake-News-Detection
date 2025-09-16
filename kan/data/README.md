# `kan.data` 模块说明（前后端契约版）

> 版本：v1（schema v1，承诺**向后兼容**）
> 目标：为 KAN（Knowledge‑aware Attention Network）提供**稳定、可复现**的数据入口与批处理（batching），并对上游/下游明确**前后端契约**。

---

## 1. TL;DR（一图流）

```
原始数据（CSV/JSONL/HF）
        │     （FieldMap/label_map/编码/去重）
        ▼
kan.data.loaders  →  List[NewsRecord]（稳定 Schema）
        │     （可选：pipelines/entity_linking & kg_fetcher 回填 entities/contexts）
        ▼
kan.data.batcher  →  {text_tok/text_vec, ent_ids, ctx_ids, …}  → encoders/NE/NE2C/head
```

---

## 2. 目录与职责边界

```
kan/data/
├─ loaders.py        # 数据加载与标准化（CSV/JSONL/HF），输出 NewsRecord
├─ batcher.py        # 三路批处理（Text/Entity/Context）与词表（Entity/Property）
└─ (本 README)       # 契约、使用方式、最佳实践
```

**边界说明**：

* **loaders**：只做 **IO 标准化与轻校验**；不做实体链接（Entity Linking, *EL*）与一跳邻接拉取（KG）。
* **pipelines**（在 `kan/pipelines/`）：`entity_linking.py` 与 `kg_fetcher.py` 负责回填 `entities/contexts`。
* **batcher**：只做 **组织与填充张量**；不包含模型前向逻辑。

---

## 3. 前后端契约（Frontend ↔ Backend Contract）

> 约定中的 “前端 Frontend” 指**调用方/业务编排方**（如 `scripts/*.py`、训练脚本、上游数据工程）；“后端 Backend” 指**本模块提供的稳定 API**（`loaders.py`、`batcher.py`）。

### 3.1 稳定 Schema v1：`NewsRecord`

```py
NewsRecord: {
  id: str,                      # 必填；若源缺失则由 text 哈希生成（blake2b）
  text: str,                    # 必填；可选 lower/strip 归一
  label: int,                   # 必填；默认二分类：0=real, 1=fake（由 label_map 归一）
  entities: List[str],          # 选填；Wikidata QID（如 "Q76"）
  contexts: Dict[str, List[str]],  # 选填；QID->邻居，邻居可为 "Qxxxx" 或 "Pxxxx=Qyyyy"
  meta: Dict[str, Any]          # 选填；追溯字段（split/fold/源字段/EL/KG 细节）
}
```

**演进策略**：

* v1 **字段名不更改**；新增只往 `meta` 扩展；
* 任何破坏性更改→必须 **v2** 并提供兼容层；
* 读取方**不得**依赖 `meta` 的内部结构（`meta.el/*.kg/*` 除外，见下）。

### 3.2 分割与折次

* `Split` 命名：`train/validation/test`（推荐）或自定义（如 `fold0`）。
* `fold`：若源数据提供，建议写入 `meta['fold']`（int）。

### 3.3 EL/KG 追溯信息（可依赖的 `meta` 子树）

* `meta['el']`：`{backend, version, language, threshold, time, mentions:[{surface,start,end,qid,score}]}`
* `meta['kg']`：`{backend, version, time, qids:[...], stats:{uniq_qids,total_q,cache_hit}, return_edges, topk, properties}`

> 这些键用于复现实验与误差分析；仅此两处 `meta` 子树可被下游**有条件依赖**。

---

## 4. `loaders.py`（数据加载与标准化）

### 4.1 能力概览

* **来源**：`csv` / `jsonl` / `hf`（HuggingFace Datasets）。
* **映射**：`FieldMap` 将源列映射至 `id/text/label/(entities/contexts/split/fold/meta)`。
* **标签归一**：`label_map`（如 `{"real":0, "fake":1}`）。
* **ID 策略**：源缺失→基于文本哈希生成 `auto_xxx`；可加 `id_prefix`。
* **容错**：可选列缺失→降级为 `[]/{}`；字符串化 JSON 自动解析。
* **去重**：按 `text` 去重（可选）。
* **编码**：`encoding`/`delimiter`/`lines` 参数化（Windows 友好）。
* **注册表**：命名空间 `dataset`，`build_loader(cfg)` 根据 `format` 选择实现。

### 4.2 关键数据结构

```py
@dataclass
class FieldMap:
  id: str|None; text: str|None; label: str|None
  entities: str|None; contexts: str|None
  split: str|None; fold: str|None
  meta: str | List[str] | None

@dataclass
class DatasetConfig:
  name: str; format: str           # 'csv'|'jsonl'|'hf'
  path: str|Path|None;             # 文件根目录
  splits: Mapping[str, str]|None   # {split -> 文件或 HF split 名}
  hf_name: str|None; hf_config: str|None
  delimiter: str = ','; encoding='utf-8'; lines=True
  fields: FieldMap = FieldMap()
  label_map: Mapping[Any,int]|None = None
  id_prefix: str|None = None
  drop_duplicates_on_text=False; lowercase_text=False; strip_text=True
```

### 4.3 使用范式

**YAML**（示例）：

```yaml
name: gossipcop
format: csv
path: ./data/gossipcop
splits: { train: train.csv, validation: dev.csv, test: test.csv }
fields: { id: nid, text: text, label: label }
label_map: { true: 0, false: 1 }
id_prefix: gc
```

**Python**：

```py
from kan.data.loaders import loader_from_config, Dataset
loader = loader_from_config(yaml.load(open('configs/data/gossipcop.yaml'), Loader=yaml.SafeLoader))
ds = Dataset(loader, preload=['train'])
recs = ds.get('train')  # -> List[NewsRecord]
```

---

## 5. `batcher.py`（三路批处理与词表）

### 5.1 输出字典（形状与语义）

| 键                         | 形状/类型               | 说明                               |
| ------------------------- | ------------------- | -------------------------------- |
| `ids`                     | `List[str]`         | 样本 ID 顺序对齐                       |
| `text`                    | `List[str]`         | 原文文本（用于调试/日志）                    |
| `text_tok.input_ids`      | `LongTensor[B,T]`   | HF 分词结果（可选）                      |
| `text_tok.attention_mask` | `LongTensor[B,T]`   | HF 掩码（可选）                        |
| `text_tok.token_type_ids` | `LongTensor[B,T]`   | 可选（BERT 类型）                      |
| `text_vec`                | `FloatTensor[B,D]`  | 句向量（可选，来自 `vectorizer`）          |
| `ent_ids`                 | `LongTensor[B,E]`   | 实体 QID 索引（保留 `<PAD>=0, <UNK>=1`） |
| `ent_mask`                | `BoolTensor[B,E]`   | 实体有效位                            |
| `ctx_ids`                 | `LongTensor[B,E,N]` | 一跳邻居 QID 索引（按实体对齐）               |
| `ctx_prop`                | `LongTensor[B,E,N]` | 关系属性索引（可选，`<PAD>=0, <NONE>=1`）   |
| `ctx_mask`                | `BoolTensor[B,E,N]` | 邻居有效位                            |

> 超限策略：`E` 超 `max_entities`、`N` 超 `max_neighbors` → **截断**；形状不变。

### 5.2 配置与依赖

```py
@dataclass
class TextConfig:
  tokenizer_backend: 'hf'|None = 'hf'
  tokenizer_name: str|None = 'bert-base-uncased'
  max_length=256; pad_to_max=True; truncation=True
  return_token_type_ids=False
  vectorizer: VectorizerConfig|None = None  # 若提供，则产出 text_vec

@dataclass
class EntityConfig:  max_entities=64
@dataclass
class ContextConfig: max_neighbors=32; keep_properties=True
@dataclass
class BatcherConfig:
  text: TextConfig; entity: EntityConfig; context: ContextConfig; device: 'cpu'（推荐）
```

* **词表**：`EntityVocab(PAD=0, UNK=1)`；`PropertyVocab(PAD=0, NONE=1)`。
* **解析**：邻居支持 `"Qxxxx"` 或 `"Pxxxx=Qyyyy"` 两种形态。
* **向量器**：对接 `kan.modules.vectorizer`（可替换后端：HF/SentenceTransformers）。

### 5.3 推荐用法

```py
batcher = Batcher(cfg)
# 用训练集预热词表（建议持久化到文件，验证/测试加载同一份）
batcher.build_vocabs(train_records)
# collate 任意 split 的样本
batch = batcher.collate(records)
```

---

## 6. 可复现性、缓存与日志

* **可复现性**：

  * loader：`label_map/id_prefix/去重规则/文本正则化` 由配置固化；
  * EL：按文本内容寻址缓存（`.cache/el`）；
  * KG：按 QID 寻址缓存（`.cache/kg`）；
  * 向量器：按“后端签名+文本”寻址缓存（`.cache/vec`）。
* **日志命名空间**：`kan.data.loaders` / `kan.data.batcher`（建议统一在 `kan.utils.logging.configure_logging()` 中设置）。
* **Windows 友好**：文件/目录用 `pathlib.Path`；显式 `encoding`；CSV 分隔符可配置。

---

## 7. 最佳实践与常见坑（FAQ）

1. **标签不是 int？** 用 `label_map` 统一到 int；未映射到的值会警告并回退到 0（可在预处理阶段清洗）。
2. **`entities/contexts` 是字符串？** 支持**字符串化 JSON**；`contexts` 中的邻居若为 `"P=Q"` 形式，将解析出 `Q` 与属性 `P`。
3. **超长文本/过多实体/邻居？** 文本在 tokenizer/向量器处 `max_length` 截断；实体/邻居在 batcher 处按 `E/N` 截断。
4. **离线/无网络环境？** 选择 `csv/jsonl` 与 `LocalProvider`/本地 `vectorizer`；关闭一切远端后端。
5. **多进程 DataLoader？** 向量器缓存写入是逐文本 `.pt`；建议 CPU 上写入，避免 GPU 设备锁资源。
6. **版本演进**：若未来切换 schema，请在 `meta['schema_version']=2` 并保留 v1 兼容层；避免破坏现有训练代码。

---

## 8. 端到端最小样例

```python
# 1) 加载
loader = loader_from_config({
  "name": "pf", "format": "jsonl", "path": "./data/pf",
  "splits": {"train": "train.jsonl"},
  "fields": {"id": "id", "text": "content", "label": "label"},
  "label_map": {"real": 0, "fake": 1},
})
records = loader.load_split("train")

# 2) 可选：EL + KG（在 pipelines 下执行，回填 entities/contexts）
# records = link_records(records, ELConfig(...))
# records = fetch_context(records, KGConfig(...))

# 3) 批处理
cfg = BatcherConfig(
  text=TextConfig(tokenizer_backend='hf', tokenizer_name='bert-base-uncased', max_length=128,
                  vectorizer=None),
  entity=EntityConfig(max_entities=32),
  context=ContextConfig(max_neighbors=16, keep_properties=True),
  device='cpu')

batcher = Batcher(cfg)
batcher.build_vocabs(records)  # 用训练集预热
batch = batcher.collate(records[:8])
```

---

## 9. 质量保障（QA 清单）

* [ ] 不同来源（CSV/JSONL/HF）在相同 `FieldMap/label_map` 下产出一致的 `NewsRecord`。
* [ ] 缺失可选字段 → `entities=[]`、`contexts={}`，不抛异常。
* [ ] `id` 缺失 → 由文本哈希生成且稳定；`drop_duplicates_on_text=True` 消除重复。
* [ ] `contexts` 中 `P=Q` 正确解析；`keep_properties=False` 时忽略属性但不改变形状。
* [ ] `E/N` 截断后形状稳定；mask 正确标注有效位。
* [ ] 向量器/分词器启用与禁用的组合均可工作（仅 text\_tok / 仅 text\_vec / 两者皆有 / 两者皆无）。
* [ ] Windows 下文件编码/路径、日志输出正常。

---

## 10. 术语对照（含一次英文）

* 实体链接（Entity Linking, **EL**）
* 知识图谱（Knowledge Graph, **KG**）
* 一跳邻居（1‑hop neighbors）
* 关系属性（Relation Property, **Pxxxx**）
* 词表/字典（Vocabulary/Dictionary）
* 归一化（Normalization）
* 截断（Truncation）
