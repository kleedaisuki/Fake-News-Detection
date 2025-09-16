先搓搓小手总结下**观测—分解—设计**，把 KAN 的“经典 Transformer ✕ 外部知识注意力”变成一套**可复现、可扩展、可组合**的工程架构与文档体系；我们完全用“搭积木”的心态来规划，后续再把 GNN 模块像乐高一样卡进来就好～(｡•̀ᴗ-)✧

---

# 观察与拆解（从论文到系统的 MECE 视图）

**目标**：复现 KAN（Knowledge-aware Attention Network），其核心流程是：

1. 对新闻文本做**实体链接**，并从知识图谱取出**一跳邻居**作为**实体上下文**(entity context)；
2. 分别用 Transformer 编码**文本、实体序列、实体上下文序列**；
3. 用两种“知识感知注意力”融合：**N-E**（News → Entities）与 **N-E2C**（News → Entities & Entity Contexts）；
4. 拼接 $p,q,r$ 后用全连接层做真假分类。这正是论文图 2 的主干语义。

* **问题表述**：输入新闻 $S$、其实体 $E=\{e_i\}$ 以及每个实体的一跳邻居集合 $EC=\{ec(e_i)\}$，输出真假标签 $y\in\{0,1\}$。实体上下文用邻居实体向量的**平均**表征（实现上就是 mean pooling）。
* **知识抽取**：先做实体链接（论文用 TagMe→Wikidata），再取**一跳邻居**构成 entity context（既包含入/出边）。&#x20;
* **编码器**：论文使用**单层** Transformer Encoder 编码文本与知识序列（词/实体向量+位置编码），FFN 维度 $d_{ff}=2048$，多头数 $h=4$。&#x20;
* **两级注意力**：
  * **N-E**：$Q=W_Q p,\ K=W_K q',\ V=W_V q'$，得到针对实体序列的权重 $\alpha$ 与汇聚表示 $q$。
  * **N-E2C**：$Q=W_Q p,\ K=W_K q',\ V=W_V r'$，用实体“活性”给上下文加权，得 $r$。
* **分类头**：拼接 $z=[p;q;r]$，经全连接 + softmax，损失为交叉熵 + L2。
* **数据与评测**：PolitiFact、GossipCop（FakeNewsNet）、PHEME；10% 验参 + 其余 5 折；指标 Precision/Recall/F1/Accuracy/AUC。&#x20;
* **消融**：去掉实体、去掉上下文、去掉注意力的多种变体（KAN\E、KAN\EC、KAN\N-E、KAN\N-E2C…），共同说明实体与上下文、两种注意力均带来增益。

> 从**第一性原理**看：我们把“算子（注意力/编码器）与组合（三路融合）”解耦，做**策略（policy）可替换**的模块化架构；后续把 GNN 当作“知识编码器”的另一策略即可热插拔替换，拓扑不动。

---

# 全流程与职责边界（Action × Components × Parameters）

## A. 流程（Action）

* **数据准备**：加载数据集 → 实体链接 → 知识检索（Wikidata 一跳）→ 缓存与版本标记。
* **训练/评估**：构造三路输入批次 → 三个编码器 → N-E 与 N-E2C 融合 → 分类头 → 指标&日志。
* **推理/复现实验**：加载缓存的实体与上下文 → 同步编码与注意力 → 输出概率/标签。

## B. 组件（Components）

1. **Dataset 层**

   * `NewsLoader`：统一字段（`id, text, label, meta`）。
   * `EntityLinker`：实体链接（可用 TagMe/其他 EL 工具）→ `entities: List[str]`。
   * `KGNeighborFetcher`：Wikidata 一跳邻居（入/出边合并）→ `contexts: Dict[entity, List[entity]]`。
   * `Vectorizer`：词/实体向量初始化（论文用 word2vec100d；实现时也可替换为可学习嵌入）。&#x20;
   * `BatchBuilder`：生成 `Batch = {text_ids, ent_ids, ctx_ids, masks…}`。

2. **Encoder 层**（三条并行支路）

   * `TextEncoder(Transformer-1L)` → $p$。
   * `EntityEncoder(Transformer-1L)` → $q'$。
   * `ContextEncoder(Transformer-1L)` → $r'$。
     （单层 Transformer、$h=4$、$d_{ff}=2048$，位置编码与残差/层归一等常规结构。）

3. **Knowledge-aware Attention 层**

   * `NEAttention`（N-E）：$Q(p),K(q'),V(q')$ → $\alpha, q$。
   * `NE2CAttention`（N-E2C）：$Q(p),K(q'),V(r')$ → $\beta, r$。

4. **Head 层**

   * `ConcatHead([p,q,r]) → Linear → Softmax`；交叉熵 + L2 正则。

5. **Evaluation/Logging 层**

   * 指标：P/R/F1/Acc/AUC（与论文一致）。
   * 交叉验证与验证集划分策略保持一致。

## C. 参数（Parameters）

* **结构超参**：隐藏维、头数、FFN 维、层数（默认为 1 层以对齐论文）等；
* **训练超参**：学习率、batch、epoch、优化器（Adam）、丢弃率等（论文 dropout=0.5；h=4；$d_{ff}=2048$）。
* **运行时模式**：bf16/fp16、`torch.compile`、DDP/FSDP（你的环境可直接受益）。
* **知识检索策略**：邻居半径（默认 1 跳）、是否过滤关系类型、上下文聚合（论文默认**平均**）。

---

# 项目文件/文档结构

```
/
├─ README.md                        # 快速开始（数据、缓存、训练、评估）
├─ STRUCTURE.md
├─ configs/
│  ├─ INDEX.md                      # 超参数导航
│  ├─ data/
│  │  ├─ README.md
│  │  ├─ common.yaml
│  │  ├─ politifact.yaml
│  │  ├─ gossipcop.yaml
│  │  └─ pheme.yaml
│  ├─ model/
│  │  ├─ README.md
│  │  ├─ transformer_text.yaml      # TextEncoder 超参
│  │  ├─ transformer_entity.yaml    # EntityEncoder 超参
│  │  ├─ transformer_context.yaml   # ContextEncoder 超参
│  │  └─ head.yaml                  # 分类头/正则系数等
│  ├─ fusion/
│  │  ├─ README.md
│  │  ├─ ne.yaml                    # N-E 注意力配置
│  │  └─ ne2c.yaml                  # N-E2C 注意力配置
│  └─ train/
│     ├─ README.md
│     ├─ base.yaml                  # 通用训练参数（bf16/compile/seed）
│     ├─ politifact_5fold.yaml
│     ├─ gossipcop_5fold.yaml
│     └─ pheme_5fold.yaml
├─ kan/
│  ├─ data/
│  │  ├─ README.md
│  │  ├─ loaders.py                 # NewsLoader：统一输出 schema
│  │  ├─ entity_linking.py          # EntityLinker 抽象；TagMe/EL 实现
│  │  ├─ kg_fetcher.py              # KGNeighborFetcher：Wikidata 一跳
│  │  ├─ vectorizer.py              # word2vec/可学习嵌入/缓存
│  │  └─ batcher.py                 # BatchBuilder：三路输入打包
│  ├─ modules/
│  │  ├─ README.md
│  │  ├─ text_encoder.py            # TransformerEncoder(1L) → p
│  │  ├─ entity_encoder.py          # TransformerEncoder(1L) → q'
│  │  ├─ context_encoder.py         # TransformerEncoder(1L) → r'
│  │  ├─ attention/
│  │  │  ├─ README.md
│  │  │  ├─ ne.py                   # Q(p),K(q'),V(q') → q
│  │  │  └─ ne2c.py                 # Q(p),K(q'),V(r') → r
│  │  └─ head.py                    # concat([p,q,r]) → logits
│  ├─ pipelines/
│  │  ├─ README.md
│  │  ├─ prepare_data.py            # 实体链接+一跳邻居抓取+缓存/去重
│  │  ├─ train_trainer.py           # transformers.Trainer 流
│  │  ├─ train_accelerate.py        # accelerate 自定义 loop
│  │  └─ evaluate.py                # 指标与交叉验证驱动
│  ├─ utils/
│  │  ├─ README.md
│  │  ├─ registry.py                # 组件注册/依赖注入
│  │  ├─ logging.py                 # 训练/评测日志
│  │  ├─ metrics.py                 # P/R/F1/Acc/AUC 计算
│  │  └─ seed.py                    # 可复现性工具
│  └─ interfaces/
│     ├─ README.md
│     ├─ linker.py                  # Protocol/ABC：EL 接口
│     ├─ kg.py                      # Protocol/ABC：KG 抽取接口
│     └─ encoders.py                # Protocol/ABC：三类编码器接口
├─ scripts/                         # 调用 prepare_data + train + eval
│  ├─ README.md
│  └─ run.py
├─ cache/                           # 实体与邻居缓存（按数据集/日期分桶）
├─ runs/                            # 日志与权重输出
├─ tests/                           # 单测（batch 形状/注意力 Q,K,V 对齐等）
├─ LICENSES/
│  └─ ...
├─ .gitignore
└─ requirements.txt
```

> 这份结构把**算子与组合**解耦：`modules/attention` 只关心 Q/K/V 的**策略**，`modules/*_encoder.py` 只关心**序列编码**，而**拓扑**在 `pipelines/train_*`；未来把**GNN**作为 `entity/context encoder` 的另一实现即可。

---

# 接口与“可组合性”设计（为 GNN 留好卡槽）

> 我们把**知识编码器**抽成策略接口 `IKnowledgeEncoder`，默认实现是 `TransformerEncoder(1L)`；后续用 `GNNEncoder`（GraphSAGE/GAT/RelGraphConv）**等价替换**，无需改动上层拓扑与注意力。

**关键接口（伪规范）**

* `ILinker.link(text) -> List[EntityID]`
* `IKG.one_hop(entity) -> List[EntityID]`
* `IKnowledgeEncoder.encode(sequence[EntityID]) -> Tensor[L,d]`
* `NEAttention(p, q') -> q`；`NE2CAttention(p, q', r') -> r`
* `IHead.forward(p,q,r) -> logits`

**数据契约（简版）**

```yaml
NewsRecord:
  id: str
  text: str
  label: int   # 0 real / 1 fake
  entities: List[str]         # Wikidata QIDs
  contexts: Dict[str, List[str]>  # QID -> neighbors (one-hop)
```

**注意力不变量**

* N-E：$Q$ 取自 $p$，$K,V$ 来自 $q'$；
* N-E2C：$Q$ 取自 $p$，$K$ 来自 $q'$，$V$ 来自 $r'$；
* 以上确保“文本语义”以 query 的形式**对知识选择性聚焦**（论文定义）。

---

# 运行与再现的“工程卫⻔”

* **可复现性**：固定随机种子、版本锁、缓存实体/邻居与词表快照；日志中记录**EL 与 KG 抽取时间戳**。
* **性能/成本**：环境建议默认 **bf16 + torch.compile**；数据侧使用 `datasets` 的 map 缓存 + `num_proc` 预处理。
* **合规性**：尊重数据与 API 的使用条款；将 TagMe/Wikidata 调用写清频控与重试策略（`DATA.md`）。
* **指标的情景性**：准确率/召回率等**依数据域而变**，不要过度外推（论文三数据集也强调了这一点）。

---

# 展望：与 GNN 的组合式创新如何无痛接入

把 `EntityEncoder` / `ContextEncoder` 的实现换成 GNN 即可：

* **局部子图构造**：以文中出现的实体为锚点，抽取 $k$-跳子图（初期仍然 1 跳与论文对齐），构造邻接/关系类型；
* **GNNEncoder**：GraphSAGE / GAT / R-GCN（关系图）输出顶点序列表征，再与 N-E / N-E2C 保持同维接口；
* **对齐与融合**：保持**Q/K/V 角色不变**，只改变 $q', r'$ 的“产生方式”，从而验证“图结构诱导”是否改进注意力的筛选效果。
* **文档**：新增 `GNN_DESIGN.md`，描述子图采样、关系建模与对比实验（Transformer vs GNN）设计。

---

# 小结（结论与下一步）

**结论**：

* 我们把 KAN 的论文结构拆成可复现的**模块化架构**：数据（EL+一跳邻居）→ 三路 Transformer 编码 → N-E 与 N-E2C 融合 → 分类头；关键细节（单层编码器、h=4、dff=2048、消融/数据与指标设置）均与论文对齐，以保证结果可比性。  &#x20;
* 架构抽象了**知识编码器的策略接口**，为后续把 GNN 替换/并联接入留好了“乐高卡槽”，不会破坏上层拓扑与注意力逻辑。
