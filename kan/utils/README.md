# `kan.utils` 使用说明（KAN Utilities）

> 版本：v0.1 • 日期：2025‑09‑16 • 适用平台：Windows/Linux/Mac（Windows 友好）
> 依赖：Python ≥3.10（推荐 3.11）、NumPy、scikit‑learn、PyTorch（可选）

本目录提供 **复现性（reproducibility）**、**指标计算（metrics）**、**注册/构建（registry/factory）** 三类通用工具，作为 KAN 项目的“工程底座”。它们遵循统一的日志规范（`kan.utils.*`）与 Doxygen 风格的中英双语注释，强调 **稳定接口、易扩展、可观测**。

---

## 目录

* [设计目标](#设计目标)
* [模块一：metrics](#模块一metrics)

  * [核心 API](#核心-api)
  * [使用示例](#使用示例)
  * [边界条件与回退](#边界条件与回退)
  * [扩展位](#扩展位)
* [模块二：registry](#模块二registry)

  * [核心 API](#核心-api-1)
  * [命名空间与契约](#命名空间与契约)
  * [使用示例](#使用示例-1)
  * [最佳实践](#最佳实践)
* [模块三：seed](#模块三seed)

  * [核心 API](#核心-api-2)
  * [训练/评估接线](#训练评估接线)
  * [注意事项](#注意事项)
* [日志与可观测性](#日志与可观测性)
* [版本与依赖](#版本与依赖)
* [常见问题（FAQ）](#常见问题faq)
* [变更记录](#变更记录)

---

## 设计目标

1. **稳定接口**：公共函数/类签名在小版本内保持兼容；关键返回字段稳定。
2. **可观测性**：所有关键路径均记录到 `kan.utils.*` 命名空间，出现退化（如 AUC 不可定义）时**告警而非崩溃**。
3. **Windows 友好**：无 fork 依赖；路径、编码、线程模型全部过 Windows 心智模型。
4. **最小依赖**：仅用标准库 + `numpy` + `scikit‑learn`；`torch` 是可选增强。
5. **可扩展**：面向配置（config‑driven）、命名空间 Registry、可插拔组件、易于插件化。

---

## 模块一：`metrics`

文件：`kan/utils/metrics.py`

为二分类/多分类/多标签提供统一指标计算：**Precision/Recall/F1/Accuracy**，若提供 `y_score`（概率或 logits），则额外计算 **ROC‑AUC（OvR/OvO）** 与 **PR‑AUC（micro/macro）**；
支持 **Top‑k accuracy**（多分类且提供概率时）；附带 **FoldAccumulator** 做折次均值/标准差聚合。

### 核心 API

* `infer_task_type(y_true) -> {'binary','multiclass','multilabel'}`
* `compute_classification_metrics(y_true, y_pred=None, y_score=None, *, labels=None, average_for_pr=('micro','macro'), average_for_roc=('ovr','ovo'), topk=(1,3), zero_division=0) -> Dict[str,float]`

  * 返回键**稳定**（缺失用 `NaN` 占位）：`accuracy`、`precision/recall/f1_{macro|micro|weighted}`、`roc_auc_{ovr|ovo}`、`pr_auc_{micro|macro}`、以及按需 `top{k}_accuracy`。
  * 若 `y_pred is None` 且给了 `y_score`：

    * 二/多分类：自动以 `argmax`（或正类阈值 0.5）得到 `y_pred`；
    * 多标签：对 `y_score>=0.5` 做逐列阈值。
* `safe_confusion_matrix(y_true, y_pred, *, labels=None, normalize=None) -> np.ndarray`

  * 当某折缺少类别时，自动以“观测到的标签并集”回退，避免抛异常。
* `FoldAccumulator`

  * `add(metrics: Mapping[str,float])`、`as_table() -> Dict[str,List[float]]`、`summary(prefix_mean='mean_', prefix_std='std_') -> Dict[str,float]`

### 使用示例

```python
from kan.utils.metrics import compute_classification_metrics, FoldAccumulator

m = compute_classification_metrics(y_true, y_pred=pred, y_score=proba)
print(m['f1_macro'], m['roc_auc_ovr'])

acc = FoldAccumulator()
for fold in range(5):
    acc.add(compute_classification_metrics(ys[fold], yp[fold], yscores[fold]))
print(acc.summary())  # {mean_f1_macro: ..., std_f1_macro: ..., ...}
```

### 边界条件与回退

* **单一类别**（本折全为 0 或全为 1）：AUC/PR‑AUC 不可定义 → 返回 `NaN` 并写入 `kan.utils.metrics` 警告日志。
* **Top‑k**：仅当 `y_score.shape==[N,C]` 的多分类才计算；遇到异常（标签未对齐等）→ `NaN`。
* **阈值**：默认 `0.5`，如需动态阈值请在外部处理后将二值 `y_pred` 传入。

### 扩展位

* 分组指标（按文本长度/实体数分桶）；
* 校准度量（Brier Score、ECE）；
* 曲线绘制（ROC/PR plot，建议单独 `plots.py`）。

---

## 模块二：`registry`

文件：`kan/utils/registry.py`

提供**强类型 + 命名空间**注册表（`Registry[T]`）与中心枢纽（`RegistryHub`），用于以统一契约构建 KAN 的各模块（text/entity/context encoder、head、attention、loss、optimizer、scheduler、dataset、tokenizer）。

### 核心 API

* `Registry(name, *, case_insensitive=True, allow_override=False)`

  * `register(key=None, *, alias=None, override=None, metadata=None)` 装饰器
  * `add(key, obj, *, alias=None, override=None, metadata=None)` / `remove` / `clear`
  * `get(key)` / `get_entry(key)` / `help(key)`（返回 docstring 第一行）
  * `build(spec_or_name, **overrides)`：从字符串或 `{type/name: ...}` 构建（若注册对象是可调用则调用之，否则直接返回对象）
  * `load('module:attr', register_as=None, alias=None, override=None)`：按点路径导入并注册
  * `freeze()` / `unfreeze()`：训练期防写保护
* `RegistryHub(case_insensitive=True, allow_override=False)`

  * `get(namespace)` / `get_or_create(namespace)` / `namespaces()`
* `build_from_config(cfg, registry, *, type_key='type', name_key='name', **overrides)`
* 预置全局 `HUB`：已经建立常用命名空间（`text_encoder/entity_encoder/context_encoder/head/attention/loss/optimizer/scheduler/dataset/tokenizer`）

### 命名空间与契约

* 统一在接口层约定：**一切组件都从对应命名空间的 Registry 构建**：

  ```python
  from kan.utils.registry import HUB, build_from_config
  TEXT = HUB.get('text_encoder')

  @TEXT.register('bert', alias=['BERT'])
  class BertTextEncoder(...):
      """BERT-based text encoder."""

  te = build_from_config(cfg['text_encoder'], TEXT)
  ```

* YAML/JSON 契约：允许 `type` 或 `name` 字段二选一；其余键作为构造参数。

### 使用示例

```python
from kan.utils.registry import Registry, RegistryHub, HUB

MODELS = Registry('model')

@MODELS.register('mlp', alias=['MLP'])
class MLP: ...

m = MODELS.build({'type': 'mlp', 'hidden': 128})

encoders = HUB.get_or_create('encoder')
encoders.add('bag', object)
```

### 最佳实践

* **命名**：键小写、别名可含驼峰；避免过度多别名；保证语义清晰。
* **覆盖策略**：默认不允许覆盖；确需覆盖时显式 `override=True`，并写入 `metadata` 记录原因/时间。
* **冻结阶段**：训练/评估环节调用 `freeze()`，避免动态注册破坏可复现性。
* **帮助信息**：确保类/函数的 docstring 第一行精炼描述，便于 `help(key)` 快速检索。

---

## 模块三：`seed`

文件：`kan/utils/seed.py`

提供全局与局部的随机性控制：一次性配置 Python/NumPy/（可选）PyTorch；可选开启确定性内核；配套 DataLoader worker 种子函数；支持 RNG 状态快照/恢复与上下文管理器。

### 核心 API

* `set_seed(seed=None, *, deterministic=True, warn_only=True, set_env_pythonhashseed=True) -> Dict`

  * 统一 `random.seed` / `numpy.seed` / `torch.manual_seed` / `torch.cuda.manual_seed_all`
  * 确定性开关：`torch.use_deterministic_algorithms(True, warn_only=...)`、`cudnn.deterministic=True`、`cudnn.benchmark=False`、`CUBLAS_WORKSPACE_CONFIG=':16:8'`
  * 写入 `PYTHONHASHSEED`（仅影响**之后**创建的子进程）
* `seed_worker(worker_id)`：适配 `DataLoader(..., worker_init_fn=seed_worker, generator=...)`
* `with_seed(seed, *, deterministic=False)`：上下文管理器，进入时设种/退出时完整恢复 RNG 状态
* `rng_state() / restore_rng_state(state)`：捕获/恢复 Python/NumPy/Torch（CPU & CUDA）
* `derive_seed(base_seed, *names) -> int`：用 SHA‑256 将 `(base, names...)` 映射到 32bit 子种子

### 训练/评估接线

```python
# 入口第一行
from kan.utils.seed import set_seed, seed_worker
seed_info = set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)

# DataLoader
import torch
from torch.utils.data import DataLoader

g = torch.Generator().manual_seed(cfg.train.seed)
loader = DataLoader(ds, num_workers=cfg.train.workers, worker_init_fn=seed_worker, generator=g)
```

### 注意事项

* 开启确定性可能 **降低性能** 或 **限制算子选择**；请在实验记录中注明。
* 设置 `PYTHONHASHSEED` **不会回溯影响**当前进程，仅影响随后创建的子进程；如需严格控制，请在启动脚本级别设置该环境变量。

---

## 日志与可观测性

* 统一使用模块级命名空间：

  * `kan.utils.metrics` / `kan.utils.registry` / `kan.utils.seed`
* 建议在入口处调用 `kan.utils.logging`（集中式配置）或 `logging.basicConfig(...)`，在 DEBUG 级别下可观测：注册/构建、覆盖、AUC 退化、worker 种子等关键事件。

---

## 版本与依赖

* Python：≥3.10（推荐 3.11）
* NumPy：与 `scikit‑learn` 兼容的版本（详见项目根 `requirements.txt`）
* scikit‑learn：用于指标计算
* PyTorch（可选）：用于设种与 CUDA 确定性；若不可用仍可正常运行 metrics/registry

---

## 常见问题（FAQ）

**Q1：为什么有时 `roc_auc_*` 或 `pr_auc_*` 为 NaN？**
A1：该折仅包含单一真实标签（如全为 0 或全为 1），AUC 不可定义。我们返回 NaN 并写告警日志，建议用 `FoldAccumulator.summary()` 的 `nanmean` 聚合。

**Q2：Top‑k accuracy 没有出现？**
A2：仅在多分类且提供 `y_score.shape == [N,C]` 时计算。若想自定义 `k`，在 `compute_classification_metrics(..., topk=(1,5,10))` 指定。

**Q3：如何支持自定义组件构建？**
A3：将组件注册到对应命名空间（如 `HUB.get('head').register('my_head')`），并在配置中写 `head: { type: my_head, ... }`。

**Q4：如何保证论文复现实验完全可复现？**
A4：入口第一行 `set_seed(...)`，训练时冻结注册表 `Registry.freeze()`，使用 `FoldAccumulator` 做统计，记录系统环境（CUDA/Driver/cuDNN）与 `deterministic` 标志。

---

## 变更记录

* **v0.1 (2025‑09‑16)**：初版发布，提供 `metrics/registry/seed` 三大模块，统一日志与契约；预置 `HUB` 命名空间。
