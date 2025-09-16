# configs/train — 训练配置说明（不含具体参数）

> 适用：Windows + Python 3.11 + PyTorch 2.8（CUDA 12.8）
> 角色定位：将 **数据配置（configs/data/**）**、**模型与融合配置（configs/model/**, configs/fusion/**）\*\* 与 **训练流程** 解耦，统一在本目录下以 YAML 驱动训练与交叉验证。

---

## 文件一览（逻辑角色）

* **base.yaml** — 训练通用基座：

  * **运行时选项**：随机种子、确定性/编译、混合精度（bf16/fp16）、日志配置路径（接入 `kan.utils.logging`）、输出目录结构等。
  * **优化与调度**：优化器类型（Adam 等）、学习率调度器（如余弦退火+热身）、梯度裁剪、梯度累积、最大轮数与早停策略。
  * **评估与保存**：评估触发策略（按 step/epoch）、主监控指标与模型保存策略（top‑k/last/best）。
  * **交叉验证策略**：固定 **10% 验证集**（用于调参）+ **对剩余样本做 K 折（默认 5 折）** 的管控开关与随机性约束。

* **politifact\_5fold.yaml** — 针对 *PolitiFact* 的训练入口：

  * 绑定 `configs/data/politifact.yaml`，并指向 `configs/model/*` 与 `configs/fusion/*` 的具体实现文件。
  * 复用 `base.yaml` 的通用设置，仅覆盖与该数据集规模/分布相关的训练细节（如批大小、训练轮数、评估频率）。
  * 保持论文设定的一致性（单层 Transformer 编码器；两级知识注意力 N‑E / N‑E2C；评价指标含 P/R/F1/Acc/AUC）。

* **gossipcop\_5fold.yaml** — 针对 *GossipCop* 的训练入口：

  * 绑定 `configs/data/gossipcop.yaml`；在数据规模更大的场景下，通常调整批大小、评估节奏与学习率调度的总步数计算方式。
  * 其他结构与 *PolitiFact* 同构，确保脚本在不同数据集之间可平移。

* **pheme\_5fold.yaml** — 针对 *PHEME* 的训练入口：

  * 绑定 `configs/data/pheme.yaml`；建议显式打开**分层（stratified）K 折**以缓解类不平衡。
  * 按论文保留 10% 固定验证集用于调参，同时在剩余样本上做 5 折交叉评估并聚合指标。

> 以上三个 *dataset\_5fold.yaml* 文件均仅承担“**把数据/模型/融合/头** 接线到 **训练流程**”的职责；**不重复**模型或融合层超参（这些在 `configs/model/*.yaml`、`configs/fusion/*.yaml` 中定义）。

---

## 训练脚本如何消费这些 YAML？

* **pipelines**：本项目提供 `kan/pipelines/train_accelerate.py` 与 `kan/pipelines/train_trainer.py` 两条训练路径（任选其一），二者都读取本目录下的训练 YAML：

  1. 装载 `data_config`（数据集与切分策略）→ 准备 `NewsRecord` 与批处理（详见 `kan.data`）；
  2. 装载 `model_config` 与 `fusion_config`（文本/实体/上下文编码器 + N‑E / N‑E2C）→ 构建模型；
  3. 依据 `train` → `optimizer/scheduler/early_stopping` 等字段启动训练循环；
  4. 评估阶段调用 `kan.utils.metrics` 计算 **Accuracy / Precision / Recall / F1 / AUC** 并写入日志与结果文件。

* **日志**：通过 `run.log_cfg` 接入集中式日志（`kan/utils/logging.py`），或留空使用内置默认（控制台 + 轮转文件）。

---

## 推荐的执行方式（示意）

```powershell
# Windows PowerShell 示例（Accelerate 路径）
python -m kan.pipelines.train_accelerate --config configs/train/politifact_5fold.yaml
python -m kan.pipelines.train_accelerate --config configs/train/gossipcop_5fold.yaml
python -m kan.pipelines.train_accelerate --config configs/train/pheme_5fold.yaml
```

---

## 设计原则回顾（与论文对齐）

* **编码器**：单层 Transformer 编码文本/实体/上下文；FFN 宽度与多头数在 `configs/model/*.yaml` 中设置。
* **知识注意力**：N‑E 与 N‑E2C 两级注意力用于融合实体与其上下文；分类头拼接 `[p;q;r]` 后做二分类。
* **评估**：固定 10% 验证集用于调参；剩余样本做 5 折交叉验证并汇总 P/R/F1/Acc/AUC。

> 这些与论文的实验设定保持一致；与训练流程相关但不属于模型结构的参数（优化器、调度、早停等）在 `base.yaml` 中集中配置，便于统一变更与复现。

---

## 小贴士（工程实践）

* 优先使用 **bf16** 混合精度与 `torch.compile`（你的 CUDA 12.8 + RTX 30 系列环境能直接收益）；
* 在大数据集（如 *GossipCop*）上适当增大批大小与梯度累积；
* 交叉验证汇总时请使用 `kan.utils.metrics.FoldAccumulator` 做均值/标准差聚合，以应对 AUC 在单折不可定义的退化情况；
* 统一通过环境变量 `KAN_LOG_CFG` 指向外部日志 YAML 时，请确保文件在仓库相对路径可见。
