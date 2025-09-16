# Fusion Configs — NE / NE2C

本目录描述 **知识感知注意力融合层（Knowledge-aware Attention）** 的**可配置策略**，用于把三路编码（文本 p、实体序列 q'、实体上下文序列 r'）融合为下游分类头所需的 `[p; q; r]`。所有键均为**策略/行为**开关，不包含与数据集绑定的数值。

## 文件说明

- `ne.yaml`  
  **News → Entities（NE）** 注意力模块配置：以新闻表示为 Query，对实体序列（Key/Value）做加权，得到实体级融合向量 `q`。  
  适用场景：仅使用实体、不引入邻接上下文的消融；或作为 NE2C 的第一级权重参考。

- `ne2c.yaml`  
  **News → Entities → Contexts（NE2C）** 注意力模块配置：以新闻为 Query、实体为 Key、上下文为 Value，得到上下文融合向量 `r`。  
  适用场景：完整 KAN 路径；当需要同时考虑实体“活性”与其邻居贡献时启用。

## 共同契约（必须遵守）

- **隐藏维一致（`d_model`）**：该值需与三路编码器输出维一致（见 `configs/model/*_encoder.yaml`）。  
- **多头一致性**：`n_heads` 需整除 `d_model`。  
- **掩码策略**：`mask_policy=auto` 时，模块会使用上游传入的 `news_mask/entity_mask/contexts_mask`。缺失时按实现约定视为“全有效”，并打印 `WARNING` 日志。  
- **日志与可观测性**：建议在训练入口调用统一日志配置（`kan.utils.logging.configure_logging`），并在 `DEBUG` 级别记录张量形状与 mask 密度，便于排查。

## 组合与调用

- **标准拓扑**：  
  `NE(p, q') → q`；`NE2C(p, q', r') → r`；随后 `Head([p, q, r]) → logits`。  
- **消融实验**：  
  仅用文本：`use_q=false, use_r=false`；仅用实体：启用 `NE`，关闭 `NE2C`；仅用上下文：启用 `NE2C`，并在 Head 关闭 `use_q`。  
- **实现后端**：  
  `impl: sdpa` 使用 PyTorch 原生 Scaled-Dot-Product Attention；必要时可切换为点积实现以便自定义行为。

## 调参指引（不绑定数值）

- 当编码器维度较大（如 BERT-base），保持 `n_heads` 与 `d_model` 的可整除关系即可；`dropout` 建议与上游编码器保持一致。  
- `return_weights=true` 仅在可解释分析阶段开启，避免显存/带宽开销。  
- `context_pooling` 初期推荐 `attn`；若希望更快实验，可切换为 `mean` 作为近似。

## 兼容性与演进

- 新增键一律给出默认值，保证旧配置可加载。  
- 若未来需要引入门控/稀疏注意力/分块注意力等新策略，建议以新增键实现（例如 `sparsity: topk`、`block_size: 128`），不破坏现有字段含义。
