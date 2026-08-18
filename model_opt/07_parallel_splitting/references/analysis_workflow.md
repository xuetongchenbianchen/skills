# 分析阶段工作流

> 本文档覆盖 Phase 0 分诊 + 分析阶段 Step 1-5 + JSON 输出规范。SKILL.md 仅保留流程概览。

---

## Phase 0：OOM 根因分诊

在进入并行切分之前，**必须先判断显存不足的根因**。大量"需要并行"的诉求源于配置不当，纠正后单卡即可运行。

### 分诊决策树

```
用户报告 OOM / 显存不足
├─ Step 1: 外部因素排查 → 发现 → 修复，不触发并行
├─ Step 2: 显存预算估算 → 理论峰值 < HBM × 0.8 → 仍有外部因素
└─ Step 3: 理论峰值 > HBM × 0.8 → 本质需要并行 → 进入分析阶段
```

> 简单 DP（多独立样本）不属于"模型并行"，用 [parallel_design.md](../../03_optimization/references/parallel_design.md) 即可，不进入本子技能。

### 外部因素排查清单（按频率排序）

| # | 检查项 | 命中表现 | 修复 |
|---|--------|---------|------|
| 1 | 注意力融合未开启 | attention score 矩阵实体化 | 启用融合注意力算子 |
| 2 | 内存碎片 | `max_memory_allocated` 远大于 `memory_allocated` | 调整分配策略 + tcmalloc |
| 3 | 序列/Batch 配置不当 | padding 浪费 / 迭代步数过多 | 调整配置 |
| 4 | 全量张量未释放 | 通信缓冲区持续存活，抵消切分收益 | 及时 `del`，复用缓冲区 |
| 5 | 混合精度未启用 | FP32 每张量翻倍 | 切 BF16 / 量化 |
| 6 | 环境配置 | 设备数不匹配 / 框架版本不配套 | 修正环境变量 |

### 显存预算估算

估算方法：
1. 识别模型中的**主导张量**——随输入规模超线性增长的张量（复杂度类型见第一步）
2. 沿 forward 路径估算各模块的峰值显存（主导张量 + 常驻部分如参数、优化器状态）
3. 判定：若估算峰值 > HBM × 0.8（排除外部因素后）→ 本质需要并行

> 0.8 阈值留出碎片与临时开销余量，可根据实际框架行为调整。

---

## 第一步：提取模型参数与模块链路

读源码，提取：
1. **模型参数**：总参数量、参数显存、框架
2. **模块链路**：每模块输入/输出 shape、主导张量、计算类型、内存复杂度
3. **子模块拆解**：复杂模块列出子模块及通信 pattern

> 关键关注点：识别主导张量的复杂度类型——O(N²) attention/pair、O(N³) 高阶交互、O(gridᵈ) 空间场、O(E×d) 边特征等。

## 第二步：模块级拆解 → 内存时间线 → 定性分类

### 2.1 内存时间线构建

沿 forward 路径逐模块追踪张量生命周期：
1. 记录每模块创建的中间张量 shape 和生命周期（模块内释放 or 跨模块存活）
2. 计算每模块"同时存活张量总大小"的峰值（串行 N block 峰值 = 单 block 峰值 + 常驻）
3. **递归检查子模块**——隐藏的超线性消费者
4. 标注阶段边界处的内存变化

### 2.2 定性分类

| 分类 | 特征 | 候选策略 |
|------|------|---------|
| elementwise | 无超线性张量 | 不切分 |
| memory_bound | 峰值 > HBM × 0.8 | SP / 空间分解 |
| compute_bound | 计算密集 | SP / DP |
| compute_bound (serial) | 多步串行无超线性 | DP |
| compute_bound (serial, superlinear input) | 多步串行接收超线性输入 | SP + DP |

### 2.3 张量分布约定表

切分后所有相关张量必须在切分维对称切。**选定策略后必须建立此表**，逐张量声明：

| 张量 | 全局 shape | 分布方式 | 每卡 shape | 消费操作 | 恢复通信 |
|------|-----------|---------|-----------|---------|---------|

> 未声明张量视为潜在 bug。

### 2.4 策略组合层级

多策略叠加按通信代价分层：**DP(外) → EP/PP(中) → SP/TP(内)**。跨节点用 DP，节点内用模型并行。

---

## 第三步：定量估算

Agent 直接计算：
1. **单卡峰值**：内存时间线最大模块峰值 + 常驻
2. **切分后每卡**：dominant_peak / world_size + 常驻，< HBM × 0.8?
3. **SP 通信量**：per_block × num_blocks × iterations（per_block = 2 × dominant_tensor_size）
4. **通信耗时**：total_comm / bandwidth
5. **PP 气泡率**：`(P-1)/(P+micro_batches-1)`（推理 micro_batch=1 时简化为 `(P-1)/P`，极大，通常不推荐）

通信开销速查：

| 操作 | 数据量 | 频率 |
|------|--------|------|
| AllToAll (SP) | (P-1)/P × tensor_size | 每 block |
| AllReduce (TP) | (P-1)/P × tensor_size | 每线性层 |
| Gather (DP) | output_size | 仅 1 次 |

---

## 第四步：方案审查 → 输出 JSON

**必须输出 JSON 文件**，从三角度论证：数学正确性 + 完备性 + 系统可行性。

文件路径：`parallel_decisions/analysis_{model}_{seq_len}_{world_size}p.json`

### ★ 确认节点

JSON 输出后用 `ask_user_question` 确认实施（展示策略对比、推荐配置、风险应对）。审查未通过 → 回第三步。

---

## （可选）第五步：Profiling 数据校准

若估算不确定：单卡跑最小输入实测各模块耗时 + 通信 benchmark 实测带宽 → 更新 JSON。

---

## SP 切分分析方法

将超线性主导张量沿高阶维度切分到多卡，AllToAll 转换视角，本地计算后聚合。

**切分维度确定原则**：
- 识别操作中对结果贡献独立的维度（可沿此维切分而不破坏计算等价性）
- 切分后每卡执行本地计算，通过 broadcast/AllToAll 补全所需的全量张量
- 最终 concat/gather 得到完整输出

**通用数学模式**：对共享索引 k 的 `Y[i,j] = Σ_k a[i,k]·b[j,k]`，切 N 后每卡持有 `a_local[N/P,k]`，需 b 全量（broadcast/AllToAll），本地计算 `Y_local[N/P,N]`，concat → 完整 `[N,N]`。各类超线性操作均可归结为此模式的实例。

**约束检查**：切分维整除 P（否则 padding）| 切后峰值 < HBM×0.8 | 通信不抵消计算节省 | 张量分布约定表一致

**实施原语**：`scatter_along_dim`、`allgather_along_dim`、`row_to_col`/`col_to_row`（AllToAll 维度转换，自动 `.contiguous()`）。`world_size=1` 时 no-op。

---

## PP 分析方法

将层划分为多 stage，数据在 stage 间流水传递。

**气泡分析**：`bubble = (P-1)/(P+micro_batches-1)`。推理 micro_batch=1 时气泡率极高。

**PP vs SP 决策**：
- 选 PP：层深大 + 多 micro_batch + 层间清晰边界
- 选 SP：层内有超线性主导张量 + 推理场景 + 需降单层显存

> 推理（micro_batch=1）下 SP 几乎总优于 PP（气泡太大，PP 不解决层内 OOM）。

---

## EP 分析方法

MoE 专家分布到不同卡，token 经 gate 路由后 AllToAll 分发。通信量与 token 数 × 嵌入维度 × MoE 层数成正比。需评估 capacity_factor 和负载均衡。

---

## JSON 输出规范

### 公理与不变量（4+2，审查通过标准）

| 编号 | 名称 | 定义 |
|------|------|------|
| 公理 1 | 必要性 | 理论峰值 > HBM × 0.8，外部因素已排除 |
| 公理 2 | 可切分性 | 每个切分操作有等价性证明，通信 ≥ 下界 |
| 公理 3 | 完备性 | 张量分布表覆盖计算图所有大张量 |
| 公理 4 | 正确性 | 切分前后输出一致（验证后 true，之前 pending） |
| 不变量 I | 显存缩放 | 切后每卡 < HBM × 0.8 |
| 不变量 II | 通信下界 | ratio ≈ 1.0 无冗余；ratio < 1.0 必有隐藏错误 |

### 顶层结构

| 字段 | 回答 | 对应公理 |
|------|------|---------|
| `model_characterization` | 模型张量图 | — |
| `split_scheme` | 切分方式 + tensor_distribution_table | 公理 3 |
| `proof` | per_operation_proofs + completeness | 公理 2, 4 |
| `quantitative` | 显存/通信/break-even | 不变量 I, II |
| `system_feasibility` | 硬件兼容 + known_pitfalls | — |
| `verification_plan` | Tier + 容差 + 指标 | 公理 4 |
| `risks` | 风险 + 应对 | — |
| `axiom_compliance` | 6 条逐一对照 | 全部 |

### 关键设计点

- **tensor_distribution_table**：必须与 §2.3 张量分布约定表一致，覆盖所有大张量
- **per_operation_proofs**：每操作独立证明 + lower_bound vs actual（ratio < 1.0 = 隐藏错误）
- **axiom_compliance**：审查时一眼可见，axiom_4 验证前为 `pending`

### 审查流程

1. `axiom_compliance` 6 条全 true（axiom_4 可 pending）
2. `proof.per_operation_proofs` 每操作 `lower_bound_met=true`
3. `tensor_distribution_table` 覆盖所有大张量
4. `system_feasibility.known_pitfalls` 每条有 mitigation
5. `verification_plan` Tier 合理、acceptance 明确

全过 → 实施。未过 → 回第三步。

### 非 2 的幂卡数处理

优先将 2 的幂部分分配给 TP（HCCL 优化好），剩余分配给 SP。序列不整除时 padding 到可整除，计算后裁剪。
