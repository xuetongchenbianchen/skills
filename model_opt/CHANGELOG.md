# 更新说明 — model_opt

## 概述

本次更新为 `model_opt` 技能新增了 **多卡推理并行切分子技能 `07_parallel_splitting`**，将原本仅覆盖单卡性能优化的四维度流程，扩展到显存容量受限场景下的多卡切分能力。`07_parallel_splitting` 是 model_opt 的**条件触发专业轨道**（非顺序 Phase），当瓶颈是显存容量而非计算效率时，从主流程 Phase 2 的 Line C 分支进入，完成后回归 Phase 4 门禁与 Phase 5 工程化提交。

同时伴随少量 references 调整：新增 `03_optimization/references/comm_optimization.md`、`decode_optimization.md`，并将 `equivalence_verification.md` 归入 `04_accuracy_assurance/references/`。

本说明围绕 `07_parallel_splitting` 的新特性展开。

---

## 新增子技能：07_parallel_splitting

### 定位与触发

- **条件触发，非顺序阶段**：仅在显存容量瓶颈时进入；计算效率瓶颈仍走主流程四维度优化。
- **双重触发路径**：
  1. Phase 2 Line C 自动触发 — profiling 的 `parse_operator_memory.py` 报告「消除 waste 后投影峰值仍 > 80% HBM」，经 OOM 分诊确认为「本质需要并行」后进入全流程。
  2. 用户直接触发 — 报告显存不足 / 需要多卡 / 要求模型并行时直接进入。
- **简单并行 vs 全流程**：仅需 DP（数据并行）等简单策略时不进入本子技能，用 `03_optimization/references/parallel_design.md` 快速参考即可；需要 SP/TP/PP/EP 等复杂切分时升级到全流程。
- **Handhand 协议**：Phase 2 Line C 分诊 → 确认本质需要并行 → 分析+实施替代 Phase 3 → 验证回归 Phase 4 门禁 → Phase 5 工程化提交（含 ADR 归档）。回退路径：`disable_parallel()` 回到单卡，回到四维度优化。

### 核心原则

- **先分诊再切分**：外部因素（配置不当、NPU 资源管理不善等）先修复，不切分；本质需要并行才触发切分流程。
- **禁止修改模型架构**：切分只改变计算的分布方式，不改变模型的数学语义和结构。
- **三角度论证**：切分方案必须从数学正确性、完备性、系统可行性三个角度论证，并输出 JSON 分析文件。

---

## 全流程（7 步 + 归档）

```
Phase 0  OOM 根因分诊 → 外部因素修复★终止 / 本质需要 → 进入分析
   ↓
分析阶段
   第一步  提取模型参数与模块链路
   第二步  模块级拆解 → 内存变化分析 → 定性分类 → 候选并行模式
   第三步  定量估算
   第四步  方案审查 → 输出 JSON（数学/完备性/系统）  ★ 确认
   （可选）第五步  Profiling 数据校准
   ↓
实施阶段
   第六步  通信 infra + 实施模式 + 冒烟测试
   ↓
验证阶段
   第七步  正确性验证：端到端测试，切分前后输出一致性   ★ 确认提交
   ↓
归档（ADR）
```

含完整错误回退路径：每个确认/验证节点失败均有明确的回退目标（回第三步 / 回第四步 / 回分诊）。

---

## 关键新特性

### 1. Phase 0 OOM 根因分诊机制

进入切分前必须判断显存不足的根因，避免对「伪 OOM」盲目切分。

- **外部因素排查清单**（6 项，按频率排序）：注意力融合未开启、内存碎片、序列/Batch 配置不当、全量张量未释放、混合精度未启用、环境配置。
- **显存预算估算**：识别随输入规模超线性增长的「主导张量」，沿 forward 路径估算各模块峰值，以 `HBM × 0.8` 为阈值判定是否本质需要并行。

### 2. 四步分析方法 + JSON 输出规范

- **第一步**：提取模型参数与模块链路，识别主导张量的复杂度类型（O(N²) attention/pair、O(N³) 高阶交互、O(gridᵈ) 空间场、O(E×d) 边特征等）。
- **第二步**：构建内存时间线（逐模块追踪张量生命周期 + 递归检查子模块）→ 定性分类（elementwise / memory_bound / compute_bound 等 5 类）→ 建立**张量分布约定表**（逐张量声明分布方式，未声明视为潜在 bug）。
- **第三步**：定量估算单卡峰值、切分后每卡显存、SP 通信量、通信耗时、PP 气泡率，并给出通信开销速查表。
- **第四步**：方案审查并输出 JSON，从三角度论证。JSON 审查标准为 **4 公理 + 2 不变量**：
  - 公理：必要性、可切分性、完备性、正确性
  - 不变量：显存缩放（切后每卡 < HBM×0.8）、通信下界（ratio < 1.0 必有隐藏错误）
- **（可选）第五步**：估算不确定时，单卡实测各模块耗时 + 通信 benchmark 实测带宽，校准 JSON。

### 3. 三种并行策略分析框架

提供 SP / PP / EP / DP 的系统化分析方法与决策依据：

- **SP（序列并行）**：将超线性主导张量沿高阶维度切分到多卡，AllToAll 转换视角，本地计算后聚合。给出**通用数学模式**——对共享索引 k 的 `Y[i,j] = Σ_k a[i,k]·b[j,k]`，各类超线性操作均可归结为此模式的实例。
- **PP（流水线并行）**：气泡分析 `bubble = (P-1)/(P+micro_batches-1)`；推理 micro_batch=1 时气泡率极高，通常不推荐。
- **EP（专家并行）**：MoE 专家分布 + gate 路由 + AllToAll 分发，需评估 capacity_factor 与负载均衡。
- **策略组合层级**：按通信代价分层 `DP(外) → EP/PP(中) → SP/TP(内)`。
- **PP vs SP 决策**：推理场景（micro_batch=1）下 SP 几乎总优于 PP。

### 4. 实施四模式 + 通信原语模板库

四种实施模式按侵入度从低到高，按项目情况选择：

| 模式 | 侵入度 | 适用策略 | 回退 | 单卡兼容 |
|------|--------|---------|------|---------|
| A 外部编排 | 最低 | DP | 换脚本 | 天然 |
| B Monkey-Patch | 中 | SP/TP/EP | disable | 天然 |
| C 子类覆写 | 中高 | 复杂组合 | 换类 | 天然 |
| D 源码内嵌 | 高 | 深度优化 | git revert | 需显式保证 |

配套**通信原语模板库**（从模板裁剪适配，所有原语 world_size=1 时 no-op）：

- `ParallelConfig` 全局配置单例 + `init_distributed`（自动 NPU/CUDA/CPU 后端切换，含 `import torch_npu` 前置处理）。
- 基础原语：`allgather_along_dim`（dim=0 用 `all_gather_into_tensor` 单 buffer 减半峰值）、`scatter_along_dim`、`allreduce_inplace`、`reduce_scatter_along_dim`（通信量减半）、`broadcast_from_rank0`、`gather_to_rank0`。
- AllToAll 维度转换：`row_to_col` / `col_to_row` / `alltoall_swap_dims`（chunk 后自动 `.contiguous()`）。
- 通信-计算重叠（NPU 专用）：`issue_on_comm_stream` / `wait_comm`，独立通信流 + Event。
- Monkey-Patch 生命周期：`register_patch` / `unregister_patch` / `enable_parallel` / `disable_parallel` / `restore_parallel` / `set_parallel_flag` / `cleanup_all_patches`。
- 混合并行 `ProcessGroupManager`：`init_groups(tp, dp, pp)`，含 dp_ranks offset 修正。

### 5. 昇腾 NPU 关键差异适配

提供 NPU vs GPU 差异对照表（后端 `hccl`/`nccl`、设备字符串、AllToAll 必须 `.contiguous()`、通信流、ReduceScatter、碎片优化 `PYTORCH_NPU_ALLOC_CONF` 等）+ 启动配置要点（CANN 环境脚本、`ASCEND_RT_VISIBLE_DEVICES`、跨节点 `HCCL_CONNECT_TIMEOUT=600`、多卡 DP 三套 RNG 同步）。

### 6. 权重处理双方案

- **在线权重切分（推荐）**：运行时各 rank 读同一份完整 checkpoint，按 rank 自动加载对应切片；给出 TP/EP weight_loader 实现要点与合并层特殊处理。
- **离线权重转换（备选）**：预切权重用于离线部署，输出 `rank_0/` ~ `rank_N/` 目录结构。

### 7. 分层正确性验证（Tier）

- **核心原则**：tolerance 是通过门禁，bit-exact 仅 debug 工具。
- **三层验证**：Tier 1 冒烟（最小输入跑通 + shape 断言）→ Tier 2a（最小输入单卡 vs 多卡 tolerance）→ Tier 2b（真实规模多配置 tolerance，最终通过标准）。
- **策略精度风险与验收标准表**：DP/SP fp32/PP 用 `atol=1e-6`；SP bf16/TP/EP 用 `atol=1e-3`。
- 验证脚本模板支持 `.npy`/`.pkl`/`.pt` 输出加载、tolerance/bit-exact/领域指标（Kabsch RMSD、cosine、spatial RMSE+ACC 等）三种对比方式。

### 8. 通信原语自检机制

`self_test()` 在搭建通信 infra 后第一步必跑（`torchrun --nproc_per_node=N`），验证 6 项原语（allgather / scatter+allgather 往返 / row_to_col+col_to_row 往返 / allreduce / broadcast / gather_to_rank0），快速排查高频坑（未 import torch_npu、contiguous 等）。

### 9. 18 条常见坑 + 通用排查流程

按频率排序的 18 条常见坑，覆盖：

- 基础（#1-10）：集合通信死锁、Shape 不匹配、AllToAll contiguous、非确定性结果、HCCL 初始化失败、并行比单卡慢、切分后仍 OOM、并行开关误触发、跨迭代 shape 变化、负索引维度歧义。
- 多节点专项（#11-13）：连接超时、拓扑发现失败、参数不一致。
- 回滚与清理专项（#14-16）：Patch 残留、分布式状态残留、通信缓冲区泄漏。
- 混合并行专项（#17-18）：ProcessGroup dp_ranks offset 错误、ReduceScatter dim≠0 需 permute。

配套通用排查流程（self_test → 冒烟 → py-spy dump → 打印 shape → 单卡 vs 多卡 → profiling 看通信占比）。

### 10. ADR 决策归档模板

提供标准化 ADR 模板，记录场景约束、Phase 0 分诊、内存时间线、定量估算、JSON 审查、实施摘要、验证结果、回退方案、未采纳方案及原因，纳入 Phase 5 工程化提交。

### 11. 两个用户确认节点

- **★ JSON 审查**（第四步后）：展示数学正确性 + 完备性 + 系统可行性 + 策略对比 + 推荐配置 + 风险应对，用 `ask_user_question` 确认实施。
- **★ 提交审核**（第七步后）：展示验证结果 + 性能收益 + ADR 路径，用 `ask_user_question` 确认提交。

### 12. 非 2 的幂卡数处理

优先将 2 的幂部分分配给 TP（HCCL 优化好），剩余分配给 SP；序列不整除时 padding 到可整除，计算后裁剪。

---

## 产出产物

| 产物 | 阶段 | 路径 | 用途 |
|------|------|------|------|
| OOM 分诊结论 | Phase 0 | stdout | 判定是否需要并行 |
| 内存时间线表 | 第二步 | stdout / JSON | 逐模块峰值定位瓶颈 |
| 分析 JSON | 第四步 | `parallel_decisions/analysis_{model}_{seq_len}_{world_size}p.json` | 切分方案论证 |
| 通信原语模块 | 第六步 | `comm/comm_primitives.py` | 通信 infra |
| 并行推理脚本 | 第六步 | `run_parallel.py` | 多卡推理入口 |
| 验证报告 | 第七步 | `verify_report.json` | 正确性验证结果 |
| ADR 决策记录 | 归档 | `parallel_decisions/ADR-NNNN-*.md` | 决策归档 |

---

## 文件清单

本次更新在 `model_opt` 下新增/调整的文件：

- 新增 `07_parallel_splitting/SKILL.md` — 子技能入口与流程概览
- 新增 `07_parallel_splitting/references/analysis_workflow.md` — Phase 0 分诊 + 分析 Step 1-5 + JSON 规范
- 新增 `07_parallel_splitting/references/implementation_guide.md` — 4 种实施模式 + 通信原语模板 + NPU 差异 + 验证模板 + ADR + 18 条坑
- 新增 `03_optimization/references/comm_optimization.md` — 通信性能优化（并行 infra 搭建后通信成瓶颈时参考）
- 新增 `03_optimization/references/decode_optimization.md` — decode 阶段优化参考
- 调整 `equivalence_verification.md` — 从 `03_optimization/references/` 归入 `04_accuracy_assurance/references/`
- 更新 `SKILL.md`（根）— 新增「多卡切分集成」章节与 07 子技能索引、Phase 2 Line C 触发说明

---

## 使用方式

- 当 model_opt 主流程在 Phase 2 触发 Line C（OOM 分诊）且确认为本质需要并行时，自动进入 `07_parallel_splitting` 全流程。
- 用户直接报告显存不足 / 需要多卡 / 要求模型并行时，亦可直接触发。
- 进入子技能后按流程阶段按需加载对应参考文档，无需一次性全部加载。
