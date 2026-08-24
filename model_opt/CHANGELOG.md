# 更新说明 — model_opt

## 概述

本次更新为 `model_opt` 技能新增了 **多卡推理并行切分子技能 `07_parallel_splitting`**，将原本仅覆盖单卡性能优化的四维度流程，扩展到显存容量受限场景下的多卡切分能力。`07_parallel_splitting` 是 model_opt 的**条件触发专业轨道**（非顺序 Phase），当瓶颈是显存容量而非计算效率时，从主流程 Phase 2 的 Line C 分支进入，完成后回归 Phase 4 门禁与 Phase 5 工程化提交。

同时伴随少量 references 调整：新增 `decode_optimization.md`，并将 `equivalence_verification.md` 归入 `04_accuracy_assurance/references/`。通信瓶颈诊断已并入 `02_bottleneck_analysis/references/profiling_to_action.md` 归因层 #8。

本说明围绕 `07_parallel_splitting` 的新特性展开。

---

## 新增子技能：07_parallel_splitting

### 定位与触发

- **条件触发，非顺序阶段**：仅在显存容量瓶颈时进入；计算效率瓶颈仍走主流程四维度优化。
- **双重触发路径**：
  1. Phase 2 Line C 自动触发 — profiling 的 `parse_operator_memory.py` 报告「消除 waste 后投影峰值仍 > 80% HBM」，经 OOM 分诊确认为「本质需要并行」后进入全流程。
  2. 用户直接触发 — 报告显存不足 / 需要多卡 / 要求模型并行时直接进入。
- **简单并行 vs 全流程**：仅需 DP（数据并行）等简单策略时不进入本子技能，用 `03_optimization/references/parallel_design.md` 快速参考即可；需要 SP/TP/PP/EP 等复杂切分时升级到全流程。
- **Handoff 协议**：Phase 2 Line C 分诊 → 确认本质需要并行 → 分析+实施+验证（支线，不替代 Phase 3） → 第八步：profiling 脚本并行化适配 → 回归主流程 Phase 3 → Phase 4 门禁 → Phase 5 工程化提交（evidence_db 纳入）。回退路径：`disable_parallel()` 回到单卡，回到四维度优化。

### 核心原则

- **先分诊再切分**：外部因素（配置不当、NPU 资源管理不善等）先修复，不切分；本质需要并行才触发切分流程。
- **禁止修改模型架构**：切分只改变计算的分布方式，不改变模型的数学语义和结构。
- **三角度论证**：切分方案必须从数学正确性、完备性、系统可行性三个角度论证，分析结果记录到 evidence_db（`parallel_splitting` 字段）。

---

## 全流程（8 步）

```
Phase 0  OOM 根因分诊 → 外部因素修复★终止 / 本质需要 → 进入分析
   ↓
分析阶段
   第一步  提取模型参数与模块链路
   第二步  模块级拆解 → 内存变化分析 → 定性分类 → 候选并行模式
   第三步  定量估算
   第四步  记录分析到 evidence_db  ★ 确认
   （可选）第五步  Profiling 数据校准
   ↓
实施阶段
   第六步  通信 infra + 实施模式 + 冒烟测试
   ↓
验证阶段
   第七步  正确性验证：端到端测试，切分前后输出一致性   ★ 确认提交
   ↓
回归前置
   第八步  Phase 1 profiling 脚本并行化适配
```

含完整错误回退路径：每个确认/验证节点失败均有明确的回退目标（回第三步 / 回第四步 / 回分诊）。

---

## 关键新特性

### 1. Phase 0 OOM 根因分诊机制

进入切分前必须判断显存不足的根因，避免对「伪 OOM」盲目切分。

- **外部因素排查清单**（6 项，按频率排序）：注意力融合未开启、内存碎片、序列/Batch 配置不当、全量张量未释放、混合精度未启用、环境配置。
- **显存预算估算**：识别随输入规模超线性增长的「主导张量」，沿 forward 路径估算各模块峰值，以 `HBM × 0.8` 为阈值判定是否本质需要并行。

### 2. 四步分析方法

- **第一步**：提取模型参数与模块链路，识别主导张量的增长阶（超线性 O(N²)+ / 线性 O(N×d)）。
- **第二步**：构建内存时间线（逐模块追踪张量生命周期 + 递归检查子模块）→ 定性分类（三问决策：切什么 → 沿哪切 → 能否叠加 DP）→ 建立**张量分布约定表**（逐张量声明分布方式，未声明视为潜在 bug）。
- **第三步**：定量估算单卡峰值、切分后每卡峰值、通信量（按切分维度区分）、通信耗时、break-even。
- **第四步**：记录分析到 evidence_db（`parallel_splitting` 字段），agent 自查关键检查项（显存阈值、通信下界、张量覆盖、pitfall mitigation）后 → ★ 确认节点。
- **（可选）第五步**：估算不确定时，单卡实测各模块耗时 + 通信 benchmark 实测带宽，校准 evidence_db 记录。

### 3. 切分维度分析框架

按**（切分位置, 切分维度）**组织分析，不预设策略名词：

- **输入处切分**（横切/竖切）：横切 batch 维 = DP；竖切 seq/空间维 = SP，需 AllToAll 交换分片。给出**通用数学模式**——对共享索引 k 的 `Y[i,j] = Σ_k a[i,k]·b[j,k]`，各类超线性操作均可归结为此模式的实例。
- **中间模块切分**：模块内切权重 = TP，AllReduce 合并；模块间切层 = PP，气泡率 `(P-1)/(P+micro_batches-1)`，推理场景通常不推荐。
- **特殊结构切分**：MoE 专家维 = EP，AllToAll 分发 + 负载均衡。其他特殊结构按结构特点分析。

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

### 9. 调试排查

问题按类型分类归属：环境/部署问题归 Phase 1，性能问题归 Phase 3，切分方法问题在本阶段处理。保留排查流程（self_test → 冒烟 → py-spy dump → 打印 shape → tolerance 对比）和 3 条切分方法常见问题（通信死锁、shape 不匹配、并行开关误触发）。

### 10. 两个用户确认节点

- **★ 方案确认**（第四步后）：展示切分方案 + 定量估算 + 等价性证明 + 风险应对，用 `ask_user_question` 确认实施。
- **★ 提交审核**（第七步后）：展示验证结果 + 性能收益，用 `ask_user_question` 确认提交。

### 11. 非 2 的幂卡数处理

优先将 2 的幂部分分配给 TP（HCCL 优化好），剩余分配给 SP；序列不整除时 padding 到可整除，计算后裁剪。

---

## 产出产物

| 产物 | 阶段 | 路径 | 用途 |
|------|------|------|------|
| OOM 分诊结论 | Phase 0 | stdout | 判定是否需要并行 |
| 内存时间线表 | 第二步 | stdout / evidence_db | 逐模块峰值定位瓶颈 |
| 分析记录 | 第四步 | `evidence_db/<id>.yaml`（`parallel_splitting` 字段） | 切分方案论证 |
| 通信原语模块 | 第六步 | `comm/comm_primitives.py`（按模板实现） | 通信 infra |
| 并行推理脚本 | 第六步 | `run_parallel.py` | 多卡推理入口（验证用） |
| 并行 profiling 脚本 | 第八步 | Phase 1 脚本的并行版本（同路径覆写或 `_parallel` 后缀） | 回归主线后 Phase 3/4 采集 L0/L1/wall-clock |
| 验证报告 | 第七步 | `verify_report.json` | 正确性验证结果 |

---

## 文件清单

本次更新在 `model_opt` 下新增/调整的文件：

- 新增 `07_parallel_splitting/SKILL.md` — 子技能入口与流程概览
- 新增 `07_parallel_splitting/references/analysis_workflow.md` — Phase 0 分诊 + 分析 Step 1-5 + 切分维度分析
- 新增 `07_parallel_splitting/references/implementation_guide.md` — 4 种实施模式 + 通信原语引用 + NPU 差异 + 验证方法 + 调试排查
- 新增 `03_optimization/references/decode_optimization.md` — decode 阶段优化参考
- 调整 `equivalence_verification.md` — 从 `03_optimization/references/` 归入 `04_accuracy_assurance/references/`
- 更新 `SKILL.md`（根）— 新增「多卡切分集成」章节与 07 子技能索引、Phase 2 Line C 触发说明

---

## 使用方式

- 当 model_opt 主流程在 Phase 2 触发 Line C（OOM 分诊）且确认为本质需要并行时，自动进入 `07_parallel_splitting` 全流程。
- 用户直接报告显存不足 / 需要多卡 / 要求模型并行时，亦可直接触发。
- 进入子技能后按流程阶段按需加载对应参考文档，无需一次性全部加载。
