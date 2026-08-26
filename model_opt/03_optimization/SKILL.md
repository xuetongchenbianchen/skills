---
name: npu-optimization-implementation
description: 优化实施：用去重/复用/掩盖/替换四维度框架实施性能优化。当用户需要实施优化方案、融合算子、预分配 buffer、编译工具（TorchScript/torch.compile）、flat forward、或换等价实现时触发。
---

# NPU 优化实施

## 运行前统一原则：NPU 资源检查

**每次需要运行代码（benchmark / profiling / 精度验证 / 功能测试等）之前，必须先用 `npu-smi info` 检查 NPU 上是否有与本任务无关的进程。若存在，先确认这些进程与当前任务无关（必要时向用户确认归属），再清理进程、释放显存，确认资源干净后才开始运行。** 无关进程会争抢算力/显存，导致性能数据失真或 OOM。

```bash
npu-smi info        # 检查各卡上的进程占用
ps -fp <PID>        # 确认进程身份与归属
kill <PID>          # 确认无关后清理（顽固进程用 kill -9）
npu-smi info        # 复查确认显存/算力已释放
```

## 定位

本阶段承接 Profiling 分析（02）的结论，将定位到的瓶颈点转化为具体的优化方案并实施。

核心流程：Profiling 定位瓶颈 → 基于四维度选择优化手段 → 方案经用户确认后实施 → 验证精度和性能 → 用户确认后提交。

## 优化四维度

所有性能优化手段本质上只做四件事：

| 维度 | 核心问题 | 典型现象 |
|------|---------|---------|
| **去重** | "这个工作是必要的吗？能和相邻工作合并吗？" | 同类算子调用次数异常多；存在可合并的独立调用 |
| **复用** | "这个结果/资源之后还会被需要吗？" | 相同尺寸 tensor 反复分配释放；同一计算结果被重复计算 |
| **掩盖** | "这段延迟能和其他工作并行吗？" | 通信和计算串行排列；计算流中有可填充的空泡 |
| **替换** | "同样的结果有没有硬件更便宜的等价写法？" | 某算子落 AI_CPU；一组拆解算子有官方融合算子；某 API 有 NPU 更友好的等价表达 |

前三者改变工作量/工作方式，第四者改变同一工作的物理执行路径——它们正交，可组合。

每个维度的详细原理、具体手段和代码模式见对应 reference (**必读**)：

| 维度 | Reference | 核心内容 |
|------|-----------|---------|
| 去重 | [eliminate_redundancy.md](references/eliminate_redundancy.md) | 合并调用、消除冗余、清理框架开销 |
| 复用 | [reuse_and_precompute.md](references/reuse_and_precompute.md) | 预计算缓存、预分配 buffer、原地操作 |
| 掩盖 | [hide_latency.md](references/hide_latency.md) | 通信-计算重叠、双 buffer 流水 |
| 替换 | [equivalent_substitution.md](references/equivalent_substitution.md) | NPU 融合算子、换等价 API、换算法 |

## 场景专用 Reference

以下文件按场景条件加载，不强制读取（npu_checklist 除外）：

| Reference | 加载条件 | 核心内容 |
|-----------|---------|---------|
| [npu_checklist.md](references/npu_checklist.md) | 每轮优化开始前必读 | NPU 已知性能陷阱的 grep 扫描清单 |
| [npu_operator_catalog.yaml](references/npu_operator_catalog.yaml) | 替换维度层 1 时加载 | 融合算子目录（被 equivalent_substitution.md 引用） |
| [compilation_tools.md](references/compilation_tools.md) | host-bound 时 | TorchScript/jit.trace/torch.compile(npu)/NPU JIT 的选择决策树、兼容性排查、编译粒度决策 |

> 多卡切分方案设计不在本子技能（属 [07_parallel_splitting](../07_parallel_splitting/SKILL.md)，Phase 0 分诊触发的条件轨道）；已多卡场景的通信优化按四维度框架走——通信-计算重叠见 [hide_latency.md](references/hide_latency.md)（掩盖），瓶颈类型判定见 [profiling_to_action.md](../02_bottleneck_analysis/references/profiling_to_action.md) 归因层 #8。

## 通用原则

- 每次优化后重新 Profiling，确认瓶颈是否转移
- GPU 最优实践在 NPU 可能反效果，必须实测验证
- 保留原始实现供 fallback
- 权重修改须保持 checkpoint 可加载且数值等价（原则见主 SKILL「核心原则 · checkpoint 兼容」，操作细节见 [reuse_and_precompute.md](references/reuse_and_precompute.md)「Checkpoint 兼容性」）
- 优化尝试失败也要记录（what + why + 实际效果），避免重复踩坑
- **四维度逻辑正交 + NPU 硬件耦合**：四个维度（去重/复用/掩盖/替换）在逻辑层面正交——它们各自回答不同的优化问题（见上表）。但在 NPU 上，维度间通过**内存分配模式**和**异步流水线**（`TASK_QUEUE_ENABLE=2`）产生硬件耦合：任何改变操作数量或操作顺序的优化（去重/替换），都可能改变 NPU 异步流水线的重叠模式，导致预期外的性能回退。
  - "一个方向的逻辑失败不影响其他方向"——逻辑层面仍然成立，不要因一个替换方案失败就放弃独立的去重方案
  - "但任何改变操作序列的优化必须用 L0 端到端 benchmark 验证"——不能只看 profiling 中的算子级数据，因为异步流水线的重叠效果只在端到端时间中体现
  - 若优化导致端到端回退但 profiling 显示算子级改善，根因是异步流水线耦合——记录此发现，可考虑通过自适应阈值（如不同输入规模用不同实现）规避
- **深度优先于广度**：对每个优化方向，穷尽探索（多种实现、完整验证）比浅尝多个方向更有价值。如果环境支持子 agent，建议对独立的方向/算子 spawn 子 agent 逐个深挖，避免因同时处理太多方向而浅尝辄止
- **编译工具优先尝试（host-bound 场景优先）**：Phase 2 分析瓶颈时同步用 npu_checklist 扫描；Phase 3 先解决其发现的 D2H 同步、AI CPU 回退等编译无法覆盖的结构性问题，然后尝试编译工具（TorchScript / torch.jit.trace / torch.compile(npu)），因为编译自动消除大量框架级开销（Python 解释器、Module.__call__、属性查找），不需要手动 inline 或预提取——host-bound 时收益最大，适用性判定见 [compilation_tools.md](references/compilation_tools.md)。编译后重新 profiling，数据更准确，再用四维度解决编译无法覆盖的剩余问题。如果编译失败，用四维度出发解决兼容性问题后重试。

## 方向放弃标准（分级）

放弃一个优化方向前，须满足该方向所属级别的全部条件：

> 本分级（Level 1/2/3）是**难度分级**（决定放弃代价），与 04 的**验证分级**（决定验证强度）是两套体系——替换级（本体系 Level 2）改动在 Phase 3 逐条验证时须过 [equivalence_verification.md](references/equivalence_verification.md) 协议，而非仅 04 的 Level 1 快速验证。

**Level 1 — 微调级**（改参数 / flag / 跳过单步操作 / 1-3 行代码改动）
- A/B benchmark 对比（优化前 vs 优化后，同一输入，≥3 次取中位数）
- 记录：改动描述、耗时对比、慢/无效的原因
- 如回退，一句话说明原因即可（如"异步流水线耦合导致回退 +0.8ms"）

**Level 2 — 替换级**（换等价实现 / 融合算子 / 改数据流路径）
- 至少 1 种实现（框架提供或自定义均可）
- 精度验证：按 [equivalence_verification.md](references/equivalence_verification.md) 协议执行（代表性样本，确认数值等价）
- A/B benchmark 对比
- 记录：实现描述、精度结果、耗时对比、失败原因归类（概念错误 / 框架 overhead / 硬件不友好 / 异步流水线耦合）
- 如仅有框架实现且失败，须尝试 1 种自定义实现后才可放弃

**Level 3 — 架构级**（重写算子 / 改变计算图结构 / 需要 checkpoint 重映射）
- 至少 2 种实现（1 框架 + 1 自定义/bare）
- 每种实现经过微基准 → 小样本 → 全量三步验证
- 每步记录具体数值（微基准 diff、小样本精度、全量耗时）
- 失败原因归类（概念错误 / 框架实现 overhead / 硬件不友好 / 其他）+ 是否尝试过调整参数重试（如换 dtype、换 shape、换输入顺序）
- 若失败原因为"框架实现 overhead"：自定义实现是强制要求——只有自定义实现也失败后才能放弃

**通用规则**（所有级别）：
- "我觉得不会通过"（未实测）始终为不合理放弃原因
- "diff 看起来有点大"（未量化）始终为不合理放弃原因
- 优化方向逻辑正交——一个方向的失败不连带放弃独立方向（见通用原则"四维度逻辑正交 + NPU 硬件耦合"）
