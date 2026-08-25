# 下界分析：还有没有优化空间、还有多少

> 跨阶段分析模块。Phase 2 用它判断优化空间，★C 用它判断是否终局。

## 目标

回答两个问题：
1. **还有没有优化空间？**
2. **还有多少？**

不回答"空间在哪一层、怎么缩小"——那是 Line B profiling 分析的工作。

## 可用数据

| 数据 | 来源 | 可靠性 | 含义 |
|------|------|--------|------|
| wall-clock | benchmark（无 profiler） | 最高 | 端到端真实时间 |
| L0 Computing | L0 step_trace | 高 | **纯 kernel 执行时间之和**（不含 kernel 间 gap） |
| L0 Free | L0 step_trace | 高 | **device 空闲时间** = kernel 间 gap 总和 |

**已验证**：L0 Computing = sum of kernel durations（差值为 0）；L0 Total（Computing + Free）/ wall-clock = 1.00-1.03（profiler 开销 0-3%）。因此 wall-clock ≈ L0 Computing + L0 Free，L0 Free ≈ wall-clock − L0 Computing = device 空闲时间。

## 下界模型

```
wall-clock ≈ L0 Computing + L0 Free

优化空间 = L0 Free + L0 Computing 中可压缩的部分
不可压缩下界 = 必要 kernel 的最小执行时间之和（难以精确计算，见下文）
```

- **L0 Free**（device 空闲时间）：优化空间的主要来源。可通过减少 dispatch 次数（去重/融合）、编译工具消除。
- **L0 Computing**（kernel 执行时间）：不完全不可压缩。去重消除冗余 kernel直接减少 Computing；融合算子减少中间结果读写可能降低总执行时间；等价替换改变 kernel 组成可能增减 Computing。但每个 kernel 的实际计算量（FLOPs）决定的最小执行时间不可压缩——除非换算法或量化。


## 不可压缩下界的估计

"必要 kernel 的最小执行时间之和"难以精确计算——哪些 kernel 是"必要的"取决于模型结构，哪些融合能减少执行时间需要实测。实际操作中通过**跨轮次观察 L0 Computing 的变化趋势**来逼近：

- 如果每轮优化后 L0 Computing 持续下降（去重/融合生效），说明 Computing 仍有压缩空间
- 如果连续 2 轮 L0 Computing 不再下降，说明已到达 kernel 执行时间的下界（剩余 kernel 都是必要的，单个 kernel 的执行时间由 CANN 实现决定）

## 终局判断

满足以下**任一**条件时，优化空间接近极限：

1. **L0 Free < wall-clock × 10%**：device 几乎不停，优化空间在 device 侧（减少 kernel 数量或换算法），Python 层优化接近极限
2. **L0 Free > 0 但连续 2 轮优化均 < 2% wall-clock 改进**：虽然 Free 仍存在但已无法有效消除
3. **所有候选被拒绝且无新候选产生**

不满足以上条件时，说明 L0 Free 仍大且有优化空间。具体空间在哪、怎么缩小，由后续 Line B profiling 分析（trace_view 的 stall 定位、operator_details 的 host 分解）确定。

## 可选：循环场景分解

当推理路径包含显式循环（for/while）时，可进一步分解：

```
wall-clock = one_time_cost + loop_cost

loop_cost = loop_device + loop_free
```

优化空间仍 = L0 Free。分解的价值在于让 agent 看到"每步的空闲时间有多少"——但具体是循环间间隙还是循环内间隙，属于 Line B 分析。

**串行依赖场景**（每步依赖前一步输出）：loop_cost = N_steps × per_step_total。当前下界基于串行假设。

**并行场景**（各步独立无数据依赖）：若 Line A 分析发现步骤可并行，下界应标注"当前下界基于串行假设，若可并行则下界更低"。但并行实现属于"掩盖"维度优化，不在下界分析范围。
