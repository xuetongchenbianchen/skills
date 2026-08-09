# 下界分析重新设计

## 1. 目标

回答两个问题：
1. **还有没有优化空间？**
2. **还有多少？**

不回答"空间在哪一层、怎么缩小"——那是 Line B profiling 分析的工作。

## 2. 可用数据与可靠性

| 数据 | 来源 | 可靠性 | 含义 |
|------|------|--------|------|
| wall-clock | benchmark（无 profiler） | 最高 | 端到端真实时间 |
| L0 Computing | L0 step_trace | 高 | **纯 kernel 执行时间之和** |
| L0 Free | L0 step_trace | 高 | **device 空闲时间** |
| L0 kernel count | L0 kernel_details | 高 | kernel 总数 |

**已验证的 evidence**：
- L0 Computing = sum of kernel durations（差值 = 0ms，确认 Computing 只含 kernel 执行，不含 kernel 间 gap）
- L0 Total（Computing + Free）/ wall-clock = 1.00-1.03（profiler 开销 0-3%，确认 L0 Total ≈ wall-clock）

由此推出：**L0 Free ≈ wall-clock − L0 Computing = device 空闲时间**。这是可靠的。

## 3. 下界模型

### 3.1 核心公式

```
wall-clock ≈ L0 Computing + L0 Free
```

- **L0 Free**（device 空闲时间）：优化空间的主要来源。可通过减少 dispatch 次数（去重/融合）、编译工具消除。
- **L0 Computing**（kernel 执行时间）：不完全不可压缩。去重消除冗余 kernel 直接减少 Computing；融合算子减少中间结果读写可能降低总执行时间；等价替换改变 kernel 组成可能增减 Computing。但每个 kernel 的实际计算量（FLOPs）决定的最小执行时间不可压缩——除非换算法或量化。

```
优化空间 = L0 Free + L0 Computing 中可压缩的部分
不可压缩下界 = 必要 kernel 的最小执行时间之和（难以精确计算，见 3.2）
```

不依赖 Roofline。直接从可观测数据构建。

Roofline 降级为可选参考量：对大 tensor 场景可作为 kernel 执行时间的理论下界参考；对小 tensor 场景不适用（瓶颈是 dispatch overhead 而非计算）。

### 3.2 不可压缩下界的估计

"必要 kernel 的最小执行时间之和"难以精确计算——哪些 kernel 是"必要的"取决于模型结构，哪些融合能减少执行时间需要实测。实际操作中通过**跨轮次观察 L0 Computing 的变化趋势**来逼近：

- 如果每轮优化后 L0 Computing 持续下降（去重/融合生效），说明 Computing 仍有压缩空间
- 如果连续 2 轮 L0 Computing 不再下降，说明已到达 kernel 执行时间的下界（剩余 kernel 都是必要的，单个 kernel 的执行时间由 CANN 实现决定）

### 3.2 终局判断

回答"还有没有空间"：

1. **L0 Free < wall-clock × 10%**：device 几乎不停，优化空间在 device 侧（减少 kernel 数量或换算法），Python 层优化接近极限
2. **L0 Free > 0 但连续 2 轮优化均 < 2% wall-clock 改进**：虽然 Free 仍存在但已无法有效消除
3. **所有候选被拒绝且无新候选产生**

不满足以上条件时，说明 L0 Free 仍大且有优化空间。具体空间在哪、怎么缩小，由后续 Line B profiling 分析（trace_view 的 stall 定位、operator_details 的 host 分解）确定。

### 3.3 可选：循环场景分解

当推理路径包含显式循环（for/while）时，可进一步分解：

```
wall-clock = one_time_cost + loop_cost

loop_cost = (L0 Computing - one_time_device) + (L0 Free - one_time_free)
          = loop_device + loop_free
```

**串行依赖场景**（每步依赖前一步输出）：

```
loop_cost = N_steps × per_step_total
per_step_total = per_step_kernel + per_step_free
```

优化空间仍 = L0 Free。分解的价值在于让 agent 看到"每步的空闲时间有多少"——但具体是循环间间隙还是循环内间隙，属于 Line B 分析。

**并行场景**（各步独立无数据依赖）：

当前下界基于串行假设（N_steps × per_step），若各步可并行则理论下界更低（max(per_step) 而非 sum）。但实际 NPU 单 compute stream 下并行需要多 stream 重构，这属于"掩盖"维度的优化方案设计，不是下界分析的范围。若 Line A 分析发现步骤可并行，下界分析应标注"当前下界基于串行假设，若可并行则下界更低"。

## 4. 与当前设计的对比

| 当前 | 新设计 |
|------|--------|
| 三档下界（Roofline / L0 Computing / wall-clock） | 两档（L0 Computing / wall-clock），Roofline 降级为可选参考 |
| gap A = L0 Computing − Roofline（归因为 kernel 效率，断言 Python 不可优化） | 不需要 gap A。L0 Computing 是不可压缩下界，减少 kernel 数量可缩小 Computing 但那是优化手段不是下界分解 |
| gap B = wall-clock − L0 Computing（归因为 host 开销） | L0 Free = wall-clock − L0 Computing = 优化空间。不做进一步因果分解（留给 Line B） |
| gap B / Tier3 > 15% 判方向 | 不在下界分析中判方向。只判断"Free 大有空间"或"Free 小接近极限" |
| wall_clock / L0_Computing < 1.1 判终局 | L0 Free < wall-clock × 10% 判终局 |

**核心变化**：
1. 删除 gap A/B 二分法和 Roofline 作为计算依据——简化为"L0 Computing 不可压缩，L0 Free 是优化空间"
2. 不在下界分析中做 Free 的因果分解或优化方向判定——留给 Line B
3. 终局判断用绝对值（L0 Free < 10%）而非比例（避免优化中比例反向）
4. 保留可选串行分解，但只用于"每步空闲多少"，不展开方向判定
