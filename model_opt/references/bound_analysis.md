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

优化空间 = L0 Free + L0 Computing 中可压缩的部分（见「下界模型」）。终局判断必须**分别对两侧给出耗尽证据**——只看 Free 侧会把"host 层到头"误判为"整体到头"。典型误判：L0 Free < 10% 时恰恰应转向 Computing 侧的 kernel 融合/去重/替换路线，而非停止优化。

判定**整体终局**须同时满足 A（Free 侧耗尽）与 B（Computing 侧耗尽）；C 为兜底。

### A. Free 侧（host/调度层）耗尽——满足任一即可
1. **L0 Free < wall-clock × 10%**：device 几乎不停，host 层优化接近极限。注意：这只宣告 Free 侧到头，**不构成整体终局**，须继续判定 B
2. **L0 Free > 0 但连续 2 轮优化均 < 2% wall-clock 改进**：Free 残留但无法有效消除

### B. Computing 侧（kernel 层）耗尽——须全部满足
Computing 的可压缩性来自三类手段：碎片算子融合（多 kernel → 1，含逐元素链/归约链的融合）、冗余消除（去重）、等价替换（更快的 kernel/算法/量化）。

1. **必要 kernel 主导**：Computing 主体由不可压缩 kernel 构成——典型如权重 GEMM（带宽/算力已饱和：cube 利用率高、mte 接近带宽上限、单 kernel 执行时间由 FLOPs 与 CANN 实现决定），且量化/换算法未获用户授权或已明确排除
2. **碎片算子库存清零**：对 op_statistic 中全部非必要 kernel（elementwise/norm/transpose/cast 及跨算子碎片链）逐条盘点，每条要么已实施，要么有**显式拒绝依据**留档——例如：融合改变 fp16 round 序列且经深层累积验证不可接受、无对应融合算子、剩余合计占比 < wall-clock × 10%
3. **跨轮趋势停滞**：连续 2 轮 L0 Computing 不再下降（融合/去重不再生效，见「不可压缩下界的估计」）

### C. 通用兜底（单独满足即可终局）
- **所有候选被拒绝且无新候选产生**

### 使用注意
- 判据数据必须来自 L0 与 wall-clock——L1 的 Utilization 受 profiler 伪影影响（barrier 注入），不可用于终局判断
- 终局判断前必须穷尽融合算子库存：**官方/框架融合算子**（`[x for x in dir(torch_npu) if 'npu_' in x.lower()]` + 昇腾官方文档）与**自定义算子**（AscendC，可精确复刻原 round 序列或内部升精度，是现成融合算子精度不达标时的最后手段）是两级库存——未评估前不计入"穷尽"，不能仅看 utilization 数字下结论
- 量化是 Computing 侧最大杠杆，但受用户授权门禁约束：未获授权时应标注"排除量化后无空间"，而非"无空间"
- B-2 的占比阈值（< wall-clock × 10%）可按项目精度要求调整，但阈值须在判断前显式声明

不满足整体终局时，剩余空间按侧别指引下一步分析：Free 侧大 → trace_view 的 stall 定位、operator_details 的 host 分解；Computing 侧 → op_statistic 的碎片算子聚类 + Line A 四维度的融合候选评估。

## 可选：循环场景分解

当推理路径包含显式循环（for/while）时，可进一步分解：

```
wall-clock = one_time_cost + loop_cost

loop_cost = loop_device + loop_free
```

优化空间仍 = L0 Free + L0 Computing 中可压缩的部分（同「下界模型」）。分解的价值在于让 agent 看到"每步的空闲时间有多少"——但具体是循环间间隙还是循环内间隙，属于 Line B 分析。

**串行依赖场景**（每步依赖前一步输出）：loop_cost = N_steps × per_step_total。当前下界基于串行假设。

**并行场景**（各步独立无数据依赖）：若 Line A 分析发现步骤可并行，下界应标注"当前下界基于串行假设，若可并行则下界更低"。但并行实现属于"掩盖"维度优化，不在下界分析范围。
