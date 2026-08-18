# 通信瓶颈优化

> **定位**：多卡推理/训练场景下，并行 infra 已搭建（由 07_parallel_splitting 或手动实施），但通信成为性能瓶颈时的诊断与优化指南。本文档属于 Phase 3 优化实施，前提是 Phase 2 已通过 profiling 确认通信是关键路径瓶颈。

## 触发条件

满足**任一**即进入本文档：
1. `parse_step_trace` 的 comm 列占总 step 时间 > 20%
2. 设备利用率 < 50% 且 comm 列有值（通信阻塞设备）
3. 多卡 wall-clock 对比单卡加速比远低于理论（如 8 卡 < 4x）
4. Phase 2 归因层 #8 确认为通信效率问题

## 诊断流程

```
Step 1  通信全景：哪些通信、占多少时间
Step 2  分类归因：每条通信属于哪类低效
Step 3  定位源码：低效通信对应哪行代码
Step 4  制定方案：按优化模式选手段
```

### Step 1：通信全景

从 profiling 提取通信操作清单：

| 数据源 | 提取内容 |
|--------|---------|
| `kernel_details.csv` | 按 `Task Type` 过滤通信类 kernel（HCCL*），统计各原语耗时 |
| `step_trace` / `parse_step_trace` | comm 列总耗时、comm/compute ratio |
| `operator_details.csv` | 通信算子的 Call Stack → 源码位置 |

产出：**通信热点表**（原语类型 × 调用次数 × 单次耗时 × 总耗时 × 源码位置）

### Step 2：分类归因

对每条通信热点，判定属于哪类低效：

| 类型 | 识别信号 | 典型场景 |
|------|---------|---------|
| 等待型 | Wait 阶段 >> Transfer 阶段；或某 rank 通信耗时远大于其他 rank | 负载不均（SP 切分不对称）、barrier 过频 |
| 冗余型 | 同一张量在短时间内被多次 AllGather；或 AllGather 结果立即被 slice 丢弃大部分 | 切分后中间层不必要重建全量、残差连接冗余聚合 |
| 串行型 | 通信 kernel 前后有设备 idle gap（无计算与之重叠） | 未使用异步通信流、未做 prefetch |
| 原语不当型 | AllGather 收集完整张量但只读 1/P（应用 ReduceScatter 或直接本地计算） | AllGather → matmul → 只取本地分片 |
| **策略性通信** | AllReduce/AllGather 的数据量 ≈ O(L²) 或单次耗时 > 100ms；通信源自 einsum/matmul 的收缩维度恰好是分片维度 | SP 下 TriMul Incoming 的 AllReduce；分片维度选择不当导致的大规模 reduce |
| 拓扑错配型 | 跨节点通信（RoCE）耗时占主导，且逻辑上可调整到节点内 | SP/TP group 跨节点、DP group 在节点内（应反过来） |

> **策略性通信的识别要点**：这类通信不是"执行层的低效"（原语选错/没有 overlap），而是"并行策略的直接后果"——分片方式决定了这个通信必须存在。优化手段不是加速通信，而是重新选择该算子的分片策略使通信消失或变为更廉价的形式。

### Step 3：源码定位

用 `operator_details.csv` 的 Call Stack 定位通信调用源码位置。若 Call Stack 断桥，用 Input Shapes + 通信数据量反推：
- AllGather `(P-1)/P × size` → 推断被聚合张量的 shape
- AllToAll 的 input/output split → 推断切分维度

### Step 4：优化模式

按四维度归类通信优化手段：

| 四维度 | 通信优化模式 | 手段 | 预期收益 |
|--------|------------|------|---------|
| **去重** | 消除冗余通信 | 删除不必要的 AllGather/Broadcast、合并重复通信、延迟聚合到最终输出 | 通信次数 ↓ |
| **复用** | 通信结果缓存 | AllGather 结果跨模块复用（而非每模块重新聚合）、通信 buffer 预分配复用 | 通信次数 ↓ + 内存 ↓ |
| **掩盖** | 通信-计算重叠 | 异步流/双 buffer 流水线、通信提前发出（prefetch）、计算与通信分离到不同 stream | 有效通信时间 → 0（理想） |
| **替换** | 换更高效原语/拓扑 | AllGather→ReduceScatter、跨节点→节点内、Ring→Tree 拓扑、通信量化压缩 | 通信量 ↓ 或带宽 ↑ |
| **重构** | 重新设计算子的并行分解 | 改变张量分布方式使收缩维度本地化、用 AllToAll 换布局代替 AllReduce 求和 | **通信消失或降级** |

> **重构 vs 替换的区别**：替换是"同一并行策略下换原语"（如 AllReduce→ReduceScatter）；重构是"改变这个算子的并行策略本身"（如从 reduce-based 改为 layout-change-based）。重构的收益通常是数量级的，但需要回到数学层面理解操作。

---

## 重构：改变算子的并行分解（策略性通信的解法）

> **触发条件**：Step 2 归因为"策略性通信"的热点。这是最高优先级的优化方向——其他模式是"加速通信"，本模式是"消除通信"。

### 方法论：并行策略审计

对每个策略性通信热点，执行以下分析（强制，不可跳过）：

```
1. 追溯数学操作
   "这个 AllReduce/AllGather 对应源码中的哪个 einsum/matmul？"
   "该操作的收缩维度（被求和的维度）是什么？"

2. 分析维度与分片的关系
   "收缩维度当前是分片的还是本地完整的？"
   如果收缩维度是分片的 → 产生 AllReduce（partial sum 需跨卡求和）
   如果输出维度需要全量但当前只有局部 → 产生 AllGather

3. 枚举替代分布
   "如果改变输入张量的分布方式，能否让收缩维度变为本地完整？"
   - 行分片 [L/P, L, c] → 列分片 [L, L/P, c]：AllToAll 转换
   - 通道分片 [L, L, c/P]：不同的通信模式
   - 分块+流水：ring_einsum（每次只 AllGather 一小块，与计算交错）

4. 成本比较
   | 方案 | 通信原语 | 通信量 | 能否被 overlap |
   当前方案 vs 替代方案的通信总成本对比

5. 决策
   如果替代方案通信成本 < 当前方案 50% → 值得重构
   如果替代方案允许 overlap 而当前不允许 → 值得重构
```

### 典型模式

| 当前策略 | 问题 | 重构为 | 收益 |
|---------|------|--------|------|
| einsum 收缩分片维度 → AllReduce | O(L²) reduce | AllToAll 转置 → 复用反方向 einsum → AllToAll 转回 | AllReduce(O(L²)) → 2×AllToAll(O(L²/P)) |
| AllGather 全量 → 大 einsum | 峰值内存 = O(L²) | ring_einsum: 分 P 块 AllGather+计算流水 | 峰值 O(L²/P)，通信被计算掩盖 |
| 两个方向的 TriMul 各自独立通信 | 通信不共享 | Incoming 复用 Outgoing 的逻辑（转置后等价） | 通信次数减半 |

---

## 去重：消除冗余通信

### 模式 1：中间层冗余 AllGather

**症状**：模型主干中间某层 AllGather 回全量 `[N,...]`，下一层又 scatter 回 `[N/P,...]`。

**诊断**：通信热点表中出现成对的 AllGather + slice/scatter，且间距极近（中间无需全量张量的计算）。

**修复**：让下游直接接收本地分片，删除中间的 AllGather。检查下游操作是否支持本地分片输入（参见 07_parallel_splitting 的张量分布约定表）。

### 模式 2：残差连接冗余聚合

**症状**：`pair = pair + op(pair)`，其中 `op` 内部 AllGather 了 pair 做计算再输出 `[N/P,...]`，但外层 `pair` 已是 `[N/P,...]`，AllGather 结果被加完就丢。

**诊断**：AllGather 的输出张量生命周期极短（仅一次加法后立即被覆盖）。

**修复**：改 `op` 内部逻辑为仅 AllGather 需要的分块（如只取对角块），而非全量。

### 模式 3：合并多次小通信

**症状**：多个小张量分别独立 AllGather/AllReduce。

**修复**：将多个小张量 cat 为一个大张量，做一次通信，再 split。减少通信启动开销。

---

## 复用：通信结果缓存

### 模式 4：跨模块 AllGather 复用

**症状**：module_A 和 module_B 都 AllGather 同一张量（如 attention bias）。

**修复**：首次 AllGather 后缓存结果，后续模块直接引用。注意生命周期管理（避免内存泄漏）。

### 模式 5：Buffer 预分配

**症状**：每次通信 malloc 新 buffer（profiling 中看到通信前有 memory allocation）。

**修复**：预分配通信 buffer 并在迭代间复用。`comm_primitives.py` 的通信封装已支持 in-place 操作。

---

## 掩盖：通信-计算重叠

### 模式 6：异步流重叠

详见 [hide_latency.md](hide_latency.md)「通信-计算重叠」节。核心判断：

1. 找到通信操作前后不依赖通信结果的**独立计算**
2. 将通信发到独立的 comm_stream，同时在默认 stream 做独立计算
3. 在需要通信结果时 `wait_stream`

**约束**：同一 communicator 的两个集合操作不能在不同流并发；`all_to_all_single` 不支持 `async_op`。

### 模式 7：双 buffer + Prefetch

适用于循环结构（如 N 个 Pairformer block）：当前 block 计算时，预发下一 block 需要的通信。

```
block_k 计算 + block_{k+1} 通信 预取
   ↓
block_{k+1} 计算（通信结果已就绪）+ block_{k+2} 通信 预取
```

---

## 替换：换更高效方案

### 模式 8：原语降级

| 当前用法 | 低效原因 | 替换为 |
|---------|---------|--------|
| AllGather → 只用 1/P 结果 | 收集了全量但只读本地相关部分 | 直接本地计算或 ReduceScatter |
| AllReduce | 累加后每卡持有完整结果但只用本地 | ReduceScatter（每卡只拿本地段） |
| 多次 Broadcast | 同源多目标 | 一次 AllGather |

### 模式 9：拓扑调整

| 问题 | 修复 |
|------|------|
| SP/TP group 跨节点（通信密集用低带宽链路） | 调整 rank 分配让 SP/TP 在节点内（HCCS），DP 跨节点 |
| 单节点内 DP group 占据所有卡 | 改为 SP(节点内) × DP(跨节点) 混合 |

rank 分配约定：`rank = dp_rank × tp_degree + tp_rank`（DP 外层跨节点，SP/TP 内层节点内）。

### 模式 10：通信量压缩

最后手段（精度有风险）：
- 通信前将张量从 fp32 → bf16/fp16，接收后恢复（通信量减半）
- 仅传差量（delta）而非全量

---

## 与 07_parallel_splitting 的回退判断

| 信号 | 判定 | 动作 |
|------|------|------|
| 通信占比 > 50% 且已无重叠空间 | 策略本身有问题（非实现问题） | 回退到 07_parallel_splitting 重新分析策略（如 SP→DP、调整 world_size） |
| 冗余通信来源是切分方案不完备 | 张量分布约定表有遗漏 | 回退到 07_parallel_splitting 第四步更新 JSON |
| 拓扑错配但 rank 分配写死在源码 | 并行 infra 设计问题 | 回退到 07_parallel_splitting 第六步重新实施 |
| 通信占比 < 30% 可通过 overlap/去重解决 | 实现层问题 | 本文档内解决，不回退 |

---

## 验证

通信优化后必须验证：
1. **正确性**：切分前后输出一致（用 `correctness_verify.py --mode tolerance`）
2. **性能**：wall-clock 改善 + comm/compute ratio 下降
3. **无回退**：通信优化不应导致计算路径变化（guard：优化前后 kernel 序列一致）
