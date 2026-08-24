# 多卡分布式训练 Profiling 数据分析指南

基于五阶段分层分析方法：从"全局粗筛"到"微观根因"，帮助在海量多卡 profiling 数据中
快速定位慢卡、识别瓶颈类型（I/O/CPU/计算/通信），并给出具体优化方向。

**核心原则**：多卡训练中，平均吞吐量没有意义。木桶效应决定整体速度取决于最慢的那张卡。
分析的第一步不是看"平均步时"，而是找出"异常离群值"。

## 分析决策流

```
[开始]
   |
   +-- 1. 全局扫描 (parse_multi_rank.py Phase 1)
   |      |- 跨 Rank Step Trace 对比 -> (T_max-T_avg)/T_avg 慢卡检测
   |      |- 通信域推断 (DP/TP/PP/EP) -> 确认可否全域对比
   |
   +-- 2. 时间线分解 (parse_multi_rank.py Phase 2)
   |      |- 分解时间: Computing vs Comm(Not Overlapped) vs Host(Not Overlapped) vs Free
   |      |- 重叠率是否过低? -> 是: 调并行策略; 否: 关键路径分析 -> 聚焦关键路径算子
   |
   +-- 3. 深度根因 (parse_multi_rank.py Phase 3 + 单卡脚本)
   |      |- 是 Comm? -> 算 R_wait -> 查包大小/对齐/带宽争抢
   |      |- 是 Compute? -> 查 Roofline (kernel_details)
   |      |- 是 Host? -> 查 JIT/同步/GC (operator_details/trace_view)
   |
   +-- 4. MoE 专项 (parse_multi_rank.py Phase 5)
          |- 查 AlltoAll Token 分布 -> 调负载均衡
```

## Phase 1: 全局扫描与慢卡定位

### 通信域分组原则

严禁跨通信域混比：
- **DP 域**：所有卡执行相同计算图，可直接对比 Step Time
- **TP/PP 域**：不同 Stage 执行不同算子，不可直接对比。仅在同一个 TP/PP 组内对比各卡耗时

`parse_multi_rank.py` 自动推断并行策略（基于通信算子类型）：
- `allReduce` → DP（数据并行）
- `allGather` / `reducescatter` → TP（张量并行）/ FSDP
- `alltoall` → EP（专家并行）/ SP（序列并行）— MoE 场景，需查 Token 分布
- `send/recv` (P2P) → PP（流水线并行）— 需关注 Bubble

### 慢卡 (Tail Card) 检测

**指标**：`(T_max - T_avg) / T_avg`
- `> 10%`：存在显著慢卡
- `> 15%`：严重慢卡

木桶效应：整体吞吐量被最慢的卡拖累。后续深度分析应聚焦最慢 rank，避免信息过载。

### 逐 Step 慢卡定位

若 step_trace 有多个 step：
- 找出 Step Time 突然飙升的 Step 编号（当前步 > 前一步 × 2.0 = SIGNAL 突升）
- 在该 Step 内对比所有 Rank 的耗时，找出最晚结束的 Rank

`parse_multi_rank.py` 自动检测逐 step 的 straggler ratio 和 step time 突升，并输出 Phase 1 结论
（锁定问题 Step 和 Rank，后续分析聚焦此范围）。

## Phase 2: 时间线分解与并行效率

### 时间构成分解

`总耗时 = Computing(含重叠) + Comm(Not Overlapped) + Host(Not Overlapped) + Free`

关注"非重叠"部分——重叠部分已隐藏，不影响总时长。Host 非重叠耗时高时直接进入 Phase 3.1-3.2 的 I/O 与 CPU 分析。

### 计算-通信重叠率

**指标**：`Overlapped / Total`
- `< 5%`：严重并行瓶颈 — 优先调整并行策略而非优化单算子
- `< 10%`：并行效率低 — 增大 micro-batch / gradient bucketing / 调整 allreduce 触发时机

### 假性重叠检测

**现象**：Profiling 显示存在计算-通信重叠，但总耗时并未下降。
**原因**：通信 DMA 与计算 MatMul 争抢 HBM 带宽。
**排查方法**：对比开启/关闭通信重叠时，同一计算算子的执行耗时是否增加。
若计算耗时增加 > 10-20%，说明重叠带来的收益被计算降速抵消。
**处理方向**：调整算子调度顺序或设置 HBM 访问优先级，而非一味追求重叠。

`parse_multi_rank.py` 在重叠 > 5% 但未重叠通信仍 > 20% 时提示假性重叠信号。

### 关键路径分析

重叠率正常时，不要直接优化单个算子——先找关键路径。

**方法**：从 Step 的最后一个算子开始，沿计算图的依赖关系（数据生产者 → 消费者）反向搜索，找出从第一个算子到最后一个算子的最长时序路径（即关键路径）。

**为什么**：关键路径上的算子决定总步时——非关键路径上的算子被关键路径掩盖，优化它们对总步时无影响。关键路径上的算子通常不到总算子数的 50%，聚焦它们可避免在无效算子上浪费时间。

**操作**：用 `parse_kernel_details.py` 按 Start Time 排序获取算子时序，结合 `parse_operator_details.py` 的 Call Stack 确定数据依赖关系（谁产出谁消费），手动反向追溯最长路径。后续 Phase 3 的深度根因分析仅针对关键路径上的 Top-N 算子。

## Phase 3: 深度根因定位

### Phase 3.4: 通信瓶颈分析

#### 同步等待比例 R_wait

**公式**：`R_wait = 1 - (T_avg / T_max)`
- `T_avg`：通信域内所有卡通信时间的平均值
- `T_max`：最慢卡的通信时间
- `R_wait > 30%`：存在严重同步慢卡问题
- `R_wait > 50%`：通信瓶颈本质是同步等待（非带宽问题）

`parse_multi_rank.py` 按通信算子类型（allGather/alltoall/allReduce）分别计算 R_wait，
并识别慢卡 rank 和快卡 rank。

#### 链路传输带宽分析

按 transport 类型（HCCS/SIO/LOCAL/PCIE/RDMA）分组检查：
- 实际带宽远低于理论带宽时的排查清单：

| 原因 | 检测方法 | 优化方向 |
|------|---------|---------|
| 小包问题 | 单次通信 < 32MB | 增大 Batch Size / 梯度融合 |
| HBM 带宽争抢 | 通信与 MatMul 并行时计算变慢 | 强制串行化或调低通信优先级 |
| 字节对齐 | 传输大小非 512B 整数倍 (HCCS) | 调整张量 shape 或 Padding |
| RDMA 重传 | 传输耗时异常 (>4s) | 查交换机拥塞/光模块/物理链路 |
| 网络配置 | UDP 端口冲突 | 查端口/交换机 PFC 配置 |

`parse_multi_rank.py` 自动检测小包占比（< 32MB）、字节对齐（512B）、RDMA 重传（transit > 4s）。

#### 私有格式转换 (TransData) 检查

检查 op_statistic 中是否存在 TransData / TransForm / FormatTransfer 算子。
这些算子表示私有格式转换开销——非 ND 格式输入会触发额外转换 kernel。
`parse_multi_rank.py` 自动检测并报告总耗时和占比。

### Phase 3.3: 计算瓶颈分析 (Roofline)

用 `parse_kernel_details.py` 获取算子的 mac_ratio、mte_ratio 等，判断：

1. **Compute Bound**：AI 大于拐点，算力接近峰值 → 检查降频/算子融合/升级硬件
2. **Memory Bound**：AI 小于拐点，带宽接近峰值 → 混合精度/ZeRO-Offload/算子融合
3. **Underutilization**（实战中最常见，60-97%）：
   - 是否运行在低性能核心（AICPU fallback）？→ `parse_op_statistic.py` 查 Core Type
   - 是否非对齐内存访问？→ 检查 shape 是否满足对齐要求
   - 是否格式转换开销？→ 查 TransData 算子

`parse_multi_rank.py` 跨 rank 对比算子耗时方差（CV > 20% 的算子标记为 SIGNAL）。

### Phase 3.1-3.2: I/O 与 CPU 瓶颈

#### I/O 瓶颈（队列深度分析法）

适用特征：Host 非重叠时间占比极大。

三级队列状态检查：
1. Device Queue 空 → NPU 空转等数据 → 进 2
2. Host Queue 空 → CPU 处理跟不上 → 查 DataLoader `num_workers`（若 = 1，提升至 CPU 核心数附近如 8~16）；检查是否从压缩包（zip/tar）读取，建议转为直接读取裸文件
3. Data Queue 空 → 存储读取慢 → 查远端存储（NAS/HDFS/网络挂载盘）；千卡训练强烈建议预处理数据并存入本地 NVMe SSD

**搬移瓶颈**（Host → Device）：检查是否启用 `pin_memory=True`（PyTorch 的 `pin_memory` 参数），以启用 DMA 加速 H2D 搬移。

用 `parse_trace_view.py` 的 Idle 成因分解定位 host 侧瓶颈。

#### CPU/Host 瓶颈

用 `parse_operator_details.py` 和 `parse_trace_view.py` 检查：
- JIT 编译：动态 Shape 触发频繁重编译
- 算子分发：大量小算子导致 CPU 频繁下发
- 同步 API：tensor.item() / reduce_all() / SyncBatchNorm
- Python GC：gc.collect 暂停（大模型 25% 性能波动可能由 GC 引起）
- CPU 资源抢占：监控进程/绑核竞争

## Phase 4: 大规模集群 (>1000 卡)

> Phase 4 需要 >1000 卡场景，`parse_multi_rank.py` 不直接覆盖。
> 以下为人工排查指南。

**核心原则**：千卡以上，"方差"比"均值"重要 100 倍。

### 吞吐量分布

绘制所有卡的 MFU 散点图。明显离群低点（如单卡 10 TFLOPS 而平均 85 TFLOPS）
直接判定为硬件故障，无需软件 Profiling。

### 硬件亚健康检测清单

- 端口振荡：查网络接口 UP/DOWN 日志
- 光模块故障：查 RDMA 链路误码率 (BER)
- 内存 ECC 纠错：查 HBM ECC 校正计数
- 散热降频：查核心温度是否超 85°C

**决策建议**：硬件亚健康节点直接隔离下线维修，不尝试软件 Workaround。
日志写入改为本地存储减少网络 I/O 干扰。

## Phase 5: MoE / AlltoAll 专项分析

传统 AllReduce 分析模型不适用于 AlltoAll 和 Expert Parallelism。

### 区分"通信慢"与"负载不均"

AlltoAll 耗时久时，先别急着查网络：
1. 结合 Profiling 查看各卡处理的 Token 数量分布
2. 若某张卡处理 Token 量远超平均（Expert Imbalance），该卡计算时间长，
   导致所有卡在 AlltoAll 同步点等待
3. **指标**：AlltoAll 总耗时跨 rank 的 CV > 0.2 = 负载不均瓶颈

`parse_multi_rank.py` Phase 5 自动检测 alltoall kernel 跨 rank 耗时方差。

### 优化方向

- 调整 MoE 路由算法 (Router)，加入负载均衡损失 (Load Balancing Loss)
- 调整专家容量因子 (Expert Capacity Factor)，限制单卡最大 Token 数

## 脚本与阶段的映射关系

| 阶段 | 脚本 | 关键指标 |
|------|------|---------|
| Phase 1 慢卡定位 | `parse_multi_rank.py` Phase 1 | (T_max-T_avg)/T_avg > 10%, Comm CV > 0.5 |
| Phase 1 通信域推断 | `parse_multi_rank.py` Phase 1 | allReduce→DP, allGather→TP, alltoall→EP, P2P→PP |
| Phase 2 重叠分析 | `parse_multi_rank.py` Phase 2 | Overlapped/Total < 5% = 严重并行瓶颈 |
| Phase 2 关键路径 | `parse_kernel_details.py` + `parse_operator_details.py` | 按 Start Time 排序 + Call Stack 确定依赖，反向追溯最长路径 |
| Phase 3.4 R_wait | `parse_multi_rank.py` Phase 3.4 | R_wait = 1-(T_avg/T_max) > 30% |
| Phase 3.4 小包/对齐 | `parse_multi_rank.py` Phase 3.4 | < 32MB 小包, 512B 对齐检查 |
| Phase 3.3 计算分析 | `parse_multi_rank.py` Phase 3.3 | 跨 rank 算子 CV > 20% |
| Phase 3.3 Roofline | `parse_kernel_details.py` | mac_ratio, mte_ratio, AICPU fallback |
| Phase 3.1-3.2 Host | `parse_operator_details.py`, `parse_trace_view.py` | Host self time, JIT, GC, sync |
| Phase 5 MoE | `parse_multi_rank.py` Phase 5 | AlltoAll CV > 0.2 = 负载不均 |

## 阈值速查

| 阈值 | 默认值 | 含义 | 配置位置 |
|------|--------|------|---------|
| tail_card_ratio | 10% | (T_max-T_avg)/T_avg 超此 = 慢卡 | thresholds.py multi_rank |
| tail_card_ratio_strong | 15% | 超此 = 严重慢卡 | thresholds.py multi_rank |
| overlap_severe_pct | 5% | 重叠率低于此 = 严重并行瓶颈 | thresholds.py multi_rank |
| overlap_low_pct | 10% | 重叠率低于此 = 并行效率低 | thresholds.py multi_rank |
| r_wait_definite | 30% | R_wait 超此 = 同步慢卡 | thresholds.py multi_rank |
| small_packet_mb | 32MB | 小于此 = 小包 | thresholds.py multi_rank |
| alignment_bytes | 512B | HCCS 对齐要求 | thresholds.py multi_rank |
| rdma_retransmission_ms | 4000ms | Transit 超此 = 疑似重传 | thresholds.py multi_rank |
| step_spike_ratio | 2.0x | Step > 前步×此 = 突升 | thresholds.py multi_rank |
| alltoall_cv_signal | 0.2 | AlltoAll CV 超此 = 负载不均 | thresholds.py multi_rank |
| comm_cv_definite | 0.50 | Comm CV 超此 = 严重不均衡 | thresholds.py multi_rank |
