# 通信分析统一方案（parse_communication.py v2 + 跨 Rank 融合）

> 修订记录：
> - 2026-08-24 v2 初版——`parse_communication.py` 单脚本增强设计（M1-M5），基于 Atlas 48 / AlphaFold3 / 8 rank 真实数据逐项验证
> - 2026-08-25 v2.1——融合 `parse_multi_rank.py`：确立"通信分析归一"架构，multi_rank 退役，跨 rank 能力收编为 `--all-ranks` 模式（见 §4）
> - 2026-08-25 v2.2——M4 简化：慢卡定位直接从两个 json 的 per-op wait join 得出（弃 step_trace 跨 rank 扫描层）；弃 DP/TP/PP/EP 策略推断（训练视角），域分组改为 @域ID 数据驱动；MoE 不单列专项，并入 alltoall 倾斜判读
> - 2026-08-25 v2.3——小包判定简化：弃 1MB/32MB 双档经验阈值（32MB 源自训练梯度分桶），改为数据驱动的延迟主导判定（transit < k×L̂）；1–32MB 档删除（与 M4 自基准、策略性归因重复）
> - 2026-08-25 v2.4——M3 时间轴分析降级：删除独立模块（直方图信号在数据内无法关闭，行动价值低），必要产出（瞬态污染防护）并入 M1 的 Top-K 头部位置标注；模块重编号：原 M4 跨 Rank → M3、原 M5 自基准 → M4
> - 2026-08-25 v2.5——R_wait 降级为画像/排序指标：删 30%/50% 两档间接判定（不均衡存在性归 M3.1 per-op join，等待/带宽定性归 wait/transit 分解；transit 主导场景同样推高 R_wait，间接判定有误报风险）
> - 2026-08-25 v2.6——重叠分析重构：低重叠降为现象测量（触发以 Comm(NotOvl) 占比为准），归因二分——串行型（可掩盖未掩盖）vs 依赖性串行（数据依赖下想并行也没法并行，overlap 收益上限为 0，正路是减通信量或改变依赖结构）
> - 2026-08-25 v2.7——全文一致性修订：删孤儿阈值 comm_cv_*（与 R_wait 同理归 M3.1 per-op join）；head_wait_ratio 语义明确为头部区段 wait 占比；补齐 unaligned_link_ratio / suspect_min_elapse_ms 阈值键；修正 §11 与 §8 的残留旧编号引用
>
> **实施状态（2026-08-25）**：P1–P4、P6 已实施完成，P1 验收标准全部通过（974/974 对齐、
> `hcom_allGather__612_546_1` 溯源输出 `parallel_utils.py(95): allgather_along_dim`、
> 总章通信疑点 Top-3 + 置信度、step_trace 与 Total Op Info 一致性检查 0.0% 偏差）；
> `parse_multi_rank.py` 已退役删除，run_analysis 的 I 节取消。P5（db 精确溯源路径）未实施——
> 遇到带 db 的数据集时按 §3.1 机制补充。
> 实施修订：§8 中 `low_bw_ratio` / `low_bw_min_size_mb` / `overlap_low_pct` 未实施（无消费方，
> 已从 thresholds.py 移除）；`--all-ranks` 参数移除（多 rank 目录恒自动启用，参数无语义）；
> alltoall kernel 耗时 CV 已接入交叉验证（§8 alltoall_cv_signal 注释相应更新）；
> §4.2/§9 的"guide 重构为多卡通信分析指南"改为**直接删除**——判读已内嵌于脚本输出，
> 独有内容（域分组机制、R_wait 误报警示、带宽算法分摊原则、依赖性串行判读）并入
> profiling_scripts_guide §parse_communication 与 profiling_to_action 归因层 #8，避免同一
> 套判读散布四处。
>
> 背景: 通信/多卡分析当前分裂为两套——`parse_communication.py`（单 rank 通信深度）与
> `parse_multi_rank.py`（跨 rank 四阶段全栈）。本方案将二者融合为单一脚本的双模式设计，
> 并与现有分析体系（profiling_to_action 归因层、单卡脚本族）显式对接。
> v2 初版基于一份真实多卡 profiling 数据（Atlas 48, AlphaFold3/TorchFold, 8 rank）的
> 逐项验证设计。
>
> 验证数据集: `/Users/kaili/Downloads/latestwork-main/rank_0/atlas48_421144_..._ascend_pt/ASCEND_PROFILER_OUTPUT`
> （communication.json 1344 op / communication_matrix.json 18 逻辑组 / trace_view.json 1.3GB /
> operator_details.csv 275MB / kernel_details.csv / ascend_pytorch_profiler_0.db）
>
> 文中所有"实测"结论均来自该数据集的验证脚本输出，可复现。

---

## 1. 现状与缺口

### 1.1 两套现状

| | parse_communication.py（H 节） | parse_multi_rank.py（I 节） |
|---|---|---|
| 视角 | 单 rank 通信深度 | 跨 rank 全栈对比 |
| 数据 | communication.json + communication_matrix.json | step_trace + communication + op_statistic + matrix + kernel_details（全部自解析） |
| 回答 | 通信慢不慢、是不是同步等待 | 慢卡在哪、重叠够不够、MoE 均不均 |

### 1.2 缺口（三类）

**缺口一：溯源断桥**——`operator_details.csv` 中**搜不到** `hcom_*` 算子名。hcom 是
COMMUNICATION 类型 kernel，只出现在 `kernel_details.csv`（无 Call Stack 列）。带 Call
Stack 的 `operator_details.csv` 只有框架层算子。通信算子的溯源必须另建对齐机制（见 §3）。

**缺口二：判据不严谨**——带宽用"低于均值 30%"判瓶颈（实测反例：allreduce 同步消息
~0MB/10GB/s 会全部误报）；小包阈值两套（communication 节 1MB、multi_rank 节 32MB）；
矩阵带宽是算法分摊值（同 HCCS：allgather 132GB/s vs alltoall 23.8GB/s），任何拿物理
峰值做绝对判据的方案都会系统性误报。

**缺口三：体系分裂**——multi_rank 的 Phase 3.3（op 跨 rank 方差）、TransData 检查与
单卡脚本（parse_op_statistic 等，均支持 `--rank N`）重复实现；multi_rank_analysis_guide
中的 I/O 队列、CPU/Host、Roofline、关键路径章节与 profiling_to_action.md 及单卡脚本
的职责重叠。两套脚本、两套阈值、两份指南，判据与话术不统一。

**缺口四：跨 rank 精度**——multi_rank 的慢卡检测是总量对比（(T_max−T_avg)/T_avg）+
wait 不均衡的受害者推断，粗粒度；无法回答"哪个通信算子上谁在让谁等"。

## 2. 数据结构实测结论

### 2.1 communication.json（逐算子明细）

```
step0
├── p2p: {}                                  # 本数据集为空
└── collective: { 1344 个条目 }
     ├── "Total Op Info"                     # 全 step 汇总（勿与明细行重复相加）
     └── "hcom_<类型>__<模型>_<序号>_<?>@<域ID>"
          ├── Communication Time Info
          │     Start Timestamp(us) / Elapse / Transit / Wait / Sync / Idle / 各占比
          └── Communication Bandwidth Info
                RDMA / HCCS / PCIE / SDMA / SIO
                每种: Transit Size(MB) / Transit Time(ms) / Bandwidth(GB/s)
                      Large Packet Ratio / Size Distribution{包大小KB: [包数, 耗时ms]}
```

- 官方定位（CANN 文档「通信性能数据解析」）：单卡所有通信算子的耗时、带宽明细。
- 时间分解语义：`Elapse = Transit + Wait/Sync + Idle`。Wait 高 = 同步等待（负载不均/
  无 overlap），Transit 高 = 带宽问题。
- `Start Timestamp(us)` 可重建通信时间轴。

### 2.2 communication_matrix.json（逻辑组 × rank 对矩阵）

```
step0
├── p2p: {}
└── collective: { 18 个逻辑组 }              # allgather-top1@域ID / alltoall-middle@域ID / ...
     每组 → { "源rank-目的rank": {...} }     # "0-1"、"0-7"
        Transport Type: LOCAL / HCCS / SIO / SDMA / RDMA
        Transit Size(MB) / Transit Time(ms) / Bandwidth(GB/s)
        Op Name: 指向 representative hcom 实例（回链 communication.json 的钥匙）
```

- 官方定位：通信小算子基本信息（size、带宽、rank），用于分析通信细节。
- `-top1/-middle/-bottom1…` 为按耗时聚类的逻辑阶段，`-total` 为汇总。

### 2.3 关键实测数据（本数据集）

- 1344 个通信算子 = allGather 974 + alltoall 363 + allReduce 6 + Total Op Info 1。
- 真实通信总量 = Total Op Info = 19461.8ms（注意：把汇总行与明细行相加会得到 38.9s，是重复计数）。
- `step_trace_time.csv` 的 `Communication` 列 = 19461.8ms，与 Total Op Info **精确相等**
  ——两份文件是同一测量的两个视角。
- Wait 占比 ~100%（19461.8ms 中 19458.9ms 等待）——同步等待主导，非带宽瓶颈。
- 多流并发实测：allGather 974 个算子中 11 处时间区间重叠（1.1%），alltoall 25 处（6.9%），
  allReduce 0 处。重叠≠乱序（见 §3.3）。
- matrix 真实倾斜样例：`alltoall-top1` 组中 0-1 传 154MB 而 0-2~0-7 各 62.5MB（2.5 倍）。
- 带宽口径陷阱：同为 HCCS，allgather 链路 132GB/s、alltoall 链路 23.8GB/s——
  后者是算法分摊（小 chunk），不是链路慢。**任何带宽判据必须以 size 为条件**。

## 3. 溯源链路验证（设计的事实基础）

### 3.1 三条路径

| 路径 | 机制 | 精度 | 可用性 |
|---|---|---|---|
| db | `COMMUNICATION_OP → STRING_IDS 解码 opName → connectionId → CANN_API(launch) 时间窗 → PYTORCH_API 时间包含` | [EXACT] 精确 | 实际场景通常无 db，降为可选 |
| trace | trace_view.json 中 `Hccl*` cpu_op 事件自带 `ts` + `args."Call stack"`，序号对齐 + ts 自校验（host_ts ≤ device_start） | [ALIGNED] 高 | 需 timeline 导出 |
| CSV 序号 | `operator_details.csv` 中 `Hccl*` 行（文件序 = 时间序）与 `communication.json` 按 `Start Timestamp` 排序的同类算子**按序号一一对应** | [RISKY] 中 | 仅需 csv/json |

序号对齐法原理：device 侧通信算子按入队顺序执行（同通信域内 HCCL FIFO），host 侧
`HcclAllGather` 调用按发起顺序记录，两序列一一对应。

### 3.2 序号对齐验证结果

- `communication.json` 中 974 个 allGather 按 Start Timestamp 排序，目标算子
  `hcom_allGather__612_546_1` 为第 372 个；
- `operator_details.csv` 第 373 行 `HcclAllGather` 的 Call Stack 即其源码位置；
- 用 trace 的 974 个 `HcclAllGather` 事件（ts 排序）交叉验证：**CSV 行序与时间序
  974/974 完全一致**；
- 与 db 路径的 connectionId 精确关联结果一致：均定位到
  `parallel_utils.py(95): allgather_along_dim ← ring_einsum_outgoing ← pairformer.py(90): forward`。

### 3.3 多流问题与四层防线

重叠检测不依赖 `kernel_details.csv` 的 Stream ID（实测 hcom 行该列全为 N/A），用时间
区间重叠率做数据内并发检测。处理层次：

1. **重叠率 < 2%** → 序号对齐有效（残余乱序风险标注）；
2. **按通信域分组对齐**：key 的 `@域ID` 后缀即分组依据，域内 FIFO，跨域并发不干扰；
3. **trace 路径用 ts 窗口匹配**替代全局序号（最近前驱 host 调用，lag 上界取 P95）；
4. **兜底降级**：仍对不上时输出**聚合栈分布**（该 host 算子所有 distinct Call Stack +
   次数 + 耗时），供 agent 用 Input Shapes 桥自行消歧。

原则：溯源错一行代码会把整个根因分析带偏，**宁可降级，不输出错误对齐**。

### 3.4 硬前提：with_stack

trace 事件 args 里的 `Call stack` 与 `operator_details.csv` 的 Call Stack 列来自同一次
`with_stack=True` 采集。关掉它，trace 与 CSV 路径**同时失效**。M1 开头先探测栈可用性，
不可用直接短路并提示"重新采集需开 with_stack"。

## 4. 架构决策：通信分析归一

**原则**：多卡场景相对单卡的核心增量是**通信**；计算/Host 侧问题在锁定问题 rank 之后，
用现有单卡脚本分析（`parse_op_statistic.py` / `parse_kernel_details.py` /
`parse_operator_details.py` / `parse_trace_view.py` 均支持 `--rank N`）。因此通信分析
收敛为**一个脚本的两个模式**，`parse_multi_rank.py` 退役。

### 4.1 能力处置表

| parse_multi_rank 能力 | 处置 | 去向与理由 |
|---|---|---|
| Phase 1 慢卡定位（(T_max−T_avg)/T_avg、逐 step straggler/spike） | **简化收编 → M3.1** | 慢卡不需要 step_trace 跨 rank 扫描——per-op wait join 直接从两个 json 得出（wait 最小的 rank 是最后到达的被等者），逐算子粒度比总量对比更精确 |
| Phase 1 通信域推断（DP/TP/PP/EP） | **降级收编 → 域分组机制** | 策略分类是分布式训练视角（本 skill 以推理优化为主，验证集即非 LLM 的 AlphaFold3）；真正需要的是 join/聚合的分组正确性——op 名的 `@域ID` 后缀数据驱动给出（§3.3 已有），不输出 DP/TP/PP/EP 标签 |
| Phase 2 重叠率 + 假性重叠 | **收编并重构 → M3.3** | 低重叠降为现象测量，归因二分（串行型可掩盖 / 依赖性串行不可掩盖），触发以 Comm(NotOvl) 占比为准——原"重叠 < 5% 即严重并行瓶颈"在 comm 占比小和真数据依赖两种场景都会误导 |
| Phase 3.4 R_wait（按类型，30%/50% 两档判定） | **降级 → M3.2 展示指标** | 判定职责归直接判据：不均衡存在性归 M3.1 per-op join（附带 straggler 身份），等待/带宽定性归 wait/transit 分解；R_wait 是间接量，transit 主导（单 rank 链路慢）同样推高 R_wait，50% 判据会把带宽问题误判为等待 |
| Phase 3.4 小包/字节对齐/RDMA 重传 | **收编 → M2** | link 级检查归 matrix 分析；小包判定改为数据驱动的延迟主导（见 M2，弃 32MB 经验阈值） |
| Phase 4 AlltoAll/MoE 耗时 CV | **并入判读规则 → M2/M3** | 不单列专项：alltoall wait 主导 + M2 rank-pair 倾斜即负载不均证据；MoE 只是 LLM 下的常见成因，科学计算模型的手工切分同样触发。kernel 耗时 CV 保留为交叉验证指标 |
| Phase 3.3 op_statistic 跨 rank 方差 | **舍弃** | 对问题 rank 跑 `parse_op_statistic.py --rank N`，单卡脚本已有 |
| TransData 检查 | **舍弃** | `parse_op_statistic.py` + profiling_to_action 归因层第 7 类（布局/格式转换）已覆盖 |
| guide 中 I/O 队列 / CPU-Host / Roofline / 关键路径章节 | **舍弃** | profiling_to_action.md + trace_view / kernel_details / operator_details 脚本已有，guide 不重复维护 |

收编判据：被收编的每一项都**直接服务于通信归因链**（谁在等谁 → 哪条链路 → 哪行源码），
数据源限于两个通信 json（step_trace 仅取重叠三列）——不越界到计算侧。

### 4.2 退役与文档处置

- `parse_multi_rank.py` 删除；`run_analysis.py` 的 I 节取消，H 节在检测到多个 `rank_N`
  目录时自动切换 `--all-ranks` 模式（见 §7）；
- `thresholds.py` 的 `multi_rank` 节删除，通信相关键迁入 `communication` 节（见 §8）；
- `multi_rank_analysis_guide.md` 重构为「多卡通信分析指南」：保留 @域ID 域分组机制、
  慢卡（straggler）判读、R_wait、带宽排查清单、alltoall 负载不均判读；I/O、CPU/Host、
  Roofline、关键路径章节删除，改为一段指回 profiling_to_action.md（见 §9）。

## 5. 模块设计

### M1 源码溯源（核心新增）

**目标**：`hcom_allGather__612_546_1` → `parallel_utils.py(95): allgather_along_dim`。

**路径决策（最终版，按可用性自动选择，输出标注置信度）**：

```
with_stack=False → 短路提示
trace_view.json 存在 → ts 窗口匹配（自校验：host_ts ≤ device_start）    [ALIGNED]
trace 缺失        → operator_details.csv 序号对齐（域分组+重叠率检测） [RISKY]
都对不上          → 聚合栈分布兜底                                      [AGGREGATED]
```

选择 trace 为主路径的理由：唯一能自校验（ts 合理性检查，实测 host 调用早于 device
起始 1.7ms）；CSV 路径无时间戳，对齐正确性无法自证，仅在 trace 缺失（如为省磁盘关闭
timeline 导出）时启用。

**算子名映射表**（进 `thresholds.py`，前缀匹配 + 模糊兜底，跨 CANN 版本命名有差异）：

```python
"comm_host_op_map": {
    "hcom_allGather":  ["HcclAllGather", "HcclAllgatherBase"],
    "hcom_allReduce":  ["HcclAllReduce"],
    "hcom_alltoall":   ["HcclAllToAllV", "HcclAllToAll"],
    "hcom_reduceScatter": ["HcclReduceScatterV", "HcclReduceScatter"],
    "hcom_broadcast":  ["HcclBroadcast"],
    "hcom_send": ["HcomSend", "HcclSend"],
    "hcom_recv": ["HcomRecv", "HcclRecv"],
}
```

**成本控制**：默认报告只对 Top-K（`trace_top_k=3`）疑点做自动溯源；完整逐算子溯源走
`--trace-source <op>` 按需模式。trace 定向抽取 `Hccl*` 事件（不全量解码），
实测 `rg` 抽取 1.3GB 约 2s，Python 流式约 10–30s。

**两种使用方式**：
- `--trace-source hcom_allGather__612_546_1`：单算子深挖（类比
  `parse_operator_details --filter`），输出完整调用栈；
- 默认报告自动溯源：H 节对 Top-3 wait-heavy 算子附源码位置，并上浮总章（见 §7）。

**瞬态污染防护（吸收原 M3，防追鬼影）**：头部区段 = 通信总时窗的前 `head_window_frac`。
两处产出：
- H1 摘要一行"wait 头部集中度"：头部区段内 wait 量占 wait 总量比例，超 `head_wait_ratio` → 单条 [SIGNAL]"疑瞬态污染"；
- Top-K wait-heavy 疑点逐条标注头部位置（Start Timestamp 是否落在头部区段），防 M1 溯源与 M3.1 join 追 warmup/编译期鬼影。

验证路径（SIGNAL 附带）：warmup 是否生效（采集规范要求预热 3 次）/ 多 step 是否复现 /
L0 对照。不做独立的 wait 直方图与假设清单——瞬态/稳态的区分证据在文件外，分析侧只做
位置标注与警示，防线在采集侧（warmup 规范）与验证侧（多 step 复现）。

### M2 Matrix 深度分析（替换现第 4 节）

| 分析 | 原理与判读规则 | 信号 |
|---|---|---|
| **rank-pair 数据倾斜** | 集合通信理想形态各 rank 收发均衡，`max(pair size)/min(pair size)` 量化最忙/最闲链路倍数。三条判读规则（缺一即误报）：① 同 transport 内比较（LOCAL 是卡内搬运，口径不同）；② 排除小消息（<`skew_min_size_mb` 的同步消息是算法结构非倾斜）；③ 按类型解读（allgather pair 倾斜 = 实现问题；alltoall 倾斜 = 负载分布不均，往往正是要找的问题） | 倾斜比 > `skew_ratio` → [SIGNAL] |
| **Transport 合理性** | 同组内同量级流量本应走同类链路（HCCS/SIO/PCIE 带宽差数量级）。**文件不含物理拓扑，不能断言某 pair "应该"走什么**，只用数据内证据：同组同量级流量出现 transport 混杂（实测：allgather-top1 中 0-1 走 SIO、0-7 走 HCCS，同为数百 MB） | 组内不对称路由 → [SIGNAL]"需对照部署拓扑确认 rank-设备映射"；同一 pair 跨组 transport 不一致 → 硬信号 |
| **慢链路定位** | 两个修正：① 排序维度从带宽改为 **Transit Time**（瓶颈在时间不在带宽）；② 带宽判据必须 **size 条件化**——小消息 time 被固定延迟（~10–20us）主导，低带宽是物理必然（实测反例：allreduce-top1 全部 pair ~0MB/10GB/s 是同步消息，现脚本的"低于均值 30%"会全部误报）。正确做法：同 transport × size 对数分档内做带宽分位数，仅大 size 档底部分位报慢链路 | 大 size 档带宽 < 档内 P50 × `bw_p50_ratio` → [SIGNAL] |
| **链路矩阵渲染** | 纯模式识别：数字表 → 空间模式（全零列=死链路，对角线独重=全本地，单行独重=单点扇出）。选流量最大逻辑组渲染，size 对数简写 | — |
| **组↔算子回链** | matrix 逻辑组是聚合视图，每条 pair 的 `Op Name` 指向 representative hcom 实例（实测：allgather-top1 的 0-1 pair → `hcom_allGather__612_604_1`）。没有此回链，matrix 层发现无法进入 M1 溯源管道 | — |
| **字节对齐**（收编自 multi_rank） | 传输大小非 512B 整数倍（HCCS 对齐要求）→ 带宽受损。仅检查 <100MB 的 link（MB 小数精度不足以做字节级判定）、跳过 LOCAL（卡内拷贝口径不同） | 非 512B 对齐 link 占比 > `unaligned_link_ratio` → [SIGNAL]，附 shape/padding 调整方向 |
| **RDMA 重传疑似**（收编自 multi_rank） | Transit Time 异常大（> `rdma_retransmission_ms`，默认 4s）疑似重传——交换机拥塞/光模块/物理链路 | → [SIGNAL]，附交叉验证清单（BER、交换机 PFC、端口冲突） |

**小包判定（延迟主导，数据驱动）**

**问题定义**：一次通信的耗时 = 固定开销（协议/启动延迟 L，与消息大小无关）+ 传输量 ÷ 带宽。小包问题 = 固定开销占大头——花在"启动通信"上的时间超过搬数据本身。修复方向只有一类：减少通信次数（合并/batching）。

**为什么阈值不能写死 MB 数**：边界 = L × 带宽，随硬件漂移。以 HCCS ~100GB/s、L≈30μs 为例：1MB 传输耗时 ~10μs < L（延迟主导，算小包）；32MB 传输耗时 ~320μs ≫ L（带宽主导，**物理上不是小包**）。v1 的 1MB 方向正确但换硬件即漂移；multi_rank 的 32MB 源自训练场景梯度分桶惯例（配的建议"梯度融合"也是训练概念），不是小包边界。

**L̂ 估计（数据驱动，按 transport 分组）**：同 transport 下，最小 size 档（< `slow_link_size_buckets[0]`，即 <1MB）内消息的 transit 时间都贴在固定开销地板上——取该档消息 transit 的 **P50 作为 L̂**。不同 transport（HCCS/RDMA/PCIE）固定开销不同，各自估计、各自判定。

**判定规则**：
1. 单次通信（matrix 中的 per-link 记录）：`transit < latency_dominated_multiple × L̂`（默认 3 倍）→ 该次通信延迟主导；
2. 汇总信号：延迟主导消息占比 > `small_packet_ratio`（0.3）→ [SIGNAL]，建议合并通信；收益上限估算 = 可合并的消息次数 × L̂（供归因映射的候选评估使用）；
3. 降级：最小档样本数 < `latency_est_min_samples`（10）→ L̂ 不可信，退回 1MB 固定阈值并标注 [DEFAULT]。

**1–32MB 档不保留（防偏移说明）**：该档想表达的两个关切各有归属，不得以固定阈值重新实现——
- "该大小消息没跑满链路" → M4 自基准（同 transport × size 档内带宽分位数，低于档内 P50×`bw_p50_ratio` 才报慢），小包判定不做带宽判断；
- "消息切得太碎可以更大" → 通信粒度由分片策略决定，归**策略性通信**（M1 溯源定位切分代码），小包判定不输出"增大粒度"类建议。

### M3 跨 Rank 通信对比（`--all-ranks` 模式）

数据源仅两个通信 json（每个 rank 一份）+ 各 rank step_trace 的重叠三列。三层产出：

**M3.1 per-op wait join（核心，慢卡定位在此得出）**：hcom 算子名含全局序号，跨 rank
同名即同一逻辑通信（实测验证，见 §3.2）。按算子名 join 所有 rank 的 communication.json：

- 对 Top-K wait-heavy 算子输出跨 rank 表（`op | rank_0 wait | rank_1 wait | ...`）；
- **straggler 判定**：集体通信中所有人等最后到达者——某 rank 在多数高 wait 算子上
  wait 显著最小（与其他 rank 的倍数差 > `straggler_min_wait_gap`）→ 它最后到达，是被等
  的 **straggler** [DEFINITE]；wait 最大者是受害者。
  慢卡定位由此直接得出，无需 step_trace 跨 rank 总量对比；
- join 按 `@域ID` 分组进行——不同域的 rank 不共享同名算子，天然不会跨组误比；
- straggler 锁定后，"它为什么慢"是计算侧问题：引导 agent 对该 rank 跑单卡脚本
  （`--rank N`），不在此脚本内实现。

**M3.2 per-rank 汇总**：各 rank 通信总量（Elapse/Wait/Transit）对比 + 按类型聚合的 R_wait（`1 − T_avg/T_max`）。R_wait **只作画像与排序指标**（哪个通信类型不均衡最严重），不设阈值判定——两个判定职责各有直接判据，R_wait 的间接判定会误报：

- "是否存在慢卡、谁在让谁等" → M3.1 per-op join [DEFINITE]（附带 straggler 身份）；
- "瓶颈本质是等待还是带宽" → wait/transit 时间分解（`wait_dominant_ratio`，现有可疑信号直接测量）。

误报实例（防偏移）：单 rank 链路降级 → 该 rank transit 主导、elapse 远超他卡 → R_wait 同样 >50%——间接判定会误判"本质是等待"，wait/transit 分解正确指向带宽问题（归 M4 慢链路）。R_wait 高只说明"分布散"，不区分散的成因是等待还是传输慢。

**M3.3 重叠对比（现象测量，非结论）**：各 rank step_trace 的 Overlapped / Comm(NotOvl)
三列直接读取（与 communication.json 是同一测量的两个视角，实测精确相等）。

**触发条件**：Comm(NotOvl) 占比高（沿用归因层 #8 触发阈值：comm 占总 step 时间
> 20%，或利用率 < 50% 且 comm 有值）→ "掩盖不足"疑点。重叠率只作刻画量不做独立
触发——comm 总量很小时低重叠无意义（原"重叠 < 5% 即报严重并行瓶颈"的两个缺陷：
comm 占比小误报、数据依赖场景误导）。

**低重叠的归因二分**（区分证据在源码，属 Line A；M3.3 只输出数据内提示——通信
kernel 前后 idle gap、多流并发占比）：

- **串行型（可掩盖未掩盖）**：通信调用点上下游存在独立计算（其他层/其他分支/可分块
  的工作），但实现是串行调度。修法：异步通信流 / 调度重排。收益上限 = 可掩盖的
  NotOvl 部分；
- **依赖性串行（不可掩盖）**：通信结果是下一步计算的直接输入（如 TP 逐层 allgather），
  依赖来自切分结构，**想并行也没法并行**。此时 overlap 路线上限为 0，正路两条：
  ①减少通信量——冗余型/原语不当型/拓扑错配型（M1 溯源 + M2 判定）；②改变依赖结构
  ——流水线/微批次/分块化（策略性，工程代价高，需 ★A 用户确认）。

**假性重叠** heuristic 保留：重叠 > `overlap_severe_pct` 但未重叠通信仍 >
`false_overlap_comm_pct` → 通信 DMA 与计算争抢 HBM 嫌疑，交叉验证"开/关重叠时同一
计算算子耗时是否增加 > 10-20%"。

**alltoall 负载不均判读**（判读规则，非专项模块）：alltoall wait 主导 + M2 rank-pair
倾斜 → Token/数据分布不均。MoE 是 LLM 场景的常见成因，科学计算模型的手工切分同样
触发——不按模型类型预设，只看数据形态。alltoall kernel 耗时跨 rank CV 保留为交叉
验证指标。优化方向指向负载再平衡（容量因子/切分配置），不预设训练侧手段（如负载
均衡损失——训练期改模型的方案，超出推理优化范围）。

**使用方式**：多 rank 目录存在时 H 节自动进入；也可 `--all-ranks` 手动调用。

### M4 带宽自基准（不写死芯片数值）

实测发现的根本问题：矩阵里的 Bandwidth 是**算法分摊后的值**（同为 HCCS：allgather
132GB/s vs alltoall 23.8GB/s，后者是小 chunk 算法形态而非链路慢），任何拿物理峰值做
绝对判据的方案都会系统性误报。设计：

1. **主判据 = 数据内自基准（self-calibrating，零配置，任意芯片可用）**：同 transport ×
   size 对数档内统计分布，档内 P50 即该环境该档实测可达值；用大 size 档的**带宽上包络**
   估计该环境实测峰值；
2. **绝对参考表降级为可选注入**：`thresholds.py` 只放 ratio/分位数参数；芯片参考表独立
   成 `soc_bw_table`（默认空 = 不启用），通过 `--soc` 或项目级 override 填充后才生效；
3. 两者互补：相对基准抓个别慢链路，绝对表（配置后）抓**所有链路一起慢的系统性问题**
   （降频、链路降级）——这是相对基准原理上的盲区。

## 6. 归因映射：与 profiling_to_action 体系的融合

通信分析的产出必须直接进入现有归因层（profiling_to_action.md 归因层第 8 类「通信开销」的细分 + 相关类别），而不是自成一套话术。模块 → 归因细分 → 收益上限口径：

| 信号来源 | 归因细分 | 反事实收益上限口径 |
|---|---|---|
| M3.1 per-op join straggler（某 rank wait 显著最小） | 等待型 | 通信 Wait 总量（修 straggler 计算侧问题后释放） |
| alltoall wait 主导 + M2 倾斜（判读规则） | 等待型（负载不均，MoE 为成因之一） | alltoall 同步点等待时间 |
| M2 组内 transport 不对称路由 | 拓扑错配型 | 跨节点通信量 ×（节点内外带宽差） |
| M1 溯源栈：同张量短时间多次 AllGather | 冗余型 | 被消除的重复通信时间 |
| M1 溯源栈：AllGather 收全量但只读 1/P、AG 后立即 slice 丢弃 | 原语不当型 | 通信量缩减比例 × 该通信时间 |
| M1 溯源栈：通信量 ≈ O(L²) / 收缩维度=分片维度 | 策略性 | 重新选择分片策略的收益估算（不可用加速通信解决） |
| 通信 kernel 前后有设备 idle gap + Line A 确认存在独立计算 | 串行型（可掩盖未掩盖，未用异步通信流） | 可掩盖的通信时间（≤ Comm(NotOvl)） |
| M3.3 低重叠 + Line A 确认通信为直接数据依赖（无独立计算） | 策略性（依赖性串行，overlap 上限为 0） | 通信量缩减比例 × Comm(NotOvl)；或改变依赖结构（流水线/分块）的重新估算 |
| M3.3 假性重叠（重叠但总时长未降） | 归因层 #10 延迟未掩盖（HBM 争抢变体） | 重叠收益被争抢抵消的部分 |
| M1 Top-K 头部集中标注（瞬态假设成立） | 非稳态问题 → 移出收益估算 | —（验证路径：warmup 核对 / 多 step 复现 / L0 对照） |
| M4 慢链路（大 size 档底部带宽） | 带宽受限（通信侧） | 档内 P50 差值 × 传输量 |
| M2 延迟主导小包（transit < k×L̂） | 协议开销 | 合并通信后的节省估算（消息次数 × L̂） |

**融合机制**：H 节信号行直接引用归因细分名（如 `[SIGNAL] 通信疑点#1 … → 归因：原语
不当型`），agent 据此在候选清单中使用 profiling_to_action §候选评估 的方法估算收益
上限，进入 ★A 确认——与 Line B 其他瓶颈类型的产出形态完全一致。

## 7. 报告集成与总章上浮

`run_analysis.py` 的机制天然支持：**所有细节节（A–H）先执行、后组装报告**，总章只是
最后拼在最前面。H 节跑完溯源后上浮，两级机制：

**1. 信号清单自动上浮（零改动复用 `_extract_signals`）**——H 节把 Top-K 疑点写成信号行
（栈压缩成单行）：

```
[SIGNAL] 通信疑点#1: hcom_allGather__612_546_1 wait 1431ms(99.99%) →
  parallel_utils.py(95): allgather_along_dim ← pairformer.py(90): forward [ALIGNED·trace]  [H]
```

**2. 总章新增专节"通信疑点 Top-K"（多行完整版）**——`parse_communication.parse()`
返回文本中疑点块用定界符包裹，`run_analysis` 切块插入总章 §3（信号清单之后），
H 节细节章保留完整版：

```
<<<COMM_SUSPECTS>>>
1. hcom_allGather__612_546_1  |  elapse 1431.7ms  |  wait 99.99%  |  序号 372/974
   源码: parallel_utils.py(95): allgather_along_dim
         ← parallel_utils.py(170): ring_einsum_outgoing
         ← pairformer.py(90): forward (PairformerBlock)
   置信度: [ALIGNED·trace] host_ts 早于 device_start 1.7ms
<<<END_COMM_SUSPECTS>>>
```

`run_analysis` 改动约 10 行（切块 + 插入），信号机制与现有约定零破坏。

**触发条件（成本控制）**：
- `communication.json` 不存在 → H 节跳过（现状）；
- 存在但无显著疑点（不满足 wait-dominant、Top op elapse < `suspect_min_elapse_ms`）→
  只输出聚合统计，跳过 trace 扫描；
- 有疑点 → 扫 trace 做 Top-K 溯源（`trace_top_k=3` 可配，`--no-trace` 可关）。

**H 节新结构**：

```
--- H. 通信（多卡） ---
H1 摘要（现有 + 慢卡提示、溯源方式与置信度、wait 头部集中度一行）
H2 按算子类型（现有）
H3 Top Ops by Elapse —— ★新增列：源码位置(项目帧) | 置信度标记 | 头部位置标注
H4 P2P Ops（现有）
H5 Matrix 深度分析（M2，含对齐/RDMA 重传/延迟主导小包判定）
H6 可疑信号（现有，扩充新信号）
H7 跨 Rank 对比（M3，仅多 rank 目录时：per-op join 慢卡定位 → per-rank 汇总/R_wait → 重叠对比 + alltoall 判读）
```

**run_analysis 集成变更**：
- H 节调用从 `parse(comm_path, matrix_path, 15)` 改为 `parse(profiling_dir, rank, 15)`
  （函数自己找文件，溯源需要 sibling 文件）；
- **I 节（parse_multi_rank）取消**：H 节检测到 `_discover_ranks(profiling_dir) >= 2`
  时自动启用 `--all-ranks`，跨 rank 对比（原 I 节职责中通信相关的部分）与单 rank 深度
  分析合并为一个 H 节输出；multi_rank 的 import 与调用移除；
- 总章上浮机制不变（信号清单 + `<<<COMM_SUSPECTS>>>` 专节），H7 的跨 rank 结论
  （锁定 straggler）同样以信号行上浮。


## 8. thresholds.py 变更

```python
"communication": {
    # --- 现有 6 项保留 ---
    "wait_dominant_ratio": 0.8,
    "per_type_wait_ratio": 0.9,
    "per_type_min_count": 10,
    "low_bw_ratio": 0.3,           # 由 M2 慢链路定位的 size 分档分位数判据替代（保留兼容）
    "low_bw_min_size_mb": 1,
    "small_packet_ratio": 0.3,     # 已有：延迟主导消息占比超此 → [SIGNAL]（M2）
    # --- 新增：M1 溯源 ---
    "trace_top_k": 3,              # 默认报告自动溯源的算子数
    "comm_host_op_map": {...},     # hcom → Hccl 名映射表
    "overlap_risk_ratio": 0.02,    # 时间区间重叠率超过此值 → 序号对齐标记 RISKY
    "host_ts_lag_p95_us": None,    # trace 路径 ts 校验 lag 上界（None=自学习 P95）
    # --- 新增：M2 Matrix / M1 瞬态防护 / M3 跨 Rank ---
    "skew_ratio": 3.0,             # rank-pair size 倾斜比（M2）
    "skew_min_size_mb": 10,
    "slow_link_size_buckets": [1, 10, 100],   # size 对数分档边界（MB）
    "bw_p50_ratio": 0.5,           # 档内低于 P50×此值 = 慢链路
    "head_wait_ratio": 0.6,        # 头部区段 wait 占比超此 → 疑瞬态污染（M1 Top-K 标注）
    "head_window_frac": 0.1,       # 头部区段 = 通信总时窗的前 10%（M1）
    "straggler_min_wait_gap": 0.5, # 跨 rank wait 差异倍数（M3.1）
    # --- M4 自基准：只放结构性参数，芯片数值放独立 soc_bw_table（默认空） ---
    "ref_min_size_mb": 10,
    # --- 融合自 multi_rank（原 "multi_rank" 节，随脚本退役整体迁入） ---
    "overlap_severe_pct": 5,       # 重叠率刻画分档（M3.3，非独立触发；触发以 Comm(NotOvl) 占比为准）
    "overlap_low_pct": 10,         # 同上：刻画"重叠充分/不足"的展示档位
    "false_overlap_comm_pct": 20,  # 假性重叠判定（M3.3）
    "r_wait_sort_top_k": 5,         # R_wait 仅作画像排序展示（M3.2，无阈值判定）
    "alltoall_cv_signal": 0.20,    # alltoall 负载不均交叉验证（M3 判读）
    "alignment_bytes": 512,        # HCCS 对齐检查（M2）
    "unaligned_link_ratio": 0.3,   # 非 512B 对齐 link 占比超此 → [SIGNAL]（M2 字节对齐）
    "rdma_retransmission_ms": 4000,# RDMA 重传疑似（M2）
    "latency_dominated_multiple": 3.0,  # transit < 此值×L̂ = 延迟主导小包（M2；L̂ 数据内估计，样本不足降级 1MB 缺省）
    "latency_est_min_samples": 10, # 最小 size 档样本数低于此 → L̂ 不可信，降级 1MB 缺省
    "suspect_min_elapse_ms": 100,  # Top op elapse 低于此不算疑点（§7 成本控制触发）
}
# "multi_rank" 节删除；tail_card_ratio / step_spike_ratio / timeline_bins / op_cv_signal /
# transdata_keywords 不迁移（慢卡扫描层、直方图模块、算子方差能力已舍弃，见 §4.1）；
# comm_cv_signal / comm_cv_definite 不迁移（总量 CV 判定与 R_wait 同理归 M3.1 per-op join，
# v2.5 逻辑）；small_packet_buckets 分档不设（1–32MB 档与 M4 自基准重复，见 M2 小包判定）
```

设计原则：阈值文件放**结构不放芯片数值**。`soc_bw_table` 独立可覆盖
（`--soc` 参数或项目级 override 注入），默认空 = 绝对判据不启用，仅自基准生效。

## 9. 配套文档更新

- `profiling_scripts_guide.md`：§parse_communication 补 H1（头部集中度）/H3（源码列+头部
  标注）/H5/H7 输出说明、
  `--trace-source`/`--all-ranks` 用法、置信度标记含义（[EXACT]/[ALIGNED]/[RISKY]/[AGGREGATED]）；
  §parse_multi_rank 条目**删除**（脚本退役）；
- `multi_rank_analysis_guide.md` → 重构为「多卡通信分析指南」：保留 @域ID 域分组机制、
  慢卡（straggler）判读、R_wait、带宽排查清单、alltoall 负载不均判读（与本文 M2/M3
  对齐）；I/O 队列、CPU/Host、Roofline、关键路径章节删除，改为一段指回
  profiling_to_action.md 与单卡脚本；
- `profiling_to_action.md`：① 三座桥表格 **Call Stack 桥新增"通信算子"一行**——
  `operator_details.csv` 的 `Hccl*` 行 + 序号对齐机制（补上通信算子断桥的修复路径）；
  ② 归因层第 8 类各细分标注判定工具（指向通信分析指南对应小节），实现 §6 的反向闭环；
- `SKILL.md`（02_bottleneck_analysis）Line B 第 4 步参考示例：增加通信溯源路径
  （"通信 wait 主导 → `--trace-source` 溯源 → 定位调用通信的 Python 函数 → 判断是否可
  overlap"）；工具映射层 multi_rank 行并入 communication 行；
- `README.md` Profiling 解析脚本表：`parse_multi_rank.py` 行并入 `parse_communication.py`
  行（双模式说明）。

## 10. 实施计划与性能预算

| 阶段 | 内容 | 工作量 | 新增耗时 |
|---|---|---|---|
| P1 | M1 溯源（trace 主路径 + CSV 序号 fallback + 域分组/重叠检测 + 总章上浮） | 主力工作 | trace 定向抽取 10–30s（条件触发）；CSV 路径 +1 次流式扫描 275MB ≈ 10s |
| P2 | M2 Matrix 深度分析 + M4 带宽自基准 + 对齐/RDMA/延迟主导小包（收编项） | 独立小改 | <1s |
| P3 | M1 瞬态污染防护（Top-K 头部位置标注 + H1 头部集中度一行，吸收原 M3）+ overlap 直接读列 | 小改 | <1s |
| P4 | M3 跨 Rank 对比（per-op wait join 慢卡定位 + per-rank 汇总/R_wait + 重叠/假性重叠 + alltoall 判读）——慢卡逻辑由 multi_rank 的 step_trace 总量对比简化为 per-op join | 中等 | 按 rank 数线性 |
| P5 | db 精确溯源路径（有 db 时自动升级置信度） | 可选 | <1s |
| P6 | **退役收尾**：删 `parse_multi_rank.py`；run_analysis I 节移除；thresholds `multi_rank` 节迁入删除；guide 重构；scripts_guide / SKILL / README 同步 | 收尾 | — |

**P1 验收标准**（用验证数据集做端到端验收）：
- 974/974 对齐一致；
- `hcom_allGather__612_546_1` 溯源输出 `parallel_utils.py(95): allgather_along_dim`；
- 总章首屏出现"通信疑点 Top-3 + 源码位置 + 置信度"；
- `Communication` 列与 Total Op Info 一致性检查通过（19461.8ms）。

**P4 验收标准**：
- per-op join 表定位 straggler，与 multi_rank 原 wait 不均衡受害者推断互为印证（一致性验证）；
- alltoall 倾斜判读与 kernel 耗时 CV 交叉验证输出一致的负载不均结论。

## 11. 边界与风险

- **名映射跨版本不稳定**：hcom→Hccl 前缀匹配 + 兜底聚合栈；p2p 场景（PP 流水）host
  调用名待实测样本补充映射；
- **多通信流并发**：按域分组 + 重叠率检测（§3.3 四层防线）；重叠率高且无 trace 时
  降级为聚合栈，不硬对；
- **溯源错误代价极高**（把 agent 带去错误源码位置）：宁可降级不硬对，置信度标记
  强制随行；
- **单 step 数据无法区分瞬态/稳态**：M1 头部标注只给位置标注与验证路径（warmup 核对/
  多 step 复现 / L0 对照），不下结论；
- **带宽自基准的盲区**：所有链路一起慢检测不出（相对基准原理限制），绝对表仅在
  配置后生效，文档需明示两者互补关系；
- **性能**：默认报告仅在"有显著通信疑点"时才付 trace 扫描成本；单卡/通信健康场景
  零新增成本；
- **退役过渡风险**：P4 收编完成前 multi_rank 与新 H7 并存，判据口径可能出现短暂分叉
  ——以本方案阈值表为唯一口径，P6 收尾前不修改 multi_rank 的任何判据（冻结待删）。

## 12. 关键设计决策记录（讨论沉淀）

| 决策 | 结论 | 依据 |
|---|---|---|
| 溯源主路径 | trace 为主、CSV 序号为 fallback、db 可选 | trace 是唯一可自校验的路径（ts 合理性）；CSV 无时间戳无法自证；db 实际场景通常不存在 |
| CSV 序号对齐的风险控制 | 域分组 + 重叠率检测 + 聚合栈兜底 | 实测重叠率 1.1%/6.9%，重叠≠乱序（974/974 仍成立），但需数据内检测 |
| 多流对齐失败怎么办 | 降级输出聚合栈分布，让 agent 用 Input Shapes 桥消歧 | 错误对齐的代价（带偏根因分析）远大于降级的信息损失 |
| 重叠分析怎么算 | 直接读 `step_trace_time.csv` 的 Overlapped/Communication/Comm(NotOvl) 三列，不推导 | 数据现成且与 communication.json 精确相等（19461.8ms），推导是绕路；收益上限 = Comm(NotOvl) 直接可读 |
| 时间轴分桶的定位 | ~~现象分类器 + 验证清单生成器~~ → v2.4 删除独立模块，保留 M1 的 Top-K 头部标注 | 直方图产出的信号在数据内无法关闭（区分瞬态/稳态的证据在文件外），行动价值低；必要的防护（防 Top 疑点是 warmup 鬼影）只需对 Top-K 加头部位置标注 |
| 带宽判据 | size 条件化 + 数据内自基准为主，芯片绝对表可选注入 | 带宽是算法分摊值（同 HCCS：132 vs 23.8GB/s），绝对物理峰值判据必然误报；写死数值导致单芯片可用 |
| matrix 倾斜判读 | 同 transport 内比较 + 排除小消息 + 按算子类型解读 | LOCAL 口径不同；~0MB 同步消息是算法结构非倾斜；allgather 倾斜=实现问题，alltoall 倾斜=负载不均（后者常是目标发现） |
| transport 合理性 | 只用数据内证据（组内不对称路由），不断言拓扑 | 文件不含物理拓扑，rank 可能在另一节点，断言"应走 HCCS"必误报 |
| Top-K 自动溯源的成本 | 条件触发（有显著疑点才扫 trace） | run_analysis D 节已全量扫 trace；H 节仅在需要时定向抽取，健康场景零成本 |
| **多卡分析归一**（v2.1） | parse_multi_rank 退役，通信能力并入 parse_communication `--all-ranks` | 多卡相对单卡的核心增量是通信；计算/Host 问题锁定 rank 后用单卡脚本（`--rank N`）分析；两套并行实现造成体系分裂与判据不一致（小包 1MB vs 32MB、带宽均值 30% 误报） |
| **慢卡定位的实现**（v2.2） | per-op wait join 直接从两个 json 得出，弃 step_trace 跨 rank 扫描层 | 集体通信中 wait 最小者即最后到达的被等者（straggler），逐算子粒度比总量对比更精确；multi_rank 的四层链路源自分布式训练心智模型，step_trace 扫描是粗筛 |
| **通信域的处理**（v2.2） | 按 `@域ID` 数据驱动分组，不做 DP/TP/PP/EP 策略推断 | 策略分类服务的是训练视角的可比性判断；join 按名分组天然不跨域误比，分组正确性由数据保证而非推断 |
| **MoE 不单列**（v2.2） | 并入 alltoall 倾斜判读规则，kernel CV 作交叉验证 | MoE 只是负载不均的 LLM 常见成因（验证集 AlphaFold3 手工切分同样触发）；原专项的优化建议（LB loss）是训练侧手段，超出推理优化范围 |
| **小包判定**（v2.3） | 只保留延迟主导一档，数据驱动：L̂ = 最小 size 档（<1MB）消息 transit 的 P50（按 transport 分组），transit < 3×L̂ 即延迟主导 | 小包 = 固定开销主导传输，边界 = L×带宽 随硬件漂移，写死 1/32MB 都是经验值（32MB 源自训练梯度分桶惯例，物理上非小包）；1–32MB 档检测的"未用满链路"与 M4 自基准重复，"粒度不够大"归策略性通信——单设固定阈值重复且误导方向 |
| **M3 时间轴降级**（v2.4） | 删除独立模块（wait 直方图 + 假设清单），必要产出并入 M1：Top-K 疑点头部位置标注 + 单条瞬态污染 [SIGNAL]；模块重编号 M4→M3、M5→M4 | M3 自己的评审结论即"现象描述器非归因器"——信号在当前数据内无法关闭，只能变成待验证清单，行动价值低；瞬态/稳态的真正防线在采集侧（warmup 规范）与验证侧（多 step 复现）；防追鬼影只需几行标注代码 |
| **R_wait 降级**（v2.5） | 只作画像/排序展示，删 30%（存在性）/50%（定性）两档判定 | 两档判定的问题都已有直接判据：存在性由 M3.1 per-op join 判定（附带 straggler 身份），"等待还是带宽"由 wait/transit 分解直接测量；R_wait 是间接量——单 rank 链路慢（transit 主导）同样推高 R_wait，50% 判据会把带宽问题误判为等待 |
| **重叠分析重构**（v2.6） | 低重叠降为现象测量，归因二分：串行型（可掩盖未掩盖，修异步流/调度）vs 依赖性串行（不可掩盖，修通信量或依赖结构）；触发以 Comm(NotOvl) 占比为准 | "重叠 < 5% → 调并行策略"隐含"总能并行"假设，真数据依赖（TP 逐层 allgather）下不成立——想并行也没法并行；且 comm 总量小时低重叠无意义。依赖性串行时 overlap 路线收益上限为 0，正路是减通信量（冗余/原语/拓扑）或改变依赖结构（策略性），两类收益口径完全不同，混淆会把候选排错序 |
| **一致性修订**（v2.7） | 删孤儿阈值 comm_cv_signal/comm_cv_definite（总量 CV 判定与 R_wait 同理归 M3.1）；head_wait_ratio 语义明确为"头部区段 wait 占比"（H1 一行）；补齐 unaligned_link_ratio / suspect_min_elapse_ms 键（正文数字不再脱离阈值表）；修正残留旧编号引用 | 全文自查：多轮修订后发现的残留引用（§11 的 M3、M2c 旧编号）与正文写死数字；阈值表必须与正文判定一一对应，否则实现时会出现两套口径 |

---

**相关文档**：
- 现有脚本：`model_opt/02_bottleneck_analysis/scripts/parse_communication.py`、`parse_multi_rank.py`（待退役）
- 分析工作流：`model_opt/02_bottleneck_analysis/SKILL.md`（Line B）+ `references/profiling_to_action.md`（三座桥 + 归因层）
- 多卡分析现状指南：`model_opt/02_bottleneck_analysis/references/multi_rank_analysis_guide.md`（待重构为通信分析指南）
- 官方数据说明：CANN「通信性能数据解析」（communication.json / communication_matrix.json 的生成与定位）
