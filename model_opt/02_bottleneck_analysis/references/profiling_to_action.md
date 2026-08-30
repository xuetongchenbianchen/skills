# Line B 分析路线：从现象到源码根因

> 方法论脊柱：**现象 → 归因 → 源码定位**。本文件只讲 Line B 自身的主分析（消费 profiling 数据定位到源码根因）；与 Line A 的联合分析（候选合并与量化、断桥回补）见 [merge_analysis.md](merge_analysis.md)。

```
现象层  观察到什么（表象 + 瓶颈类型）
  ↓ 为什么慢 / 属于哪类开销
归因层  属于哪类浪费（识别信号 + 量化上限）
  ↓ 哪行源码导致的
源码层  源码中的具体位置 + 根因分析
```

> 脚本已自动完成异常定位（Top-N 排序、DEFINITE/SIGNAL 信号、Suspect Kernels），agent 直接以脚本输出的显著信号为起点，不需要自行"找离群"。

> 推理纪律：**纵向不跳层**——沿脊柱逐层推进，先确认"确实是这个算子的问题"再深入下一层；**横向多源互证**——单信号判断模糊时（如利用率 50% 难判 host/device）从多个文件/维度交叉验证收敛，真问题会在多维度同时留痕（典型：L0 vs L1 交叉验证分离 profiler 注入开销、GPU vs NPU 跨平台对比定位差距环节）。

## 1. 现象层：观察到什么

只回答"看起来是什么问题"，不回答为什么。从 profiling 读出表象并判定瓶颈类型：

- 设备利用率低 + Free 大 → Host-Bound（host 喂不动设备）
- 利用率高 + mac 占主导 → Compute-Bound 嫌疑
- 利用率高 + mte 占主导 → Memory-Bound 嫌疑
- Communication(Not Overlapped) 占比高 → Comm-Bound 嫌疑
- empty_tensor 高频 + Free 大 → Allocator-Bound 嫌疑

瓶颈类型决定归因层往哪个方向查（Host-Bound 查 host 侧浪费、Compute-Bound 查 compute 饱和…）。判定用利用率/硬件占比/通信占比，具体阈值是负载相关默认值（见工具映射层），非普适判据。

## 2. 归因层：属于哪类浪费

把"慢"归到一类可消除的浪费。**十类浪费的定义与收益上限见 [waste_taxonomy.md](waste_taxonomy.md)（跨线统一分类）**；本节按类别给出 Line B 侧的识别信号（profiling 现象）与典型场景。

1. **① 显式同步开销**——信号：host 侧出现 D→H 同步类操作且占比高。典型：HuggingFace Trainer 每步 grad clip/NaN check 调 .item()
2. **② dispatch/调度开销**——信号：host dispatch 时间占比高、设备 idle 但 host 在框架层忙碌。归因时**必须产出分层构成**（数据来源：trace_view §0b Host 开销分层 + operator_details 按 Category 拆解），它是 03 编译门槛与工具选择的判定输入：
   - **②a Python/Module 机制层**（Module.__call__ / hook / __getattr__ / 解释器）——jit.script、flat forward 等 eager 手段可达
   - **②b aten dispatch 链**（aten::op → aclnnXxx 的逐算子调度，含 metadata op）——仅层次 3 图编译可达
   - **②c aclnn tiling/launch**（CANN host 侧参数计算与下发）——仅层次 3 图编译可达
   - 判定：②a 占比显著（如 ≥30%）→ eager 框架手段先行并测得终点再决定编译；②b/②c 主导 → eager 框架手段收益有限，直接评估图编译
3. **③ 内存管理阻塞**——信号：内存管理类 host 时间占比高、高频分配释放
4. **④ 在线编译/重编译**——信号：编译事件贯穿全程（非仅预热期）
5. **⑤ 内存带宽受限**——信号：mte 占比远大于 mac、带宽利用率接近峰值
6. **⑥ compute 饱和**——信号：mac 高 + 并行度满 + 利用率高（总可优化空间 <10% 时此类别为主）
7. **⑦ 布局/格式转换**——信号：非 ND 格式占比高 / Transpose·Cast 类耗时多
8. **⑧ 通信开销**——信号：通信 Wait 占比高，或 Transit 主导但带宽低于同 size 档基准。细分（判读规则与判定工具见 [profiling_scripts_guide.md](profiling_scripts_guide.md) §parse_communication，脚本输出自带提示）：
   - **等待型**——某 rank 计算侧慢导致其余 rank 等（straggler）；或同步点/barrier 过频
   - **冗余型**——同一张量被重复通信（短时间多次 AllGather、AG 结果大量丢弃）
   - **串行型**——通信可与独立计算重叠但调度串行；无独立计算则为依赖性串行，不可掩盖
   - **原语不当型**——集合通信原语选错（AllGather 收全量但只读 1/P）
   - **策略性**——通信量由分片策略本质决定（如收缩维度=分片维度），修法是重新分片而非加速通信
   - **拓扑错配型**——跨节点通信逻辑上可调整到节点内
   - **协议开销型**——消息被固定延迟主导（小包），修法是合并通信

   > 触发阈值：comm 列占总 step 时间 > 20%，或设备利用率 < 50% 且 comm 列有值，或多卡 wall-clock 加速比远低于理论。慢链路 / RDMA 重传疑似属环境侧问题——产出环境报告候选，不进代码优化候选。
9. **⑨ 小算子碎片**——信号：短 kernel 占比高、kernel 数极多
10. **⑩ 延迟未掩盖**——信号：device idle 但有可并行计算、多流并发占比低

### 当信号不匹配以上任何类别时

在 operator_details 中发现高 host self time 但不在上述 10 类的识别信号中时，按以下路径推理归因：

1. 用 `--filter <op_name>` 查看 call stack，判断是框架内部操作还是业务代码
2. 若是框架内部操作（TorchScript 函数名、aten:: 前缀、框架容器索引等）→ 归因到"dispatch/调度开销"（第 2 类），并按 ②a/②b/②c 细分到子层
3. 若是业务代码 → 检查是否属于框架调用链开销（Module.__call__、__getattr__），归因到"dispatch/调度开销"（第 2 类）的 ②a 子层
4. 若 device time 为 0 且不属于以上 → 检查是否为内存/元数据操作被错误归类，归因到"内存管理阻塞"（第 3 类）

## 3. 源码定位

Profiling 只能告诉你"哪个算子慢"，要动手优化必须先跨接到"源码里哪一行"。跨接依赖以下字段作"桥"——**桥全部依赖采集参数**：缺对应开关则字段不生成、桥即断裂，只能按降级路径做有限推断（采集参数见 [profiling_collection.md](../../01_preparation/references/profiling_collection.md)）。

### 桥接工具

| 桥 | 字段 / 文件 | 采集前提 | 作用 |
|----|------------|---------|------|
| **Call Stack** | `operator_details.csv` 的 `Call Stack` 列 | CPU activity + `with_stack=True` | 算子调用 → Python 源码行号（唯一直接映射） |
| **通信算子 Call Stack** | `Hccl*` 行/事件，经 `parse_communication.py --trace-source` 对齐回链 | 同上 | hcom 通信算子无直接行，经 host 侧 `Hccl*` 记录回链到调用源码 |
| **Input Shapes** | `kernel_details.csv` / `operator_details.csv` 的 `Input Shapes` 列 | `record_shapes=True` | 区分同一算子的不同调用点；断桥时反推计算语义 |
| **下发时序** | `trace_view.json` 的 flow / `connection_id` / `opCompile` 事件（`parse_trace_view.py`） | NPU 采集即有（源码映射另需 `with_stack`） | 下发链与编译停顿；自带 Call Stack，适合设备空等/下发延迟类问题 |

### 定位路径

- **设备侧问题**（某算子慢）：脚本信号 → `--filter` 该算子 → Call Stack 定位源码行；同一算子多调用点用 Input Shapes 区分。
- **host-device 交互问题**（设备空闲/下发延迟/编译停顿）：信号来自 step_trace（Free 大）与 trace_view（Bound Regions / idle 成因）；入口用**下发时序**——`connection_id` 配对定位"设备在等哪个 host 操作"，再从该 host 操作的 Call Stack 落到源码。

**通用规则**：
- **断桥**（"(no stack)"，框架 codegen 算子）：先 Input Shapes 反推语义；反推不出移交合并阶段（[merge_analysis.md](merge_analysis.md)「断桥与回补」）
- **伪影验证**：异步流水线（TASK_QUEUE_ENABLE=2）下 host-device gap 可能是 profiler 伪影，先 L0 交叉验证再作因果推断

例：`op_statistic` 发现 Transpose 占 15% → `operator_details --filter Transpose` 用 Call Stack 定位源码行 → `kernel_details --filter` 用 Input Shapes 确认调用点 → 产出根因追踪记录。

## 脚本信息不够时的深入方法

当脚本输出不足或桥梁断裂时，直接读原始文件：

| 想了解什么 | 去哪里 | 看什么 |
|---|---|---|
| 某算子实际 input shape | kernel_details.csv | Input Shapes 列 |
| 某次分配时系统内存多满 | operator_memory.csv | Allocation Total Allocated(MB) |
| 某算子在 forward 中的位置序列 | kernel_details.csv | 按 Start Time 排序搜目标算子 |
| 完整 Python 调用链 | operator_details.csv | Call Stack 列 |
| 两个 kernel 间真实 gap | kernel_details.csv | Start Time − 上一kernel 的(Start+Duration) |
| 某 step 独立数据 | kernel_details.csv | 按 Step Id 列过滤 |
| L0 vs L1 采集差异 | 两份 profiling | 分别跑脚本对比，L1 bubble 可能被 profiler barrier 夸大 |
