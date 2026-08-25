# 实施与验证指南

> 覆盖分析通过后的实施、验证、记录回归与调试，以及回归 Phase 1 的并行 profiling 采集。分析阶段见 [analysis_workflow.md](analysis_workflow.md)。

---

## 实施主线路径

第四步按以下顺序推进（各节详述）：

1. **接入决策**：切分落在输入边界还是模型内部（→「切分怎么接入模型」）
2. **通信基建**：copy 基座 `comm_primitives.py`，按张量分布约定表从 `comm_recipes.py` 取用组合函数（→「通信原语」）
3. **实现切分**：按分析的 `tensor_distribution_table` 逐模块实现并行 forward，权重按分布表切片加载（→「切分怎么接入模型」「权重处理」）
4. **启动冒烟**：`torchrun --nproc_per_node=N` 启动——先 source CANN 环境脚本（见 [00 环境参考](../../00_adaptation/references/environment_reference.md)，含环境变量清单），`ASCEND_RT_VISIBLE_DEVICES` 与 `--nproc_per_node` 一致；最小输入跑通 + shape 断言（冒烟即验证 Tier 1）
5. **分层验证**：verify_split.py 单卡 vs 多卡（→「正确性验证」）

## 切分怎么接入模型

切分代码落在哪里，由切分发生在哪一级决定：

**输入边界——模型零改动**：batch 维可切时（叠加 DP 扩吞吐），外层脚本分发样本、收集输出，模型完全不改，天然单卡兼容，换脚本即回退。

**模型内部——替换相应模块的 forward**：切 seq / 权重 / 层 / 分支都要进入模型内部。并行版 forward 的写法只取决于张量分布表（逐张量配通信原语），剩下的差别只在**替换机制**——工程选择，不是分析决策：

| 替换机制 | 适用 | 回退 |
|---------|------|------|
| Monkey-Patch（默认） | 源码不改或不宜改，绝大多数场景 | `disable_parallel()` 可逆 |
| 子类覆写 | 并行逻辑与原逻辑差异大 | 换类 |
| 源码内嵌 | 长期核心需求 | git revert；需保证 `is_parallel()=False` 时行为不变 |

**接入约定**（项目侧自行实现，`verify_split.py` 与 evidence_db 按此对接）：patch 怎么写是通用 Python 能力，不在此展开；需要遵守的是三条契约——
1. 提供 `enable_parallel()` / `disable_parallel()` **可逆对**：patch 前保存原始 forward 引用，disable 时恢复
2. `world_size=1` 时 enable 为 no-op（单卡兼容）
3. 切分开关用**模块级 guard**，未切分模块不检测全局 flag（避免误触发，见「调试排查」）

**全程保持切分（进入/退出决策）**：进入时每卡从完整张量取本地分片（零拷贝 slice），或 rank 0 scatter 到各卡（节省初始内存）——不假设输入已是本地分片，跨迭代场景首轮可能是全局 `[N, ...]`、后续轮已是本地 `[N/P, ...]`，用 shape 显式判断（`shape[0] == N`）后取片。退出时尽量不退——下游模块能在切分数据上运行就继续传递本地分片；个别操作必须用全局数据（如需要完整二维索引的 gather）时临时 AllGather、用完立即释放；仅在最终输出时 AllGather 回全量。避免在中间重建 GB 级全局张量。

---

## 通信原语

三个文件、三种使用方式——**使用方式决定文件形态**：

| 文件 | 内容 | 使用方式 |
|------|------|---------|
| [scripts/comm_primitives.py](../scripts/comm_primitives.py) | 通用基座：基础原语 1:1 对应 torch.distributed，ws=1 → no-op，NPU → 自动 `.contiguous()` | **整文件 copy** 到项目 `comm/comm_primitives.py` |
| [scripts/comm_recipes.py](../scripts/comm_recipes.py) | 切分场景组合：沿维聚合/取片（`allgather_along_dim`、`local_chunk` 等）、分片维切换（`row_to_col`/`col_to_row`）、重叠工具（`issue_on_comm_stream`/`wait_comm`，供回归主流程后的 hide_latency 优化取用，切分实施阶段不引入） | **按需取用**——按张量分布约定表挑选函数 copy 进项目 comm 模块，import 行改为项目路径 |
| [scripts/comm_self_test.py](../scripts/comm_self_test.py) | 环境自检：验证多卡通信本身通不通（基础原语 + 组合函数往返） | **原地运行，不 copy**：`torchrun --nproc_per_node=N comm_self_test.py` |

copy 基座后：`from comm.comm_primitives import init_distributed, ParallelConfig, ...`。所有原语在 `world_size=1` 时 no-op，均接受 `group` 参数（混合并行时 ws/rank 按子组计算）。

### 混合并行的通信组构建

多策略叠加（如切权重 × 切数据维度）时需创建子通信组。各维度的 rank 分配：切权重维度的 group 取连续 rank 段，切 batch 维度的 group 需含 `rank % (tp×pp)` 偏移（固定步长会导致 `new_group` 报错），切层维度的 group 取相同 stage 的 rank。

> rank 排列的性能约束：高频通信的组（切权重维度，每层集合通信）放节点内（HCCS 高带宽），跨节点链路留给低频通信。

---

## 权重处理

切分后，被切维度上的权重需要按 rank 取对应分片，未被切的权重在各 rank 间复制。具体哪些权重需要切、沿哪个维度切，由 `tensor_distribution_table` 决定。

**原则**：逐权重对照 `tensor_distribution_table` —— 在切分维度上的权重按 `rank` 取切片，不在切分维度上的权重整份加载。合并存储的权重需按合并方式确定 slice 顺序。加载后遍历子模块触发后处理（量化 scale 调整等）。

**在线切分**（推荐）：各 rank 读同一份完整 checkpoint，按 `rank` 自动加载切片。在每个并行模块中实现 `weight_loader()`。

**离线转换**（备选）：预切权重输出为 `rank_0/` ~ `rank_N/` 目录。与切分配置绑定，改策略或 world_size 需重新转换。

> 切分维度、切片大小必须与 evidence_db 中 `parallel_splitting.tensor_distribution_table` 一致。

---

## 正确性验证

> 方案的数学等价性在分析阶段论证（★ 方案确认）；本节验证实施正确性与浮点容差，不可跳过。

> ★ 验证必须基于模板脚本 [scripts/verify_split.py](../scripts/verify_split.py)：agent 填写 `build_model` / `build_sample` / `run_inference` / `enable_parallel` 四个函数，禁止重写比较框架（`compare_outputs` / `bit_exact_diff` / 分层逻辑）。防作弊设计见脚本头部注释。

**baseline 不可得时（权重本身超单卡）**：按超限原因选替代方案——

| 超限原因 | baseline 替代方案 | 验证的是什么 |
|---------|-----------------|------------|
| 激活/持久状态主导 | 缩小输入，单卡与多卡跑相同小输入 | 同一模型同一输入下的切分正确性 |
| 权重本身超单卡 | 缩小模型（减层/减宽、保持同一切分逻辑），单卡与多卡跑同缩配版 | 切分逻辑正确性 |
| 权重本身超单卡 | 逐层对比：单层加载完整权重比中间量 | 每层切分实施正确性 |
| （辅证） | 多配置互证：P=2 与 P=4 对比 | 随 P 变化的 bug（不作主证） |

缩配 baseline 在 `build_model` 中参数化缩小配置，验证记录注明缩配参数。

**使用流程**：
1. copy `scripts/verify_split.py` 到项目根目录
2. 填写 4 个函数：`build_model` / `build_sample` / `run_inference` / `enable_parallel`
3. 切分前：`python verify_split.py --mode baseline --split-type <order_preserved|order_changed>`
4. 切分后：`torchrun --nproc_per_node=N verify_split.py --mode verify --split-type <同上>`
5. 检查 `verify_report.json` 中 `overall=true` 才算通过

**核心原则**：tolerance 是通过门禁，bit-exact 仅 debug 工具。

| 方法 | 用途 |
|------|------|
| tolerance（首选） | 单卡 vs 多卡 `allclose(atol, rtol)` |
| bit-exact（仅 debug） | tolerance 未通过时逐元素定位差异 |
| 领域指标（可选） | 结构化输出用领域特定指标 |

### 验收标准

> 理论依据：浮点加法不满足结合律，切分改变归约顺序导致数值差异（BF16 典型误差 10⁻⁵ 级）。切可切维（如 batch）不改变归约顺序→误差极低；切归约维（如权重输出维、归约维）改变聚合顺序→误差增大。见 [analysis_workflow.md](analysis_workflow.md) 第三步「等价性论证」。

| 切分影响 | 精度风险 | 验收 |
|---------|---------|------|
| 不改变计算顺序（切 batch / fp32 切 seq / 切层） | 极低 | allclose(atol=1e-6) |
| 改变累加顺序/聚合方式（bf16 切 seq / 切权重 / 切专家） | 中 | allclose(atol=1e-3) |

> 具体容差应根据模型精度要求和数值范围调整。若模型用于评测或 RL（对数值一致性敏感），需收紧容差或固定归约顺序。

### 验证分层

| Tier | 内容 | 目的 |
|------|------|------|
| 1 冒烟 | 最小输入跑通 + shape 断言 | 逻辑不报错 |
| 2a | 最小输入单卡 vs 多卡 tolerance | 快速定位逻辑错误 |
| 2b | 真实规模多配置 tolerance | 最终通过标准 |

### 对比方法

将单卡和多卡在相同输入下的输出转为可比较的格式后比较：
- 张量：直接 `allclose`
- 嵌套结构（dict/list）：递归逐 key 比较
- 结构化数据：按需选择领域特定指标，阈值在比较前声明

tolerance 未通过时，用最小输入做 bit-exact 逐元素对比定位差异来源。

验证通过 → ★ 确认提交。未通过 → 回 [analysis_workflow.md](analysis_workflow.md) 修正切分方案（第二步维度选择 / 第三步等价性论证），再重新实施。

## 记录与回归（验证通过后）

★ 提交确认通过后，将完整案例记录到 evidence_db——分析阶段的内存时间线、切分方案与张量分布表、定量估算与等价性论证、验证结果（verify_report 摘要 + 缩配参数若有）、known_pitfalls。Schema 见 [06_evidence_db/schema.md](../../06_evidence_db/schema.md) 的 `parallel_splitting` 顶层字段——切分案例与优化案例分立记录，后续优化案例用 `depends_on` 指向本案例。

记录完成后回归主流程：Phase 0 完成适配精度验证（golden 对齐，多卡配置下执行）→ Phase 1 采集并行基线（L0/wall-clock；多卡采集适配见 [01_preparation 的 profiling_collection](../../01_preparation/references/profiling_collection.md)「多卡并行推理采集」）→ 进入 Phase 2 瓶颈分析。

---

## 调试排查

### 问题归属

| 问题 | 归属 |
|------|------|
| HCCL 初始化失败、连接超时等环境问题 | Phase 0（[00_adaptation](../../00_adaptation/SKILL.md) 0.1 环境准备） |
| 通信死锁、shape 不匹配等切分问题 | 本阶段（已知坑见下） |
| 并行比单卡慢 | 回归 Phase 2（见「性能问题归属」） |

排查第一步：原地运行 `comm_self_test.py`（`torchrun --nproc_per_node=N`）——通过则问题在切分实现逻辑，失败则是环境/部署问题。之后的通用排查（最小输入冒烟、打印 shape、抓栈）属常规调试能力，不在此展开；切分逻辑错误的排查 = 对照 `tensor_distribution_table` 找实现与契约的偏差。

### 性能问题归属

并行比单卡慢时，回归主流程 Phase 2：`run_analysis.py` 的 H 节自动做跨 Rank 通信分析（straggler / 重叠归因二分 / 慢链路自基准），按 [profiling_to_action.md](../../02_bottleneck_analysis/references/profiling_to_action.md) 归因层 #8 细分问题类型（切分过度 → 策略性，回本子技能调整方案；重叠不足 → 串行型，走四维度掩盖；链路次优 → 拓扑错配型），再选择对应优化手段。
