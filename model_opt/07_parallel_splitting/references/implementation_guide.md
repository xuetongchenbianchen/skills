# 实施与验证指南

> 覆盖分析通过后的实施、验证与调试。分析阶段见 [analysis_workflow.md](analysis_workflow.md)。

---

## 实施模式

四种模式按侵入度从低到高，根据项目情况选择。

| 模式 | 侵入度 | 适用场景 | 回退 | 单卡兼容 |
|------|--------|---------|------|---------|
| A 外部编排 | 最低 | 输入处切 batch 维 | 换脚本 | 天然 |
| B Monkey-Patch | 中 | 输入处切 seq 维 / 中间模块切分 | disable | 天然 |
| C 子类覆写 | 中高 | 并行逻辑与原逻辑差异大 | 换类 | 天然 |
| D 源码内嵌 | 高 | 长期核心需求 | git revert | 需显式保证 |

- **A**：模型 forward 外层做样本分发与 gather，模型侧最小改动
- **B**：源码不改，启动脚本注册并行版 forward
- **C**：继承基类覆写需要并行的方法
- **D**：`if is_parallel(): return self._forward_parallel(...)`，需保证 `is_parallel()=False` 时行为不变

---

## 通信原语

### 基础原语

基础通信原语（all_gather, all_reduce, reduce, reduce_scatter, broadcast, gather, scatter, all_to_all）及便捷函数、通信-计算重叠、自检，已实现为可直接 copy 的脚本：

→ [scripts/comm_primitives.py](../scripts/comm_primitives.py)

copy 到项目 `comm/comm_primitives.py`，`from comm.comm_primitives import init_distributed, ParallelConfig, ...`。所有原语在 `world_size=1` 时 no-op。

### Monkey-Patch 生命周期

基础原语是通用的，如何把并行 forward 接入模型取决于项目结构。核心三件套：

- `register_patch(cls, method_name, parallel_fn)` — 保存原始 forward，登记并行版
- `enable_parallel()` — 替换为并行版（ws=1 时 no-op）
- `disable_parallel()` — 恢复原始（可逆回退）

补充：`set_parallel_flag(enabled)` 仅切 flag 不触碰 patch（模块级 guard），`cleanup_all_patches()` 彻底清理（不可恢复）。

### 混合并行 ProcessGroupManager

多策略叠加（如切权重 × 切数据维度）时需创建子通信组。各维度的 rank 分配：切权重维度的 group 取连续 rank 段，切 batch 维度的 group 需含 `rank % (tp×pp)` 偏移（固定步长会导致 `new_group` 报错），切层维度的 group 取相同 stage 的 rank。

> 所有原语接受 `group` 参数，混合并行时 ws/rank 按子组计算。

---

## NPU 关键差异

| 项目 | NPU | GPU |
|------|-----|-----|
| 通信后端 | `hccl` | `nccl` |
| 设备字符串 | `'npu'`/`'npu:0'` | `'cuda'` |
| 同步 | `torch.npu.synchronize()` | `torch.cuda.synchronize()` |
| AllToAll | chunk 后必须 `.contiguous()` | 通常不需要 |
| 通信流 | `torch.npu.Stream()` | `torch.cuda.Stream()` |
| 碎片优化 | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | 同名但不同值 |

启动：source CANN 环境脚本 + `ASCEND_RT_VISIBLE_DEVICES` 与 `--nproc_per_node` 一致 + 跨节点 `HCCL_CONNECT_TIMEOUT=600`。

---

## 权重处理

切分后，被切维度上的权重需要按 rank 取对应分片，未被切的权重在各 rank 间复制。具体哪些权重需要切、沿哪个维度切，由 `tensor_distribution_table` 决定。

**原则**：逐权重对照 `tensor_distribution_table` —— 在切分维度上的权重按 `rank` 取切片，不在切分维度上的权重整份加载。合并存储的权重需按合并方式确定 slice 顺序。加载后遍历子模块触发后处理（量化 scale 调整等）。

**在线切分**（推荐）：各 rank 读同一份完整 checkpoint，按 `rank` 自动加载切片。在每个并行模块中实现 `weight_loader()`。

**离线转换**（备选）：预切权重输出为 `rank_0/` ~ `rank_N/` 目录。与切分配置绑定，改策略或 world_size 需重新转换。

> 切分维度、切片大小必须与 evidence_db 中 `parallel_splitting.tensor_distribution_table` 一致。

---

## 正确性验证

> ★ 验证必须基于模板脚本 [scripts/verify_split.py](../scripts/verify_split.py)：agent 填写 `build_model` / `build_sample` / `run_inference` / `enable_parallel` 四个函数，禁止重写比较框架（`compare_outputs` / `bit_exact_diff` / 分层逻辑）。防作弊设计见脚本头部注释。

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

> 理论依据：浮点加法不满足结合律，切分改变归约顺序导致数值差异（BF16 典型误差 10⁻⁵ 级）。切可切维（如 batch）不改变归约顺序→误差极低；切归约维（如权重输出维、归约维）改变聚合顺序→误差增大。见 [analysis_workflow.md](analysis_workflow.md)「通信设计」节。

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

验证通过 → ★ 确认提交。未通过 → 回第四步。

---

## 并行 Profiling 采集

切分验证通过后回归 Phase 1 主线时，**带 profiling 的并行推理脚本需按并行方式构建**（否则后续无法采集数据）。该脚本在 Phase 1 完成后供 Phase 2/4 使用。

**构建原则**：复用 Phase 1 脚本骨架（profiler 参数、路径规范不变），只改推理入口和启动方式。复用第六步的 `init_distributed()` + `enable_parallel()`。

| 维度 | Phase 1 单卡版 | 并行版 |
|------|---------------|--------|
| 启动 | `python script.py` | `torchrun --nproc_per_node=N script.py` |
| 推理入口 | `run_inference(model, input_data)` | `run_inference_parallel(model, input_data, cfg)` |
| 输出路径 | `profiling/<timestamp>/` | `profiling/<timestamp>/rank_<n>/` |
| wall-clock | 单进程取中位数 | 各 rank 独立计时，取 max 作为端到端时间 |

推荐直接覆写 Phase 1 原脚本（保留 git 历史），用 `disable_parallel()` 可还原单卡行为。L0 多卡目录用 `run_analysis.py --rank N` 分析。

---

## 调试排查

### 问题分类与归属

| 问题类型 | 归属 | 示例 |
|---------|------|------|
| 环境/部署 | Phase 1 | HCCL 初始化失败、连接超时 |
| 切分方法 | 本阶段 | 通信死锁、shape 不匹配 |
| 性能 | Phase 3 | 并行比单卡慢 |

### 排查流程

1. `self_test()`：验证通信原语无问题
2. 冒烟测试（最小输入）：定位逻辑错还是规模问题
3. `py-spy dump`：死锁时抓各 rank 栈
4. 打印 shape：对照 `tensor_distribution_table`
5. 单卡 vs 多卡 tolerance 对比：未通过用 bit-exact 定位

### 切分方法常见问题

- **通信死锁**：各 rank 通信原语数量/顺序不一致 → 对照 `tensor_distribution_table` 确保对称
- **Shape 不匹配**：切了主导张量没切配套 mask/index → 未声明的张量视为潜在 bug
- **并行开关误触发**：全局 `is_parallel()` 被未切分模块检测 → 用模块级 guard

---

> 通信成为瓶颈时，用 Phase 2 的 profiling 分析定位通信低效类型（见 [profiling_to_action.md](../../02_bottleneck_analysis/references/profiling_to_action.md) 归因层 #8），按四维度框架选择优化手段。诊断核心读四个量：通信占比（切分是否过度）、Not-Overlapped 通信（重叠是否充分）、带宽利用率（链路是否次优）、空闲气泡（调度是否失配）。
