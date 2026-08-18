# 实施与验证指南

> 覆盖分析通过后的实施、验证、归档与调试。分析阶段见 [analysis_workflow.md](analysis_workflow.md)。

---

## 实施模式

四种模式按侵入度从低到高，根据项目情况选择。

| 模式 | 侵入度 | 适用策略 | 回退 | 单卡兼容 |
|------|--------|---------|------|---------|
| A 外部编排 | 最低 | DP | 换脚本 | 天然 |
| B Monkey-Patch | 中 | SP/TP/EP | disable | 天然 |
| C 子类覆写 | 中高 | 复杂组合 | 换类 | 天然 |
| D 源码内嵌 | 高 | 深度优化 | git revert | 需显式保证 |

### 模式 A：外部编排（DP 首选）

通信只在首尾，模型 forward 外层做样本分发与 gather。模型侧最小改动：给迭代方法加 `rank`/`world_size`/`seed` 可选参数（默认单卡行为不变）。

### 模式 B：Monkey-Patch（SP/TP/EP）

源码不改，启动脚本注册并行版 forward（生命周期见下方通信原语模板）。

### 模式 C：子类覆写

并行逻辑与原逻辑差异大时，继承基类覆写需要并行的方法。

### 模式 D：源码内嵌

并行是长期核心需求时直接改源码：`if is_parallel(): return self._forward_parallel(...)`。可做算子级优化（通信-计算重叠、GEMM 融合）。需保证 `is_parallel()=False` 时行为不变。

---

## 通信原语模板

> 从模板裁剪适配到具体项目，不要直接复制运行。所有原语在 world_size=1 时 no-op，保证单卡路径零改动。

```
# 全局配置单例 ParallelConfig（单例，get()/reset()）
#   is_parallel / rank / world_size / device('npu'|'cuda'|'cpu') / group / comm_stream
#
# init_distributed(backend='hccl'):
#   try: import torch_npu           # ★必须先 import 再探测，否则 torch.npu.is_available() 恒 False
#   except: pass                    #   （hccl 后端由 torch_npu 注册，提前导入无副作用）
#   if WORLD_SIZE not in env or == 1:  # 单卡 no-op
#       set_device(0); cfg.is_parallel=False; return
#   local_rank = LOCAL_RANK
#   if has_npu:  set_device(local_rank); device='npu'   # backend 保持 'hccl'
#   elif has_cuda: set_device(local_rank); device='cuda'; backend='nccl'  # 自动切换
#   else: device='cpu'; backend='gloo'
#   dist.init_process_group(backend)
#   cfg.rank, cfg.world_size = dist.get_rank(), dist.get_world_size()
#   cfg.is_parallel = (world_size > 1)
#   if is_parallel and device=='npu': cfg.comm_stream = torch.npu.Stream()  # 通信流

# --- 基础原语（world_size=1 → 直接返回输入）---
#
# allgather_along_dim(t, dim=0, group=None):  [L/P,..]→[L,..]
#   dim==0: 用 all_gather_into_tensor（单 buffer，峰值内存减半，避免全量 buffer 抵消切分收益）
#   其他维度: 回退 list + cat
#
# scatter_along_dim(t, dim=0, group=None):    [L,..]→[L/P,..]
#   t.chunk(ws,dim)[rank]，contiguous 时 clone()，非 contiguous 时 .contiguous()
#
# allreduce_inplace(t, op=SUM, group=None):   dist.all_reduce(t.contiguous())
#
# reduce_scatter_along_dim(t, dim=0, op=SUM, group=None):  AllReduce+slice 等价，通信量减半
#   dim==0: reduce_scatter(output, input_list=t.chunk(ws,dim=0))
#   其他维度: permute→dim0→reduce_scatter→permute_back
#   ★用于 TriMul Incoming 等 AllReduce 后立刻取本地行的场景
#
# broadcast_from_rank0(t, group=None):        dist.broadcast(t.contiguous(), src=0)
#
# gather_to_rank0(t, group=None):             rank0 返回 cat; 其他返回 None
#   ★group 参数：混合并行时传入子组，ws/rank 按子组计算而非全局

# --- AllToAll 维度转换（★chunk 后必须 .contiguous()）---
#
# row_to_col(t, group=None):  [N/P,N,c] → chunk dim=1 → all_to_all → cat dim=0 → [N,N/P,c]
# col_to_row(t, group=None):  [N,N/P,c] → chunk dim=0 → all_to_all → cat dim=1 → [N/P,N,c]
# alltoall_swap_dims(t, dim_partition, dim_full, group=None):
#   通用版：将分片从 dim_partition 移到 dim_full
#   chunk dim_full → all_to_all → cat dim_partition

# --- 通信-计算重叠（NPU 专用）---
#
# issue_on_comm_stream(comm_fn):
#   在独立通信流上执行 comm_fn，返回 event 供计算流等待
#   if comm_stream is None: comm_fn(); return None     # 单卡或非 NPU 直接执行
#   comm_stream.wait_stream(current_stream)            # 等数据就绪
#   with stream(comm_stream): comm_fn()
#   event = Event(); event.record(comm_stream); return event
#
# wait_comm(event):
#   if event is not None: current_stream.wait_event(event)
#
# 典型用法:
#   evt = issue_on_comm_stream(lambda: all_gather_into_tensor(out, t))
#   # ... 计算流上做不依赖通信结果的工作 ...
#   wait_comm(evt)  # 现在可以使用通信结果

# --- Monkey-Patch 生命周期 ---
#
# register_patch(cls, method_name, parallel_fn):
#   保存原始 forward → 登记 parallel_fn（不立即替换）
#   setattr(cls, f'_original_{method_name}', original)  # 供并行版内部调用
#
# unregister_patch(cls, method_name):
#   恢复原始 forward，删除 _original_ 属性（精确移除单个 patch）
#
# enable_parallel():                替换为 fn（ws=1 时 no-op）
# disable_parallel():               恢复原始（可恢复，临时回退单卡路径）
# restore_parallel():               等价 enable_parallel()（从 disable 状态恢复）
# set_parallel_flag(enabled):       仅切 flag，不触碰 patch（模块级 guard 用）
# cleanup_all_patches():            彻底清理（不可恢复，回滚时调用）

# --- 混合并行 ProcessGroupManager ---
#
# init_groups(tp_degree, dp_degree, pp_degree):
#   assert tp×dp×pp == world_size
#   rank = dp_rank×tp×pp + pp_rank×tp + tp_rank
#
#   tp_group: 连续 rank 段 [r//tp×tp, r//tp×tp+tp)
#   dp_group: ★offset = rank % (tp×pp)  修正！
#             dp_ranks = [d × tp×pp + offset  for d in range(dp_degree)]
#             旧实现用固定 {0, tpp, 2×tpp, ...} 对所有 rank 都构造同一份，
#             不含 offset!=0 的 rank 自身 → new_group 报错/组语义错乱
#   pp_group: 相同 stage：(r // tp) % pp 相同的 rank
#
# 属性: tp_rank = rank % tp_degree | dp_rank = rank // (tp×pp) | pp_rank = (rank//tp) % pp
# ★所有原语接受 group 参数，混合并行时 ws/rank 按子组计算

# --- 自检 ---
#
# self_test():  验证基础原语在当前环境下工作正常
#   用法: torchrun --nproc_per_node=4 -c \
#     "from comm_primitives import init_distributed, self_test; init_distributed(); self_test()"
#   检查项（各 rank 打印 PASS/FAIL）:
#     1. allgather:         各 rank 填不同值，gather 后验证顺序
#     2. scatter/allgather:  往返一致性
#     3. row_to_col/col_to_row:  往返 allclose(atol=1e-6)
#     4. allreduce:          各 rank 填 rank+1，reduce 后验证总和
#     5. broadcast:          rank0 填 42.0，验证传播
#     6. gather_to_rank0:    rank0 验证 cat 顺序
#   ★搭建通信 infra 后第一步必跑，排查 #5(未 import torch_npu) / #3(contiguous) 等高频坑
```

---

## 昇腾 NPU 关键差异

| 项目 | NPU | GPU |
|------|-----|-----|
| 后端 | `hccl` | `nccl` |
| 设备字符串 | `'npu'`/`'npu:0'` | `'cuda'` |
| 同步 | `torch.npu.synchronize()` | `torch.cuda.synchronize()` |
| AllToAll | **必须 `.contiguous()`** | 通常不需要 |
| 通信流 | `torch.npu.Stream()` + `Event` | 同 API，`cuda.Stream` |
| ReduceScatter | `dist.reduce_scatter`（HCCL 支持） | 同 |
| 碎片优化 | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | 同名但不同值 |

启动：source CANN 环境脚本 + `ASCEND_RT_VISIBLE_DEVICES` 与 `--nproc_per_node` 一致 + 跨节点 `HCCL_CONNECT_TIMEOUT=600`。多卡 DP 需控三套 RNG（`torch`/`random`/`np.random`）。

---

## 编排脚本骨架

```python
import os, torch, torch.distributed as dist
dist.init_process_group('hccl')
rank, world_size = dist.get_rank(), dist.get_world_size()
torch.npu.set_device(int(os.environ['LOCAL_RANK']))
model = load_model(...)
runner = ParallelRunner(model, rank, world_size)
for batch in dataloader:
    result = runner.run(batch, seed)
    if rank == 0: save_output(result)
```

启动：`torchrun --nproc_per_node=N run_parallel.py`

---

## 权重处理

切分后权重的加载方式是每个并行项目都要面对的。两种方式按项目情况选择。

### 在线权重切分（推荐）

运行时各 rank 读同一份完整 checkpoint，按 `rank` 自动加载对应切片。在每个并行模块中实现 `weight_loader()`：

- **TP 权重加载**：按 `tp_rank` 取对应切片。切输出维的层取列切片，切输入维的层取行切片
- **EP 权重加载**（MoE）：按 `ep_rank` 过滤专家——只保留本卡负责的专家权重，丢弃其他
- **合并层**（多个权重合并存储时）：需特殊处理 slice 顺序，按合并方式交错取而非连续取
- **加载后处理**：遍历子模块触发权重后处理（量化 scale 调整、格式转换等）

```
# 模型类实现 load_weights()：遍历权重文件 → 匹配到各模块 weight_loader()
# 模型类实现 process_weights_after_loading()：
#   for _, module in self.named_modules():
#       post = getattr(module, 'process_weights_after_loading', None)
#       if post is not None: post()
```

> 切分维度、切片大小必须与分析 JSON 的 `tensor_distribution_table` 一致。

### 离线权重转换（备选）

预切权重用于离线部署，输出为 `rank_0/` ~ `rank_N/` 目录结构，每个 rank 只包含该卡需要的权重切片。

```
# 转换脚本伪代码：
#   for rank in range(world_size):
#       for name, param in model.state_dict():
#           sliced = parallel_module.weight_loader(param, rank=rank)
#           save(sliced, f"rank_{rank}/{name}")
```

> 离线预切的权重与切分配置绑定。改了策略或 world_size 必须重新转换。

---

## 正确性验证

核心原则：**tolerance 是通过门禁，bit-exact 仅 debug 工具**。

- **tolerance**（首选）：单卡 vs 多卡 `allclose(atol, rtol)` 判定
- **bit-exact**（仅 debug）：tolerance 未通过时，最小输入逐元素定位差异
- **领域指标**（可选）：结构化输出用领域特定指标替代逐元素对比

### 策略精度风险与验收标准

| 策略 | 精度风险 | 验收标准 |
|------|---------|---------|
| DP / SP fp32 / PP | 极低 | allclose(atol=1e-6) |
| SP bf16 / TP / EP | 累加顺序/聚合/路由变化 | allclose(atol=1e-3) |

> 具体容差应根据模型精度要求和数值范围调整。

### 验证分层（Tier）

| Tier | 内容 | 目的 |
|------|------|------|
| Tier 1 冒烟 | 最小输入跑通 + shape 断言 | 逻辑不报错 |
| Tier 2a | 最小输入单卡 vs 多卡 tolerance | 快速定位逻辑错误 |
| Tier 2b | 真实规模多配置 tolerance | 最终通过标准 |

验证通过 → ★ 确认提交（用 `ask_user_question` 展示验证结果 + 性能收益 + ADR 路径）。未通过 → 回第四步分析（排查见下方调试节）。

### 验证脚本模板

```
# 输出加载: .npy→np.load | .pkl→pickle+tensor→numpy | .pt→torch.load(cpu)+tensor→numpy
# 返回 numpy 数组或 dict（值已转 numpy）

# tolerance（首选）: dict→递归逐key | else→np.allclose + max/mean abs diff
# bit-exact（仅debug）: dict→递归逐key | else→np.array_equal + n_different
# 领域指标（按需实现）: extract(single,key) vs extract(parallel,key) → compute_fn → threshold
#   常见 compute_fn: Kabsch RMSD / cosine similarity / spatial RMSE+ACC
```

---

## ADR 模板

```markdown
# ADR-NNNN: {模型} {规模} {卡数}卡 {策略}

## 场景约束
- 模型/规模/目标/硬件

## Phase 0 分诊
- 外部因素检查结果 / 显存估算 + 判定

## 内存时间线
| 模块 | 峰值 | OOM? | 备注 |

## 定量估算
- 切分后每卡显存 / 通信量 / break-even / 不推荐方案原因

## JSON 审查
- 策略 / 张量分布表 / 公理合规

## 实施摘要
- 模式 / 原语 / 冒烟结果

## 验证结果
- Tier 2a/2b 结果 / 性能

## 回退方案
- disable_parallel() / cleanup_all_patches() / git revert

## 未采纳方案及原因
```

---

## 调试排查

### 常见坑（按频率排序）

| # | 问题 | 根因 | 解法 | 预防 |
|---|------|------|------|------|
| 1 | 集合通信死锁（hang 无报错） | 各 rank 通信原语数量/顺序不一致 | 全 rank 必须调用；`py-spy dump` 抓栈对比 | 先跑 `self_test()` 验证原语；写代码时对照张量分布表确保每 rank 通信对称 |
| 2 | Shape 不匹配 | 切了主导张量没切配套 mask/index | 对照张量分布约定表，所有相关张量对称切 | 建张量分布表时逐张量声明，未声明张量视为潜在 bug |
| 3 | `all_to_all_single expects contiguous` | `chunk()` 返回 view | chunk 后 `.contiguous()`，或用 `row_to_col`/`col_to_row` | 统一用封装好的 `row_to_col`/`col_to_row`，不裸调 `all_to_all` |
| 4 | 非确定性结果 | RNG 各卡不同步 / NPU 算子非确定性 | 逐样本播种；tolerance 通过即提交 | 推理脚本入口统一播种三套 RNG（`torch`/`random`/`np.random`） |
| 5 | HCCL 初始化失败 | 未 `import torch_npu` / 环境未配置 | 见通信原语模板初始化 | `init_distributed` 模板已含 `try: import torch_npu`，勿删除 |
| 6 | 并行比单卡还慢 | 通信 > 计算节省 | 回第三步重算 break-even；加通信-计算重叠 | 实施前先算 break-even；用 `issue_on_comm_stream` 做通信-计算重叠 |
| 7 | 切分后仍 OOM | 入口没切 / 残差未释放 / buffer 抵消 | 入口 scatter + 残差用本地分片 + buffer 复用 | 入口第一时间 scatter；AllGather 用 `all_gather_into_tensor` 单 buffer 减半峰值 |
| 8 | 并行开关误触发子模块 | 全局 `is_parallel()` 被未切分模块检测 | 进入前 `disable_parallel()` 或模块级 guard | 用 `set_parallel_flag()` 做模块级 guard，不依赖全局 flag |
| 9 | 跨迭代 shape 变化 | 首轮全量、后续本地分片，代码假设固定形状 | 用 `shape[0]` 判断，不假设输入形状 | 并行路径用 `tensor.shape[0]` 动态推断，不硬编码 |
| 10 | 负索引维度歧义 | 切分后维度数变化，负索引指向不同 | 并行路径用正整数维度索引 | 并行路径统一用正整数 dim 索引，禁用负索引 |

**多节点专项**：#11 连接超时（`MASTER_ADDR` 设网卡 IP + `HCCL_CONNECT_TIMEOUT=600`）| #12 拓扑发现失败（确认 `HCCL_IF_BASE_PORT` + 实测带宽）| #13 参数不一致（rank 0 broadcast 权重）

**回滚与清理**：#14 Patch 残留（`cleanup_all_patches()` 重注册）| #15 分布式状态残留（`destroy_process_group` + reset + reinit）| #16 通信缓冲区泄漏（buffer 交替复用 + `del` 临时张量 + 监控 alloc 差值）

**混合并行专项**：#17 ProcessGroup dp_ranks offset 错误（dp_ranks 必须含 `rank % (tp×pp)` 偏移，固定 `{0, tpp, 2×tpp, ...}` 会导致 `new_group` 报错或子组语义错乱）| #18 ReduceScatter dim≠0 需 permute（`dist.reduce_scatter` 只支持 dim=0，其他维度需 permute→reduce_scatter→permute_back）

### 通用排查流程

1. 先跑 `self_test()`（`torchrun --nproc_per_node=N`）：验证通信原语本身无问题
2. 再跑冒烟（最小输入）：定位逻辑错还是规模问题
3. `py-spy dump`：死锁时抓各 rank 栈
4. 打印 shape：对照张量分布约定表
5. 单卡 vs 多卡：tolerance 首选，未通过用 bit-exact 小输入 debug
6. profiling 看通信占比：慢时先确认 break-even

---

> **后续通信性能优化**：并行 infra 搭建后若通信成为瓶颈（comm 占比 > 20%），见 [comm_optimization.md](../../03_optimization/references/comm_optimization.md)。
