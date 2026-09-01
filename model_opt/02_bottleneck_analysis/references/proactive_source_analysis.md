# Line A 源码分析
> 本文件只讲 Line A 自身的主分析；与 Line B 的联合分析见 [merge_analysis.md](merge_analysis.md)。

### 目标

从代码出发主动发现优化机会，产出 profiling 无法提供的信息：

| 层 | profiling 给不了的 |
|----|-------------------|
| 模型结构层 | 组件构成、重复结构、权重共享、生成模式 |
| 实现逻辑层 | 数据流路径、中间结果生命周期、控制流的真实取值、数据访问模式 |
| 算法层 | 计算的数学语义、可等价重组性、是否存在更优算法/融合 pattern |

很多高价值问题（同源 Linear 独立调用、KV cache 动态扩容、cross-attention 重复计算）在 profiling 中看不出异常——只有理解代码结构才能识别，这正是 Line A 存在的意义。

产出物是结构化报告（`analysis/round_{N}/line_a_report.md`）：**三层事实记录 + 由事实推导的疑点清单**。只记"存在什么机会"，不设计"怎么改"（方案是 Phase 3 的职责）。

### 分析节奏

第一轮做全面审视（覆盖下方所有分析对象和维度）。后续轮次不需要重新通读全部源码——结构性问题第一轮已发现,但**每次优化修改会改变计算结构和代码组织,需要重新审视**：

- **修改部分审视**：上次修改引入的新代码结构是否产生了新的问题或机会？
  
- **计算语义深审**：不满足于"读一遍代码发现显而易见的冗余", 而是对当前计算链逐操作追问？

### 穿透框架层

很多模型的推理路径被框架层层包裹, 直接看入口代码看到的是框架调度逻辑而非模型计算。agent 必须穿透框架到达真实的模型实现代码。

**识别框架 wrapper**:看到 `model.generate()` / `trainer.predict()` / `pipeline()` 时, 不要停在这里——这是框架入口, 不是模型逻辑。

**定位真实 forward**:沿调用链向下, 跳过 `Module.__call__` / hook dispatch / mixin 方法, 找到项目自己的 `forward()` 实现。

**常见框架穿透路径**:
- HuggingFace: `generate()` → `_generate_*()` → `model()` → `modeling_xxx.py:XxxModel.forward()`
- vLLM: `LLMEngine.step()` → `model_runner.execute_model()` → `model.forward()`
- 自定义脚本: `main()` → `model(input)` → 找到 `nn.Module` 子类的 `forward()`

#### 穿透层级认知

推理调用链从外到内分为多层，每层都可能贡献 host 开销。不要只分析 `forward()`——**每一层**都要认知到：

从入口到算子，列出调用链经过的所有层（如 `generate()` → `Module.__call__` → `forward()` → `F.linear` → `aten::matmul`），记录每层的职责（调度 / hook / 计算）与代码归属（框架层 / 项目层）。这张层级清单是 Line A 的结构产出，记入事实层（结构层）。

#### 框架开销识别

穿透框架层不只是"识别"框架代码,还要判断框架 wrapper 的开销是否值得消除。对**热路径上的框架 wrapper**（热路径由生成模式推导,不依赖 profiling）,检查是否存在以下模式：

- 热路径中通过 `module(x)` 调用子模块,但该模块不需要 hooks/grad——调用链经过的 `__call__` + `__getattr__` 是否是不必要的开销？
- 热路径中反复访问 `self.xxx` 属性——每次 `__getattr__` 的 dict 搜索是否累积成显著开销？
- 热路径中使用 `nn.Linear` 等框架 API——中间经过的 `F.linear` 分支判断和权重转置是否可以跳过？
- 热路径中用 `List[Tensor]` 索引——索引操作是否触发了不必要的设备侧 op？


### findings.yaml 的组织：facts 与 suspects

**事实与疑点分离**是 Line A 的核心机制：

- **事实层**（facts 列表）：三层理解的结构化记录，每条一个事实。只记客观事实，不记判断。
  
- **疑点层**（suspects 列表）：从事实推导出的可优化点。

**粒度**：事实按"计算块/主导张量"粒度记录。

**范围收敛**：分析范围由生成模式推导的热路径收敛——热路径（每步执行的代码）逐计算块记录；非热路径（一次性初始化、预处理）只记一条 component 事实，不展开。

报告由 `line_a_report.py` 按 layer/kind 自动归入对应章节渲染，agent 只需组织好 yaml 条目，不需要关心报告排版。

#### facts：六种 kind 及 content 记录要点

三层理解（模型结构 → 实现逻辑 → 算法）对应 facts 的 layer 取值；每个 layer 下按 kind 分条记录：

| kind | layer | content 应记录 | 示例 |
|------|-------|---------------|------|
| `component` | structure | 组件类型、重复度、参数量级、生成模式中的角色 | "32 层 MHA attention，decode 每步执行" |
| `dataflow` | implementation | 主导张量的变换链（输入→输出逐步变换，标注每步算子语义） | "hidden_states 经 q/k/v 三个 Linear 独立投影，view+transpose 调整头维度" |
| `lifecycle` | implementation | 中间结果的 创建点 → 最后使用点 → 释放点；是否跨步存活 | "attention weights 每步新建用完即弃；KV cache 跨步存活持续扩容" |
| `controlflow` | implementation | 分支的推理期真实取值（恒真/恒假/数据依赖）及依据 | "if self.training 推理期恒假（model.eval() 保证）" |
| `access` | implementation | 索引/切片/维度操作；索引是否依赖输入内容 | "gather 按动态索引访问，索引实际循环前已知" |
| `algo` | algorithm | 计算块的数学表达 + 当前实现路径 | "RMSNorm = x·w/√(mean(x²)+ε)，拆解为 Cast→Square→Mean→Add→Rsqrt→Mul→Cast" |

填写要点：

- **`component` 字段**：实现逻辑层/算法层的事实填归属组件名——热路径覆盖检查靠它关联架构行（脚本校验 hot 组件必须有同名 component 的 implementation 事实）
- **`hot: true`**：架构行上标记生成模式推导的热路径组件（每步执行）
- **生成模式**：作为一条 component 事实记录（自回归 / 扩散迭代 / encoder-decoder）——它决定热路径
- **重复度是影响范围的天然乘数**：块内任何疑点的影响 ×N，记在 component 事实里

#### suspects：从事实推导

suspects推导分两层：

**第一层：根问题（完备的生成机制）**。对每条事实（每个热路径计算块）问四个问题，任一得到"是"即存在机会、产出一条 suspects 条目：

- 必要性：这段代码是必要的吗？输出被使用吗？更少的工作能达到同一结果吗？ 
- 重复性：这个结果之前算过或存过吗？能跨调用/跨步复用吗？
- 独立性：有没有与这段工作无数据依赖的其他工作可以并行？
- 表达：同一数学结果有没有物理代价更低的等价写法？

**第二层：判据表（已模式化的常见答案）**。根问题的常见回答已沉淀为下表——content 命中即按表取值，作为加速器使用。**未命中的机会照样产出 suspects**：字段手工填写（`suspicion` 描述机会、`waste_class` 按 [waste_taxonomy.md](waste_taxonomy.md) 归类、`dimension` 对应命中的根问题），稳定复现的新模式追加进判据表（开放积累）。

| 匹配 kind | content 中的事实形态 | suspicion | waste_class | dimension |
|---|---|---|---|---|
| component | 同一输入馈入多个同型算子/组件 | 同输入多路独立投影（合并机会） | ② / ⑨ | eliminate |
| component | 同构块重复 N 次 | （放大器：与其他判据组合，标注 ×N） | — | — |
| component | 权重跨模块共享/复用 | 布局统一/预计算机会 | ⑤ / ⑥ | reuse |
| component | 每步执行（hot 标记） | （优先级依据，不产出独立条目） | — | — |
| dataflow | 变换链"变过去又变回来"（permute/contiguous 往返、格式往返） | 冗余变换 | ⑦ | eliminate |
| lifecycle | 中间结果创建后仅使用一次即弃 | 可融合/原地操作机会 | ⑨ | substitute / reuse |
| lifecycle | 每步重建但内容跨步不变 | 预计算/缓存机会 | ② / ⑥ | reuse |
| lifecycle | 大张量跨步驻留不释放 | 内存复用机会 | ③ | reuse |
| controlflow | 推理期恒假分支、训练遗留逻辑（dropout p=0 等） | 死代码 | ② / ⑥ | eliminate |
| controlflow | 每步重评估的不变条件 | 条件外提机会 | ① / ② | eliminate |
| access | 动态索引实际在循环前已确定 | 访问模式简化机会 | ⑤ / ⑨ | substitute |
| access | size-1 维切片引发下游额外维度变换 | 形状规范化机会 | ⑦ | eliminate |
| access | Python 逐元素/逐行操作张量 | 向量化机会 | ② / ⑨ | substitute |
| algo | 连续 element-wise/norm 拆解序列 | 融合 pattern 机会（对照 [npu_operator_catalog.yaml](../../03_optimization/references/npu_operator_catalog.yaml)） | ⑨ | substitute |
| algo | 数学表达可经结合律/分配律等价重组 | op 数/中间张量削减机会 | ⑨ / ⑥ | substitute |
| algo | 同类功能存在复杂度更低的算法 | 算法替换机会 | ⑥ | substitute |
| algo | 循环不变计算出现在循环内 | 外提机会 | ⑥ | eliminate |
| algo | 硬件不亲和表达（转置 GEMM、one-hot×矩阵、einsum 字符串） | 等价 API 替换机会 | ⑥ / ⑦ | substitute |

- **waste_class**：编号对应十类浪费统一分类（[waste_taxonomy.md](waste_taxonomy.md)）
- **填写纪律**：`suspicion` 只记"存在什么机会"，不设计"怎么改"；`location` 统一格式"文件:行 函数"

### 报告落盘

**填写 findings.yaml → 运行脚本渲染 → 报告落盘**：

```bash
python <skill_path>/02_bottleneck_analysis/scripts/line_a_report.py \
    --findings analysis/round_{N}/line_a_findings.yaml
# 报告输出到 findings 同目录的 line_a_report.md；非法输入拒绝渲染（fail loudly）
```

findings.yaml schema：

```yaml
meta:
  model: <string>              # 模型名/架构
  source_commit: <string>      # 被分析源码版本（git rev-parse HEAD）
  round: <int>                 # 优化轮次（0 = 首次分析）
  scope: <string, optional>    # 后续轮次：仅审视上次修改部分 + 语义深审

facts:                           # 事实层：三层记录
  - id: F-001
    layer: structure | implementation | algorithm
    kind: component | dataflow | lifecycle | controlflow | access | algo
    component: <string, optional>   # 归属组件/计算块名（热路径覆盖检查与关联用）
    hot: <bool, optional>           # 仅架构行：是否热路径（生成模式推导，每步执行）
    content: <string>               # 按「三层模板」的一行事实

suspects:                        # 疑点层
  - id: LA-001
    fact_refs: [F-001, F-002]    # 必填：依据的事实 id
    layer: structure | implementation | algorithm
    location: "modeling_llama.py:214 LlamaAttention.forward"  # 文件:行 + 函数（join key，格式统一）
    suspicion: "同输入多路独立投影"          # 机会描述，非方案
    dimension: eliminate | reuse | hide | substitute
    waste_class: <string, optional>          # 对应 10 类浪费（判据映射列；判据外机会可空，合并阶段补）
    impact_qualitative: "32×重复，decode 每步执行"  # 定性影响：重复度×频率
    confidence: high | medium | low
```

### 产出

- `analysis/round_{N}/line_a_report.md`（+ findings.yaml 源文件）
- 疑点进入合并分析：与 `line_b_report.md` 关联/合成/去重/补盲，产出 `candidates.md` → 进入 Phase 3 方案设计（见 [execution_protocol.md](../../references/execution_protocol.md)）
