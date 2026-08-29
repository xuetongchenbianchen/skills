# Line A 源码分析重构设计

> 目标：重构 `02_bottleneck_analysis/references/proactive_source_analysis.md`（+ 可选新增 `line_a_report.py`），使 Line A 产出结构化、可审计的源码理解报告。
> 状态：**已实施（2026-08-26）**。v1 的 profiling 耦合、v2 的"消费 Line B 层数据"均已废弃——双线完全解耦；v4 确定轮次目录结构、分类策略（判据↔浪费类别映射）与合并产物形态。实施时两个待确认项均按建议落地：脚本保留、事实粒度=计算块/主导张量 + 热路径收敛。

## 1. 定位：Line A 与 Line B 完全解耦

**原则：Line A 只产出 profiling 无法提供的信息，且不消费 Line B 的任何产出。** 凡 profiling 可见的（哪个算子慢、占比多少、调用几次、哪层 host 开销大）归 Line B，Line A 不重复分析、也不依赖其结果定位重点。

Line A 产出三类 profiling 给不了的理解：

| 层 | profiling 给不了的 |
|----|-------------------|
| 模型结构层 | 组件构成、重复结构、权重共享、生成模式 |
| 实现逻辑层 | 数据流路径、中间结果生命周期、控制流的真实取值、数据访问模式 |
| 算法层 | 计算的数学语义、可等价重组性、是否存在更优算法/融合 pattern |

**Line A 自身的优先级机制是语义的**：由生成模式推导热路径（自回归 decode 的每步执行代码 = 热路径），不需要 profiling 数据收敛分析范围。

**双线的交叉消费统一收敛到合并阶段（★A 前）一个点**，两份报告各自封卷后交叉：

| 合并阶段动作 | 消费物 |
|-------------|--------|
| **关联合成**：以位置/算子/调用链层为键关联两线发现，同一问题双来源归并，形成"结构根因 + 量化影响"的完整候选（详见 §5.2） | 两份报告 |
| 覆盖性检查：Line B 报告中 >10% 的调用链层、10 类浪费、全部 DEFINITE/WARNING 是否有候选覆盖 | Line B 报告 + 合并候选清单（候选来源不限 A/B——如框架层 dispatch 开销高的候选可完全由 Line B 归因层给出） |
| 覆盖缺口回补：发现未覆盖的热层 → 定向回补 Line A（有界返工，显式记录） | — |

> v1 的 quant_refs 自动量化、v2 的"层级数据作为 Line A 审视输入"均已废弃——后者制造了虚假的串行依赖，违背双线并行的设计本意。

## 2. 信息组织：事实层 + 疑点层

**事实与疑点分离**是本设计的核心机制：

- **事实层**（报告 §1–3）：三层理解的结构化记录。只记客观事实，不记判断——"Q/K/V 三个 Linear 各自接收同一份 hidden_states"是事实，"可合并"不是。
- **疑点层**（报告 §4）：从事实推导出的可优化点。**每条疑点必须引用事实 id**——形成"依据 → 结论"的可审计推导链。无事实依据的疑点不允许出现（防止凭空脑补）。

## 3. 事实层模板

### 3.1 架构理解（结构层）

| 组件 | 类型 | 重复度 | 参数量级 | 生成模式中的角色 |
|------|------|--------|---------|----------------|
| LlamaAttention ×32 | attention(MHA) | 32 | … | decode 每步执行（热路径） |
| q_proj/k_proj/v_proj | Linear | 32×3 | … | 每步执行 |

- 生成模式一句话：自回归 / 扩散迭代 / encoder-decoder 等——它决定哪些组件是"每步热路径"
- 重复度是影响范围的天然乘数：块内任何疑点的影响 ×N

### 3.2 实现逻辑理解（按推理路径，逐路径填写）

| 维度 | 记录内容 | 形式 |
|------|---------|------|
| 数据流 | 主导张量的变换链（输入→输出逐步变换） | 链式列表，标注每步算子语义 |
| 生命周期 | 中间结果的 创建点 → 最后使用点 → 释放点 | 表：张量/中间结果 \| 创建 \| 最后使用 \| 释放 \| 跨步存活? |
| 控制流 | 每个分支的推理期真实取值 | 表：分支 \| 推理期取值（恒真/恒假/数据依赖）\| 依据 |
| 访问模式 | 索引/切片/维度操作清单 | 表：操作 \| 方式（gather/scatter/切片/逐元素）\| 索引是否输入相关 |

### 3.3 算法理解

| 计算块 | 数学表达 | 当前实现路径 |
|--------|---------|-------------|
| attention | softmax(QKᵀ/√d)V | 逐算子拆解（MatMul→Softmax→MatMul…） |
| RMSNorm | x·w/√(mean(x²)+ε) | Cast→Square→Mean→Add→Rsqrt→Mul→Cast |

## 4. 疑点判据（事实 → 疑点的推导规则）

疑点发现的机制化：每类事实形态对应固定判据，逐条对照三层事实记录。

**分类策略——不建平行浪费分类**：Line B 的 10 类浪费（profiling_to_action 归因层）是**收益归因维度**（反事实收益上限的度量依据、优先级覆盖门禁的轴）；Line A 判据是**结构形态维度**（定位到源码的依据）。候选 = 两者交点。判据表每条标注"对应浪费类别"，Line A 疑点产出时天然落入 10 类框架，不维护第二套分类体系。

**穷尽性不靠判据枚举**——判据是加速器不是过滤器，三层保证：① Line A 侧事实记录全覆盖（每个热路径计算块有事实记录，判据外的机会仍可产出）；② Line B 侧 DEFINITE/WARNING 全追踪 + 10 类覆盖门禁；③ 合并侧双向覆盖表落盘检查（见 §5.2）。判据表开放积累（同 npu_checklist）。

### 结构层判据

| 事实形态 | 疑点 | 浪费类别 |
|---------|------|---------|
| 同一输入馈入多个同型算子/组件 | 合并候选（一次大算子替代 N 路独立调用） | ② dispatch / ⑨ 碎片 |
| 同构块重复 N 次 | 块内机会 ×N（疑点标注重复度） | —（放大器，非独立类别） |
| 权重跨模块共享/复用 | 布局统一/预计算机会 | ⑤ 带宽 / ⑥ compute |
| 组件在生成模式中每步执行 | 热路径标记（疑点优先级依据） | —（优先级机制） |

### 实现逻辑层判据

| 事实形态 | 疑点 | 浪费类别 |
|---------|------|---------|
| 变换链"变过去又变回来"（permute/contiguous 往返、格式往返） | 冗余变换 | ⑦ 布局/格式转换 |
| 中间结果创建后仅使用一次即弃 | 可融合/原地操作候选 | ⑨ 碎片 |
| 每步重建但内容跨步不变 | 预计算/缓存候选 | ② dispatch / ⑥ compute |
| 大张量跨步驻留不释放 | 内存复用候选 | ③ 内存管理阻塞 |
| 推理期恒假分支、训练遗留逻辑（dropout p=0 等） | 死代码 | ② dispatch / ⑥ compute |
| 每步重评估的不变条件 | 条件外提候选 | ① 显式同步 / ② dispatch |
| 动态索引实际在循环前已确定 | 访问模式简化候选 | ⑤ 带宽 / ⑨ 碎片 |
| size-1 维切片引发下游额外维度变换 | 形状规范化候选 | ⑦ 布局/格式转换 |
| Python 逐元素/逐行操作张量 | 向量化候选 | ② dispatch / ⑨ 碎片 |

### 算法层判据

| 事实形态 | 疑点 | 浪费类别 |
|---------|------|---------|
| 连续 element-wise/norm 拆解序列 | 融合 pattern 候选（对照 `npu_operator_catalog.yaml`） | ⑨ 碎片 |
| 数学表达可经结合律/分配律等价重组 | op 数/中间张量削减候选 | ⑨ 碎片 / ⑥ compute |
| 同类功能存在复杂度更低的算法 | 算法替换候选 | ⑥ compute 饱和 |
| 循环不变计算出现在循环内 | 外提候选 | ⑥ compute |
| 硬件不亲和表达（转置 GEMM、one-hot×矩阵、einsum 字符串） | 等价 API 替换候选 | ⑥ compute / ⑦ 布局 |

**与 npu_checklist 的分工**：checklist 管"已知坑的机械匹配"（grep 命中即报，属 Phase 3 前置扫描）；上述判据管"需要语义理解才能判定的机会"。Line A 报告不重复收录 checklist 能直接命中的条目，引用其结果即可。

**疑点纪律**：只记"存在什么机会"，不设计"怎么改"（方案是 Phase 3 的事）；只记 profiling 看不见的结构性依据（"这个 Transpose 占 15%"归 Line B，Line A 记"布局约定不统一导致 permute 链"）。

## 5. 报告与合并产物结构

### 5.1 Line A 报告（`analysis/round_{N}/line_a_report.md`）

```
# Line A 源码分析报告（round N）
## 1. 架构理解                    # §3.1 模板
## 2. 实现逻辑理解                # §3.2 四张事实表（按推理路径）
## 3. 算法理解                    # §3.3 模板
## 4. 疑点汇总
   id | 层 | 位置 | 依据事实(id) | 疑点（机会描述） | 维度 | 浪费类别 | 影响范围（定性：重复度×频率）
```

报告在 §4 封卷——**疑点量化不属于本报告**，发生在合并阶段。Line A 报告一经落盘不再修改；合并阶段发现的覆盖缺口以"定向回补"方式追加条目并注明缘由。

### 5.2 合并分析（`analysis/round_{N}/candidates.md`）——★A 的直接输入

**合并不是两份报告的拼接，而是一个分析过程**：散落的信息点只有交叉关联后才成为完整候选——Line B 知道"Transpose 占 15%"但不知道为什么，Line A 知道"布局约定不统一导致 permute 链"但不知道值多少钱。合并由 agent 执行以下动作：

| 动作 | 说明 |
|------|------|
| **关联** | 以位置 / 算子 / 调用链层为键，把 Line A 疑点与 Line B 信号对齐——同一代码位置的两个侧面（结构成因 vs 时间开销） |
| **合成** | 候选 = 结构根因（Line A）+ 量化影响（Line B），形成"为什么慢 + 省多少"的完整叙述；反事实收益上限按其浪费类别的估算方法计算 |
| **去重** | 两线各自发现的同一问题归并为一条，标注双来源（LA-id + LB-信号）——双来源候选置信度更高，排序时可加权 |
| **补盲（双向）** | Line B 信号无 Line A 事实支撑 → 走桥梁做根因追踪，或定向回补 Line A；Line A 疑点无 Line B 现象 → 判定为冷路径（排除，附依据）或 profiling 盲区（保留，收益上限标注不确定） |
| **覆盖检查** | 三张表落盘（见下），缺口走定向回补并留修订记录 |

```
# 第 N 轮候选清单（★A 输入）
## 1. 候选清单（按反事实收益上限降序）
   id | 来源（LA-xx + LB-信号；双来源标注） | 问题 + 位置 | 结构根因（Line A） | 浪费类别 | 反事实收益上限 | 风险等级
## 2. 覆盖检查
   2.1 十类浪费优先级覆盖表        # execution_protocol 现有表格式：有候选/已排除(附依据)/不适用
   2.2 调用链层覆盖               # Line B 报告中 >10% 的层 × 对应候选或排除依据
   2.3 DEFINITE/WARNING 追踪表    # 发现来源|内容|桥梁|源码位置|根因|候选（Line B 门禁产出格式）
## 3. 未量化项与待定项             # 含 profiling 盲区候选（收益不确定标注）
```

## 6. 脚本角色（收缩后）

解耦后 `line_a_report.py` 不再量化，角色收缩为**校验 + 渲染**：

1. findings.yaml schema 校验（疑点必须引用存在的事实 id；三层事实节非空；location/dimension 枚举合法）
2. 完整性检查：热路径组件在 §2 有审视记录（热路径来自生成模式语义推导，不依赖 profiling）
3. 渲染落盘 + stdout 摘要（疑点数/各层数/热路径覆盖）

**脚本化收益**（按对整合分析的价值排序）：

1. **Join key 稳定**：合并分析以位置/算子为键关联两线发现，Line B 的 Call Stack 位置格式固定，schema 强制 Line A 的 `location` 统一格式后关联是精确匹配——手写格式的漂移会让 join 退化为模糊匹配，漏配难察觉
2. **机械门禁**：脚本拒绝渲染非法输入，完整性从 agent 自查（可跳过）变为强制——与 parse 脚本 DEFINITE 信号、verify_split 防作弊设计同哲学
3. **evidence_db 落库是映射**：findings.yaml 的 facts/suspects 与 schema 的 `phenomenon.signals`/`analysis_path.steps` 天然对应，Phase 5 记案例时机械映射而非从 markdown 重新提取
4. **跨轮可 diff**：固定结构 + 逐轮延续的 id（F/LA 编号），"新轮次强制重置"可审计
5. **注意力卸载**（次要）：格式不占 agent 工作记忆，语义质量更高

成本：脚本维护（约 150–200 行）、schema 学习（一次性）。**结论：建议保留**——收益 1+2+3 压倒成本，其中机械门禁只有脚本能给。

## 7. findings.yaml Schema（修订）

```yaml
meta:
  model: <string>
  source_commit: <string>
  round: <int>
  scope: <string, optional>      # 后续轮次：仅审视上次修改部分 + 语义深审

facts:                           # 事实层：三层记录
  - id: F-001
    layer: structure | implementation | algorithm
    kind: component | dataflow | lifecycle | controlflow | access | algo   # 事实类型（对应 §3 模板行）
    content: <string>            # 按 §3 模板的一行事实

suspects:                        # 疑点层
  - id: LA-001
    fact_refs: [F-001, F-002]    # 必填：依据的事实 id
    layer: structure | implementation | algorithm
    location: "modeling_llama.py:214 LlamaAttention.forward"
    suspicion: "同输入多路独立投影"          # 机会描述，非方案
    dimension: eliminate | reuse | hide | substitute
    waste_class: <string, optional>          # 对应 10 类浪费（来自判据映射列；判据外机会可空，合并阶段补）
    impact_qualitative: "32×重复，decode 每步执行"  # 定性影响；定量在合并阶段
    confidence: high | medium | low
```

## 8. 存放位置：轮次目录

主目录新建 `analysis/`，按轮次建子目录，同轮的两份分析与合并产物同处：

```
<workspace>/
└── analysis/                      # Phase 2 分析产物（提交——结论性记录）
    └── round_{N}/
        ├── line_a_findings.yaml   # Line A 事实+疑点源（脚本输入）
        ├── line_a_report.md       # Line A 报告（脚本渲染或手写）
        ├── line_b_report.md       # Line B 报告（run_analysis.py --output 落盘）
        └── candidates.md          # 合并产物（★A 直接输入，§5.2）
```

- 与 `profiling/`（原始大体积数据，不提交）解耦：分析结论进 git，原始数据不进
- 跨轮对比直接 diff `analysis/round_{1..N}/candidates.md`；"新轮次强制重置"有物证可查（每轮独立目录）
- 需在 standardized_operations 标准目录结构中增加 `analysis/`（提交范围）

## 9. 联动修改清单

| 文件 | 修改 |
|------|------|
| `references/proactive_source_analysis.md` | 按 §2–§5 重构；删除四维度提问/Profiling 量化/审计表三节；三层判据（含浪费类别映射列）吸收其有效内容；「穿透层级量化」保留结构认知、量化数据归属 Line B |
| `references/execution_protocol.md` | 「Line A 产出完整性门禁」收窄为 Line A 自身完整性（疑点须有 fact_refs、三层事实完整、热路径有审视）；「>10% 层须有候选」移至候选清单门禁；优先级覆盖表/根因追踪表/候选清单**落盘化**到 `analysis/round_{N}/candidates.md`（门禁从 agent 自查变为有落盘物可查） |
| `02_bottleneck_analysis/SKILL.md` | 双线执行改为真并行；Line A 流程改为"三层记录事实 → 判据推导疑点 → 报告落盘"；合并步骤升级为**整合分析**（关联/合成/去重/补盲，§5.2），产出 `analysis/round_{N}/candidates.md`；run_analysis 以 `--output` 落盘到轮次目录 |
| `model_opt/SKILL.md` | Phase 2 要点 Line A 改为"按三层记录事实并推导疑点，报告落盘 `analysis/`"；量化归属合并阶段 |
| `references/profiling_scripts_guide.md` | 若保留脚本：脚本列表加 line_a_report.py 行 |
| `references/standardized_operations.md` | 标准目录结构加 `analysis/`（提交；含轮次子目录说明） |

## 10. 实施决议

1. **脚本去留**：**保留**——join key 稳定 / 机械门禁 / evidence_db 映射三项收益压倒维护成本（§6）。
2. **事实粒度**：**计算块/主导张量 + 热路径收敛**——语义只存在于计算块级别，逐算子记录会重复 Line B 的 op_statistic 职责。
3. **candidates.md 生成**：**agent 分析产出**——合并是判断力工作（关联/合成/去重/补盲），不可脚本化。
