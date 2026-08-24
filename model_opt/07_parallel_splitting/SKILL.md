---
name: npu-parallel-splitting
description:
  多卡推理并行切分的判断、分析、实施与验证。model_opt 的子技能（07_parallel_splitting），当用户报告显存不足、需要多卡推理、或要求模型并行时触发；也由 Phase 1 的多卡切分前置判定（OOM 分诊）条件触发。
---

# 多卡推理切分

## 在 model_opt 流程中的位置

本子技能是 model_opt 的**条件触发的专业轨道**，不是顺序 Phase。当瓶颈是显存容量（而非计算效率）时，从主流程分支进入，完成后回归主流程。

**触发方式**：
1. **Phase 1 分诊触发**：[01_preparation/SKILL.md](../01_preparation/SKILL.md)「多卡切分前置判定」分诊确认"本质需要并行" → 进入本子技能全流程
2. **用户直接触发**：用户报告显存不足/需要多卡/要求模型并行 → 直接进入全流程

**与各 Phase 交互**：Phase 1 分诊（见 [01_preparation/SKILL.md](../01_preparation/SKILL.md)「多卡切分前置判定」）确认本质需要并行 → 进入本子技能完成分析+实施+验证（步骤 1-7）→ 验证通过后回归 Phase 1 主线，完成带 profiling 的并行推理脚本并采集并行基线 → 进入 Phase 2 瓶颈分析 → Phase 3 四维度优化 → Phase 4 门禁 → Phase 5 提交（evidence_db 纳入）。简单并行（DP only）不进入本子技能，用 [parallel_design.md](../03_optimization/references/parallel_design.md) 即可。回退路径：`disable_parallel()` 回到单卡，回到 model_opt 四维度优化。

## 核心原则

- **专注模型的多卡推理切分**：让模型在多卡上正确运行，每卡承担合理份额的计算与显存
- **禁止修改模型架构**：切分只改变计算的分布方式，不改变模型的数学语义和结构
- **分诊前置**：分诊在 [01_preparation/SKILL.md](../01_preparation/SKILL.md)「多卡切分前置判定」完成，外部因素在分诊阶段已排除；进入本技能即确认本质需要并行
- **三角度论证**：切分方案必须从数学正确性、完备性、系统可行性三个角度论证，分析结果记录到 [evidence_db](../06_evidence_db/schema.md)（`parallel_splitting` 字段）

## 全流程

```
前置：Phase 1 分诊（见 01_preparation/SKILL.md「多卡切分前置判定」）已确认本质需要并行
   ↓
分析阶段（agent 直接读源码 + 计算）
   第一步  什么在吃显存？（提取参数 → 内存时间线）
   第二步  切哪个维度？（定性分类 → 候选并行模式）
   第三步  通信代价与可行性（定量估算 → break-even）
   第四步  记录分析到 evidence_db  ★ 确认
   （可选）第五步  Profiling 数据校准
   ↓                                          详情: analysis_workflow.md
实施阶段
   第六步  通信 infra + 实施模式 + 冒烟测试     详情: implementation_guide.md（含通信原语模板）
   ↓
验证阶段
   第七步  正确性验证：基于 verify_split.py 模板
          copy → 填 4 函数 → baseline(单卡) → verify(多卡) → verify_report.json   ★ 确认提交
   ↓                                          详情: implementation_guide.md + scripts/verify_split.py
完成后回归 Phase 1 主线 → 完成带 profiling 的并行推理脚本 → 采集并行基线 → 进入 Phase 2
```
**注意**：第七步要端到端测试，所以得留好锚点，要选一组模型在切分之前可以单卡跑通的配置，先单卡跑通；之后同等配置下，多卡也跑通；然后对比切分前后模型的输出是否在容差范围。具体操作：copy [scripts/verify_split.py](scripts/verify_split.py) 到项目 → 填写 4 个函数 → `python verify_split.py --mode baseline` 采基线 → `torchrun --nproc_per_node=N verify_split.py --mode verify` 验证 → 检查 `verify_report.json`
**错误回退**：第四步检查未过→回第三步 | 第六步冒烟失败→回第四步 | 第七步验证未过→回第四步 | （可选第五步 profiling 校准结果用于更新 evidence_db，更新后仍不达标→回第四步）

## 输出产物

| 产物 | 阶段 | 路径 | 用途 |
|------|------|------|------|
| 内存时间线表 | 第一步 | stdout / evidence_db | 逐模块峰值定位瓶颈 |
| 分析记录 | 第四步 | `evidence_db/<id>.yaml`（`parallel_splitting` 字段） | 切分方案论证 |
| 通信原语模块 | 第六步 | `comm/comm_primitives.py`（copy 自 [scripts/comm_primitives.py](scripts/comm_primitives.py)） | 通信 infra |
| 并行推理脚本 | 第六步 | `run_parallel.py` | 多卡推理入口（验证用） |
| 验证脚本 | 第七步 | `verify_split.py`（copy 自 [scripts/verify_split.py](scripts/verify_split.py)） | 切分前后输出对比模板 |
| 验证报告 | 第七步 | `verify_report.json` | 正确性验证结果 |

## 确认节点

- **★ 方案确认**（第四步后）：展示切分方案 + 定量估算 + 等价性证明 + 风险应对，用 `ask_user_question` 确认实施
- **★ 提交审核**（第七步后）：展示 `verify_report.json`（overall + 各 tier 结果）+ 性能收益，用 `ask_user_question` 确认提交

## 调试与排查

问题分类归属 + 排查流程 + 切分方法常见问题见 [implementation_guide.md](references/implementation_guide.md#调试排查)。

## 子文档索引

**按流程加载**：进入哪个阶段就读对应文档，无需一次性全部加载。

| 阶段 | 文档 | 内容 |
|------|------|------|
| 分析 | [analysis_workflow.md](references/analysis_workflow.md) | 切分只做两件事 → 推理优先 → 什么在吃显存 → 切哪个维度 → 通信代价 → 维度速查 → 通信设计 → 记录确认 |
| 实施 + 验证 + 调试 | [implementation_guide.md](references/implementation_guide.md) | 4 种模式 + 通信原语引用 + NPU 差异 + 验证方法 + 并行 Profiling 采集 + 调试排查 |
| 通信原语脚本 | [scripts/comm_primitives.py](scripts/comm_primitives.py) | 基础原语（allgather/scatter/allreduce/reduce_scatter/broadcast/gather/alltoall）+ 通信-计算重叠 + 自检 |
| 验证模板 | [scripts/verify_split.py](scripts/verify_split.py) | 切分前后输出对比（分层 tolerance + bit-exact debug + verify_report.json） |
| 案例记录 | [06_evidence_db/schema.md](../06_evidence_db/schema.md) | `parallel_splitting` 字段定义 |

**场景提问**：接到多卡请求后用 `ask_user_question` 按优先级问（max 4 题/次）：① 场景/指标/独立样本数 → ② 模型架构/规模 → ③ 硬件拓扑/HBM。已明确的维度跳过。
