---
name: npu-parallel-splitting
description:
  多卡推理并行切分的判断、分析、实施与验证。model_opt 的子技能（07_parallel_splitting），当用户报告显存不足、需要多卡推理、或要求模型并行时触发；也由 Phase 0 适配冒烟的 OOM 分诊条件触发。
---

# 多卡推理切分

## 在 model_opt 流程中的位置

本子技能是 model_opt 的**条件触发的专业轨道**，不是顺序 Phase。当瓶颈是显存容量（而非计算效率）时，从主流程分支进入，完成后回归主流程。

**触发方式**：
1. **Phase 0 冒烟分诊触发**：[00_adaptation/SKILL.md](../00_adaptation/SKILL.md)「冒烟测试与 OOM 分诊」分诊确认"本质需要并行" → 进入本子技能全流程
2. **用户直接触发**：用户报告显存不足/需要多卡/要求模型并行 → 直接进入全流程

**与各 Phase 交互**：Phase 0 冒烟分诊（见 [00_adaptation/SKILL.md](../00_adaptation/SKILL.md)「冒烟测试与 OOM 分诊」）确认本质需要并行 → 进入本子技能完成全流程（第一步~第六步，见下方「全流程」）→ 验证通过后回归 Phase 0 完成适配精度验证 → Phase 1 采集并行基线（L0/wall-clock）→ 进入 Phase 2 瓶颈分析 → Phase 3 四维度优化 → Phase 4 门禁 → Phase 5 提交（evidence_db 纳入）。简单并行（DP only）不进入本子技能，直接配置 `ASCEND_RT_VISIBLE_DEVICES` 多卡即可。回退路径：`disable_parallel()` 回到单卡，回到 model_opt 四维度优化。

## 核心原则

- **专注模型的多卡推理切分**：让模型在多卡上正确运行，每卡承担合理份额的计算与显存
- **禁止修改模型架构**：切分只改变计算的分布方式，不改变模型的数学语义和结构
- **三角度论证**：切分方案必须从数学正确性、完备性、系统可行性三个角度论证，分析结果记录到 [evidence_db](../06_evidence_db/schema.md)（`parallel_splitting` 字段）

## 全流程

```
前置：Phase 0 冒烟分诊（见 00_adaptation/SKILL.md「冒烟测试与 OOM 分诊」）已确认本质需要并行
   ↓
分析阶段（agent 读源码分析 + 脚本估算）             详情: analysis_workflow.md
   第一步  什么在吃显存？（提取参数 → 内存时间线）
   第二步  切哪个维度？（定性分类 → 候选并行模式）
   第三步  通信代价与可行性（estimate_split.py 定量估算 → break-even）+ 数学等价性论证（代数恒等）
   ↓
 ★ 方案确认（`ask_user_question`：展示切分方案 + 定量估算 + 等价性论证 + 风险应对 → 确认/裁剪）
   ↓
实施阶段                                         详情: implementation_guide.md（含通信原语模板）
   第四步  复用通信原语模板（comm_primitives.py）+ 按分析逻辑实现切分 + 冒烟测试
   ↓
验证阶段                                         详情: implementation_guide.md + scripts/verify_split.py
   第五步  正确性验证——数学等价已在分析阶段论证，本步验证**实施正确性与浮点容差**
          copy verify_split.py → 填 4 函数 → baseline(单卡/缩配) → verify(多卡) → verify_report.json
   ↓
 ★ 提交审核（`ask_user_question`：展示 verify_report.json（overall + 各 tier 结果）+ 性能收益 → 确认提交）
   ↓
记录与回归
   第六步  记录 evidence_db 切分案例（方案 + 估算 + 等价性论证 + 验证结果 + 已知坑；与优化案例分立，后者 depends_on 本案例）
          → 回归 Phase 0 完成适配精度验证 → Phase 1 采集并行基线 → 进入 Phase 2
```
**注意**：第五步要端到端测试，需选一组切分前可单卡跑通的配置作为锚点；权重超单卡、拿不到单卡 baseline 时的替代方案（缩小输入/缩小模型/逐层对比）见 implementation_guide「正确性验证」。具体操作：copy [scripts/verify_split.py](scripts/verify_split.py) 到项目 → 填写 4 个函数 → `python verify_split.py --mode baseline` 采基线 → `torchrun --nproc_per_node=N verify_split.py --mode verify` 验证 → 检查 `verify_report.json`

**错误回退**：第三步 break-even 不成立→回第二步换切分维度 | ★方案确认被否→回分析调整 | 第四步冒烟失败→按 implementation_guide「调试排查」处理，方案性问题回第二步 | 第五步验证未过→回第三步修正等价性论证，或回第二步换维度

## 调试与排查

问题归属（环境 / 切分 / 性能三分路由）+ 性能问题归属见 [implementation_guide.md](references/implementation_guide.md#调试排查)。

## 子文档索引

**按流程加载**：进入哪个阶段就读对应文档，无需一次性全部加载。

| 阶段 | 文档 | 内容 |
|------|------|------|
| 分析 | [analysis_workflow.md](references/analysis_workflow.md) | 切分只做两件事 + 图拓扑约束 → 什么在吃显存 → 切哪个维度 → 通信代价与等价性论证 → 通信效率层次 → 方案确认 |
| 实施 + 验证 + 调试 | [implementation_guide.md](references/implementation_guide.md) | 实施主线路径 + 接入决策（边界/内部 + 替换机制）+ 通信原语（基座/组合/自检三件）+ 权重处理 + 验证方法 + 调试排查 |
| 通信基建 | [scripts/comm_primitives.py](scripts/comm_primitives.py) / [comm_recipes.py](scripts/comm_recipes.py) / [comm_self_test.py](scripts/comm_self_test.py) | 基座整文件 copy；切分组合函数按需取用；环境自检原地运行 |
| 估算脚本 | [scripts/estimate_split.py](scripts/estimate_split.py) | 第三步显存/通信/效率估算：填 spec → 原地运行（无需 copy）→ 读报告排序 |
| 验证模板 | [scripts/verify_split.py](scripts/verify_split.py) | 切分前后输出对比（分层 tolerance + bit-exact debug + verify_report.json） |
| 案例记录 | [06_evidence_db/schema.md](../06_evidence_db/schema.md) | `parallel_splitting` 字段定义 |

**场景提问**：接到多卡请求后用 `ask_user_question` 按优先级问（max 4 题/次）：① 场景/指标/独立样本数 → ② 模型架构/规模 → ③ 硬件拓扑/HBM。已明确的维度跳过。
