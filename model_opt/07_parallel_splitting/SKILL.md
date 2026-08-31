---
name: npu-parallel-splitting
description:
  多卡推理并行切分的判断、分析、实施与验证。model_opt 的子技能（07_parallel_splitting），当用户报告显存不足、需要多卡推理、或要求模型并行时触发；也由 Phase 0 适配冒烟的 OOM 分诊条件触发。
---

# 多卡推理切分

## 运行前统一原则：NPU 资源检查

**每次需要运行代码（benchmark / profiling / 精度验证 / 功能测试等）之前，必须先用 `npu-smi info` 检查 NPU 上是否有与本任务无关的进程。若存在，先确认这些进程与当前任务无关（必要时向用户确认归属），再清理进程、释放显存，确认资源干净后才开始运行。** 无关进程会争抢算力/显存，导致性能数据失真或 OOM。

```bash
npu-smi info        # 检查各卡上的进程占用
ps -fp <PID>        # 确认进程身份与归属
kill <PID>          # 确认无关后清理（顽固进程用 kill -9）
npu-smi info        # 复查确认显存/算力已释放
```

## 在 model_opt 流程中的位置

本子技能是 model_opt 的**条件触发的专业轨道**，不是顺序 Phase。当瓶颈是显存容量（而非计算效率）时，从主流程分支进入，完成后回归主流程。

**触发方式**：
1. **Phase 0 冒烟分诊触发**：[00_adaptation/SKILL.md](../00_adaptation/SKILL.md)「冒烟测试与 OOM 分诊」分诊确认"本质需要并行" → 进入本子技能全流程
2. **用户直接触发**：用户报告显存不足/需要多卡/要求模型并行 → 直接进入全流程

进入本子技能完成全流程（第一步~第六步，见下方「全流程」）→ 验证通过后回归 Phase 0 完成适配精度验证。回退路径：`disable_parallel()` 回到单卡，回到 model_opt 四维度优化。

## 核心原则

- **专注模型的多卡推理切分**：让模型在多卡上正确运行，每卡承担合理份额的计算与显存
- **禁止修改模型架构**：切分只改变计算的分布方式，不改变模型的数学语义和结构
- **三角度论证**：切分方案必须从数学正确性、完备性、系统可行性三个角度论证；张量分布约定表在方案确认后落盘为工作契约（供实施/验证对照），验证通过后随切分案例记录到 [evidence_db](../06_evidence_db/schema.md)（`parallel_splitting` 字段）


## 全流程

```
前置：Phase 0 冒烟分诊（见 00_adaptation/SKILL.md「冒烟测试与 OOM 分诊」）已确认本质需要并行
   ↓
分析阶段（agent 读源码分析 + 脚本估算）             详情: analysis_workflow.md
   第一步  什么在吃显存？（提取参数 → 内存时间线 → OOM 根因组件清单）
   第二步  切哪个维度？（定性分类 → 候选并行模式）
   第三步  通信代价与可行性（有真实多卡环境时先 `comm_self_test.py --timing` 校准 α/带宽
          → 填 spec（计算耗时由脚本按 FLOPs 解析估算，无需单卡实测）
          → estimate_split.py 定量估算 → 候选裁决：净收益≤0 剪掉；差异超出误差带（~30%）延迟定胜负，
          带内并列候选才允许以工程量决胜）+ 数学等价性论证（代数恒等 + 数值噪声预期，见 analysis_workflow）
   ↓
 ★ 方案确认（`ask_user_question`：展示切分方案 + 定量估算 + 等价性论证 + 风险应对 → 确认/裁剪）
   ↓
实施阶段                                         详情: implementation_guide.md（含通信原语模板）
   第四步  复用通信原语模板（comm_primitives.py）+ 按分析逻辑实现切分 + 冒烟测试
   ↓
验证阶段                                         详情: implementation_guide.md + scripts/verify_split.py
   第五步  正确性验证——数学等价已在分析阶段论证，本步验证**实施正确性与浮点容差**
          （切分只验切分：固定 1 条代表性真实样本 + 最小冒烟输入，全量精度回归
          由回归主流程后的 Phase 0「0.5 精度验证」（golden 对齐）承担，不在本步消耗）
          copy verify_split.py → 填 4 函数 → baseline(单卡/缩配) → verify(多卡) → verify_report.json
   ↓
 ★ 提交审核（`ask_user_question`：展示 verify_report.json（overall + 各 tier 结果）+ 性能收益 → 确认提交）
   ↓
记录与回归
   第六步  记录 evidence_db 切分案例（方案 + 估算 + 等价性论证 + 验证结果 + 已知坑；与优化案例分立，后者 depends_on 本案例）
          → 回归 Phase 0 完成适配精度验证 → Phase 1 采集并行基线 → 进入 Phase 2
```
**注意**：第五步要端到端测试，需选一组切分前可单卡跑通的配置作为锚点；权重超单卡、拿不到单卡 baseline 时的替代方案（缩小输入/缩小模型/逐层对比）与 verify_split.py 完整操作流程（含必填的 `--split-type`）见 implementation_guide「正确性验证」。

**错误回退**：第三步 break-even 不成立→回第二步换切分维度 | ★方案确认被否→回分析调整 | 第四步冒烟失败→按 implementation_guide「调试排查」处理，方案性问题回第二步 | 第五步验证未过→按 implementation_guide「验证未过处置」路由：最小输入未过=实施错误，回第四步；仅真实样本超档→消融定界（fp32 通信 / fp32 行并行 GEMM / 双跑一致性），定位单点则修复重验；三项均无法消除、偏差符合形状级噪声预期（见 analysis_workflow「等价性论证」）且方向一致→**例外审批**：用户确认 + evidence_db 留痕（复现条件、容差边界、消融证据），禁止无依据放宽容差后重跑"通过"

## 调试与排查

问题归属（环境 / 切分 / 性能三分路由）+ 性能问题归属见 [implementation_guide.md](references/implementation_guide.md#调试排查)。

## 子文档索引

**按流程加载**：进入哪个阶段就读对应文档，无需一次性全部加载。

| 阶段 | 文档 | 内容 |
|------|------|------|
| 分析 | [analysis_workflow.md](references/analysis_workflow.md) | 切分的两个判定 + 拓扑定哪里可切 → 什么在吃显存 → 切哪个维度 → 通信代价与等价性论证 → 通信效率层次 → 方案确认 |
| 实施 + 验证 + 调试 | [implementation_guide.md](references/implementation_guide.md) | 实施主线路径 + 接入决策（边界/内部 + 替换机制）+ 通信原语（基座/组合/自检三件）+ 权重处理 + 验证方法 + 调试排查 |
| 通信基建 | [scripts/comm_primitives.py](scripts/comm_primitives.py) / [comm_recipes.py](scripts/comm_recipes.py) / [comm_self_test.py](scripts/comm_self_test.py) | 基座整文件 copy；切分组合函数按需取用；环境自检原地运行（`--timing` 兼作通信参数校准，供 spec hardware 覆盖） |
| 估算脚本 | [scripts/estimate_split.py](scripts/estimate_split.py) | 第三步显存/通信/收益估算：填 spec（含实测 α；计算耗时由脚本 FLOPs 解析估算）→ 原地运行（无需 copy）→ 读报告排序与推荐结论 |
| 验证模板 | [scripts/verify_split.py](scripts/verify_split.py) | 切分前后输出对比（分层 tolerance + bit-exact debug + verify_report.json） |
| 案例记录 | [06_evidence_db/schema.md](../06_evidence_db/schema.md) | `parallel_splitting` 字段定义 |

**场景提问**：接到多卡请求后用 `ask_user_question` 按优先级问（max 4 题/次）：① 场景/指标/序列规模与负载形态 → ② 模型架构/规模 → ③ 硬件拓扑/HBM。已明确的维度跳过。
