# 执行协议（agent 程序约束）

> 本文件是迭代流程中三个用户确认节点的详细门禁。方法论（分析思路 + 方案设计思路）在 [SKILL.md](../SKILL.md) 和 [profiling_to_action.md](../02_bottleneck_analysis/references/profiling_to_action.md)，本文件只管"执行时必须完成的程序约束"。

## 确认节点 A：方案审核（Phase 3 内：方案设计 → 实施之间）

Phase 2 给出瓶颈定位（candidates.md 问题点清单），Phase 3 将每个问题点转化为方案。方案设计完成后、逐条实施前，**必须**向用户展示方案清单并等待确认。

### 前置检查（展示前）

1. **Phase 2 产物完整**：`analysis/round_{N}/` 下 line_a_report.md、line_b_report.md（脚本生成，内部完整性由脚本保证）与 candidates.md
2. **方案完备**：`analysis/round_{N}/solutions.md` 中，candidates.md §1 的**每个问题点**都有对应方案，或附"无方案解释"（不可行 / 收益不足 / 依赖前置改动等）——不允许静默遗漏

### 展示与确认

向用户展示 solutions.md 方案清单：

1. 每条方案包含：对应问题点（问题 + 位置 + 结构根因）、方案内容、预期收益（对照该问题点的反事实收益上限）、风险等级
2. 无方案的问题点逐条展示解释，供用户判断是否要求重新设计
3. 使用 `ask_user_question` 询问用户：
   - 方案清单是否合适、有无遗漏
   - 哪些方案实施、优先级如何
   - 是否有额外想尝试的方向
4. 根据用户反馈调整方案清单——之后**逐条实施**确认的方案（每条实施后 Level 1 快速精度验证；实施中放弃须满足 [03_optimization/SKILL.md](../03_optimization/SKILL.md)「方向放弃标准（分级）」）

## 确认节点 B：提交前审核（Phase 4 → Phase 5 之间）

本批优化的精度验证和 profiling 确认均通过后、git commit 前，**必须**向用户展示本批总结并等待确认：

1. **对照 ★A 确认的方案清单逐条交代**：已实施（实际效果：性能数据 + 精度数据）/ 已放弃（附原因，须满足 [03_optimization/SKILL.md](../03_optimization/SKILL.md)「方向放弃标准（分级）」）/ 已回退（附原因）
2. 确认 evidence_db 案例已按 [schema](../06_evidence_db/schema.md) 记录到项目工作目录的 `evidence_db/` 下（展示案例文件路径）
3. 使用 `ask_user_question` 询问用户：
   - 是否确认提交本批优化
   - 是否需要回退某些改动
4. 用户确认后才执行 git commit

> 案例未记录 = 不允许提交。与"精度未通过 = 不允许提交"同等约束力。

## 确认节点 C：继续优化确认（Phase 5 之后）

git commit 完成后，**必须**向用户展示本轮总结并询问是否继续：

1. 展示本轮优化总结（性能提升 + 精度状态）
2. 展示当前剩余瓶颈（最新 profiling 数据）
3. 使用 `ask_user_question` 询问用户：是否继续下一轮优化？
4. 用户确认继续 → 回到 Phase 2 开启下一轮（重新采集 L1，并基于新的采集数据进行优化而不是沿着原来的路径继续做）
5. 用户确认停止 → 结束
