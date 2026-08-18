---
name: npu-parallel-splitting
description:
  多卡推理并行切分的判断、分析、实施与验证。model_opt 的子技能（07_parallel_splitting），当用户报告显存不足、需要多卡推理、或要求模型并行时触发；也由 Phase 2 的 Line C（OOM 分诊）条件触发。
  核心原则：先分诊（外部因素 vs 本质需要），再分析（4 步 + JSON 输出），确认后实施（通信 infra + 推理脚本），最后验证（输出一致性）。
---

# 多卡推理切分

## 在 model_opt 流程中的位置

本子技能是 model_opt 的**条件触发的专业轨道**，不是顺序 Phase。当瓶颈是显存容量（而非计算效率）时，从主流程分支进入，完成后回归主流程。

**触发方式**：
1. **Phase 2 Line C 触发**：profiling Parallelism Trigger 报告"消除 waste 后投影峰值仍 > 80% HBM" → OOM 分诊确认"本质需要并行" → 进入本子技能全流程
2. **用户直接触发**：用户报告显存不足/需要多卡/要求模型并行 → 直接进入全流程

**与各 Phase 交互**：Phase 2 Line C 分诊确认 → 本子技能分析+实施替代 Phase 3 → 验证回归 Phase 4 门禁 → ADR 纳入 Phase 5 提交。简单并行（DP only）不进入本子技能，用 [parallel_design.md](../03_optimization/references/parallel_design.md) 即可。回退路径：`disable_parallel()` 回到单卡，回到 model_opt 四维度优化。

## 核心原则

- **做好模型的多卡推理切分**：让模型在多卡上正确运行，每卡承担合理份额的计算与显存
- **禁止修改模型架构**：切分只改变计算的分布方式，不改变模型的数学语义和结构
- **先分诊再切分**：外部因素（配置不当、NPU 资源管理不善等）→ 修复，不切分；本质需要并行 → 触发切分流程
- **三角度论证**：切分方案必须从数学正确性、完备性、系统可行性三个角度论证，输出 JSON 分析文件

## 全流程

```
Phase 0  OOM 根因分诊 → 外部因素修复★终止 / 本质需要 → 进入分析
   ↓                                          详情: analysis_workflow.md Phase 0
分析阶段（agent 直接读源码 + 计算）
   第一步  提取模型参数与模块链路
   第二步  模块级拆解 → 内存变化分析 → 定性分类 → 候选并行模式
   第三步  定量估算（analysis_workflow.md 参考）
   第四步  方案审查 → 输出 JSON（数学/完备性/系统）  ★ 确认
   （可选）第五步  Profiling 数据校准
   ↓                                          详情: analysis_workflow.md
实施阶段
   第六步  通信 infra + 实施模式 + 冒烟测试     详情: implementation_guide.md（含通信原语模板）
   ↓
验证阶段
   第七步  正确性验证：端到端测试，同样的输入，对比模型切分前后输出是否一致   ★ 确认提交
   ↓                                          详情: implementation_guide.md（含验证脚本模板）
归档（ADR）                                    详情: implementation_guide.md
```
**注意**：第七步要端到端测试，所以得留好锚点，要选一组模型在切分之前可以单卡跑通的配置，先单卡跑通；之后同等配置下，多卡也跑通；然后对比切分前后模型的输出是否在容差范围
**错误回退**：Phase 0 外部因素→修复重分诊 | 第四步 JSON 审查未过→回第三步 | 第六步冒烟失败→回第四步 | 第七步验证未过→回第四步 | （可选第五步 profiling 校准结果用于更新 JSON，更新后仍不达标→回第四步）

## 输出产物

| 产物 | 阶段 | 路径 | 用途 |
|------|------|------|------|
| OOM 分诊结论 | Phase 0 | stdout | 判定是否需要并行 |
| 内存时间线表 | 第二步 | stdout / JSON | 逐模块峰值定位瓶颈 |
| 分析 JSON | 第四步 | `parallel_decisions/analysis_{model}_{seq_len}_{world_size}p.json` | 切分方案论证 |
| 通信原语模块 | 第六步 | `comm/comm_primitives.py`（按模板实现） | 通信 infra |
| 并行推理脚本 | 第六步 | `run_parallel.py` | 多卡推理入口 |
| 验证报告 | 第七步 | `verify_report.json` | 正确性验证结果 |
| ADR 决策记录 | 归档 | `parallel_decisions/ADR-NNNN-*.md` | 决策归档 |

## 确认节点

- **★ JSON 审查**（第四步后）：展示数学正确性 + 完备性 + 系统可行性 + 策略对比 + 推荐配置 + 风险应对，用 `ask_user_question` 确认实施
- **★ 提交审核**（第七步后）：展示验证结果 + 性能收益 + ADR 路径，用 `ask_user_question` 确认提交

## 调试与排查

常见坑 1-18 + 通用排查流程见 [implementation_guide.md](references/implementation_guide.md#调试排查)。

## 子文档索引

**按流程加载**：进入哪个阶段就读对应文档，无需一次性全部加载。

| 阶段 | 文档 | 内容 |
|------|------|------|
| Phase 0 + 分析 + JSON | [analysis_workflow.md](references/analysis_workflow.md) | OOM 分诊 → Step 1-5 → SP/TP/PP/EP 理论 → JSON 规范 |
| 实施 + 验证 + 调试 | [implementation_guide.md](references/implementation_guide.md) | 4 种模式 + 通信原语模板 + NPU 差异 + 验证模板 + ADR + 18 条坑 |

**场景提问**：接到多卡请求后用 `ask_user_question` 按优先级问（max 4 题/次）：① 场景/指标/独立样本数 → ② 模型架构/规模 → ③ 硬件拓扑/HBM。已明确的维度跳过。
