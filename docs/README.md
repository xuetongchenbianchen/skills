# docs 目录索引

历史改进方案文档按「**解决的问题 → 创建时间 → 文档**」三级组织。一级目录是方案要解决的问题，二级目录是文档创建日期（依据 git 首次提交时间及文档内标注日期）。

## 目录结构与问题对应

| 一级目录 | 解决的问题 |
|----------|-----------|
| `profiling_analysis_gaps/` | profiling 数据解析与分析能力不足：字段捕获不全、处理不充分、分析维度有缺口、脚本分类规则错误 |
| `analysis_methodology_gaps/` | 分析方法论与知识框架存在缺口：等价变换方法论薄弱、Phase 2 缺源码结构分析线、框架外发现无法沉淀 |
| `agent_process_discipline/` | agent 流程执行与决策纪律不足：有知识但流程松散导致遗漏优化点、不从第一性原理决策、只读 SKILL.md 不深入 references |
| `attribution_and_bound_analysis/` | 性能归因与优化上界估计不准：self-time 排序误导优化方向、缺少三档上界与 gap 分解理论、下界分析需重新设计 |
| `case_study_validation/` | 真实模型实测验证 skill 实效：MACE、7 模型 22 轮优化等案例的评估结论与改进输入 |
| `parallel_splitting_capability/` | 显存受限场景下缺少多卡并行切分能力：切分方法论、成本建模与 `07_parallel_splitting` 子技能设计 |
| `adaptation_opt_merge/` | model_adaptation 与 model_opt 双 skill 拆分导致流程入口假设错误、切分判定错位与内容重叠：合并为单一全流程 skill 的方案 |

## 文件清单

### profiling_analysis_gaps/ — profiling 数据解析与分析能力不足
- `2026-07-14/profiling_analysis_gap_analysis.md` — 对比 msprof-analyze 22 个 Checker，找出可补充的分析维度
- `2026-07-16/skills_improvement_plan.md` — 综合两阶段审计 + diff_profiling 审查的完整改进路线
- `2026-07-23/parse_coverage_audit.md` — Phase 1 审计：原始信息字段是否都被脚本捕获
- `2026-07-23/parse_processing_audit_phase2.md` — Phase 2 审计：已捕获字段的处理是否充分
- `2026-07-24/parse_scripts_data_analysis.md` — 逐脚本说明数据采集与分析逻辑
- `2026-08-09/host_category_dispatch_chain_analysis.md` — parse_operator_details.py Host Category 分类规则修复
- `2026-08-24/communication_analysis_enhancement_design.md` — 通信分析增强设计（parse_communication.py v2）

### analysis_methodology_gaps/ — 分析方法论与知识框架缺口
- `2026-07-14/skill_improvement_analysis.md` — 等价替换独立为第四维度等方法论缺口回应
- `2026-07-14/opt_explore_proposal.md` — 框架外新发现的探索与沉淀方案（opt_explore）
- `2026-07-15/dual_line_analysis_proposal.md` — Phase 2 双线分析模型：源码结构线 + profiling 数据线

### agent_process_discipline/ — agent 流程执行与决策纪律
- `2026-07-16/process_enforcement_improvement.md` — 从"知识丰富流程松散"到"知识 + 流程门禁"
- `2026-07-16/first_principles_enforcement.md` — 第一性原理决策倾向强化（parse 脚本增强 + 决策规则）
- `2026-07-16/realtest_issue_and_fixes.md` — 实测暴露的四个问题及修复

### attribution_and_bound_analysis/ — 性能归因与优化上界
- `2026-08-03/opt_line_research.md` — 低层瓶颈定位与三档优化上界的理论调研
- `2026-08-03/improvement_proposal.md` — 从"时间花在哪"升级为"优化空间在哪"的框架级方案
- `2026-08-09/bound_analysis_redesign.md` — 下界分析重新设计：还有没有空间、还有多少

### case_study_validation/ — 真实模型实测验证
- `2026-07-31/mace_eval_improvement_proposal.md` — 基于 MACE-MP-0 实测的 skills 改进方案
- `2026-08-03/OPTIMIZATION_SUMMARY.md` — 7 模型 22 轮 NPU 推理优化总结报告
- `2026-08-03/selected_single_card_models.csv` — 单卡评测模型选择清单

### parallel_splitting_capability/ — 多卡并行切分能力
- `2026-08-18/CHANGELOG.md` — 07_parallel_splitting 子技能新增说明
- `2026-08-21/parallel_splitting_general_analysis.md` — 并行切分通用分析：动作原理、成本建模、通信设计与性能诊断

### adaptation_opt_merge/ — 适配与优化 skill 合并
- `2026-08-25/merge_model_adaptation_into_model_opt.md` — model_adaptation 并入 model_opt 成为 Phase 0 的合并方案：入口判定、切分分诊迁移、职责归属调整
