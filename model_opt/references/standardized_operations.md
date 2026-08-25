# 标准化操作规范

Profiling 采集、精度对比等操作必须遵循统一的规范，确保每次执行的环境和流程一致。agent 应根据具体项目编写或修改脚本，但必须满足以下约束。

## 标准项目目录结构

全流程统一的项目目录布局，Phase 0 项目初始化时建立（见 [00_adaptation/SKILL.md](../00_adaptation/SKILL.md) 0.1），后续所有 Phase 的产出都落在这个结构里：

```
<project>/
├── set_env.sh              # 一键环境启动（主目录）
├── run_inference.py        # 推理入口（主目录）
├── weights/                # 模型权重与配置（不提交）
├── scripts/                # 模型相关脚本：精度验证、benchmark、profiling 采集等（提交）
├── profiling/              # profiling 采集输出（不提交）
├── golden/                 # 迁移前 golden 输出（Phase 0 适配精度对齐，不提交）
├── baseline/               # 优化前 baseline 输出（Phase 1 优化回归，不提交）
├── comparison_records/     # 精度对比结果留存（提交，提交门禁证据）
└── evidence_db/            # 优化案例记录（提交）
```

- git clone 下来的模型库按此结构重组：已有脚本归入 `scripts/`，权重移动或软链到 `weights/`，推理入口保持在主目录
- 提交范围按目录决定：源代码与结论性记录（`scripts/`、`evidence_db/`、`comparison_records/`）提交；原始大体积数据（`weights/`、`profiling/`、`golden/`、`baseline/`）不提交

`.gitignore`：

```gitignore
# 原始大体积数据（目录级忽略）
weights/
profiling/
golden/
baseline/

# 运行时临时文件
__pycache__/
*.pyc
kernel_meta/
*.log
.venv/
```

## Profiling 采集规范

1. **环境变量**：采集脚本中必须在 `import torch_npu` 之前设置：
   - `TASK_QUEUE_ENABLE=2`（Host-Device 异步流水）
   - `CPU_AFFINITY_CONF=1`（CPU 绑核）
2. **输出路径**：保存到 `<workspace>/profiling/<YYYYMMDD_HHMMSS>/`，禁止使用 `/tmp` 或无时间戳的固定路径
3. **软链接**：每次采集后更新 `profiling/latest` 指向最新结果
4. **运行日志**：采集过程的 stdout/stderr 同时保存到输出目录下的 `run.log`
5. **一致性**：同一项目的多次采集必须使用相同的输入数据、warmup 步数、推理参数
6. **全覆盖 + 口径对齐**：wall-clock、L0、L1 三种测量必须覆盖完全相同的代码范围，且包含模型全部功能代码（禁止禁用任何功能组件）。详见 [profiling_collection.md](../01_preparation/references/profiling_collection.md) §三种性能测量及其覆盖范围。

## 精度对比规范

1. **对比对象**：始终与原始未优化的 baseline 对比，禁止与中间版本自比
2. **指标声明**：对比前必须明确使用的指标和阈值，不可事后调整
3. **结果留存**：对比结果（指标数值 + 判定结论）保存为文件，保存到 `<workspace>/comparison_records/<YYYYMMDD_HHMMSS>/`中作为提交门禁证据
4. **可复现**：对比脚本必须是自包含的（给定输入路径即可独立运行），不依赖临时变量或交互输入

## 脚本设计原则

agent 应为每个项目编写适配的采集/对比脚本。关键约束：一键可运行、时间戳防覆盖、关键配置参数化（不硬编码）、GPU/NPU 脚本保持相同参数和数据（仅设备不同）。

## 配置变更验证规范

1. **A/B 对比必须**：任何对 NPU 设置 / 环境变量 / 模型配置的变更，必须在变更前后各采集一次 L0 profiling，对比关键指标（utilization、kernel count、host time by category、相关算子开销）。只允许一个变量不同。
2. **采纳条件**：新配置在目标指标上改善且不退化其他指标时才采纳。某指标改善但另一退化时，必须在确认节点 A 中说明取舍理由。
3. **禁止盲目变更**：不得仅基于推理变更配置而不验证。profiling 数据是变更的唯一依据——"理论上应该更好"不是变更理由。
