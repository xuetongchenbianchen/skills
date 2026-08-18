---
name: npu-adaptation-preparation
description: NPU 适配前期准备：代码理解、CANN 环境搭建、测试数据、profiling 采集与精度验证脚本构建。当用户需要把模型在 NPU 上跑通、搭建/诊断 CANN 环境、采集基线 profiling、或构建精度对比脚本时触发。
---

# NPU 适配前期准备

---

## 一、模型代码理解

**目标**：摸清推理路径，避免在错误位置改代码。

### 系统性探索顺序
从全局到局部：包入口（`__init__.py`）→ 模型类定义（含 forward）→ 推理入口（main/CLI/pipeline）。关注 config.json 中的架构参数（hidden_size, num_layers, num_heads 等）。

### 已有文档阅读
- 阅读项目 README、设计文档
- 若有已有适配文档（如其他版本、同系列模型），列出差异点：**可复用 vs 需纠偏**

### 特殊输入识别
- 非文本输入（图像、蛋白质序列、音频等）需确认预处理路径和 tokenizer/encoder 是否独立
- 检查 `collate_fn` / `DataCollator` 是否有设备绑定逻辑

---

## 二、环境准备

完整环境变量清单见 [environment_reference.md](references/environment_reference.md)。

### CANN 环境诊断
```bash
npu-smi info                          # 芯片型号、驱动版本、卡数、空闲卡
ls $ASCEND_HOME_PATH/opp/             # OPP 算子包
source ~/Ascend/ascend-toolkit/latest/set_env.sh  # 激活工具链
```

### 多卡管理
```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3   # 逻辑编号从 0 开始，可能与物理编号不一致
```

### 版本配套确认
```python
import torch, torch_npu, torchair
print(torch.__version__, torch_npu.__version__, torchair.__version__)
# CANN 版本：cat $ASCEND_HOME_PATH/version.cfg
```
> 版本不配套是最常见的环境问题，优先确认 torch_npu 与 CANN 版本对应表。

### 设备兼容层（init_device 模板）

```python
def init_device(device_str: str):
    if device_str.startswith("npu"):
        import torch_npu
        torch.npu.set_compile_mode(jit_compile=False)
        torch_npu.npu.config.allow_internal_format = False
    return torch.device(device_str)
```

> 默认关闭 jit_compile 和 allow_internal_format 确保确定性。若 profiling 显示相关瓶颈，可逐个开启并验证精度无退化。

---

## 三、项目 Git 初始化

```bash
cd /path/to/project
git init
git checkout -b optimize/main
```

### .gitignore 必须覆盖的大文件

```gitignore
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
*.h5
profiling/
*.prof
*.trace.json
ASCEND_PROFILER_OUTPUT/
__pycache__/
*.pyc
kernel_meta/
.venv/
```

---

## 四、测试数据准备

**原则**：小而代表性，显式覆盖多个维度，不能只按长度分桶。

### 覆盖性检查清单

构造或抽样时逐项确认，每个维度至少 1-2 条代表：

| 维度 | 说明 | 示例 |
|------|------|------|
| **输入规模** | 最短、中位、最长 | 1 token / 512 / 2048 |
| **Batch 形态** | batch=1 和 batch>1；不同 padding 比例 | 全填充 / 混合长度 |
| **边界条件** | padding 边界、mask 形状、空/极短输入 | 全 pad、仅 1 有效 token |
| **数值范围** | 正常 + 极值（接近 dtype 上溢/下溢） | fp16 max 附近值 |
| **语义多样性** | 不同任务域/场景/输入类型 | LLM：代码/数学/对话/长上下文 |
| **推理路径** | 触发不同代码分支 | prefill vs decode；有/无目标 |

### 性能测试集

统计稳定的端到端时间测量。选取标准：
- 代表真实分布（有生产日志则按真实分布抽样）
- 50–200 条，含 warmup（前 N 条不计入统计）
- 用于优化前后对比 wall-clock 均值和 P95

### L1 Profiling 样本

从性能测试集中选 1 条生产 shape 分布的中位样本用于 L1 采集。L1 数据量大（可达数 GB）、分析重，只需落在生产主流瓶颈 regime 内即可——同 regime 内的 shape 差异只带来线性缩放，不改变瓶颈画像结构，无需多采。

L0 + wall-clock 在性能测试集上跨 shape 采集作为兜底：每轮优化后验证收益时，若某 shape 的 L0 结构（L0 Free / L0 Computing 比例）与主流 regime 质变，或优化收益不符预期，再对该 shape 做 targeted L1。精度校验独立用丰富数据集覆盖多维 shape/content。

---

## 五、Profiling 采集体系构建

完整代码模板和框架适配方案见 [profiling_collection.md](references/profiling_collection.md)。

### 输出路径规范

```
<workspace>/profiling/
├── YYYYMMDD_HHMMSS/
│   ├── *.csv
│   └── trace_view.json
└── latest -> YYYYMMDD_HHMMSS
```

强制规则：必须用 `tensorboard_trace_handler` + 时间戳路径；禁止 `/tmp`；禁止 `export_chrome_trace`（不产出 `step_trace_time.csv`）。

### 采集级别

| 级别 | 内容 | 数据量 |
|------|------|--------|
| **L0** | 仅 NPU 活动 | 小 |
| **L1** | CPU + NPU + 算子详情 + 调用栈 + 内存 + AI Core 指标 | 大 |

### 三种性能测量

wall-clock（无 profiler 真实时间）、L0（设备执行时间）、L1（瓶颈分析数据）。三者**必须覆盖完全相同的代码范围**。

### 采集前必设

```bash
export TASK_QUEUE_ENABLE=2    # Host-Device 异步流水，接近生产环境真实性能
export CPU_AFFINITY_CONF=1    # CPU 绑核，减少调度抖动
```

### 关键规则

- **不改业务代码的 CUDA 写法**：`import torch_npu` + `transfer_to_npu` 自动转换
- **多推理路径分离采集**：prefill/decode 等分别建立独立 profiling 段
- **GPU/NPU 对称采集**：跨平台对比时 GPU 用相同 schedule + `CUDA` activity

### 环境预检

采集前运行：
```bash
python <skill_path>/01_preparation/scripts/validate_profiling_env.py --device npu:x --output-dir ./profiling
```

---

## 六、精度验证脚本构建

**目标**：保存可信 baseline 输出，构建可一键运行的对比脚本。

### Baseline 来源与对比策略

遵循 [04_accuracy_assurance/SKILL.md](../04_accuracy_assurance/SKILL.md)：按优先级确认基线来源，按输出类型选择距离函数，阈值在比较前声明。本阶段只构建脚本框架，不定义具体方法论。

### 输出保存

原则：离线可加载、不依赖设备、可复现。

```python
import numpy as np, json
np.save(f"baseline/{sample_id}_output.npy", output.cpu().float().numpy())
with open(f"baseline/{sample_id}_tokens.json", "w") as f:
    json.dump({"tokens": token_ids}, f)
```

### 对比脚本要求

自包含（给定 baseline 目录和当前输出目录即可独立运行），指标和阈值显式声明，运行后输出判定结论。

### 确定性条件

推理：`model.eval()` + `torch.no_grad()` + 固定输入 + 关闭随机性。详见 04_accuracy_assurance「一、前提条件」。
