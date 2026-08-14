# Level 2 任务层验证执行流程

## 定位

Level 2 是版本提交前的正式精度门禁。当一个优化阶段结束（可能包含了多次改动），准备提交版本时执行。不管优化类型是什么（数值等价还是近似算法），提交前都需要通过 Level 2。

**必须与 Profiling 确认性能收益一起作为提交前的两项门禁**——两项均通过后才可进入用户确认提交流程。

## 核心原则

- Level 2 比较的是 **baseline（优化前的已验收版本）** vs **optimized（当前要提交的版本）** 在标准评测集上的任务指标
- 评测规范必须有出处（model card / 论文 / 官方 README），不可自行编造
- 数据集必须是模型发布时使用的标准评测集，不可用随意样本代替
- 指标必须与官方报告一致，复用相同计算方式
- 阈值必须在评测前声明，不可看到结果后调整

## 执行步骤

### Step 1：确定评测规范

从模型出发，提取评测三要素：

| 字段 | 说明 | 示例 |
|------|------|------|
| 评测数据集 | 名称 + 子集/split | ImageNet-1K val, LibriSpeech test-clean |
| 评测指标 | 名称 + 计算方式 + 含义 | Top-1 Accuracy（分类精度，越高越好）|
| 允许退化量 | 在运行前声明 | 退化 < 1%（相对） |

信息源优先级：
1. HuggingFace Model Card（通常有 Evaluation Results 表格）
2. 原论文（Main results table）
3. GitHub README / benchmark 目录
4. 官方 eval 脚本（`eval.py`、`benchmark.py`）

### Step 2：获取评测数据集

获取路径优先级：
1. **本地已有**：检查项目中是否已有 `data/`、`datasets/` 等目录
2. **HuggingFace Datasets**：`datasets.load_dataset("dataset_name", split="test")`
3. **官方下载链接**：论文或 README 中给出的 URL
4. **框架内置**：`torchvision.datasets`、`torchaudio.datasets` 等

操作规范：
- 下载前先询问用户确认存储位置
- 记录数据集版本/hash，确保可复现
- 如果数据集过大（>10GB），先用小子集验证流程通畅，再跑全量

### Step 3：构建评测脚本

评测脚本必须满足：
- **与官方 eval 逻辑一致**：优先复用模型仓库自带的 eval 脚本，只改 device/路径
- **输入管道与发布一致**：前处理参数、tokenizer 版本必须匹配
- **指标计算使用标准库**（evaluate、pycocotools、torchmetrics、sklearn.metrics 等）
- **结果可复现**：固定 seed，记录所有超参

脚本结构：
```python
# task_eval.py
"""
模型: {model_name}
数据集: {dataset_name} ({split})
指标: {metric_name}
允许退化量: {threshold}（运行前声明）
"""

def load_dataset(): ...
def load_model(checkpoint_path, device): ...
def evaluate(model, dataset): ...

def main():
    # 1. 加载数据集
    # 2. 分别加载 baseline 和 optimized 模型
    # 3. 各自独立推理 + 计算指标
    # 4. 配对比较，输出判定结果
    ...
```

### Step 4：执行评测并判定

**关键：baseline 和 optimized 必须各自独立跑完整评测**，不共享任何中间结果。

判定逻辑：
```
退化量 = baseline 指标 - optimized 指标

退化量 <= 允许退化量  →  PASS
退化量 >  允许退化量  →  FAIL
```

允许退化量的确定：
- 由用户/项目声明（最优先）
- 用户未声明时的参考默认值：1% 相对退化为警告线，2% 为失败线
- 阈值必须在评测前写入脚本，不可看到结果后调整

关注 worst case：
- 不只看总平均指标，也关注不同数据切片的表现
- 如果总指标通过但某个子集退化严重，需要记录并报告

### Step 5：输出报告

```
=== Level 2 任务层验证报告 ===
模型: xxx
优化阶段: xxx（本阶段包含的改动概述）
评测数据集: xxx (样本数: N)
评测指标: xxx（含义: 衡量什么能力）

Baseline 实测: X.XX
Optimized 实测: X.XX
退化量: X.XX%（相对）

官方公开数值（校验用）: X.XX（来源: model card / 论文）

预设阈值: < Y% 退化
判定: PASS / FAIL

Profiling 门禁: PASS / FAIL（性能收益: X.Xx 加速比）
综合判定: 两项均 PASS → 可提交

复现信息:
- 评测脚本: path/to/eval.py
- 数据集位置: path/to/data
- 运行命令: python eval.py --args
```

## 常见模型类型的标准评测集

| 模型类型 | 标准评测集 | 获取方式 |
|---------|-----------|---------|
| 图像分类 | ImageNet-1K val (50K) | `torchvision` 或手动下载 |
| 目标检测 | COCO val2017 (5K) | `pycocotools` + 官方下载 |
| 语义分割 | ADE20K val / Cityscapes val | HF datasets 或官方下载 |
| LLM (通用) | WikiText-103 / C4 (ppl) | `datasets.load_dataset` |
| LLM (任务) | MMLU / HellaSwag / ARC | `lm-evaluation-harness` |
| 文生图 | COCO-30K (FID) / DrawBench (CLIP) | COCO val caps + 生成 |
| 语音识别 | LibriSpeech test-clean/other | `datasets.load_dataset` |
| 机器翻译 | WMT test sets | `sacrebleu` 内置 |

## 异常处理

| 情况 | 处理 |
|------|------|
| Model card 无评测结果 | 搜索论文 → 搜索第三方 benchmark → 询问用户 |
| 评测数据集需注册/付费 | 告知用户，询问替代方案或由用户提供 |
| 官方 eval 脚本不兼容 NPU | 最小改动适配（仅改 device），不改 eval 逻辑 |
| 评测耗时过长 | 先用 10% 子集估算趋势，确认方向正确后跑全量 |
| 基线数值与实测有系统偏差 | 检查前处理/后处理一致性，可能是版本差异，需向用户确认 |

## 与 Level 1 的关系

- Level 2 是最终裁判，Level 1 是快速筛查——两者不互相替代
- Level 1 通过多次不等于 Level 2 通过（精度问题可能多步叠加后才显现）
- Level 2 通过证明"任务表现在容差内"，但不证明"计算无 bug"——如果需要定位具体问题仍需 Level 3
