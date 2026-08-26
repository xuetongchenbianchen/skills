---
name: npu-accuracy-assurance
description: 精度保证：优化回归基线管理、分层验证、精度调试。当用户需要验证优化后精度、构建优化前后精度对比脚本、对齐训练过程、或调试精度问题时触发。
---

# NPU 精度保证

精度验证的本质是**定义一个距离函数 d(baseline, optimized)，然后检查 d < 阈值**。

本文件按执行顺序组织：前提条件 → 确认基线 → 确定比什么 → 选择怎么比 → 执行验证 → 不通过时处理。

## 运行前统一原则：NPU 资源检查

**每次需要运行代码（benchmark / profiling / 精度验证 / 功能测试等）之前，必须先用 `npu-smi info` 检查 NPU 上是否有与本任务无关的进程。若存在，先确认这些进程与当前任务无关（必要时向用户确认归属），再清理进程、释放显存，确认资源干净后才开始运行。** 无关进程会争抢算力/显存，导致性能数据失真或 OOM。

```bash
npu-smi info        # 检查各卡上的进程占用
ps -fp <PID>        # 确认进程身份与归属
kill <PID>          # 确认无关后清理（顽固进程用 kill -9）
npu-smi info        # 复查确认显存/算力已释放
```

---

## 一、前提条件：确定性保证

精度对比的一切前提是输出可复现。不确定的输出无法做有意义的对比。

**推理场景**：`model.eval()` + `torch.no_grad()` + 固定输入 + 关闭随机性。若结果不可复现，排查：随机性未关闭、JIT 编译引入不确定性、NPU 内部格式转换引入偏差。

**训练场景**（要求更严格，是单步对齐的前提）：
- 固定所有随机种子：`random.seed()` / `np.random.seed()` / `torch.manual_seed()` / `torch.npu.manual_seed_all()`
- `torch.backends.cudnn.deterministic = True`，关闭 `benchmark`
- `torch.use_deterministic_algorithms(True)`（如适用）
- DataLoader 设置 `worker_init_fn` 固定各 worker 种子
- NPU 侧关闭私有格式等可能引入差异的配置（如 `FLAGS_npu_storage_format=0`）
- 确认 NPU 确定性计算模式已开启

**验证方法**：同一输入跑两次，确认 bit-exact（推理）或 diff 在自然波动范围内（训练）。不通过则先修复确定性问题，再进入后续流程。

---

## 二、确认基线

精度对比的前提是有一个可信 baseline。来源不对，整个结论无效。

**基线语义（本子技能 = 优化回归）**：baseline 固定为**优化前 NPU 自身输出**，由 Phase 1 采集保存（见 [01_preparation/SKILL.md](../01_preparation/SKILL.md)「三、精度回归脚本构建」）。优化验证回答的是"优化有没有引入退化"，只要求优化前后对齐，**不依赖外部 golden**。与迁移前 golden（GPU/CPU 原始实现输出）的对齐属于 Phase 0 适配验证（见 [00_adaptation/SKILL.md](../00_adaptation/SKILL.md) 0.5），不在本阶段范围。

**切分场景（07 轨道）**：锚点配置（单卡可跑通）以切分前单卡输出为 baseline；生产规模（单卡放不下）以切分后、优化前的并行输出为 baseline——切分正确性由 `verify_split.py` 等价性验证背书，切分不改变模型数学语义。

**强制规则**：
- 始终与优化前的原始 baseline 对比，禁止与中间版本自比
- 先确认 baseline 可用（文件齐全、输入一致），再写对比脚本
- 样例级对齐 ≠ 模型级对齐（部分样本通过只能证明该范围内的对齐）

**自一致性验证**：优化前必须验证 baseline 自身确定性——用相同输入运行原始模型两次，确认输出一致。若 baseline 不确定，精度对比的阈值必须大于 baseline 自身波动。

---

## 三、确定比什么（验证边界）

"最终产物"有两层，选错层会导致验证结论无意义：

| 层次 | 说明 | 特点 |
|------|------|------|
| **模型直接输出** | forward 的返回值（logits、energy、图片、坐标等） | 始终可获取，快速 |
| **下游功能指标** | 用模型输出完成任务的评价（benchmark 分数、RMSD 等） | 对齐下游消费者真正关心的东西 |

**判断规则**：从模型直接输出到下游消费之间，是否有采样、随机过程、或非线性放大？
- **没有** → 验证边界 = 模型直接输出（直接输出等价即保证下游等价）
- **有** → 验证边界 = 下游功能指标（直接输出不等不代表功能退化）

### 常见模型类型速查

| 模型类型 | 直接输出 | 验证边界 | 推荐度量 |
|----------|----------|----------|----------|
| LLM（贪心） | logits | 直接输出 | cosine + max_abs |
| LLM（采样） | logits | 下游功能指标 | MMLU / HumanEval 等 benchmark |
| CV 分类 | logits | 直接输出 + Top-1 | cosine + accuracy |
| CV 检测/分割 | bbox/mask | 下游功能指标 | mAP / mIoU |
| Diffusion | 像素矩阵 | 下游功能指标 | LPIPS / FID |
| 力场（MACE 等） | energy/forces | 直接输出 | 相对误差 + max_abs |
| 蛋白质结构 | 3D 坐标 | 下游功能指标 | RMSD / TM-score |

更多模型类型的详细说明见 [model_family_hints.md](references/model_family_hints.md)。

---

## 四、选择怎么比（距离函数 + 阈值）

### 距离函数

按输出类型选择，且必须匹配下游消费者认为"等价"的语义：

**连续向量**（logits、forces、embedding）：
- cosine similarity + max abs diff，两者同时满足才通过
- **排除无效位置**：计算前按 attention_mask 过滤 padding 位置

**聚合标量**（energy、loss、score）：
- 相对误差 |a-b|/|a|；baseline 接近 0 时用绝对误差

**分布**（attention weights、概率分布）：
- KL / JS 散度。**前提**：输入必须是归一化概率分布；对 raw logits 应先 softmax 或直接用 cosine

**离散序列**（token ids、标签）：
- 完全匹配率。注意自回归中 logits 微小 diff 可能导致 argmax 翻转

**图片/矩阵**（diffusion 输出、feature map）：
- 感知级度量：LPIPS / FID / SSIM（像素级 diff 无语义意义）

**领域功能指标**（AI4S）：
- 蛋白质：RMSD / TM-score；天气：空间 RMSE / ACC；分子动力学：轨迹偏差 / 物理量守恒

### 阈值

阈值不是固定的，由三个因素决定：

1. **dtype 上下文**：fp32 自然 diff ~1e-6；fp16/bf16 ~1e-3。阈值应高于自然 diff 但低于"功能错误"的 diff。
2. **自然波动基准（必测）**：在 baseline 模型上用相同输入运行 2-3 次，计算各次之间的距离函数值。**此步不可跳过**——不测就无法区分"优化引入的差异"和"硬件固有抖动"。该值即为阈值的下界。
3. **验证范围**：单步等价用紧阈值；累积多步的端到端验证用松阈值。

参考起点：fp32 单步 cosine >= 0.9999 / max_abs < 1e-4；fp16 单步 cosine >= 0.999 / max_abs < 1e-2。最终以自然波动基准为准。

**阈值必须在比较前声明，不可事后调整。**

---

## 五、执行验证

### 推理场景：分级验证

**Level 1 快速验证**（每次修改后）：
- 样本：少量代表性输入（覆盖边界 case）
- 比较：模型直接输出，按输出类型选距离函数
- 目的：快速确认改动没有引入 bug

**Level 1.5 任务级降级验证**（Level 1 不通过时触发）：
- 触发条件：Level 1 不通过，**且**模型属于"输出到消费间有随机性"类型（LLM 采样、Diffusion 等）
- 做法：用**极小子集**的下游任务评测给出方向性信号
  - LLM：从论文 benchmark 中抽 10-20 题，确认分数在报告值合理范围内
  - Diffusion：对 5-10 个固定 prompt 生成图片，计算 LPIPS/FID
  - 蛋白质：对 3-5 个已知结构计算 TM-score
- 判定：任务级分数在官方报告值 ±2% 范围内则通过

**Level 2 全量验证**（每批提交前，提交门禁之一）：
- 样本：完整测试集，覆盖所有推理路径，关注 worst case
- 比较：模型直接输出 + 下游功能指标（如适用）
- **必须与 Profiling 确认收益一起作为提交前的两项门禁**

**Level 3 精度调试**（Level 1/2 不通过时）：
- 逐层对比定位退化起始位置（注册 hook 保存中间输出，逐层计算距离函数）
- 详见 [debugging_guide.md](references/debugging_guide.md)

### 训练场景：三层递进验证

训练精度对齐从微观到宏观逐层验证，每层有独立的距离函数和阈值：

#### Layer 1：单步数值对齐（微观）

固定相同输入、相同初始权重、关闭所有随机性，对比单步训练的数值。

| 对齐点 | 距离函数 | 参考阈值 |
|--------|----------|----------|
| 前向输出 | cosine + max abs + 相对误差 | 相对误差 < 5% |
| Loss 值 | 绝对误差 + 相对误差 | 相对误差 < 1% |
| 反向梯度 | cosine + 梯度范数比 + max abs | cosine > 0.999 |
| 参数更新 | 相对误差 + max abs | 相对误差 < 1% |

问题定位：逐模块比对 → 模块内二分 → 定位到具体算子。

#### Layer 2：训练过程对齐（中观）

跑多步或完整 epoch，对比训练曲线和动态行为。

| 对齐点 | 距离函数 | 参考阈值 |
|--------|----------|----------|
| Loss 曲线 | 平均相对误差 + 最大绝对偏差 | 平均相对误差 < 5% |
| 收敛速度 | step 数比值 | 偏差 < 10% |
| 训练稳定性 | loss 方差比 + grad norm 曲线 | 无异常震荡或发散 |
| 优化器状态 | state_dict 逐项对比 | 完全一致 |

#### Layer 3：最终结果对齐（宏观）

| 对齐点 | 距离函数 | 参考阈值 |
|--------|----------|----------|
| 任务指标 | 任务相关（accuracy / F1 / BLEU 等） | 在多次运行自然波动范围内 |
| 最终权重 | cosine + max abs | cosine > 0.99 |
| 泛化一致性 | 指标分布差异 | 无系统性偏差 |

不通过时：Layer 1 → 逐层二分定位算子；Layer 2 → 检查超参/数据一致性；Layer 3 → 检查累积误差/随机性。

---

## 六、常见误区

1. 只验证第一步就判定精度正确——某些 bug 只在后续步骤暴露
2. 将浮点累积偏差误判为 bug——关键看 diff 增长是否平滑
3. 与优化后的模型自己对比——必须与原始未优化 baseline 对比
4. 只看平均指标——必须关注 worst case
5. 对离散输出要求 bit-exact——自回归/随机采样中不可能
6. 未确认基线来源就写对比脚本，或看到 mismatch 后临时放宽阈值
7. Level 1 不通过就判定失败——对"输出到消费间有随机性"的模型，应升级到任务级验证
8. 训练场景只看 loss 曲线——必须同时验证单步梯度和最终模型质量

---

## 参考资料

| 文件 | 内容 | 加载时机 |
|------|------|----------|
| [model_family_hints.md](references/model_family_hints.md) | 各模型类型的验证边界和度量详细说明 | 遇到不在速查表中的模型类型时 |
| [debugging_guide.md](references/debugging_guide.md) | Level 3 精度调试的定位方法 | 验证不通过需要定位问题时 |
| [baseline_policy.md](references/baseline_policy.md) | baseline 完整性检查清单和报告措辞 | 使用 Phase 1 baseline 前或撰写结论时 |
