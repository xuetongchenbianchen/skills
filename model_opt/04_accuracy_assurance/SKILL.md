---
name: NPU 精度保证
description: Phase 4 精度保证。确保 NPU 优化后的模型精度正确，包括基线管理、分层验证和精度调试。当用户关注精度验证、输出对比或精度问题定位时触发。执行前参见根 SKILL.md 全流程。
---

# NPU 精度保证

## 核心思路

推理精度验证有两种方式，分别对应不同的时机和成本：

**Level 1（快速验证）** 的逻辑是输出对比——比较 baseline 和 optimized 的输出，不需要 ground truth。采用两步瀑布流，严格在前、宽松在后：

```
Step 1（固定规则）：直接比较原始输出 tensor
  ├── 通过 → Level 1 通过，结束
  └── 不通过 → 进入 Step 2
Step 2（需设计）：比较任务代理量 Φ(Y)，用距离 D 量化变化
  ├── 通过 → 任务级可接受，Level 1 通过
  └── 不通过 → 精度退化，Level 1 拒绝
```

Step 1 问的是"输出是否几乎一致"——通过意味着基于输出的任何任务指标必然也一致，无需进一步检查。Step 2 问的是"输出虽然变了，但任务还能正常完成吗"——需要针对具体模型设计验证方案：选择什么可观测量 Φ（从输出中提取什么性质）、用什么距离 D（怎么量化变化）、阈值多少（多大变化可接受）。

关键在于 Step 2 选对 Φ——它必须捕捉"决定任务成败的那个性质"。选错 Φ 会导致误判。

**优化类型决定预期路径**：
- **数值等价优化**（算子融合、编译、等价 Kernel）→ 预期 Step 1 通过
- **近似算法优化**（量化、FP16、剪枝）→ 预期 Step 1 不通过，需要 Step 2 兜底

若声明数值等价但 Step 1 未通过，说明优化实现有 bug——应修复而非降级到 Step 2。

**Level 2（全量验证）** 的逻辑是任务评测——在完整测试集上跑标准评测 pipeline，直接看任务分数有没有退化：

```
完整测试集 + ground truth → 标准评测 pipeline → 任务分数 → 比较是否退化
```

不需要选 Φ，直接用领域标准的评测方式。这是精度的最终裁判。

推理和训练的验证复杂度差异很大：

- **推理**：比较最终推理产物即可，验证链路短
- **训练**：需要三层递进验证——单步数值对齐 → 训练过程对齐 → 最终结果对齐

## 基线管理

精度对比的第一步是确认 baseline 来源和资格。来源不对，整个结论无效。

**资格要求**：baseline 必须已通过适用任务的验收，并记录锁定其运行条件：
- 模型架构、权重与配置
- tokenizer、特征处理和确定性后处理
- 推理参数、随机策略与 seed 规则
- 运行时框架版本、精度模式、硬件型号

**两种场景的 baseline 不同**：

| 场景 | baseline 是什么 |
|------|----------------|
| 首次适配（模型第一次上 NPU） | 官方公布结果或用户 GPU 实际输出 |
| 推理加速优化（已跑通后做优化） | 优化前的已验收 NPU 模型 |

**首次适配时的来源优先级**：官方基线（README/benchmark）→ 用户 GPU 基线 → CPU 仅作调试辅助。详见 [baseline_policy.md](references/baseline_policy.md)

**通用强制规则**：
- 阈值必须在比较前声明，不可看到结果后放宽
- 样例级对齐 ≠ 模型级对齐
- baseline 变化时，已有验证结论自动失效

## 推理验证体系

### Level 1 快速验证（每次修改后）

每完成一次推理优化改动，立即执行 Level 1。目的是快速确认改动没有引入精度退化。

#### 瀑布流验证逻辑

Level 1 采用两步瀑布流判定——第一步严格、第二步宽松。第一步通过则必然意味着第二步也通过，因此可以提前结束。

```
Step 1（固定规则）：比较原始输出 tensor
  ├── 通过 → Level 1 通过，结束
  └── 不通过 → 进入 Step 2
        ↓
Step 2（需设计）：比较任务代理量（Φ/D/阈值，由 validation_plan.yaml 定义）
  ├── 通过 → 任务级可接受，Level 1 通过
  └── 不通过 → 精度退化，Level 1 拒绝
```

**为什么可以这样做**：Step 1 检查的是"输出是否几乎一致"。如果输出一致，那么基于输出的任何任务指标必然也一致。严格检查通过蕴含宽松检查通过。

**每条优化声明预期路径**：
- 数值等价优化（算子融合、编译、等价 Kernel）→ 预期 Step 1 通过
- 近似算法优化（量化、FP16、剪枝）→ 预期 Step 1 不通过，需要 Step 2 兜底

若声明数值等价但 Step 1 未通过 → 说明优化实现有 bug，不应降级到 Step 2，而应修复代码。

#### Step 1：原始输出比较（固定规则，无需设计）

对所有样本的模型最终输出 Y_base 和 Y_opt 执行：

1. **硬不变量检查**：shape 一致、无 NaN/Inf、值域合理 → 任一失败则直接拒绝
2. **数值距离**：
   - `cosine = cosine_similarity(flatten(Y_base), flatten(Y_opt))`
   - `max_diff = max(abs(Y_base - Y_opt))`
3. **通过条件**：所有样本 cosine > 0.99999 且 max_diff < baseline_variance × 10
   - `baseline_variance` 通过同一输入在 baseline 上跑 2-3 次测量最大差异得到
   - 若 baseline 完全确定性（bit-exact），则 max_diff 阈值取 1e-5 作为浮点安全余量

**操作要点**：关闭随机性 → 同输入分别跑 baseline 和 optimized → 保存输出 → 计算上述指标 → 判定。

#### Step 2：任务代理量比较（需设计，由 validation_plan.yaml 定义）

当 Step 1 不通过时（输出有明显数值差异），需要判断"虽然输出不一样了，但任务还能正常完成吗"。

这需要针对具体模型的任务设计验证方案：选择合适的 Φ（从输出中提取什么）、D（怎么量化差异）、阈值（多大差异可接受）。

核心推理链：

```
论文/model card → 理解任务 → 任务成败取决于输出的什么性质 → 选 Φ 捕捉该性质 → 选 D 量化 Φ 的变化 → 用户确定阈值
```

方案确定后冻结复用，后续每次需要 Step 2 时直接按方案执行比较。

详细的方案设计方法论见 [precision_validation_design.md](references/precision_validation_design.md)，各模型类型的具体 Φ/D 选择见 [model_family_hints.md](references/model_family_hints.md)。

#### 验证方案产出物：`validation_plan.yaml`

验证方案的推导结果必须保存为 `<workspace>/accuracy/validation_plan.yaml`。该文件在首次需要 Step 2 判定前生成，此后冻结复用。

**触发时机**：Phase 3 中第一条优化实施完成后、首次执行 Level 1 比较前。此时 agent 进入本模块完成 plan 设计，设计完成后再执行 Level 1。即使第一条优化预期走 Step 1 通过，plan 也应提前设计好以备后续近似优化使用。

**门禁**：该文件不存在 或 必填字段为空 → 任何已产生的精度验证结论无效。Agent 必须先补全 plan（需用户确认阈值），再对所有已实施的优化重新执行精度验证。优化代码本身无需撤销，但在 plan 补全前禁止执行新的 Level 1 比较。

详细的字段填写方法和各模型类型的推导示例见 [validation_plan_guide.md](references/validation_plan_guide.md)。

**Schema**：

```yaml
model: <模型名>
created_at: <ISO timestamp>

# --- 任务理解 ---
task:
  description: <一句话任务描述>
  source: <信息来源，如 "model card: https://..." 或 "config.json: model_type=resnet">
  decision_function: <从模型原始输出到最终任务结果的具体操作链>
  sensitive_to: <决策函数对输出的什么性质敏感>
  robust_to: <什么变化不影响决策>

# --- Φ 选择（单样本级）---
phi:
  definition: <对第 i 个样本的输出 Y_i，Φ(Y_i) = 什么>
  reasoning: <为什么选这个 Φ —— 必须引用 task.sensitive_to 做因果论证>
  anti_example: |
    <回答：能否构造一种情况使 "Φ(base) ≈ Φ(opt)（Φ 判定没问题）但任务实际失败"？>
    <如果能 → 说明 Φ 有盲区，写出具体反例，并修改 Φ 或增加 secondary>
    <如果不能 → 写出为什么不能（因果性论证）>
  secondary: <辅助 Φ，覆盖主 Φ 的盲区。如主 Φ 已有因果性保证则写 null>

# --- D 选择（单样本级）---
distance:
  definition: <d_i = D(Φ(Y_base_i), Φ(Y_opt_i)) = 什么>
  interpretation: <d_i 的含义：什么值代表好，什么值代表差>

# --- 阈值（用户设定）---
threshold:
  value: <>
  source: <>  # 只允许：causal:<解释> / empirical:<引用文档段落> / baseline_variance:<实测数据> / user_specified:<用户给定>
  # 禁止 source 写 "凭经验"、"一般认为"、"通常"、"参考网上"

# --- 样本设计 ---
samples:
  count: <32-100>
  source: <数据集名 + split + 抽取方式，如 "bridge_orig val, 按 trajectory 长度分层抽样">
  strategy: <分层方式>
  includes_hard_cases: <是否包含决策边界附近的困难样本，如何选取>
  fixed_seed: <随机种子>
  storage_path: <样本数据保存路径>
```

**字段完整性规则**（任一违反 → 文件无效 → 等同于不存在）：

| 字段 | 规则 |
|------|------|
| `task.decision_function` | 不可只写笼统词（"分类"/"生成"），必须写具体操作（argmax/cosine/decode/...） |
| `phi.reasoning` | 必须引用 `task.sensitive_to`，形成 "因为 X 敏感，所以 Φ 捕捉 X" 的推理链 |
| `phi.anti_example` | 不可为空。必须明确回答"能/不能构造 Φ 判定通过但任务失败的情况"并给出论证 |
| `threshold.source` | 必须以 `causal:` / `empirical:` / `baseline_variance:` / `user_specified:` 开头 |
| `samples.includes_hard_cases` | 不可为 false 或空 |
| `samples.source` | 必须指向真实数据集（格式：`数据集名 split, 抽取方式`）。禁止含 "random"/"synthetic"/"torch.randn"/"np.random"。仅当标准数据集确实不可获取时允许写 "manual_domain_valid:<理由>" |

#### 执行比较

- **样本**：从真实数据集中抽取的代表性输入（32-100），覆盖典型输入、边界 case、历史失败样例。禁止使用随机生成或合成数据——原因见 01_preparation §三
- **比较对象**：模型最终公开输出（不是中间层）
- **条件**：同输入、同配置、同 seed（seed 配对条件见方案设计文档）
- **度量**：Step 1 用固定规则（cosine + max_diff）；Step 2 按 plan 中的 Φ 和 D 计算
- **关注 worst case**：报告均值和最大偏差样本，不只看平均

#### 判定

**Step 1 不通过 + 声明数值等价**：
- 说明优化实现有 bug → 不降级到 Step 2，修复代码后重新验证

**Step 1 不通过 + 声明近似算法 → 进入 Step 2**：
- 硬不变量失败（shape/NaN/Inf/值域）→ 拒绝
- 所有样本 d_i 在阈值内 → 通过
- 代理指标处于不确定区域 → 标记为"有风险"，继续开发但在 Level 2 时重点关注

#### Level 1 结论约束

- 通过的正确表述："数值层快速检查未发现明显退化"
- 不可声称"精度已保证"或"任务表现无影响"——那是 Level 2 的结论
- 多次 Level 1 通过的累积不等于 Level 2 通过

### Level 2 全量验证（每批提交前，提交门禁之一）

当一个优化阶段结束（可能包含了多次改动），准备提交版本时执行。

- **样本**：完整测试集，覆盖所有推理路径，关注 worst case 而非平均值
- **比较对象**：模型直接输出 + 下游功能指标（如果直接输出的不确定性决定下游表现）
- **度量**：直接输出用适当的距离函数；下游指标用领域功能度量（accuracy / FID / WER / mAP 等）
- **目的**：确认累积改动无精度退化——Level 2 通过证明"任务表现在容差内"，但不证明"计算无 bug"

**必须与 Profiling 确认收益一起作为提交前的两项门禁**——两项均通过后才可进入用户确认提交流程。

任务层验证的详细执行流程见 [task_level_eval_workflow.md](references/task_level_eval_workflow.md)。

### Level 3 精度调试（验证未通过时）

当 Level 1 或 Level 2 检测到精度退化，进入调试流程定位问题根因。

**逐层精度追踪（核心方法）**：在优化前模型和优化后模型中，对每一层的中间输出注册 hook 保存，逐层计算距离函数，定位第一个显著退化的层。

适用场景：
- 多层模型累积精度退化
- 疑似特定算子引入的精度问题
- Level 1 数值等价路径失败，需要找到 bug 位置

注意：逐层对比的阈值应比最终输出宽松——单层误差经后续层放大后才在最终输出上显现。

详见 [debugging_guide.md](references/debugging_guide.md)

## 训练场景：三层递进验证

训练精度对齐比推理复杂得多，需要从微观到宏观逐层验证。

### Layer 1：单步数值对齐（微观）

固定相同输入、相同初始权重、关闭所有随机性，对比 GPU 与 NPU 单步训练的数值。

| 对齐点 | 方法 | 指标 | 参考阈值 |
|--------|------|------|----------|
| 前向输出 | 逐层 hook 抓取激活值 | cosine sim + max abs diff + 相对误差 | 单步相对误差 < 5% |
| Loss 值 | 同一输入对比 loss 标量 | 绝对误差 + 相对误差 | 相对误差 < 1% |
| 反向梯度 | `loss.backward()` 后取 `param.grad` | cosine sim + 梯度范数比 + max abs diff | cosine > 0.999 |
| 参数更新 | `optimizer.step()` 后对比权重变化量 | 相对误差 + max abs diff | 相对误差 < 1% |

问题定位方法：逐模块比对 → 模块内二分 → 定位到具体算子。

### Layer 2：训练过程对齐（中观）

跑多步或完整 epoch，对比训练曲线和动态行为。

| 对齐点 | 方法 | 指标 | 参考阈值 |
|--------|------|------|----------|
| Loss 曲线 | 逐 step 对比 GPU/NPU loss 值 | 平均相对误差 + 最大绝对偏差 | 平均相对误差 < 5% |
| 收敛速度 | 对比达到目标 loss 的 step 数 | step 数比值 | 偏差 < 10% |
| 训练稳定性 | 对比 loss 方差、梯度范数变化 | loss 滑动方差比 + grad norm 曲线 | 无异常震荡或发散 |

### Layer 3：最终结果对齐（宏观）

完整训练后，对比最终模型质量。

| 对齐点 | 方法 | 指标 | 参考阈值 |
|--------|------|------|----------|
| 任务指标 | 在相同验证/测试集上评估 | accuracy / F1 / BLEU / perplexity 等 | 在 GPU 多次运行自然波动范围内 |
| 最终权重 | 对比训练结束后的 state_dict | cosine sim + max abs diff | cosine > 0.99 |

### 训练对齐验证流程

```
Layer 1 单步对齐 ── 通过 ──→ Layer 2 过程对齐 ── 通过 ──→ Layer 3 最终结果
      │                            │                            │
    不通过                        不通过                        不通过
      ↓                            ↓                            ↓
  逐层二分定位问题算子        检查超参/数据一致性         检查累积误差/随机性
```

## 确定性保证

NPU 结果必须可复现。推理和训练各有侧重：

**推理确定性条件**（精度对比的前提，必须在比较前确认）：
```python
model.eval()
torch.manual_seed(42)
torch.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.config.allow_internal_format = False
# 确认无 dropout / random 路径
```

验证方法：同一输入跑两次，确认结果 bit-exact。不确定意味着 NPU 配置有问题，对比结果不可信。

若结果不可复现，排查：随机性未关闭、JIT 编译引入不确定性、NPU 内部格式转换引入偏差。

**训练确定性条件**（更严格，是单步对齐的前提）：
- 固定所有随机种子：`random.seed()` / `np.random.seed()` / `torch.manual_seed()` / `torch.npu.manual_seed_all()`
- `torch.backends.cudnn.deterministic = True`，关闭 `benchmark`
- `torch.use_deterministic_algorithms(True)`（如适用）
- DataLoader 设置 `worker_init_fn` 固定各 worker 种子
- NPU 侧关闭私有格式等可能引入差异的配置（如 `FLAGS_npu_storage_format=0`）
- 确认 NPU 确定性计算模式已开启

## 常见误区

1. 只验证第一步就判定精度正确——某些 bug 只在后续步骤暴露
2. 将浮点累积偏差误判为 bug——关键看 diff 增长是否平滑
3. 与优化后的模型自己对比——必须与原始未优化的 baseline 对比
4. 只看平均指标——必须关注 worst case
5. 对离散输出要求 bit-exact——自回归/随机采样中不可能
6. 未确认基线就写对比脚本——基线来源不对，整个结论无效
7. 看到 mismatch 后临时放宽阈值——阈值必须在比较前声明
8. 训练场景只看 loss 曲线——必须同时验证单步梯度和最终模型质量
9. 跳过单步对齐直接看收敛——单步就有问题时，收敛结果不可信
10. 未固定随机性就做训练对齐——确定性是单步对齐的前提
11. 机械选 cosine 作为万能指标——cosine 对某些模型没有任务预测力（如检索模型需要 NC@k）
12. 数值等价路径失败后用任务分数替代——数值声明必须用数值证据满足
13. 把有限样本的输出验证声称为数学等价——有限经验测试不能证明数学等价

## 结论措辞规范

| 实际做了什么 | 可以说 | 不可以说 |
|-------------|--------|---------|
| Level 1 输出对比通过 | 快速检查未发现明显退化 | 精度已保证 |
| Level 2 全量评测通过 | 任务表现在容差内，精度验收通过 | 数学等价 |
| 少量样本对齐 | 样例级结果对齐 | 模型已完成精度对齐 |
| 仅 CPU vs NPU 对比 | 调试性比较，不作为最终结论 | 精度已对齐 |

## 参考资料索引

| 文件 | 加载时机 |
|------|----------|
| [precision_validation_design.md](references/precision_validation_design.md) | Level 1 首次接触模型，需要设计验证方案时 |
| [model_family_hints.md](references/model_family_hints.md) | 需要按模型类型查找具体 Φ/D 选择时 |
| [baseline_policy.md](references/baseline_policy.md) | 需要确认基线来源或向用户询问时 |
| [task_level_eval_workflow.md](references/task_level_eval_workflow.md) | Level 2 执行任务层验证时 |
| [debugging_guide.md](references/debugging_guide.md) | Level 3 定位精度问题时 |
| [checklists.md](references/checklists.md) | 需要快速核对配置项时 |

## 配套脚本（参考实现）

以下脚本位于本 skill 的 `scripts/` 目录下，作为参考实现。agent 应根据具体项目的输出格式和对比需求编写适配的对比脚本，但设计原则必须一致：一键可运行、指标和阈值在运行前声明、结果保存为文件。执行时需使用脚本的完整路径。

| 脚本 | 用途 | 适用场景 |
|------|------|----------|
| `scripts/compare_inference.py` | 比较两份推理输出 | 输出为标准格式文件时可直接使用 |
| `scripts/compare_loss.py` | 比较两份训练日志中的标量信号 | 日志格式规整时可直接使用 |
| `scripts/scan_baseline_bundle.py` | 扫描用户提供的基线目录 | 首次接触基线目录时快速了解结构 |
