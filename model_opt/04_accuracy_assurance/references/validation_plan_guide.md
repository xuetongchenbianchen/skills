# validation_plan.yaml 填写指南

## 定位

本文档指导 agent 正确填写 `accuracy/validation_plan.yaml` 的每个字段。首次为某个模型设计验证方案时加载（Phase 3 中首次需要 Level 1 时触发）。

核心原则：**每个字段的值必须是推导结果，不是默认值。推理过程通过 reasoning / source / anti_example 字段展示。**

---

## 好的填法 vs 糊弄的填法

### task.decision_function

| 糊弄 | 正确 |
|------|------|
| `"分类"` | `"logits [B, 1000] → argmax(dim=-1) → class_id"` |
| `"特征提取"` | `"embedding [B, 768] → cosine_sim(query, gallery) → argmax → retrieved_id"` |
| `"文本生成"` | `"logits [vocab] → argmax(greedy) → next_token → repeat → full_sequence"` |
| `"语音识别"` | `"CTC logits [B, T, vocab] → argmax → collapse_repeats_and_blanks → text"` |

规则：必须写出具体操作（argmax / cosine / decode / threshold / softmax+sample），不写笼统任务名。

### phi.definition

| 糊弄 | 正确 |
|------|------|
| `"cosine similarity"` | 这是 D 不是 Φ |
| `"模型输出"` | 太笼统 |
| `"embedding"` | 哪个？shape？ |

正确写法：
- `"Φ(Y_i) = argmax(Y_i, dim=-1) — 第 i 个样本的预测类别 ID（标量）"`
- `"Φ(Y_i) = Y_i[:, 0, :] — 第 i 个样本的 CLS token embedding（768-dim vector）"`
- `"Φ(Y_i) = CTC_decode(argmax(Y_i, dim=-1)) — 第 i 个样本解码后的文本字符串"`
- `"Φ(Y_i) = greedy_generate(model, prompt_i, max_tokens=32) — 第 i 个 prompt 的生成 token 序列"`

规则：Φ 是从模型输出 Y_i 中提取一个东西。写清楚 "对单个样本做什么操作" 和 "输出的类型/shape"。

### phi.anti_example

| 糊弄 | 正确 |
|------|------|
| `"无法构造"` （无论证） | 见下方完整示例 |
| `""` （空） | 文件无效 |
| `"不存在这种情况"` （无论证） | 文件无效 |

**正确示例 1 — 能构造（说明 Φ 有盲区）**：

```
分类模型，主 Φ = cosine(logits)：

能构造：
  Y_base = [1.01, 1.00, 0, 0, ..., 0]  (1000-dim)
  Y_opt  = [1.00, 1.01, 0, 0, ..., 0]
  cosine(Y_base, Y_opt) ≈ 0.99999 → Φ 判定"没问题"
  但 argmax(Y_base)=0, argmax(Y_opt)=1 → 预测翻转 → 任务失败

结论：cosine 作为唯一 Φ 存在盲区。
修正：主 Φ 改为 argmax agreement，cosine 降为 secondary。
```

**正确示例 2 — 不能构造（因果性论证）**：

```
分类模型，主 Φ = argmax(Y_i)：

不能构造：
  如果 argmax(Y_base_i) == argmax(Y_opt_i)，则该样本预测的类别相同。
  预测类别相同 → 分类任务对该样本的结果必然相同。
  这是因果性保证（argmax 一致 → 任务不变），不依赖任何经验假设。
  因此不存在 "Φ 通过但任务失败" 的情况。
```

**正确示例 3 — 不能构造（但有条件限制）**：

```
LLM，主 Φ = greedy_decode(Y_i)，条件 do_sample=False：

不能构造（限定条件下）：
  greedy decode 完全由每步 argmax(logits) 决定。
  如果 baseline 和 optimized 每步的 argmax 相同 → token 序列必然相同。
  因果性保证：生成序列一致 → 文本一致 → 任务结果一致。
  
  但注意：此论证仅在 do_sample=False 时成立。
  若 do_sample=True，则输出依赖随机数，同 seed 不保证相同
  （优化可能改变随机数消费顺序）→ 此 Φ 不适用于 sampling 路径。
```

### threshold.source

**核心原则：阈值的数值由用户（人）设定，agent 不通过实验自行确定阈值。**

agent 的职责是建立验证体系（选择 Φ 和 d），但"多大偏差可以接受"是业务决策，由人给出。
agent 可以向用户提供参考信息（如 model_family_hints.md 中的经验范围），帮助用户做决定，但最终数值必须由用户确认。

| 糊弄 | 正确 |
|------|------|
| `"凭经验选 0.999"` | 无效 |
| `"通常认为够了"` | 无效 |
| `"参考网上说法"` | 无效 |
| `"0.99"` （无 source） | 无效 |

正确写法（必须以指定前缀开头）：

| 前缀 | 含义 | 示例 |
|------|------|------|
| `causal:` | 因果性保证（逻辑推导） | `"causal: argmax agreement=1 → 预测必一致，无需余量"` |
| `empirical:` | 引用文档/论文经验值 | `"empirical: model_family_hints.md「CV/视觉模型」— FP16 cos>0.999 时 5 模型无退化"` |
| `baseline_variance:` | baseline 自身方差测量 | `"baseline_variance: 同输入跑 3 次 max_diff=1e-7，取 1e-5 为安全线"` |
| `user_specified:` | 用户直接给定 | `"user_specified: 用户要求 cosine > 0.9999，基于业务容忍度"` |

注意：不再使用 `calibrated:` 前缀。阈值不应由 agent 通过实验校准得出——这会导致"自己考试自己打分"的循环。

---

## 完整推导示例

### 示例 1：ResNet50（分类）

```yaml
model: resnet50
created_at: "2026-08-12T08:00:00"

task:
  description: "ImageNet 1000-class image classification"
  source: "config.json: num_classes=1000, architecture=resnet50; timm model card"
  decision_function: "logits [B, 1000] → argmax(dim=-1) → class_id"
  sensitive_to: "top-1 位置的 logit 是否仍然最大（排序是否翻转）"
  robust_to: "非 top-1 位置的数值变化；logits 整体加/减常数"

phi:
  definition: "Φ(Y_i) = argmax(Y_i, dim=-1) — 第 i 个样本的预测类别 ID"
  reasoning: >
    task.sensitive_to = "top-1 排序是否翻转"。
    argmax 直接检查 top-1 是否相同，与 sensitive_to 完全对齐。
  anti_example: |
    不能构造：
    argmax(Y_base_i) == argmax(Y_opt_i) → 预测类别相同 → 任务结果相同。
    因果性保证：不存在 argmax 一致但分类任务失败的情况。
  secondary: "cosine_similarity(Y_base_i, Y_opt_i) — 监测数值偏移量级"

distance:
  definition: "d_i = 1 if argmax(Y_base_i)==argmax(Y_opt_i) else 0"
  interpretation: "d_i=1 → 该样本预测不变；d_i=0 → 预测翻转"

threshold:
  value: "d_i = 1 (每个样本 argmax 都必须一致)"
  source: "causal: argmax agreement=1 → 预测必一致，无需余量"

samples:
  count: 100
  strategy: "ImageNet val 中按 top-1 confidence 分层：高置信 50 + 低置信(margin<0.1) 30 + 边界 case 20"
  includes_hard_cases: "是 — 低置信样本即 top-1 和 top-2 logit 差距小于 0.1 的样本，这些最可能因 FP16 翻转"
  fixed_seed: 42
  storage_path: "accuracy/test_samples/"
```

### 示例 2：DINOv2（特征提取/检索）

```yaml
model: dinov2_base
created_at: "2026-08-12T08:00:00"

task:
  description: "通用视觉特征提取，下游用 kNN(k=20) 做 ImageNet 分类"
  source: "DINOv2 论文 Table 1: Linear/kNN evaluation on ImageNet; model card"
  decision_function: "CLS embedding [768] → cosine_sim(query, all_gallery) → top-k → majority vote → class"
  sensitive_to: "query embedding 与 gallery 的相对距离排序（最近邻是否改变）"
  robust_to: "embedding 整体缩放（不改 cosine rank）；远离 query 的 gallery 点"

phi:
  definition: "Φ(Y_i) = L2_normalize(Y_i.last_hidden_state[:, 0, :]) — 第 i 个样本的归一化 CLS embedding (768-dim)"
  reasoning: >
    task.sensitive_to = "相对距离排序"。
    归一化后的 embedding 方向决定了 cosine ranking。
    Φ 取归一化 CLS embedding，D 用 cosine 量化方向偏移。
  anti_example: |
    能构造（理论上）：
    如果 gallery 中存在两个点 A, B 与 query 距离极近且 A 稍近于 B，
    embedding 的微小方向偏移可能让 B 变得比 A 近 → top-1 翻转。
    
    但实际风险：cos(emb_base, emb_opt) > 0.9999 时偏移角 < 0.8°，
    在 DINOv2 768-dim 空间中跨越 Voronoi 边界的概率极低。
    model_family_hints.md 记录 cos>0.9999 时 NC@20 实测不变。
    
    结论：cosine 不是因果性保证，但在足够严格的阈值下经验安全。
    标记为"经验性安全"而非"因果性保证"。
  secondary: null

distance:
  definition: "d_i = cosine_similarity(Φ(Y_base_i), Φ(Y_opt_i))"
  interpretation: "d_i=1.0 → 方向完全一致；d_i=0.999 → 偏移约 2.6°"

threshold:
  value: 0.9999
  source: "empirical: model_family_hints.md — DINOv2 FP16 实测 cos>0.9999 时 NC@20 不变"

samples:
  count: 50
  strategy: "随机抽取 ImageNet val 图片，覆盖不同类别和图像复杂度"
  includes_hard_cases: "是 — 包含 10 张纹理复杂的小物体图片（embedding 区分度低的 case）"
  fixed_seed: 42
  storage_path: "accuracy/test_samples/"
```

### 示例 3：Qwen3-1.7B（自回归 LLM）

```yaml
model: qwen3_1_7b
created_at: "2026-08-12T08:00:00"

task:
  description: "Causal Language Model, 自回归文本生成"
  source: "config.json: architectures=[Qwen3ForCausalLM]; HuggingFace model card"
  decision_function: "logits [vocab_size] → argmax (greedy, do_sample=False) → next_token → repeat max_new_tokens 次 → token_sequence"
  sensitive_to: "每一步的 top-1 token 是否相同 — 第一个不同的 token 之后整个序列发散"
  robust_to: "低概率 token 的绝对值变化；logits 整体偏移（不改 argmax）"

phi:
  definition: "Φ(Y_i) = greedy_decode(model, prompt_i, max_new_tokens=32, do_sample=False) — 第 i 个 prompt 的生成 token 序列"
  reasoning: >
    task.sensitive_to = "每步 top-1 是否相同"。
    Φ 直接取完整 greedy 生成序列。如果序列一致 → 每步 argmax 都一致 → 任务结果一致。
  anti_example: |
    不能构造（限定 do_sample=False）：
    greedy decode 完全确定性 — 每步 argmax(logits) 决定下一个 token。
    如果 baseline 和 optimized 的 greedy 序列完全相同，
    则每步 argmax 必然相同 → 文本必然相同 → 任务结果必然相同。
    因果性保证。
    
    注意限制条件：仅在 do_sample=False 时成立。
    若 do_sample=True → 不可使用此 Φ（需改用 logits KL divergence）。
  secondary: "per-token logit margin: max(logits) - second_max(logits) at first diverging position"

distance:
  definition: "d_i = prefix_match_length(seq_base_i, seq_opt_i) / max_new_tokens — 一致前缀比例"
  interpretation: "d_i=1.0 → 32 个 token 完全一致；d_i=0.5 → 前 16 个一致后面发散"

threshold:
  value: "d_i >= 0.8 (允许尾部少量 token 不同)"
  source: "user_specified: 用户设定允许尾部 20% token 不同，基于对 LLM 长序列精度衰减的业务容忍度"

samples:
  count: 50
  strategy: "多样化 prompt：短 prompt(5词) 20个 + 中 prompt(20词) 20个 + 长 prompt(100词) 10个"
  includes_hard_cases: "是 — 包含数学推理/代码生成等需要长链推理的 prompt（logit margin 普遍低）"
  fixed_seed: 42
  storage_path: "accuracy/test_samples/"
```

### 示例 4：Stable Diffusion（图像生成）

```yaml
model: stable_diffusion_v1_4
created_at: "2026-08-12T08:00:00"

task:
  description: "Text-to-Image generation (Latent Diffusion)"
  source: "model_index.json: StableDiffusionPipeline; 论文 arXiv:2112.10752"
  decision_function: "text → CLIP encode → noise_pred(t) × 50 steps → final_latent → VAE decode → image"
  sensitive_to: "noise_pred 的逐值精度（误差经多步迭代放大）；最终图像的语义与 prompt 对齐"
  robust_to: "单步 noise_pred 的极小偏差（若未经多步放大）"

phi:
  definition: "Φ(Y_i) = CLIP_score(decode(final_latent_i), prompt_i) — 第 i 个 prompt 的图文对齐分数"
  reasoning: >
    扩散模型的最终目标是生成与 prompt 语义对齐的图像。
    当优化引入精度损失时（如 FP16），输出 latent 必然不同，
    但只要语义质量不退化，任务仍然完成。
    CLIP score 量化图文语义对齐，是单样本最佳代理指标。
  anti_example: |
    能构造：CLIP score 高但图像有局部伪影（CLIP 对局部缺陷不敏感）。
    但这是单样本代理指标的固有限制，Level 2 用 FID (N 张统计) 覆盖。
    Level 1 阶段 CLIP score 作为单样本代理指标已是最佳选择。
  secondary: "PSNR(image_base_i, image_opt_i) — 辅助像素级检查"

distance:
  definition: "d_i = CLIP_score(image_opt_i, prompt_i) / CLIP_score(image_base_i, prompt_i)"
  interpretation: "d_i≈1.0 → 质量相当；d_i<0.95 → 质量下降"

threshold:
  value: "CLIP score ratio >= 0.98"
  source: "empirical: model_family_hints.md「图像生成模型」— CLIP Score 退化<2%时人工评估无感知差异"

samples:
  count: 30
  strategy: "多样化 prompt: 人物 10 + 风景 10 + 抽象概念 5 + 复杂场景 5"
  includes_hard_cases: "是 — 包含细节丰富的复杂场景 prompt（对精度最敏感）"
  fixed_seed: 42
  storage_path: "accuracy/test_samples/"
```

---

## 与现有文档的关系

本指南不替代 `precision_validation_design.md`（方法论）和 `model_family_hints.md`（经验值），而是指导如何将那些文档中的知识**落实到具体的 yaml 字段中**。

推导时的加载顺序：
1. 先读 `precision_validation_design.md` 理解第一性原理和设计步骤
2. 再读 `model_family_hints.md` 查找该模型类型的经验数据
3. 最后用本指南确认 yaml 每个字段的填写质量
