# 模型类型与精度对比策略

## 目的

不同模型类型的"应该比什么"和"重点看什么"完全不同。进入对齐前先判断模型类型，避免用错指标或遗漏关键检查点。

## 经验验证的安全阈值映射

以下映射基于五模型（ResNet50、DINOv2、CLIP、Wav2Vec2、SD v1.4）NPU 推理优化实验建立：

| 场景 | 数值条件 | 任务影响 | 结论 |
|------|---------|---------|------|
| FP32 图编译（JIT/compile） | bit-exact 或 cos > 0.999999 | 5 模型实测任务均不变 | 数值可替代任务层（经验性结论） |
| FP16 降精度 | cos > 0.999 | 5 个模型均证明任务不退化 | 可替代（配合安全阈值） |
| 调度器/算法替换 | 像素 cos 低但 CLIP 语义 cos 高 | 需语义层判定 | 像素不可替代，需语义验证 |
| INT8/蒸馏/剪枝 | 偏差可能非单调 | 需全量任务层评测 | 不可替代 |

这些映射是经验性的，新模型首次使用时应通过校准实验验证是否适用。

## NLP / 序列模型（Transformer Encoder/Decoder，非生成）

数值层：
- 对比对象：logits、中间 hidden states、attention weights
- 指标：cosine sim > 0.999、max abs diff
- 条件约束：相同输入、相同 padding/mask 策略

任务层：
- 指标：accuracy、F1、BLEU、Rouge（取决于下游任务）
- 触发条件：量化、蒸馏、head pruning 等改变模型能力的优化

重点关注：
- tokenizer 一致性（不同版本可能结果不同）
- attention mask 和 padding 策略
- position encoding 的边界处理

## CV / 视觉模型（CNN、ViT、检测/分割）

数值层：
- 对比对象：特征图、最终 logits 或检测框
- 指标：逐像素/逐元素 cosine sim > 0.999；检测框 IoU
- 条件约束：相同预处理（归一化、resize、crop 顺序和参数）

任务层：
- 指标：Top-1/Top-5 Accuracy、mAP、mIoU
- 触发条件：量化、剪枝、输入分辨率变更等改变模型容量/精度的优化
- 样本要求：标准验证集（ImageNet val、COCO val 等）

重点关注：
- 图像预处理管道（归一化参数、resize 插值方式、crop 策略）
- NPU 上的 internal format 对卷积的影响
- batch norm 在训练/推理模式下的行为差异

## LLM / 自回归文本生成

数值层：
- 对比对象：logits（连续）、生成 token 序列（离散）
- 指标：logits cosine sim > 0.999；greedy decode token 匹配率
- 条件约束：相同 prompt、seed、temperature=0（或 do_sample=False）、max_length

任务层：
- 指标：perplexity、BLEU、Rouge、人工评估
- 触发条件：量化、蒸馏、KV cache 压缩、speculative decoding 等改变了解码路径
- 样本要求：覆盖不同长度和领域的输入

重点关注：
- sampling 参数必须完全一致（temperature、top_k、top_p）
- KV cache 在长序列 decode 过程中的累积偏差
- position encoding 的长度外推边界

## 图像生成模型（扩散模型、GAN、VAE、Flow）

### 数值层

- 对比对象：生成器/去噪网络的中间输出（如 noise prediction）或最终图像像素
- 指标：cosine sim > 0.999、PSNR > 30dB、mean abs diff < 0.01
- 条件约束：相同随机种子、相同采样算法、相同迭代步数、相同条件输入（prompt/class label/reference image）

适用的优化类型：算子替换、内存优化、数据格式变更、图编译——这些不改变生成路径，只改实现。

### 任务层

当优化改变生成路径（换采样算法、减迭代步数、改条件强度等），输出像素必然不同，数值层对比无意义。此时需要**在特征空间**验证生成质量是否退化。

**核心指标：FID (Fréchet Inception Distance)**

原理：用预训练视觉模型将两组图像映射到特征空间，比较两组特征的分布距离。

计算过程：
1. 分别生成两组图像（优化前 N 张、优化后 N 张）
2. 每张图过预训练视觉模型（Inception-v3），没有时就下载模型，得到特征向量
3. 对每组 N 个特征向量拟合高斯分布：均值 μ + 协方差矩阵 Σ
4. 计算两个高斯分布之间的 Fréchet 距离：
   ```
   FID = ‖μ₁ - μ₂‖² + Tr(Σ₁ + Σ₂ - 2√(Σ₁·Σ₂))
         ──────────      ──────────────────────────
          中心距离              形状差异
   ```

FID 越小越好：
| FID 值 | 含义 |
|--------|------|
| < 5 | 几乎无差异 |
| < 15 | 差异可接受 |
| 15–30 | 需要关注 |
| > 30 | 质量退化 |

辅助指标：
- CLIP Score：衡量 text-image 对齐度（文生图场景）
- IS (Inception Score)：衡量清晰度 + 多样性
- LPIPS：衡量感知相似度（有参考图时）

操作要求：
- 输入条件集多样化（不同主题/风格/复杂度），≥ 25 种
- 生成样本数 ≥ 30（建议 50+），每张用不同 seed
- 特征提取器在对比前确定，不可中途更换
- 阈值在运行前声明

### 何时用哪层

| 优化类型 | 验证层 | 原因 |
|---------|--------|------|
| 算子替换/融合 | 数值层 | 生成路径不变 |
| 内存/格式优化 | 数值层 | 不影响计算结果 |
| 图编译 | 数值层 | 等价变换 |
| 换采样算法/solver | 任务层 | 路径不同，输出必然不同 |
| 减少迭代步数 | 任务层 | 可能欠采样 |
| 改条件强度/guidance | 任务层 | 改变了生成引导 |
| 量化/蒸馏 | 两层都做 | 既可能引入数值偏差，也可能改变生成能力 |

### 常见误区

- ❌ 用 raw pixel cosine 做跨路径对比 — 路径不同则 pixel 必然不同，cosine 低不代表质量差
- ❌ FID 样本量太少 — 高斯估计不稳定，结论不可靠
- ❌ 只做任务层不做数值层 — 可能掩盖计算 bug（FID 对个别样本偏差不敏感）
- ❌ 只做数值层宣称精度通过 — 如果路径也变了，数值层无法覆盖

重点关注：
- 随机种子需在正确的 device 上创建 Generator
- 采样算法的超参一致性（sigma schedule、solver order 等）
- 后处理（VAE decode、归一化反变换）的参数一致

## 图神经网络（GNN）

数值层：
- 对比对象：节点/边/图级 embedding、logits
- 指标：cosine sim、max abs diff
- 条件约束：相同图构建顺序和邻接矩阵排序

任务层：
- 指标：节点分类 accuracy、图分类 accuracy、链接预测 AUC
- 触发条件：量化、采样策略变更、聚合方式变更

重点关注：
- 图构建顺序和邻接矩阵排序
- scatter / gather 相关算子在 NPU 上的行为
- batch 内 padding 或 mask 处理

## 科学计算 / AI4S

### PDE / CFD 类
数值层：loss、逐样本物理量场的 cosine sim。
任务层：验证集误差（MAE/RMSE）、长时滚动是否发散。
重点关注：网格预处理、边界条件编码、时间步累积误差。

### 分子/材料性质预测
数值层：逐样本预测值的相对误差。
任务层：MAE、RMSE、R²。
重点关注：原子/键特征构造、标准化与反标准化。

### 时序预测
数值层：逐步预测值 cosine sim。
任务层：MAE、MSE、MAPE、多步滚动预测后半段是否漂移。
重点关注：滑窗切分、时间特征编码、teacher forcing 逻辑。

## 音频 / 语音模型（Wav2Vec2、Whisper、CTC/Seq2Seq ASR）

数值层：
- 对比对象：CTC logits 或 decoder logits
- 指标：logits cosine + CTC/greedy decode 文本匹配率
- 条件约束：相同音频输入、相同采样率、padding 策略一致

任务层：
- 指标：WER (Word Error Rate)、CER
- 触发条件：量化、换 decoder、修改 padding/bucket 策略
- 样本要求：LibriSpeech test-clean 或对应标准集

重点关注：
- Padding Buckets 在桶边界处可能引入偏差（padding 区域计算对有效区域的微小影响）
- CTC argmax 对数值偏差有较强鲁棒性——logits cosine=0.984 时解码文本仍可能 100% 一致
- 变长输入时，确保 logits 裁剪回原始长度再比较

经验阈值（Wav2Vec2 实测）：
- cos > 0.9999 且 rel_L2 < 0.001 → WER 完全一致，可跳过全量评测
- cos > 0.999 且 rel_L2 < 0.01 → WER 变化 ≤ 0.01%，建议抽样验证
- cos < 0.999 或 rel_L2 > 0.01 → 必须跑全量 WER

## 关键经验教训

### 指标选择的因果性

最好的 Φ 是具有因果性保证的指标——即 Φ 满足某个条件时，能数学保证任务不退化：
- NC@k = 1.0 → k-NN 预测必不变（DINOv2）
- CTC decode match = 100% → WER 必不变（Wav2Vec2）
- Agreement = 100% → Top-1 必不变（分类模型）

相比之下，cosine > 0.999 只是统计相关——不能数学保证任务不变，但经验上足够可靠。

### 不同模型对偏差的鲁棒性不同

- 分类模型对 FP16 偏差高度鲁棒（cos > 0.999 时 50K 样本零退化）
- 扩散模型对量化极敏感（噪声预测的小误差经多步去噪放大）
- 特征检索模型对均匀噪声鲁棒但对定向偏移敏感

### 便宜的任务指标可以在 Level 1 直接使用

某些任务指标计算成本极低，不需要等 Level 2：
- LLM 的 PPL（一次前向传播）
- CTC decode 文本匹配率（一次 argmax + 比较）
- 分类的 Agreement（一次 argmax + 比较）

这些可以直接集成到 Level 1 中，提供比纯数值层更强的判断力。

## 使用原则

- 先按模型类型选比较对象，再运行脚本
- 用户未说明模型类型时，先问清任务类型和最终指标
- 混合任务模型不要只看一个指标，同时保留任务指标和中间输出证据
- 不要机械查表选 cosine——先理解"任务成败取决于什么"，再选能捕捉那个性质的 Φ
