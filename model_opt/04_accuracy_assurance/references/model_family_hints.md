# 模型类型与验证策略（详细版）

> 速查表见主文件 SKILL.md「三、确定比什么」。本文件提供各模型类型的详细推理过程，供遇到边界情况或不在速查表中的模型时参考。

## 判断方法

对每种模型，回答两个问题：
1. 模型直接输出是什么？（forward 返回值）
2. 直接输出是否确定性决定下游结果？（输出到消费间是否有采样、随机过程、非线性放大）

是 → 验证边界 = 模型直接输出；否 → 验证边界 = 下游功能指标。

---

## NLP / 序列模型（Transformer Encoder/Decoder）

**直接输出**：logits（连续向量）

**验证边界**：
- 贪心解码 / 非自回归：logits 确定性决定输出 → 比直接输出
- 采样解码：sampling 引入随机性 → 比 benchmark 分数
- 下游任务（分类、NER）：logits 确定性决定预测 → 比直接输出 + 任务指标

**度量**：logits → cosine + max_abs；token 序列（贪心）→ 完全匹配率；benchmark → MMLU / HumanEval / BLEU / Rouge

## CV / 视觉模型（CNN、ViT、检测/分割）

**直接输出**：feature map / logits / bbox / mask

**验证边界**：
- 分类：logits 确定性决定预测 → 比直接输出 + Top-1 accuracy
- 检测/分割：输出经后处理（NMS、阈值化）→ 比后处理后结果

**度量**：feature map → cosine + max_abs；分类 → cosine + accuracy；检测 → mAP / mIoU

## 生成式模型（LLM、扩散模型、VAE）

**直接输出**：logits（LLM）/ 像素矩阵（diffusion）/ 潜在表示（VAE）

**验证边界**：
- LLM 贪心：比 logits
- LLM 采样：比 benchmark 分数（不同 seed 产生不同序列都合法）
- Diffusion：比图片语义（像素 diff 无意义）
- VAE 重构：比 LPIPS 或固定 seed 下对比

**度量**：logits → cosine + max_abs；benchmark → 任务分数；图片 → LPIPS / FID / SSIM；概率分布 → KL 散度

## 图神经网络（GNN）

**直接输出**：节点/图级 embedding 或 logits

**验证边界**：
- 节点/图分类：logits 确定性决定预测 → 比直接输出 + accuracy
- 属性预测（如 MACE energy/forces）：输出是物理量，下游直接消费 → 比直接输出

**度量**：embedding → cosine + max_abs；物理量 → max_abs + 相对误差（力的绝对值影响轨迹，不能只看方向）

## 科学计算 / AI4S

AI4S 场景的特殊性：模型直接输出和"下游消费者真正关心的东西"可能不一致。

### 分子动力学 / 力场（MACE 等）

**直接输出**：energy（标量）、forces（向量）、stress（矩阵）
**验证边界**：直接输出（下游 MD 模拟器直接消费，确定性决定）
**度量**：energy → 相对误差；forces → max_abs + cosine；stress → max_abs

### 蛋白质结构预测

**直接输出**：3D 坐标
**验证边界**：下游功能指标（坐标 diff 不代表结构等价）
**度量**：RMSD（结构整体偏差）、TM-score（结构相似度，比 RMSD 更语义化）

### 天气/气候预测

**直接输出**：物理场（温度、湿度、风速等网格数据）
**验证边界**：下游功能指标（逐点 diff 不代表预报质量）
**度量**：空间 RMSE、异常相关系数（ACC）

### 时序预测

**直接输出**：预测序列
**验证边界**：直接输出（确定性预测）或下游功能指标（多步滚动有累积误差）
**度量**：MAE / MSE / MAPE；多步滚动预测后半段是否漂移
