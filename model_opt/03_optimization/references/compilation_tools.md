# 编译工具使用方法论

当 profiling 显示 dispatch 开销占主导，或瓶颈为可融合的算子碎片化，且不存在融合算子时加载。

## 前提条件：输入 shape 必须可控

编译产物绑定 shape——shape 变则重编译。使用前必须确认：

| Shape 状况 | 可否编译 | 处理 |
|------------|---------|------|
| 静态 / 有限离散 | ✅ | 直接编译或 per-shape 缓存 |
| 已做 bucket | ✅ | padding 到 bucket 边界 |
| 无约束动态 | ❌ | 先做 bucket，否则放弃 |

Bucket 典型做法：序列长度按 [64, 128, 256, 512, 1024, 2048] 向上 padding；batch size 固定或限 2-3 个值。

---

## 为什么需要编译

eager 模式每个 op 独立走完 dispatch 链：

```
Python → aten::op (~10-20μs) → aclnnXxx (~8μs) → device kernel (~5μs)
```

四维度优化减少 op 数量，但不改变每个 op 独立 dispatch 的事实。编译做两件 eager 做不到的事：

1. **dispatch 消除**（host 侧）：多个 op 合并为一次图调度；
2. **跨算子 kernel 融合**（device 侧）：相邻 elementwise/norm/bias 链融合为单 kernel，减少中间结果读写——除非手写融合算子，此项用四维度手段在 eager 下不可达。

---

## 工具选择

按 profiling 指出的开销层选择**消除范围匹配**的工具：Python/Module 层开销为主（框架循环重、算子少）→ 层次 1/2 足够；aten→aclnn dispatch 链或可融合算子链为主 → 直接层次 3，跳过层次 1/2（它们够不着 dispatch 链，先试只是浪费一轮验证）。层次从低到高仅在开销归属不明确时作为尝试顺序（复杂度与风险随层次递增）。

### 层次 1: torch.jit.script

消除 Python 解释器开销（Module.__call__ / Python 循环）。保留 aten→aclnn→device dispatch。**门槛最低——开销归属不明确时的默认起点。**

- 支持 `out=` 和 in-place ops，可与预分配 buffer 组合
- 随机 op 的 RNG 可能与 eager 不同 → 随机 op 留在 Python 外部
- 依赖运行时值的动态参数不兼容 → fallback 到 trace

### 层次 2: torch.jit.trace

同层次 1 的消除范围，但只记录操作序列，绕过类型限制。

- shape-specific：不同 shape 触发重新 trace（开销较小，可 per-shape 缓存）
- 数据依赖控制流会被固化（if/else 基于 tensor 值的分支被 trace 时的路径替代）

### 层次 3: torch.compile(npu) / GE 图编译

消除整个 dispatch 链，多 op 融合为 1 graph。**收益最大，复杂度最高。**

- 不兼容 op（`ge_converter is not implemented`）→ 数学等价拆解，仍不行则 `suppress_errors=True` 让该 op graph break 回退 eager
- 不支持 `out=` 和 in-place → 改为普通赋值
- `dynamic=True` 在 GE 后端可能不生效 → per-shape 缓存
- 首次编译耗时长（分钟级）→ 缓存编译结果
- graph break 过多会抵消收益——break 点产生 eager↔compiled 切换开销

### 层次 4: NPU JIT (`jit_compile=True`)

CANN kernel 编译缓存。仅用于 profiling 显示频繁 online-compile 事件时。动态 shape 下可能卡死，**不推荐首选**。

---

## 编译粒度

| 粒度 | dispatch 次数 | 适用 |
|------|-------------|------|
| 编译循环内单步函数 | N 次（循环次数） | 单步计算量大，N 小 |
| 编译含循环的整个函数 | 1 次 | 单步计算量小，N 大 |

判断标准：单步计算时间 > 图 dispatch 开销 → 编译单步可行；否则必须编译整个循环。实测确定。

---

## 与四维度优化的顺序

1. **先做结构性优化**：改变"算什么"的优化编译器不会替你做，无论最终是否编译都有效。其中 D2H 同步（`.item()`）、AI CPU 回退必须先行——编译不解决这些，且会导致 graph break。
2. **eager 收敛后编译**——编译的前置条件：
   - 四维度方案已全部实施或被证据否决；
   - 剩余 host 开销的构成经判定为"eager 不可达"为主（产出：02 的 trace_view §0b Host 开销分层 + operator_details 按 Category 拆解——②b/②c 算子调用层占大头，②a Python/Module 层占比小）。构成相反（Python/Module 层显著）时，flat forward 等 eager 框架层手段先做并测得 eager 终点，再决定编译。
   - eager 终点（收益停滞 + 开销构成数据）记录到 evidence_db，作为编译必要性的测量依据。
3. **编译落地后复验**——对已并入图内的 eager 优化用开关做保留/回退 A/B，防止单项负贡献被编译整体收益掩盖；随后重新 profiling，数据更干净。
4. **编译后再做剩余四维度优化**——针对编译后暴露的瓶颈。

---

## 放弃条件

- Shape 无法收敛且业务不允许 bucket
- 算子不兼容且无法等价替换、graph break 后收益归零
- 编译期 OOM 或编译时间 > 30 分钟
- 精度不达标
- 编译后比 eager 更慢——先归因再决定：粒度不当（单步计算量太小 / 循环未包含）→ 调整编译粒度重试；发生在 device 耗时远大于 host 的场景（异步流水已掩盖逐算子 dispatch 开销，整图 launch 固定开销反超）→ 该场景不适用图编译，回退 eager 并记录到 evidence_db

回退后记录失败原因到 evidence_db。
