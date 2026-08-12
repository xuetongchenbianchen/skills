# 编译工具使用方法论

> 当 profiling 显示 host-bound（dispatch 开销占主导）时加载。

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

四维度优化减少 op 数量，但不改变每个 op 独立 dispatch 的事实。编译工具将多个 op 合并为一次调度。

---

## 工具选择（按门槛从低到高）

### 层次 1: torch.jit.script

消除 Python 解释器开销。保留 aten→aclnn→device dispatch。**门槛最低，首选。**

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

1. **先解决结构性问题**——D2H 同步（`.item()`）、AI CPU 回退。编译不解决这些，且会导致 graph break。
2. **然后编译**——自动消除 Python/Module.__call__/属性查找等框架开销。编译后重新 profiling，数据更干净。
3. **编译后再做剩余四维度优化**——针对编译后暴露的真正瓶颈。

不要在未编译的 eager 代码上做大量微优化——很多框架开销会被编译自动消除。

---

## 放弃条件

- Shape 无法收敛且业务不允许 bucket
- 算子不兼容且无法等价替换、graph break 后收益归零
- 编译期 OOM 或编译时间 > 30 分钟
- 精度不达标
- 编译后比 eager 更慢（粒度不当）

回退后记录失败原因到 evidence_db。
