# 编译工具使用方法论

> 编译工具跨四维度（去重/复用/掩盖/替换），不专属任何一个维度。本文件是独立 reference，当 profiling 显示 host-bound 时加载。

## 为什么需要编译工具

eager 模式下每个 op 的 host 开销来自 dispatch 链的多个层次：

```
Python 调用 → aten::op (PyTorch dispatch) → aclnnXxx (CANN dispatch) → device kernel
              ~10-20μs                        ~8μs                        ~5μs
```

四维度优化（去重/复用/掩盖/替换）可以减少 op 数量或单 op 开销，但**不改变每个 op 仍需独立 dispatch 的事实**。当 op 数量无法再减、单 op dispatch 开销无法再降时，编译工具通过"合并多个 op 为一次调度"来消除 dispatch 链开销。

## 工具层次与选择

按"易用性 × 收益"排序，从低门槛到高收益：

### 层次 1: TorchScript (`torch.jit.script`)

**消除范围**：Python 解释器开销（Python→C++ IR）。保留 aten→aclnn→device 的 dispatch 链。

**适用场景**：profiling 显示 Python 层开销大（operator_details 中 Python 函数调用占 host time 显著）。**首选项**——门槛最低，不需要处理 op 兼容性。

**限制**：
- 随机性 op（如采样类操作）的 RNG 在 TorchScript 中可能与 eager 不同，导致采样结果不一致。**验证方法**：将随机 op 留在 Python 中调用，不放入 script 函数。
- 依赖运行时值的动态参数（如 topk 的 k 参数依赖 tensor shape）不兼容。**fallback**：用 `torch.jit.trace`（见层次 2）。
- 函数参数类型必须可推断（不支持 Python 动态类型）。

**优势**：支持 `out=` 参数和 in-place ops（`mul_`/`add_`/`div_`），可与预分配 buffer 策略组合。

**使用模式**：
```
1. 提取热路径函数为 @torch.jit.script
2. 随机 op 留在 Python，script 函数返回中间结果
3. 验证精度（RNG 差异可能导致采样不同）
4. 若精度通过，测量 wall-clock 确认收益
```

### 层次 2: torch.jit.trace

**消除范围**：同 TorchScript，但不编译源码——只记录操作序列。绕过 TorchScript 的类型限制。

**适用场景**：TorchScript 因类型限制（如动态参数依赖运行时值）无法编译时使用。

**限制**：
- **shape-specific**：只对 trace 时的输入 shape 有效，不同 shape 会触发重新 trace（非重新编译，开销较小）。
- 不支持数据依赖的控制流（if/else 基于 tensor 值的分支会被固化）。

**使用模式**：
```
1. 用示例输入 trace 目标函数
2. per-shape 缓存：按输入 shape 缓存 trace 结果，避免每次调用都 trace
3. 验证精度
```

### 层次 3: torch.compile(npu) / GE 图编译

**消除范围**：整个 dispatch 链（Python→aten→aclnn→device），多个 op 融合为 1 个 graph，1 次 dispatch。收益最大。

**适用场景**：TorchScript 已用且 host 开销仍高；或 profiling 显示 dispatch 开销（aten+aclnn 层）占 host time 主导。**追求极限性能时使用**。

**限制**：
- **不兼容 op**：部分 op 无 GE converter 实现（报错 `ge_converter is not implemented`）。**处理方法**：① 查找数学等价替换（将复合算子拆解为基本算子组合，如 `f(a,b,c)=a+b*c` 类的 fused op 拆为 `a + b*c`）；② 仍不兼容则用 `suppress_errors=True` 让该 op 回退 eager（graph break），编译其余部分。
- **不支持 `out=` 参数和 in-place ops**：GE 图编译需要明确的 tensor 生命周期。**处理方法**：去掉 `out=` 和 in-place，改为普通赋值（GE 会自动管理中间 tensor 内存）。
- **编译时间长**：首次编译可能数分钟到数十分钟。**处理方法**：按 shape 缓存编译结果。
- **`dynamic=True` 可能不生效**：部分后端（如 GE）不支持动态 shape graph，`dynamic=True` 被忽略后不同 shape 仍触发重新编译。**fallback**：per-shape 编译缓存。

**使用模式**：
```
1. 确认 profiling 显示 dispatch 开销占主导
2. 检查目标函数是否有不兼容 op → 数学等价替换
3. 去掉 out= 和 in-place ops
4. 设置 suppress_errors=True 处理无法替换的 op（graph break 回退 eager）
5. 编译目标函数，首次调用触发编译
6. 验证精度（图融合可能改变浮点计算顺序）
7. 测量 wall-clock 确认收益
8. 若需支持多 shape，实现 per-shape 缓存
```

### 层次 4: NPU JIT (`torch.npu.set_compile_mode(jit_compile=True)`)

**消除范围**：CANN 层的 kernel 编译缓存。

**适用场景**：profiling 显示在线编译（online-compile）事件频繁。

**限制**：动态 shape 导致反复重编译时可能卡死。**不推荐作为首选**——优先用层次 1-3。

## 兼容性排查方法论

编译失败时，按以下流程系统性排查：

1. **看错误信息中的 op 名**——确定哪个 op 不兼容
2. **查该 op 是否有数学等价写法**——将复合算子拆解为基本算子组合（如 fused op `f(a,b,c)` 拆为等价的基本运算），或使用近似但精度可接受的等价 API
3. **替换后重试编译**——确认等价替换是否解决兼容性
4. **仍不兼容则用 graph break**——设置 suppress_errors 让不兼容 op 回退 eager，编译其余部分。注意：graph break 会在 break 点产生 eager↔compiled 切换开销，break 点过多会抵消编译收益
5. **验证精度**——等价替换和 graph break 都可能改变数值结果

## 编译粒度决策

编译粒度决定了 dispatch 开销是否被有效分摊：

| 粒度 | 示例 | dispatch 次数 | 适用场景 |
|------|------|-------------|---------|
| **单步函数** | 编译循环内的单步计算函数 | N（循环次数）× 1 | 循环次数少（<10）|
| **含循环的函数** | 编译包含 for 循环的整个 sample 函数 | 1 | 循环次数多（>10）|

**关键洞察**：编译单步函数在循环中调用时，每次调用有 1 次图 dispatch 开销。循环 N 次则 N 次 dispatch。如果单步函数的计算量小（图 dispatch 开销占比过高），可能**比 eager 更慢**。

编译包含循环的函数时，整个循环只有 1 次图 dispatch，dispatch 开销被分摊到接近零。但循环内的控制流（如随机性 op）会触发 graph break，break 后的部分回退 eager。

**决策标准**：单步计算时间 > 图 dispatch 开销时，编译单步可行；单步计算时间 < 图 dispatch 开销时，必须编译整个循环。图 dispatch 开销的具体值因后端而异，需通过 A/B benchmark 实测确定。

## 与四维度优化的关系

编译工具和四维度优化的推荐顺序：

1. **先解决编译无法覆盖的结构性问题**——D2H 同步（`.item()`/`.all()` 在热路径中阻塞 pipeline）、AI CPU 回退（op 无 AI Core 实现）。这些问题编译不解决，且可能导致 graph break。用 npu_checklist 扫描发现。
2. **然后尝试编译工具**——TorchScript 自动消除 Python 解释器开销、Module.__call__ 开销、属性查找开销等框架级开销，不需要手动 inline 或预提取。编译后重新 profiling，数据更准确（框架噪音被消除，真正的结构性瓶颈更清晰）。
3. **编译后用四维度优化解决剩余问题**——针对编译后 profiling 暴露的瓶颈（如 buffer 分配、op 兼容性、循环结构等）用去重/复用/替换手段优化。注意：TorchScript 支持 `out=` 和 in-place，可与预分配 buffer 组合；GE 编译不支持，需要去掉。

**不要在未编译的 eager 代码上做大量微优化**——很多 eager 层面的框架开销（Module.__call__、属性查找、Python 函数调用）会被编译自动消除。先编译再优化，避免在编译后会自动消失的问题上浪费时间。

## 放弃条件

- 算子不兼容且无法数学等价替换、也无法 graph break（如不兼容 op 在循环内每步执行）
- 图太大导致编译期 OOM 或编译时间 > 30 分钟
- 精度不达标且无法通过调整编译选项修复
- 编译后比 eager 更慢（编译粒度不当，dispatch 开销 > 融合收益）
- `dynamic=True` 不生效且输入 shape 变化频繁（每 shape 编译一次不可接受）

回退后记录失败原因到 evidence_db，包括：编译工具、失败原因、尝试的替代方案、实际效果。
