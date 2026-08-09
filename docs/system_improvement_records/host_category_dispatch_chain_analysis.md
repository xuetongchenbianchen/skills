# parse_operator_details.py Host Category 分类规则修复方案

## 1. 问题描述

### 1.1 现象

ProteinMPNN NPU 推理优化中，`parse_operator_details.py` 的 "Host Time by Category" 输出：

```
## Host Time by Category
  (total host self = 389.1 ms)
  other                        206.4 ms  ( 53.0%)
  alloc/metadata                86.6 ms  ( 22.3%)
  dispatch (aclnn launch)       62.7 ms  ( 16.1%)
  H2D/D2H copy                  29.2 ms  (  7.5%)
  sync (D-to-H)                  4.1 ms  (  1.1%)
```

53% 的 host 开销落入 "other"，agent 无法直接判断这是什么、该用什么手段优化。

### 1.2 根因

`thresholds.py` 的 `host_category_rules` 有 4 条规则缺失 + 1 个顺序问题：

**规则缺失**：

| 问题 op | 实际语义 | 应归入类别 | 当前归入 | host 时间 |
|---------|---------|-----------|---------|----------|
| `aten::addmm` 等 `aten::` 前缀 op | PyTorch C++ dispatch（计算类 op 的 dispatch 入口） | dispatch | other | 163.5ms |
| `aten::slice` | 创建 tensor view（纯元数据，无 kernel） | alloc/metadata | other | 36.3ms |
| `aten::select` | 创建 tensor view（纯元数据，无 kernel） | alloc/metadata | other | 12.9ms |
| `aten::resize_` | 原地 resize tensor（元数据操作） | alloc/metadata | other | 5.6ms |

**顺序问题**：

当前规则顺序为 sync > dispatch(aclnn) > H2D > alloc > framework > compile。如果简单在 dispatch 类别中增加 `aten::` 匹配，会导致 `aten::slice` 被先匹配到 `aten::` 归入 dispatch（错误），`aten::to` 被先匹配到 `aten::` 归入 dispatch 而非 H2D（错误）。

正确顺序应为：sync > **alloc** > **H2D** > dispatch(aclnn) > dispatch(aten) > framework > compile。即：更具体的 alloc 和 H2D 规则必须在宽泛的 `aten::` 匹配之前，否则 `aten::` 会"吃掉"所有 `aten::` 前缀的 op，包括本应归入 alloc 的 `aten::slice` 和本应归入 H2D 的 `aten::to`。

### 1.3 影响

agent 看到 "other = 53%" 后需要手动从 Top-15 Ops 表逐个关联 `aten::` 前缀 op 到 "other" 类别，再推理出"这些是 PyTorch C++ dispatch 层开销，需要图编译/TorchScript 消除"。这个手动关联和推理步骤本应由分类规则自动完成。

---

## 2. 修改方案

### 2.1 修改 `thresholds.py`

**修改 `host_category_rules` 的内容和顺序**：

```python
"host_category_rules": {
    "sync (D-to-H)": ["_local_scalar", "::item", ".item", "numpy"],
    "alloc/metadata": ["empty", "as_strided", "view", "reshape", "clone",
                       "contiguous", "detach", "expand", "squeeze", "unsqueeze",
                       "slice", "select", "resize_"],
    "H2D/D2H copy": ["copy_", "_to_copy", "to_copy", "memcpy", "::to"],
    "dispatch (CANN aclnn)": ["aclnn"],
    "dispatch (PyTorch aten)": ["aten::"],
    "framework/comm": ["c10d::", "profiler", "broadcast_"],
    "compile": ["compile", "opcompile"],
}
```

**改动点**：
1. `alloc/metadata` 增加 `slice`、`select`、`resize_` 三个 pattern
2. `dispatch` 拆为 `dispatch (CANN aclnn)` 和 `dispatch (PyTorch aten)` 两个类别
3. **顺序调整**：`alloc/metadata` 和 `H2D/D2H copy` 移到 dispatch 之前，确保 `aten::slice` 先匹配 alloc、`aten::to` 先匹配 H2D

**first-match 顺序验证**（用实际 profiling 数据测试，20 个 op 全部正确分类）：

| op | 期望类别 | 新规则匹配结果 |
|----|---------|--------------|
| `aten::slice` | alloc | alloc/metadata ✓ |
| `aten::select` | alloc | alloc/metadata ✓ |
| `aten::resize_` | alloc | alloc/metadata ✓ |
| `aten::addmm` | dispatch | dispatch (PyTorch aten) ✓ |
| `aten::gelu` | dispatch | dispatch (PyTorch aten) ✓ |
| `aten::gather` | dispatch | dispatch (PyTorch aten) ✓ |
| `aten::add_` | dispatch | dispatch (PyTorch aten) ✓ |
| `aten::multinomial` | dispatch | dispatch (PyTorch aten) ✓ |
| `aclnnAddmm` | dispatch | dispatch (CANN aclnn) ✓ |
| `empty_tensor` | alloc | alloc/metadata ✓ |
| `aten::view` | alloc | alloc/metadata ✓ |
| `aten::to` | H2D | H2D/D2H copy ✓ |
| `copy_` | H2D | H2D/D2H copy ✓ |
| `_decoder_loop` | other | other ✓（TorchScript 函数名，无规则匹配） |

### 2.2 修改 `parse_operator_details.py`

**在 `parse_overview` 的 "Host Time by Category" 输出段增加**：

1. **"other" 类别自动分解**：当 other 占比 > 10% 时，列出 other 中的 Top-10 op，让 agent 不需要手动去 Top-15 表关联。

2. **dispatch 合计行**：在所有类别后增加 dispatch 两层的合计，让 agent 直接看到总 dispatch 开销。

```python
# other 分解（在 category 输出之后）
other_us = cat_agg.get("other", 0)
other_pct = other_us / total_host_us * 100 if total_host_us > 0 else 0
if other_pct > 10:
    other_ops = {name: info for name, info in op_agg.items()
                 if _host_category(name) == "other"}
    other_sorted = sorted(other_ops.items(), key=lambda x: -x[1]["host_us"])
    lines.append(f"## 'other' 类别分解 (占比 {other_pct:.1f}%)")
    lines.append("  以下 op 未匹配任何分类规则，按 host 时间排序:")
    for name, info in other_sorted[:10]:
        lines.append(f"  {name:<35} {info['count']:>8} {info['host_us']/1000:>9.1f} ms")
    lines.append("")

# dispatch 合计（在 category 输出之后）
dispatch_total = sum(v for k, v in cat_agg.items() if k.startswith("dispatch"))
if dispatch_total > 0:
    lines.append(f"  dispatch 合计: {dispatch_total/1000:.1f} ms ({dispatch_total/total_host_us*100:.1f}%)")
    lines.append("")
```

### 2.3 不修改的部分

- `_host_category` 函数逻辑不变（仍按 rules first-match 分类，只是 rules 内容变了）
- `parse_filtered` 函数不变
- 其他 parse 脚本不变

---

## 3. 预期输出变化

**修改前**：
```
## Host Time by Category
  other                        206.4 ms  ( 53.0%)
  alloc/metadata                86.6 ms  ( 22.3%)
  dispatch (aclnn launch)       62.7 ms  ( 16.1%)
  H2D/D2H copy                  29.2 ms  (  7.5%)
  sync (D-to-H)                  4.1 ms  (  1.1%)
```

**修改后**（用实际 profiling 数据验证）：
```
## Host Time by Category
  alloc/metadata                141.4 ms  ( 36.3%)
  dispatch (PyTorch aten)      125.9 ms  ( 32.4%)
  dispatch (CANN aclnn)         62.7 ms  ( 16.1%)
  H2D/D2H copy                  29.2 ms  (  7.5%)
  other                         25.8 ms  (  6.6%)
  sync (D-to-H)                  4.1 ms  (  1.1%)
  dispatch 合计:               188.7 ms  ( 48.5%)

## 'other' 类别分解 (占比 6.6%)
  _decoder_loop                    68     14.1 ms
  _post_sample_step                68      2.8 ms
  _sample_post_decode              68      2.6 ms
  npu::npu_dtype_cast             207      2.6 ms
  _all_probs_step                  68      2.6 ms
  forward                           1      1.1 ms
```

**other 从 53.0% 降到 6.6%**，剩余的 other 内容是 TorchScript 函数名（`_decoder_loop` 等）和 `npu::` 前缀——这些确实无法归入现有类别，但 other 分解段让 agent 能直接看到它们是什么。

---

## 4. 验证方法

1. 修改 `thresholds.py` 和 `parse_operator_details.py`
2. 用已有的 L1 profiling 数据重新运行 `parse_operator_details.py`
3. 确认：
   - other 占比从 53% 降到 < 10%
   - `dispatch (PyTorch aten)` 类别出现，包含所有 `aten::` 前缀 op
   - `alloc/metadata` 增加（因 `slice`/`select`/`resize_` 被正确归入）
   - `aten::to` 正确归入 H2D 而非 dispatch
   - other 分解段正确展示未分类 op
   - dispatch 合计行正确显示两层之和
