#!/usr/bin/env python3
"""解析 kernel_details.csv — 逐 kernel 执行详情（含硬件单元耗时拆分）。

这是信息最丰富的 profiling 文件之一。包含逐 kernel 的：执行时间、
wait time、硬件单元耗时拆分（mac/mte1/mte2/vec/scalar）、shape、
Block Dim（并行度）以及 cube 利用率。

相比其他 profiling 文件的独特价值：
- 硬件单元利用率（逐 kernel 区分 compute-bound 与 memory-bound）
- 逐 kernel 的 shape 与并行度（Block Dim）
- 顺序 kernel 流与 wait time 模式
- 识别低效 kernel（耗时长但利用率低）

用法:
    python parse_kernel_details.py <profiling_dir> [--rank N] [--top-k 15]
        [--small-threshold 5.0] [--wait-threshold 500]
"""

import argparse
import heapq
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output, stream_csv, safe_float, format_duration_ms


def parse(profiling_dir: str, rank=None, top_k: int = 15,
          small_threshold: float = 5.0, wait_threshold: float = 500.0) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "kernel_details.csv"

    if not csv_path.exists():
        return f"[kernel_details] 文件未找到: {csv_path}"

    total_rows = 0
    total_dur_us = 0.0
    total_wait_us = 0.0

    # Accelerator core 拆分
    core_stats = defaultdict(lambda: {"count": 0, "dur_us": 0.0})

    # 硬件单元聚合（duration 加权 — 按 kernel 执行时间加权，反映时间真实分布）
    aic_kernels = 0
    aic_dur_sum = 0.0
    aic_mac_wsum = 0.0
    aic_mte1_wsum = 0.0
    aic_mte2_wsum = 0.0
    aic_scalar_wsum = 0.0
    aic_fixpipe_wsum = 0.0
    aic_icache_wsum = 0.0
    aic_icache_dur_sum = 0.0
    aic_icache_max = 0.0
    format_counts = defaultdict(int)

    # 硬件单元聚合（AI_VECTOR_CORE）
    aiv_kernels = 0
    aiv_dur_sum = 0.0
    aiv_vec_wsum = 0.0
    aiv_mte2_wsum = 0.0
    aiv_mte3_wsum = 0.0
    aiv_scalar_wsum = 0.0
    aiv_icache_wsum = 0.0
    aiv_icache_dur_sum = 0.0
    aiv_icache_max = 0.0

    # 小 kernel 跟踪
    small_count = 0
    small_dur_total = 0.0
    small_type_count = defaultdict(int)

    # Kernel duration 分布（5 个桶）
    dur_buckets = {"<5us": 0, "5-20us": 0, "20-50us": 0, "50-200us": 0, ">200us": 0}

    # Block Dim 分布（duration 加权，仅 AI_CORE/AI_VECTOR_CORE）
    block_dim_dur = {"1": 0.0, "2-8": 0.0, "9-28": 0.0, "29+": 0.0}
    block_dim_total_dur = 0.0
    _bd_bounds = threshold("kernel_details", "block_dim_buckets", [8, 28])

    # 提取 threshold 值到循环外（避免热循环中重复调用）
    _suspect_min_dur = threshold("kernel_details", "suspect_min_duration_us", 10)
    _suspect_mac = threshold("kernel_details", "suspect_mac_ratio", 0.2)
    _compute_bound_mac = threshold("kernel_details", "compute_bound_mac_ratio", 0.5)
    _suspect_vec = threshold("kernel_details", "suspect_vec_ratio", 0.05)
    _cube_low = threshold("kernel_details", "cube_low_util", 50)

    # Cube 利用率跟踪（仅 AI_CORE，duration 加权）
    cube_util_wsum = 0.0
    cube_util_dur_sum = 0.0
    cube_util_min = float("inf")
    cube_low_util_count = 0
    cube_total_count = 0

    # 可疑 kernel：高 duration 但低 compute ratio（两种 core 类型）
    suspect_heap = []
    # 真正的 compute-bound：高 duration 且高 compute ratio（replace/quant 目标）
    compute_bound_heap = []

    # Wait time 分布桶
    _wb = threshold("kernel_details", "wait_buckets_us", [100, 500, 2000])
    _wk = [f"<{_wb[0]}us", f"{_wb[0]}-{_wb[1]}us", f"{_wb[1]}-{_wb[2]}us", f">{_wb[2]}us"]
    wait_buckets = {k: 0 for k in _wk}

    # 高 wait kernel 及上下文
    all_kernels = []

    for row in stream_csv(csv_path):
        total_rows += 1
        dur = safe_float(row.get("Duration(us)", 0))
        wait = safe_float(row.get("Wait Time(us)", 0))
        total_dur_us += dur
        total_wait_us += wait

        name = row.get("Name", "?")
        op_type = row.get("Type", "?")
        core = row.get("Accelerator Core", "?")
        block_dim = int(safe_float(row.get("Block Dim", 0)))
        cube_util = safe_float(row.get("cube_utilization(%)", 0))
        start_time = safe_float(row.get("Start Time(us)", 0))
        stream_id = row.get("Stream ID", "?").strip()
        input_formats = row.get("Input Formats", "").strip()

        core_stats[core]["count"] += 1
        core_stats[core]["dur_us"] += dur

        # Format 分布 (A2)：非 ND format = layout 转换开销
        if input_formats:
            for fmt in input_formats.replace(";", " ").split():
                fmt = fmt.strip().strip('"')
                if fmt:
                    format_counts[fmt] += 1

        # 硬件单元 ratio
        if core == "AI_CORE" and dur > 0:
            aic_kernels += 1
            mac_ratio = safe_float(row.get("aic_mac_ratio", 0))
            mte1_ratio = safe_float(row.get("aic_mte1_ratio", 0))
            mte2_ratio = safe_float(row.get("aic_mte2_ratio", 0))
            scalar_ratio = safe_float(row.get("aic_scalar_ratio", 0))
            fixpipe_ratio = safe_float(row.get("aic_fixpipe_ratio", 0))
            icache_miss = safe_float(row.get("aic_icache_miss_rate", 0))
            aic_dur_sum += dur
            aic_mac_wsum += mac_ratio * dur
            aic_mte1_wsum += mte1_ratio * dur
            aic_mte2_wsum += mte2_ratio * dur
            aic_scalar_wsum += scalar_ratio * dur
            aic_fixpipe_wsum += fixpipe_ratio * dur
            if icache_miss > 0:
                aic_icache_wsum += icache_miss * dur
                aic_icache_dur_sum += dur
                if icache_miss > aic_icache_max:
                    aic_icache_max = icache_miss
            if cube_util > 0:
                cube_util_wsum += cube_util * dur
                cube_util_dur_sum += dur
                cube_total_count += 1
                if cube_util < cube_util_min:
                    cube_util_min = cube_util
                if cube_util < _cube_low:
                    cube_low_util_count += 1
            # 可疑：高 duration 但 compute ratio 低
            if dur > _suspect_min_dur and mac_ratio < _suspect_mac:
                entry = (dur, total_rows, name, core, mac_ratio,
                         mte1_ratio + mte2_ratio, row.get("Input Shapes", ""), block_dim)
                if len(suspect_heap) < top_k:
                    heapq.heappush(suspect_heap, entry)
                elif dur > suspect_heap[0][0]:
                    heapq.heapreplace(suspect_heap, entry)
            # 真正的 compute-bound：高 duration 且高 mac ratio（replace/quant 目标）
            if dur > _suspect_min_dur and mac_ratio >= _compute_bound_mac:
                cb_entry = (dur, total_rows, name, core, mac_ratio, block_dim, row.get("Input Shapes", ""))
                if len(compute_bound_heap) < top_k:
                    heapq.heappush(compute_bound_heap, cb_entry)
                elif dur > compute_bound_heap[0][0]:
                    heapq.heapreplace(compute_bound_heap, cb_entry)

        elif core == "AI_VECTOR_CORE" and dur > 0:
            aiv_kernels += 1
            vec_ratio = safe_float(row.get("aiv_vec_ratio", 0))
            aiv_mte2_ratio = safe_float(row.get("aiv_mte2_ratio", 0))
            aiv_mte3_ratio = safe_float(row.get("aiv_mte3_ratio", 0))
            aiv_scalar_ratio_val = safe_float(row.get("aiv_scalar_ratio", 0))
            aiv_icache_miss = safe_float(row.get("aiv_icache_miss_rate", 0))
            aiv_dur_sum += dur
            aiv_vec_wsum += vec_ratio * dur
            aiv_mte2_wsum += aiv_mte2_ratio * dur
            aiv_mte3_wsum += aiv_mte3_ratio * dur
            aiv_scalar_wsum += aiv_scalar_ratio_val * dur
            if aiv_icache_miss > 0:
                aiv_icache_wsum += aiv_icache_miss * dur
                aiv_icache_dur_sum += dur
                if aiv_icache_miss > aiv_icache_max:
                    aiv_icache_max = aiv_icache_miss
            # 可疑：高 duration 但 vec ratio 低
            if dur > _suspect_min_dur and vec_ratio < _suspect_vec:
                entry = (dur, total_rows, name, core, vec_ratio,
                         aiv_mte2_ratio + aiv_mte3_ratio, row.get("Input Shapes", ""), block_dim)
                if len(suspect_heap) < top_k:
                    heapq.heappush(suspect_heap, entry)
                elif dur > suspect_heap[0][0]:
                    heapq.heapreplace(suspect_heap, entry)

        # 小 kernel
        if dur < small_threshold and dur > 0:
            small_count += 1
            small_dur_total += dur
            small_type_count[op_type] += 1

        # Duration 分布
        if dur > 0:
            if dur < 5: dur_buckets["<5us"] += 1
            elif dur < 20: dur_buckets["5-20us"] += 1
            elif dur < 50: dur_buckets["20-50us"] += 1
            elif dur < 200: dur_buckets["50-200us"] += 1
            else: dur_buckets[">200us"] += 1

        # Block Dim (仅 AI_CORE/AI_VECTOR_CORE，block_dim=0 无意义)
        if block_dim > 0 and dur > 0 and core in ("AI_CORE", "AI_VECTOR_CORE"):
            if block_dim == 1:
                block_dim_dur["1"] += dur
            elif block_dim <= _bd_bounds[0]:
                block_dim_dur["2-8"] += dur
            elif block_dim <= _bd_bounds[1]:
                block_dim_dur["9-28"] += dur
            else:
                block_dim_dur["29+"] += dur
            block_dim_total_dur += dur

        # Wait time 桶
        if wait < _wb[0]:
            wait_buckets[_wk[0]] += 1
        elif wait < _wb[1]:
            wait_buckets[_wk[1]] += 1
        elif wait < _wb[2]:
            wait_buckets[_wk[2]] += 1
        else:
            wait_buckets[_wk[3]] += 1

        # 存储用于上下文分析（含 start time + stream，用于时间分组）
        all_kernels.append({"name": name, "type": op_type, "dur": dur, "wait": wait,
                            "start": start_time, "stream": stream_id})

    if total_rows == 0:
        return f"[kernel_details] 空文件: {csv_path}"

    # 检测 fusible 序列：连续的小 kernel（同一 stream，按 start time 排序）
    # 当 stream 交错时，文件行序 ≠ 时间序；按 stream 分组并
    # 按 Start Time 排序，使"连续"指同一 stream 上时间相邻。
    fusible_sequences = []  # (total_dur, count, start_idx, op_types, stream)
    SMALL_THRESH = threshold("kernel_details", "fusible_small_us", 10.0)  # us
    MIN_LEN = threshold("kernel_details", "fusible_min_length", 5)

    # 每 stream 的 all_kernels 原始位置索引，按 start time 排序
    stream_order = defaultdict(list)
    for idx, k in enumerate(all_kernels):
        stream_order[k["stream"]].append(idx)
    for s in stream_order:
        stream_order[s].sort(key=lambda i: all_kernels[i]["start"])

    for s, idxs in stream_order.items():
        i = 0
        while i < len(idxs):
            ki = all_kernels[idxs[i]]
            if ki["dur"] > 0 and ki["dur"] < SMALL_THRESH:
                j = i
                seq_total = 0
                while j < len(idxs) and all_kernels[idxs[j]]["dur"] > 0 and all_kernels[idxs[j]]["dur"] < SMALL_THRESH:
                    seq_total += all_kernels[idxs[j]]["dur"]
                    j += 1
                seq_len = j - i
                if seq_len >= MIN_LEN and seq_total > threshold("kernel_details", "fusible_min_total_us", 100):
                    types = [all_kernels[idxs[k]]["type"] for k in range(i, j)]
                    fusible_sequences.append((seq_total, seq_len, idxs[i], types, s))
                i = j
            else:
                i += 1

    lines = []
    lines.append("# Kernel Details 分析")
    lines.append(f"数据来源: {csv_path}")
    lines.append(f"Kernel 总数: {total_rows:,}  |  Compute: {total_dur_us/1000:.1f}ms  |  Wait: {total_wait_us/1000:.1f}ms")
    lines.append("")

    # --- 1. Accelerator Core 分布 ---
    lines.append("## 1. Accelerator Core 分布")
    for core, info in sorted(core_stats.items(), key=lambda x: -x[1]["dur_us"]):
        pct = info["dur_us"] / total_dur_us * 100 if total_dur_us > 0 else 0
        lines.append(f"  {core}: {info['count']:,} 个 kernel, {info['dur_us']/1000:.1f}ms ({pct:.1f}%)")
    lines.append("")

    # --- 2. 硬件单元利用率 ---
    lines.append("## 2. 硬件单元利用率（duration 加权）")
    if aic_kernels > 0:
        lines.append(f"  AI_CORE ({aic_kernels} 个 kernel, {aic_dur_sum/1000:.1f}ms):")
        wmac = aic_mac_wsum / aic_dur_sum
        wmte1 = aic_mte1_wsum / aic_dur_sum
        wmte2 = aic_mte2_wsum / aic_dur_sum
        lines.append(f"    mac (compute):  {wmac:.3f}")
        lines.append(f"    mte1 (load):    {wmte1:.3f}")
        lines.append(f"    mte2 (store):   {wmte2:.3f}")
        lines.append(f"    scalar:         {aic_scalar_wsum/aic_dur_sum:.3f}")
        lines.append(f"    fixpipe:        {aic_fixpipe_wsum/aic_dur_sum:.3f}")
        if aic_icache_dur_sum > 0:
            lines.append(f"    icache miss:    avg={aic_icache_wsum/aic_icache_dur_sum:.3f}  max={aic_icache_max:.3f}")
        avg_mac = wmac
        avg_mte = wmte1 + wmte2
        if avg_mte > avg_mac * threshold("kernel_details", "hw_dominance_ratio", 1.5):
            lines.append(f"    - Memory 主导: mte ({avg_mte:.3f}) >> mac ({avg_mac:.3f})")
        elif avg_mac > avg_mte * threshold("kernel_details", "hw_dominance_ratio", 1.5):
            lines.append(f"    - Compute 主导: mac ({avg_mac:.3f}) >> mte ({avg_mte:.3f})")
    if aiv_kernels > 0:
        lines.append(f"  AI_VECTOR_CORE ({aiv_kernels} 个 kernel, {aiv_dur_sum/1000:.1f}ms):")
        lines.append(f"    vec (compute):  {aiv_vec_wsum/aiv_dur_sum:.3f}")
        lines.append(f"    mte2 (load):    {aiv_mte2_wsum/aiv_dur_sum:.3f}")
        lines.append(f"    mte3 (store):   {aiv_mte3_wsum/aiv_dur_sum:.3f}")
        lines.append(f"    scalar:         {aiv_scalar_wsum/aiv_dur_sum:.3f}")
        if aiv_icache_dur_sum > 0:
            lines.append(f"    icache miss:    avg={aiv_icache_wsum/aiv_icache_dur_sum:.3f}  max={aiv_icache_max:.3f}")
    # Format 分布 (A2)：非 ND format 表明存在 layout 转换开销
    if format_counts:
        total_fmt = sum(format_counts.values())
        non_nd = sum(c for f, c in format_counts.items() if f != "ND" and f != "N/A")
        lines.append(f"  Input Formats: {dict(sorted(format_counts.items(), key=lambda x: -x[1]))}")
        if (non_nd / total_fmt > threshold("kernel_details", "non_nd_format_ratio", 0.1)) if total_fmt else False:
            lines.append(f"  - {non_nd}/{total_fmt} ({non_nd/total_fmt*100:.0f}%) 非 ND input format — layout 转换开销。op_statistic 的 Transpose/Cast 聚合可确认占比")
    if cube_total_count > 0:
        avg_cube = cube_util_wsum / cube_util_dur_sum
        lines.append(f"  Cube 利用率: avg={avg_cube:.1f}%, min={cube_util_min:.1f}%, "
                     f"low(<50%)={cube_low_util_count}/{cube_total_count}")
    if aic_kernels == 0 and aiv_kernels == 0:
        lines.append("  [!] 所有 ratio 均为 0 — 请检查采集时是否使用 aic_metrics=PipeUtilization")
    lines.append("")

    # --- 3. Kernel Duration 分布 ---
    sec_num = 3
    lines.append(f"## {sec_num}. Kernel Duration 分布")
    dur_counted = sum(dur_buckets.values())
    for bucket, count in dur_buckets.items():
        pct = count / dur_counted * 100 if dur_counted > 0 else 0
        bar = "█" * int(pct / 3)
        lines.append(f"  {bucket:>8}: {count:>6} ({pct:>5.1f}%) {bar}")
    short_ratio = (dur_buckets["<5us"] + dur_buckets["5-20us"]) / dur_counted * 100 if dur_counted > 0 else 0
    lines.append(f"  短 kernel 占比 (<20us): {short_ratio:.1f}%")
    if short_ratio > threshold("kernel_details", "short_kernel_dominant", 60):
        lines.append(f"  - 多数 kernel 非常短。减少 op 数量可能比优化单个 op 收益更大。op_statistic 的 fragmentation signal + trace_view 的 dispatch latency 可交叉确认")
    lines.append("")

    sec_num += 1
    lines.append(f"## {sec_num}. 小 kernel（duration < {small_threshold}us）")
    if small_count > 0:
        small_pct = small_count / total_rows * 100
        lines.append(f"  数量: {small_count:,} ({small_pct:.1f}% 占所有 kernel)  |  累计: {small_dur_total/1000:.2f}ms")
        lines.append(f"  Top 类型:")
        for t, c in sorted(small_type_count.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {t}: {c}")
    else:
        lines.append(f"  未发现")
    lines.append("")

    # --- 5. Block Dim 分布 ---
    sec_num += 1
    lines.append(f"## {sec_num}. Block Dim 分布（并行度，duration 加权）")
    if block_dim_total_dur > 0:
        for bucket, dur_sum in block_dim_dur.items():
            pct = dur_sum / block_dim_total_dur * 100
            bar = "█" * int(pct / 3)
            lines.append(f"  Dim {bucket:>5}: {dur_sum/1000:>8.1f}ms ({pct:>5.1f}%) {bar}")
        low_par_ratio = block_dim_dur["1"] / block_dim_total_dur
        if low_par_ratio > threshold("kernel_details", "low_parallelism_ratio", 0.1):
            lines.append(f"  - Block Dim=1 占 {low_par_ratio*100:.1f}% 计算时间: shape 可能太小，无法并行")
    else:
        lines.append("  无 AI_CORE/AI_VECTOR_CORE kernel")
    lines.append("")

    # --- 6. Wait Time 分布（事实陈述）---
    sec_num += 1
    lines.append(f"## {sec_num}. Wait Time 分布")
    for bucket, count in wait_buckets.items():
        pct = count / total_rows * 100 if total_rows > 0 else 0
        lines.append(f"  {bucket:>10}: {count:>7} ({pct:.1f}%)")
    # TASK_QUEUE 检测：若所有 wait 均匀偏高，async pipeline 可能未生效
    all_waits = [k["wait"] for k in all_kernels]
    if all_waits and len(all_waits) > 20:
        avg_wait = sum(all_waits) / len(all_waits)
        sorted_waits = sorted(all_waits)
        median_wait = sorted_waits[len(sorted_waits) // 2]
        if median_wait > threshold("kernel_details", "median_wait_threshold_us", 100):
            lines.append(f"  [SIGNAL] Wait time 普遍偏高 (avg={avg_wait:.0f}us, median={median_wait:.0f}us)")
    lines.append("")

    # --- 7. 可疑信号（诊断）---
    sec_num += 1
    lines.append(f"## {sec_num}. 可疑信号")
    lines.append("  [DEFINITE]=可直接行动  [SIGNAL]=异常，需结合其他维度交叉验证")
    lines.append("")

    # 7a. 可疑 kernel
    suspect_sorted = sorted(suspect_heap, key=lambda x: -x[0])
    if suspect_sorted:
        lines.append("  [SIGNAL] 高 duration、低 compute ratio — 交叉验证: 用 --filter <op> 查看 shape 分布")
        header = f"  {'Name':<42} {'Core':<5} {'Dur(us)':>8} {'Compute':>8} {'Move':>8} {'BDim':>5} {'Shapes'}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for dur, _, name, core, compute_ratio, move_ratio, shapes, bdim in suspect_sorted:
            shapes_clean = shapes.replace("\n", " ").replace(";", "|").replace('"', '')[:30]
            core_short = "AIC" if "AI_CORE" == core else "AIV"
            lines.append(
                f"  {name:<42} {core_short:<5} {dur:>8.1f} "
                f"{compute_ratio:>7.3f} {move_ratio:>7.3f} {bdim:>5} {shapes_clean}")
        lines.append("")

    # 7a-bis. 真正的 compute-bound kernel（高 duration + 高 compute ratio）
    cb_sorted = sorted(compute_bound_heap, key=lambda x: -x[0])
    if cb_sorted:
        lines.append("  [SIGNAL] 真正的 compute-bound（高 duration + 高 compute ratio）— replace/quantize/split 目标")
        header = f"  {'Name':<42} {'Dur(us)':>8} {'mac':>6} {'BDim':>5} {'Shapes'}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for dur, _, name, core, mac_ratio, bdim, shapes in cb_sorted:
            shapes_clean = shapes.replace("\n", " ").replace(";", "|").replace('"', '')[:30]
            lines.append(f"  {name:<42} {dur:>8.1f} {mac_ratio:>5.2f} {bdim:>5} {shapes_clean}")
        lines.append("")

    # 7b. 高 wait 上下文
    high_wait_indices = [i for i, k in enumerate(all_kernels) if k["wait"] > wait_threshold]
    if high_wait_indices:
        lines.append(f"  [SIGNAL] 高 wait kernel (wait > {wait_threshold:.0f}us) — 在 trace_view 中查找原因")
        lines.append("")
        top_waits = sorted(high_wait_indices, key=lambda i: -all_kernels[i]["wait"])[:min(top_k, 8)]
        for rank_idx, idx in enumerate(top_waits, 1):
            k = all_kernels[idx]
            lines.append(f"  [{rank_idx}] #{idx} {k['name']}  wait={format_duration_ms(k['wait'])}  stream={k['stream']}")
            # 同一 stream 上的时间邻居（非文件顺序）
            s_order = stream_order.get(k["stream"], [])
            pos = s_order.index(idx) if idx in s_order else -1
            context = 2
            if pos >= 0:
                lo = max(0, pos - context)
                hi = min(len(s_order), pos + context + 1)
                for p in range(lo, hi):
                    ci = s_order[p]
                    ck = all_kernels[ci]
                    marker = " <<<" if ci == idx else ""
                    lines.append(f"      [{ci}] {ck['type']:<20} dur={ck['dur']:>7.1f}us  wait={ck['wait']:>7.0f}us{marker}")
            lines.append("")

    # 7c. Fusible 算子序列
    if fusible_sequences:
        fusible_sorted = sorted(fusible_sequences, key=lambda x: -x[0])
        total_fusible = sum(s[0] for s in fusible_sorted)
        lines.append(f"  [SIGNAL] Fusible 序列: {len(fusible_sorted)} 个序列，每个含 ≥{MIN_LEN} 个连续小 kernel (<{SMALL_THRESH}us)")
        lines.append(f"    Fusible 序列总耗时: {total_fusible/1000:.2f}ms ({total_fusible/total_dur_us*100:.1f}% 占 compute)")
        lines.append(f"    按累计耗时排序的 Top {min(top_k, 5)}:")
        for total, count, start_idx, types, s in fusible_sorted[:5]:
            tc = Counter(types).most_common(3)
            type_str = ", ".join(f"{t}:{c}" for t, c in tc)
            lines.append(f"      {total/1000:.2f}ms  {count} 个 kernel  at #{start_idx}  stream={s}  类型: {type_str}")
        lines.append("    - 交叉验证: 检查这些是否能 fuse (equivalent_substitution layer 1) 或 batch。")
        lines.append("")

    if not suspect_sorted and not cb_sorted and not high_wait_indices and not fusible_sequences:
        lines.append("  无")
        lines.append("")

    return "\n".join(lines)


def parse_filtered(profiling_dir: str, filters: list, rank=None, top_k: int = 15) -> str:
    """过滤模式：对特定算子的深入分析。"""
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "kernel_details.csv"

    if not csv_path.exists():
        return f"[kernel_details] 文件未找到: {csv_path}"

    filter_lower = [f.lower() for f in filters]

    matched = []
    all_seq = []  # (index_in_file, is_matched, name, type, dur, wait, start, stream)
    idx = 0
    for row in stream_csv(csv_path):
        name = row.get("Name", "")
        op_type = row.get("Type", "")
        dur = safe_float(row.get("Duration(us)", 0))
        wait = safe_float(row.get("Wait Time(us)", 0))
        start_ts = safe_float(row.get("Start Time(us)", 0))
        stream_id = row.get("Stream ID", "?").strip()
        is_match = any(f in name.lower() or f in op_type.lower() for f in filter_lower)
        all_seq.append((idx, is_match, name, op_type, dur, wait, start_ts, stream_id))
        if is_match:
            matched.append(row)
        idx += 1

    lines = []
    lines.append(f"# Kernel Details — 过滤式深入分析")
    lines.append(f"数据来源: {csv_path}")
    lines.append(f"过滤: {', '.join(filters)}")
    lines.append(f"匹配的 kernel: {len(matched)}")
    lines.append("")

    if not matched:
        lines.append("没有 kernel 匹配该过滤条件。")
        return "\n".join(lines)

    # --- 1. 摘要 ---
    total_dur = sum(safe_float(r.get("Duration(us)", 0)) for r in matched)
    total_wait = sum(safe_float(r.get("Wait Time(us)", 0)) for r in matched)
    lines.append("## 1. 摘要")
    lines.append(f"  数量: {len(matched)}")
    lines.append(f"  总 duration: {total_dur/1000:.2f} ms")
    lines.append(f"  总 wait: {total_wait/1000:.2f} ms")
    lines.append(f"  平均 duration: {total_dur/len(matched):.1f} us")
    lines.append(f"  平均 wait: {total_wait/len(matched):.1f} us")
    lines.append("")

    # --- 2. Shape - Performance 相关性 ---
    lines.append("## 2. Shape - Performance 相关性")
    shape_groups = defaultdict(list)
    for r in matched:
        shape = r.get("Input Shapes", "").replace("\n", " ").replace('"', '').strip()
        if not shape:
            shape = "(空)"
        shape_groups[shape].append(r)

    shape_perf = []
    for shape, rows in shape_groups.items():
        durs = [safe_float(r.get("Duration(us)", 0)) for r in rows]
        waits = [safe_float(r.get("Wait Time(us)", 0)) for r in rows]
        shape_perf.append((sum(durs), shape, len(rows), durs, waits, rows))
    shape_perf.sort(key=lambda x: -x[0])

    header = f"  {'Shape':<40} {'Count':>6} {'Avg Dur':>8} {'Min':>7} {'Max':>7} {'Total(ms)':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for total, shape, count, durs, waits, rows in shape_perf[:top_k]:
        avg_d = sum(durs) / len(durs)
        lines.append(
            f"  {shape[:40]:<40} {count:>6} {avg_d:>7.1f}us "
            f"{min(durs):>6.1f} {max(durs):>6.1f} {total/1000:>10.2f}"
        )
    lines.append("")

    # --- 3. 逐实例硬件拆分（最慢的 N 个）---
    lines.append(f"## 3. 最慢实例 — 硬件拆分")
    matched_sorted = sorted(matched, key=lambda r: -safe_float(r.get("Duration(us)", 0)))
    lines.append(f"  (按 duration 排序的 Top {min(top_k, len(matched_sorted))}，展示逐实例硬件 ratio)")
    lines.append("")

    for i, r in enumerate(matched_sorted[:top_k], 1):
        name = r.get("Name", "?")
        dur = safe_float(r.get("Duration(us)", 0))
        wait = safe_float(r.get("Wait Time(us)", 0))
        core = r.get("Accelerator Core", "?")
        bdim = int(safe_float(r.get("Block Dim", 0)))
        shape = r.get("Input Shapes", "").replace("\n", " ").replace('"', '').strip()[:50]

        lines.append(f"  [{i}] {name}  dur={dur:.1f}us  wait={wait:.0f}us  block_dim={bdim}")
        lines.append(f"      shape: {shape}")

        if core == "AI_CORE":
            mac = safe_float(r.get("aic_mac_ratio", 0))
            mte1 = safe_float(r.get("aic_mte1_ratio", 0))
            mte2 = safe_float(r.get("aic_mte2_ratio", 0))
            scalar = safe_float(r.get("aic_scalar_ratio", 0))
            fixpipe = safe_float(r.get("aic_fixpipe_ratio", 0))
            cube = safe_float(r.get("cube_utilization(%)", 0))
            lines.append(f"      AI_CORE: mac={mac:.3f} mte1={mte1:.3f} mte2={mte2:.3f} scalar={scalar:.3f} fixpipe={fixpipe:.3f} cube={cube:.1f}%")
        elif core == "AI_VECTOR_CORE":
            vec = safe_float(r.get("aiv_vec_ratio", 0))
            mte2 = safe_float(r.get("aiv_mte2_ratio", 0))
            mte3 = safe_float(r.get("aiv_mte3_ratio", 0))
            scalar = safe_float(r.get("aiv_scalar_ratio", 0))
            lines.append(f"      AI_VECTOR: vec={vec:.3f} mte2={mte2:.3f} mte3={mte3:.3f} scalar={scalar:.3f}")
        lines.append("")

    # --- 4. Wait time 上下文：该 op 的前后内容 ---
    lines.append("## 4. Wait Time Context")
    matched_waits = [(safe_float(r.get("Wait Time(us)", 0)), i)
                     for i, r in enumerate(matched)]
    avg_wait = total_wait / len(matched)
    max_wait_val = max(w for w, _ in matched_waits) if matched_waits else 0
    lines.append(f"  平均 wait: {avg_wait:.0f}us, 最大 wait: {max_wait_val:.0f}us")
    lines.append("")

    # 在完整序列中找到匹配 kernel 的位置以展示邻居
    high_wait_matched = []
    for seq_idx, (file_idx, is_match, name, op_type, dur, wait, start_ts, stream_id) in enumerate(all_seq):
        if is_match and wait > avg_wait * threshold("kernel_details", "filter_high_wait_multiplier", 3) and wait > threshold("kernel_details", "filter_high_wait_min_us", 200):
            high_wait_matched.append((wait, seq_idx, stream_id))
    high_wait_matched.sort(key=lambda x: -x[0])

    if high_wait_matched:
        # 按 stream 分组并按 start time 排序，用于时间邻居查找
        pf_stream_order = defaultdict(list)
        for si, entry in enumerate(all_seq):
            pf_stream_order[entry[7]].append(si)
        for s in pf_stream_order:
            pf_stream_order[s].sort(key=lambda si: all_seq[si][6])
        lines.append(f"  wait > 3 倍均值 ({avg_wait*3:.0f}us) 的实例，展示同 stream 时间邻居:")
        lines.append("")
        for rank_idx, (wait_val, seq_idx, stream_id) in enumerate(high_wait_matched[:min(8, top_k)], 1):
            lines.append(f"  [{rank_idx}] wait={format_duration_ms(wait_val)} 位于位置 #{seq_idx}  stream={stream_id}")
            s_order = pf_stream_order.get(stream_id, [])
            pos = s_order.index(seq_idx) if seq_idx in s_order else -1
            context = 3
            if pos >= 0:
                lo = max(0, pos - context)
                hi = min(len(s_order), pos + context + 1)
                for p in range(lo, hi):
                    ci = s_order[p]
                    _, is_m, n, t, d, w = all_seq[ci][:6]
                    marker = " <<<" if ci == seq_idx else ""
                    match_tag = "*" if is_m else " "
                    lines.append(f"      {match_tag}[{ci}] {t:<20} dur={d:>7.1f}us  wait={w:>7.0f}us{marker}")
            lines.append("")
    else:
        lines.append("  没有 wait time 显著偏高的实例。")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--filter", nargs="+", default=None,
                        help="按算子名称/类型过滤（子串匹配，大小写不敏感）。"
                             "指定后仅展示匹配算子的逐 kernel 详情。")
    parser.add_argument("--small-threshold", type=float, default=5.0,
                        help="用于识别小 kernel 的 duration 阈值 (us)")
    parser.add_argument("--wait-threshold", type=float, default=500.0,
                        help="用于高 wait 上下文分析的 wait time 阈值 (us)")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    if args.filter:
        result = parse_filtered(args.profiling_dir, args.filter, args.rank, args.top_k)
    else:
        result = parse(args.profiling_dir, args.rank, args.top_k,
                       args.small_threshold, args.wait_threshold)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
