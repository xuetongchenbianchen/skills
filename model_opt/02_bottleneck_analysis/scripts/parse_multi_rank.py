#!/usr/bin/env python3
"""跨 Rank 对比分析 — 多卡 profiling 的分层分析工具。

基于四阶段方法论实现:
  Phase 1: 全局扫描与慢卡定位 — (T_max-T_avg)/T_avg 慢卡检测、通信域推断
  Phase 2: 时间线分解与并行效率 — 计算-通信重叠率、假性重叠检测
  Phase 3: 深度根因定位 — 通信 R_wait 同步等待比例、小包/对齐分析、算子跨 rank 方差
  Phase 4: MoE/AlltoAll 负载不均衡检测 — Token 分布 CV 分析

当多卡 profiling 目录包含 rank_0 ~ rank_N 子目录时，本脚本跨所有 rank
提取并对比 step_trace、communication、op_statistic、communication_matrix、
kernel_details 数据，定位:
  - Straggler rank / Tail Card（拖慢全局的慢卡）
  - 通信域与并行策略（DP/TP/PP/EP 推断）
  - 计算-通信重叠效率与假性重叠
  - R_wait 同步等待比例与 straggler 受害者
  - 小包/字节对齐问题
  - AlltoAll/MoE Token 分布不均衡

用法:
    python parse_multi_rank.py <profiling_dir> [--top-k 15] [--output out.txt]
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    find_ascend_profiler_output, read_csv_all, safe_float, safe_int, threshold,
)
import parse_step_trace
import parse_communication


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _discover_ranks(profiling_dir: str) -> list:
    """Return sorted list of (rank_id, ascend_dir) for all rank_N dirs."""
    base = Path(profiling_dir)
    ranks = []
    for rd in sorted(base.glob("rank_*")):
        if not rd.is_dir():
            continue
        try:
            rid = int(rd.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        ascend_dir = find_ascend_profiler_output(str(rd))
        ranks.append((rid, ascend_dir))
    return ranks


def _cv(values):
    """Coefficient of variation (std/mean). Returns 0 for single-element lists."""
    if len(values) < 2:
        return 0.0
    m = statistics.mean(values)
    if m == 0:
        return 0.0
    return statistics.stdev(values) / m


def _median(values):
    return statistics.median(values) if values else 0.0


def _load_step_totals(ascend_dir):
    """Aggregate step_trace for a single rank. Returns dict of totals in us."""
    csv_path = ascend_dir / "step_trace_time.csv"
    rows = read_csv_all(csv_path)
    if not rows:
        return None
    step_data = [parse_step_trace._row_totals(r) for r in rows]
    agg_keys = ["computing", "free", "comm_not_ovl", "comm_raw", "overlapped",
                "stage", "bubble", "total"]
    tot = {k: sum(s[k] for s in step_data) for k in agg_keys}
    tot["step_count"] = len(step_data)
    return tot


def _load_comm_summary(ascend_dir):
    """Parse communication.json for a single rank. Returns aggregated by_type dict + p2p count."""
    comm_path = ascend_dir / "communication.json"
    if not comm_path.exists():
        return None
    data = json.loads(comm_path.read_text(encoding="utf-8"))
    by_type = defaultdict(lambda: {"count": 0, "elapse": 0, "transit": 0, "wait": 0, "sync": 0, "idle": 0})
    total = {"elapse": 0, "transit": 0, "wait": 0, "sync": 0, "idle": 0}
    p2p_count = 0
    for step, sd in data.items():
        for opname, info in sd.get("collective", {}).items():
            if opname == "Total Op Info":
                continue
            ti = info.get("Communication Time Info", {})
            elapse = safe_float(ti.get("Elapse Time(ms)", 0))
            transit = safe_float(ti.get("Transit Time(ms)", 0))
            wait = safe_float(ti.get("Wait Time(ms)", 0))
            sync = safe_float(ti.get("Synchronization Time(ms)", 0))
            idle = safe_float(ti.get("Idle Time(ms)", 0))
            op_type = parse_communication.extract_op_type(opname)
            by_type[op_type]["count"] += 1
            by_type[op_type]["elapse"] += elapse
            by_type[op_type]["transit"] += transit
            by_type[op_type]["wait"] += wait
            by_type[op_type]["sync"] += sync
            by_type[op_type]["idle"] += idle
            total["elapse"] += elapse
            total["transit"] += transit
            total["wait"] += wait
            total["sync"] += sync
            total["idle"] += idle
        p2p_count += len(sd.get("p2p", {}))
    return {"by_type": dict(by_type), "total": total, "p2p_count": p2p_count}


def _load_op_stats(ascend_dir):
    """Load op_statistic.csv → {op_type: {"count": n, "total_us": us}}."""
    csv_path = ascend_dir / "op_statistic.csv"
    rows = read_csv_all(csv_path)
    if not rows:
        return {}
    result = {}
    for row in rows:
        op_type = row.get("OP Type", "?")
        result[op_type] = {
            "count": safe_int(row.get("Count", 0)),
            "total_us": safe_float(row.get("Total Time(us)", 0)),
        }
    return result


def _load_comm_matrix(ascend_dir):
    """Parse communication_matrix.json → list of (opname, link, bw, size, transport, transit_ms)."""
    mat_path = ascend_dir / "communication_matrix.json"
    if not mat_path.exists():
        return []
    data = json.loads(mat_path.read_text(encoding="utf-8"))
    entries = []
    for step, sd in data.items():
        for opname, links in sd.get("collective", {}).items():
            for link, lv in links.items():
                bw = safe_float(lv.get("Bandwidth(GB/s)", 0))
                size = safe_float(lv.get("Transit Size(MB)", 0))
                transport = lv.get("Transport Type", "?")
                transit_ms = safe_float(lv.get("Transit Time(ms)", 0))
                entries.append((opname[:40], link, bw, size, transport, transit_ms))
    return entries


def _load_step_per_step(ascend_dir):
    """Per-step step_trace data for straggler step detection. Returns list of step dicts or None."""
    csv_path = ascend_dir / "step_trace_time.csv"
    rows = read_csv_all(csv_path)
    if not rows:
        return None
    return [parse_step_trace._row_totals(r) for r in rows]


def _load_comm_size_distribution(ascend_dir):
    """Extract Size Distribution from communication.json Total Op Info.

    Returns {link_type: [(size_mb, count, transit_ms), ...]} aggregated across steps.
    Used for small packet detection (Phase 3.4).
    """
    comm_path = ascend_dir / "communication.json"
    if not comm_path.exists():
        return {}
    data = json.loads(comm_path.read_text(encoding="utf-8"))
    result = defaultdict(list)
    for step, sd in data.items():
        total_info = sd.get("collective", {}).get("Total Op Info", {})
        bw_info = total_info.get("Communication Bandwidth Info", {})
        for link_type, bw in bw_info.items():
            dist = bw.get("Size Distribution", {})
            if isinstance(dist, dict):
                for size_str, vals in dist.items():
                    if isinstance(vals, list) and len(vals) >= 2:
                        size_mb = safe_float(size_str)
                        count = safe_float(vals[0])
                        transit_ms = safe_float(vals[1])
                        result[link_type].append((size_mb, count, transit_ms))
    return dict(result)


def _load_alltoall_shapes(ascend_dir):
    """Extract alltoall kernel Input Shapes from kernel_details.csv.

    Returns list of (name, input_shapes_str, duration_us) for alltoall-related kernels.
    Used for MoE load imbalance detection (Phase 4).
    """
    from common import stream_csv
    csv_path = ascend_dir / "kernel_details.csv"
    if not csv_path.exists():
        return []
    results = []
    for row in stream_csv(csv_path):
        name = row.get("Name", "")
        if "alltoall" not in name.lower() and "all2all" not in name.lower():
            continue
        shapes = row.get("Input Shapes", "")
        duration = safe_float(row.get("Duration(us)", 0))
        results.append((name, shapes, duration))
    return results


# === Step Trace 跨 Rank 对比 ===

def _section_step_trace(ranks_data, top_k):
    """Phase 1: Cross-rank step_trace comparison + straggler (Tail Card) detection."""
    L = []
    L.append("# Phase 1: 跨 Rank Step Trace 对比与慢卡定位")
    L.append("")

    # Build table
    comp_vals = [d["step"]["computing"] / 1000 for _, d in ranks_data]
    free_vals = [d["step"]["free"] / 1000 for _, d in ranks_data]
    comm_vals = [d["step"]["comm_not_ovl"] / 1000 for _, d in ranks_data]
    total_vals = [d["step"]["total"] / 1000 for _, d in ranks_data]
    util_vals = [d["step"]["computing"] / d["step"]["total"] * 100 if d["step"]["total"] > 0 else 0
                 for _, d in ranks_data]

    header = (f"  {'Rank':>5} {'Total(ms)':>10} {'Comp(ms)':>10} {'Free(ms)':>10} "
              f"{'Comm(NO)':>10} {'Util%':>7}")
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    for (rid, d), total, comp, free, comm, util in zip(
            ranks_data, total_vals, comp_vals, free_vals, comm_vals, util_vals):
        L.append(f"  {rid:>5} {total:>10.1f} {comp:>10.1f} {free:>10.1f} {comm:>10.1f} {util:>6.1f}%")
    L.append("")

    # Statistics
    med_total = _median(total_vals)
    avg_total = statistics.mean(total_vals)
    max_total = max(total_vals)
    min_total = min(total_vals)
    L.append(f"  Total:   min={min_total:.1f}  avg={avg_total:.1f}  median={med_total:.1f}  max={max_total:.1f}  CV={_cv(total_vals):.3f}")
    L.append(f"  Compute: min={min(comp_vals):.1f}  median={_median(comp_vals):.1f}  max={max(comp_vals):.1f}  CV={_cv(comp_vals):.3f}")
    L.append(f"  Comm:    min={min(comm_vals):.1f}  median={_median(comm_vals):.1f}  max={max(comm_vals):.1f}  CV={_cv(comm_vals):.3f}")
    L.append("")

    # Tail Card detection: (T_max - T_avg) / T_avg
    if avg_total > 0:
        tail_ratio = (max_total - avg_total) / avg_total
        tail_pct = tail_ratio * 100
        worst_rid = ranks_data[total_vals.index(max_total)][0]
        L.append(f"  Tail Card Ratio: (T_max - T_avg) / T_avg = {tail_pct:.1f}%  (最慢: rank_{worst_rid})")
        if tail_ratio > threshold("multi_rank", "tail_card_ratio_strong", 0.15):
            L.append(f"  [DEFINITE] 严重慢卡 (Tail Card): rank_{worst_rid} 比平均慢 {tail_pct:.0f}%，"
                     f"超过 {threshold('multi_rank', 'tail_card_ratio_strong', 0.15)*100:.0f}% 阈值")
            L.append("    木桶效应：整体吞吐量被最慢的卡拖累，平均步时不代表真实性能")
            L.append(f"    后续深度分析应聚焦 rank_{worst_rid}（问题设备已锁定）")
        elif tail_ratio > threshold("multi_rank", "tail_card_ratio", 0.10):
            L.append(f"  [DEFINITE] 慢卡 (Tail Card): rank_{worst_rid} 比平均慢 {tail_pct:.0f}%，"
                     f"超过 {threshold('multi_rank', 'tail_card_ratio', 0.10)*100:.0f}% 阈值")
        else:
            L.append(f"  各卡步时差异在正常范围内 (<{threshold('multi_rank', 'tail_card_ratio', 0.10)*100:.0f}%)")
    L.append("")

    # Per-step straggler detection (if multiple steps)
    per_step_data = [d.get("step_per_step") for _, d in ranks_data if d.get("step_per_step")]
    problem_step = None
    # Only lock down a problem rank when tail card ratio actually exceeds threshold
    problem_rank = worst_rid if (avg_total > 0 and
                                tail_ratio > threshold("multi_rank", "tail_card_ratio", 0.10)) else None
    if per_step_data and all(len(d) == len(per_step_data[0]) for d in per_step_data) and len(per_step_data[0]) > 1:
        L.append("## 逐 Step 慢卡分析")
        n_steps = len(per_step_data[0])
        # Step time spike detection: step time > previous * ratio = sudden spike
        spike_ratio = threshold("multi_rank", "step_spike_ratio", 2.0)
        prev_step_avg = None
        for step_idx in range(n_steps):
            step_totals = [d[step_idx]["total"] / 1000 for d in per_step_data]
            step_avg = statistics.mean(step_totals)
            step_max = max(step_totals)
            step_id = per_step_data[0][step_idx].get("step_id", str(step_idx))
            # Per-step straggler
            if step_avg > 0 and (step_max - step_avg) / step_avg > threshold("multi_rank", "tail_card_ratio", 0.10):
                worst_rid_step = ranks_data[step_totals.index(step_max)][0]
                L.append(f"  [SIGNAL] Step {step_id}: rank_{worst_rid_step} 最慢 ({step_max:.1f}ms vs avg {step_avg:.1f}ms, "
                         f"+{(step_max-step_avg)/step_avg*100:.0f}%)")
                if problem_step is None:
                    problem_step = step_id
                    problem_rank = worst_rid_step
            # Step time spike (sudden surge)
            if prev_step_avg is not None and step_avg > prev_step_avg * spike_ratio:
                L.append(f"  [SIGNAL] Step {step_id}: Step Time 突然飙升 ({step_avg:.1f}ms vs 前步 {prev_step_avg:.1f}ms, "
                         f"{step_avg/prev_step_avg:.1f}x) — 检查该 step 是否有编译/动态 shape/异常事件")
                if problem_step is None:
                    problem_step = step_id
            prev_step_avg = step_avg
        L.append("")

    # Phase 1 conclusion: lock down problem step and rank for subsequent analysis
    L.append("## Phase 1 结论")
    if problem_rank is not None or problem_step is not None:
        parts = []
        if problem_step is not None:
            parts.append(f"问题 Step: {problem_step}")
        if problem_rank is not None:
            parts.append(f"问题 Rank: rank_{problem_rank}")
        L.append(f"  锁定: {', '.join(parts)}")
        L.append("  后续所有深度 Profiling 分析（Phase 2-5）应聚焦此 Step 和 Rank，无需分析所有卡")
    elif avg_total > 0 and tail_ratio > threshold("multi_rank", "tail_card_ratio", 0.10):
        L.append(f"  锁定: 问题 Rank: rank_{worst_rid}")
        L.append("  后续所有深度 Profiling 分析应聚焦此 Rank")
    else:
        L.append("  无显著慢卡 — 各 rank 表现一致，后续按全域分析")

    # Comm imbalance
    comm_cv = _cv(comm_vals)
    if comm_cv > threshold("multi_rank", "comm_cv_definite", 0.50):
        L.append(f"  [DEFINITE] 通信时间跨 rank 严重不均衡 (CV={comm_cv:.2f}) — "
                 f"min={min(comm_vals):.1f}ms, max={max(comm_vals):.1f}ms")
        L.append("    某些 rank 通信耗时远超其他 — 可能是 TP/PP 切分不均或 straggler 导致的 wait 膨胀")
    elif comm_cv > threshold("multi_rank", "comm_cv_signal", 0.30):
        L.append(f"  [SIGNAL] 通信时间跨 rank 不均衡 (CV={comm_cv:.2f}) — "
                 f"min={min(comm_vals):.1f}ms, max={max(comm_vals):.1f}ms")

    # Compute imbalance (should be near-zero in balanced scenario)
    comp_cv = _cv(comp_vals)
    if comp_cv > threshold("multi_rank", "compute_cv_signal", 0.05):
        L.append(f"  [SIGNAL] Computing 时间跨 rank 不均衡 (CV={comp_cv:.3f}) — "
                 f"min={min(comp_vals):.1f}ms, max={max(comp_vals):.1f}ms")
        L.append("    Computing 本应一致，不均衡说明计算负载分配不均或某些 rank 有额外计算")

    L.append("")
    return "\n".join(L)


# === 通信域推断与并行策略识别 ===

def _section_parallel_strategy(ranks_data, top_k):
    """Phase 1: Infer parallel strategy (DP/TP/PP/EP) from communication op types."""
    L = []
    L.append("# Phase 1: 通信域推断与并行策略识别")
    L.append("")
    L.append("  通信域分组原则: DP 域内各卡执行相同计算图，可直接对比 Step Time;")
    L.append("  TP/PP 域内不同 Stage 执行不同算子，不可直接对比耗时。")
    L.append("")

    # Collect all comm op types across ranks
    all_types = set()
    for _, d in ranks_data:
        if d.get("comm"):
            all_types.update(d["comm"]["by_type"].keys())

    if not all_types:
        L.append("  (无通信算子，可能是纯单卡推理)")
        L.append("")
        return "\n".join(L)

    # Infer parallel strategy
    strategies = []
    if "allReduce" in all_types:
        strategies.append(("DP (数据并行)", "allReduce", "各卡执行相同计算图，可直接对比 Step Time"))
    if "allGather" in all_types or "reducescatter" in all_types:
        strategies.append(("TP (张量并行) / FSDP", "allGather / reducescatter",
                          "不同 rank 执行不同分片，同一 TP 组内可对比"))
    if "alltoall" in all_types:
        strategies.append(("EP (专家并行) / SP (序列并行)", "alltoall",
                          "MoE 场景 — 需检查 Token 分布是否均衡 (见 Phase 4)"))
    # Check P2P (send/recv) from communication.json
    has_p2p = False
    for _, d in ranks_data:
        if d.get("comm_p2p_count") and d["comm_p2p_count"] > 0:
            has_p2p = True
            break
    if has_p2p:
        strategies.append(("PP (流水线并行)", "send/recv (P2P)",
                          "不同 Stage 执行不同层 — 不可跨 Stage 对比，需关注 Bubble"))

    L.append("## 检测到的并行策略")
    for name, ops, note in strategies:
        L.append(f"  - {name}: 通信算子 {ops}")
        L.append(f"    {note}")
    L.append("")

    # Domain grouping guidance
    n_ranks = len(ranks_data)
    if "allReduce" in all_types and not has_p2p:
        L.append(f"  [INFO] 检测到 allReduce 且无 P2P — {n_ranks} 个 rank 可能在同一 DP 域内，"
                 "可直接对比 Step Time")
    elif "allGather" in all_types and "alltoall" in all_types:
        L.append(f"  [INFO] 检测到 allGather + alltoall — 可能是 TP+EP 混合并行 (MoE 场景)")
        L.append("    跨 rank 对比时需按 TP 组分组，不可全域混比")
    L.append("")

    return "\n".join(L)


# === 计算-通信重叠分析 (Phase 2) ===

def _section_overlap(ranks_data, top_k):
    """Phase 2: Compute-Comm overlap ratio analysis."""
    L = []
    L.append("# Phase 2: 计算-通信重叠与并行效率分析")
    L.append("")

    overlap_vals = [d["step"]["overlapped"] / 1000 for _, d in ranks_data]
    comm_raw_vals = [d["step"]["comm_raw"] / 1000 for _, d in ranks_data]
    comm_novl_vals = [d["step"]["comm_not_ovl"] / 1000 for _, d in ranks_data]
    total_vals = [d["step"]["total"] / 1000 for _, d in ranks_data]

    has_overlap = any(v > 0 for v in overlap_vals) or any(v > 0 for v in comm_raw_vals)
    if not has_overlap:
        L.append("  (无通信重叠数据 — step_trace 中 Overlapped/Communication 均为 0)")
        L.append("")
        return "\n".join(L)

    header = f"  {'Rank':>5} {'Total(ms)':>10} {'Ovl(ms)':>10} {'CommRaw(ms)':>12} {'CommNO(ms)':>11} {'Ovl%':>7}"
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    for (rid, d), total, ovl, comm_raw, comm_novl in zip(
            ranks_data, total_vals, overlap_vals, comm_raw_vals, comm_novl_vals):
        ovl_pct = ovl / total * 100 if total > 0 else 0
        L.append(f"  {rid:>5} {total:>10.1f} {ovl:>10.1f} {comm_raw:>12.1f} {comm_novl:>11.1f} {ovl_pct:>6.1f}%")
    L.append("")

    # Overlap ratio analysis
    ovl_ratios = [ovl / t * 100 if t > 0 else 0 for ovl, t in zip(overlap_vals, total_vals)]
    avg_ovl = statistics.mean(ovl_ratios)
    L.append(f"  平均重叠率 (Overlapped / Total): {avg_ovl:.1f}%")

    if avg_ovl < threshold("multi_rank", "overlap_severe_pct", 5):
        L.append(f"  [DEFINITE] 严重并行瓶颈: 计算-通信重叠率仅 {avg_ovl:.0f}% (<{threshold('multi_rank', 'overlap_severe_pct', 5)}%)")
        L.append("    通信几乎完全暴露在关键路径上 — 优先调整并行策略 (微批次/TP/PP切分) 而非优化单算子")
    elif avg_ovl < threshold("multi_rank", "overlap_low_pct", 10):
        L.append(f"  [SIGNAL] 并行效率低: 计算-通信重叠率 {avg_ovl:.0f}% (<{threshold('multi_rank', 'overlap_low_pct', 10)}%)")
        L.append("    仍有大量通信未重叠 — 可增大 micro-batch / gradient bucketing / 调整 allreduce 触发时机")
    else:
        L.append(f"  重叠充分 (≥{threshold('multi_rank', 'overlap_low_pct', 10)}%)")
    L.append("")

    # False overlap detection (heuristic)
    avg_comm_novl = statistics.mean([v / t * 100 if t > 0 else 0 for v, t in zip(comm_novl_vals, total_vals)])
    if avg_ovl > 5 and avg_comm_novl > threshold("multi_rank", "false_overlap_comm_pct", 20):
        L.append(f"  [SIGNAL] 疑似假性重叠: 重叠 {avg_ovl:.0f}% 但未重叠通信仍占 {avg_comm_novl:.0f}%")
        L.append("    通信 DMA 可能与计算 MatMul 争抢 HBM 带宽 — 重叠未带来真实收益")
        L.append("    交叉验证: 对比开启/关闭通信重叠时同一计算算子的执行耗时是否增加 (>10-20%)")
    L.append("")

    return "\n".join(L)

def _section_communication(ranks_data, top_k):
    """Phase 3.4: Cross-rank communication comparison + R_wait sync straggler analysis."""
    L = []
    L.append("# Phase 3.4: 跨 Rank 通信深度分析")
    L.append("")

    # Check if any rank has communication data
    comm_ranks = [(rid, d["comm"]) for rid, d in ranks_data if d.get("comm")]
    if not comm_ranks:
        L.append("  (无 communication.json 数据 — 非多卡场景或 profiler_level < Level1)")
        L.append("")
        return "\n".join(L)

    # Per-rank total comm breakdown
    elapse_vals = [comm["total"]["elapse"] for _, comm in comm_ranks]
    wait_vals = [comm["total"]["wait"] for _, comm in comm_ranks]
    transit_vals = [comm["total"]["transit"] for _, comm in comm_ranks]

    header = f"  {'Rank':>5} {'Elapse(ms)':>11} {'Transit(ms)':>12} {'Wait(ms)':>10} {'Wait%':>6}"
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    for rid, comm in comm_ranks:
        t = comm["total"]
        wait_pct = t["wait"] / t["elapse"] * 100 if t["elapse"] > 0 else 0
        L.append(f"  {rid:>5} {t['elapse']:>11.1f} {t['transit']:>12.1f} {t['wait']:>10.1f} {wait_pct:>5.1f}%")
    L.append("")

    med_elapse = _median(elapse_vals)
    min_wait, max_wait = min(wait_vals), max(wait_vals)
    L.append(f"  Elapse: min={min(elapse_vals):.1f}  median={med_elapse:.1f}  max={max(elapse_vals):.1f}  CV={_cv(elapse_vals):.3f}")
    L.append(f"  Wait:   min={min_wait:.1f}  median={_median(wait_vals):.1f}  max={max_wait:.1f}  CV={_cv(wait_vals):.3f}")
    L.append("")

    # Wait imbalance — straggler victim detection
    if min_wait > 0 and max_wait / min_wait > threshold("multi_rank", "wait_imbalance_ratio", 2.0):
        victim_rank = comm_ranks[wait_vals.index(max_wait)][0]
        fast_rank = comm_ranks[wait_vals.index(min_wait)][0]
        L.append(f"  [DEFINITE] Wait 时间严重不均衡: rank_{victim_rank} 等待 {max_wait:.1f}ms "
                 f"vs rank_{fast_rank} 等待 {min_wait:.1f}ms ({max_wait/min_wait:.1f}x)")
        L.append(f"    rank_{victim_rank} 是 straggler 的受害者（等待最久的 rank），"
                 f"rank_{fast_rank} 可能是 straggler（等待最少的 rank 先到先走）")
        L.append(f"    交叉验证: 用 step_trace 确认 rank_{victim_rank} 的 Comm(Not Overlapped) 是否最高")
    elif _cv(wait_vals) > threshold("multi_rank", "comm_cv_signal", 0.30):
        L.append(f"  [SIGNAL] Wait 时间跨 rank 有方差 (CV={_cv(wait_vals):.2f}) — "
                 f"部分 rank 等待较多")

    # Transit near-zero (synchronization-bound)
    total_transit = sum(transit_vals)
    total_elapse = sum(elapse_vals)
    if total_elapse > 0 and total_transit / total_elapse < 0.05:
        L.append(f"  [DEFINITE] Transit 仅占 {total_transit/total_elapse*100:.1f}% — "
                 f"通信几乎全是等待 (synchronization-bound)")
        L.append("    所有 rank 的通信时间都是 wait 而非 transit — 通信瓶颈在同步等待，不是带宽")

    L.append("")

    # Per-op-type comparison table
    L.append(f"## 按通信算子类型跨 Rank 对比 (Top {top_k})")
    all_types = set()
    for _, comm in comm_ranks:
        all_types.update(comm["by_type"].keys())

    type_data = {}
    for t in sorted(all_types):
        vals = []
        for rid, comm in comm_ranks:
            agg = comm["by_type"].get(t, {})
            vals.append((rid, agg.get("elapse", 0), agg.get("count", 0),
                        agg.get("wait", 0), agg.get("transit", 0)))
        type_data[t] = vals

    # Sort by median elapse
    sorted_types = sorted(type_data.items(),
                          key=lambda x: -_median([v[1] for v in x[1]]))

    rank_ids = [rid for rid, _ in comm_ranks]
    header = f"  {'Type':<14} {'Metric':>8} " + " ".join(f"r{rid:>1}" for rid in rank_ids) + f" {'CV':>6}"
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    for t, vals in sorted_types[:top_k]:
        elapse_list = [v[1] for v in vals]
        count_list = [v[2] for v in vals]
        wait_list = [v[3] for v in vals]
        L.append(f"  {t:<14} {'Elapse':>8} " + " ".join(f"{v/1:>6.0f}" for v in elapse_list) + f" {_cv(elapse_list):>5.2f}")
        L.append(f"  {'':>14} {'Count':>8} " + " ".join(f"{v:>6d}" for v in count_list) + f" {'':>6}")
        L.append(f"  {'':>14} {'Wait':>8} " + " ".join(f"{v/1:>6.0f}" for v in wait_list) + f" {_cv(wait_list):>5.2f}")
    L.append("")

    # R_wait analysis: R_wait = 1 - (T_avg / T_max) per comm type
    L.append("## 同步等待比例 R_wait 分析")
    L.append(f"  公式: R_wait = 1 - (T_avg / T_max)，T = 各 rank 的通信耗时")
    L.append(f"  R_wait > {threshold('multi_rank', 'r_wait_definite', 0.3):.0%} = 存在同步慢卡问题")
    L.append("")
    r_wait_threshold = threshold("multi_rank", "r_wait_definite", 0.3)
    for t, vals in sorted_types[:top_k]:
        elapse_list = [v[1] for v in vals]
        t_avg = statistics.mean(elapse_list)
        t_max = max(elapse_list)
        if t_max > 0:
            r_wait = 1 - (t_avg / t_max)
        else:
            r_wait = 0
        worst_rid = vals[elapse_list.index(t_max)][0]
        fast_rid = vals[elapse_list.index(min(elapse_list))][0]
        # Skip R_wait signal for types with negligible total time (noise)
        if t_max < 100:
            L.append(f"  {t:<14}: T_max={t_max:.0f}ms (< 100ms, R_wait 不计算 — 耗时可忽略)")
            continue
        L.append(f"  {t:<14}: R_wait = {r_wait:.2f} (T_avg={t_avg:.0f}ms, T_max={t_max:.0f}ms)  "
                 f"慢卡 rank_{worst_rid}, 快卡 rank_{fast_rid}")
        if r_wait > r_wait_threshold:
            L.append(f"    [DEFINITE] {t} 存在严重同步慢卡: R_wait={r_wait:.0%} > {r_wait_threshold:.0%}")
            if r_wait > 0.5:
                L.append("      超过 50% — 通信瓶颈本质是同步等待 (非带宽问题)")
                L.append("      → 如果是同步慢: 查负载均衡 (哪张卡计算慢)")
                L.append("      → 如果是通信慢: 查网络/带宽")
    L.append("")

    return "\n".join(L)


# === Op Statistic 跨 Rank 对比 ===

def _section_op_statistic(ranks_data, top_k):
    """Phase 3.3: Cross-rank op_statistic comparison — find compute ops with high variance."""
    L = []
    L.append("# Phase 3.3: 跨 Rank 算子耗时对比")
    L.append("")

    # Collect all op types
    all_ops = set()
    for _, d in ranks_data:
        all_ops.update(d["ops"].keys())

    # For each op, collect total_us across ranks
    op_data = {}
    grand_total = 0
    for op in all_ops:
        vals = []
        for rid, d in ranks_data:
            vals.append(d["ops"].get(op, {}).get("total_us", 0))
        op_data[op] = vals
        grand_total += max(vals)  # rough grand total for share calculation

    # Sort by max total time across ranks
    sorted_ops = sorted(op_data.items(),
                        key=lambda x: -max(x[1]))

    rank_ids = [rid for rid, _ in ranks_data]
    header = f"  {'Op Type':<30} " + " ".join(f"r{rid:>1}" for rid in rank_ids) + f" {'CV':>6} {'MaxShr':>7}"
    L.append(f"## 按总耗时排序的 Top {min(top_k, len(sorted_ops))} 算子")
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    for op, vals in sorted_ops[:top_k]:
        max_val = max(vals)
        share = max_val / grand_total * 100 if grand_total > 0 else 0
        cv = _cv(vals)
        L.append(f"  {op[:30]:<30} " + " ".join(f"{v/1000:>6.0f}" for v in vals) +
                 f" {cv:>5.2f} {share:>6.1f}%")
    L.append("")

    # High-variance ops
    cv_signal = threshold("multi_rank", "op_cv_signal", 0.20)
    min_share = threshold("multi_rank", "op_cv_min_share", 0.01)
    high_var_ops = []
    for op, vals in sorted_ops:
        max_val = max(vals)
        share = max_val / grand_total if grand_total > 0 else 0
        if share < min_share:
            continue
        cv = _cv(vals)
        if cv > cv_signal:
            high_var_ops.append((op, vals, cv, share))

    if high_var_ops:
        L.append(f"## 高跨 Rank 方差算子 (CV > {cv_signal}, share > {min_share*100:.0f}%)")
        for op, vals, cv, share in high_var_ops[:top_k]:
            max_rid = ranks_data[vals.index(max(vals))][0]
            min_rid = ranks_data[vals.index(min(vals))][0]
            L.append(f"  [SIGNAL] {op[:30]:<30} CV={cv:.2f} | "
                     f"rank_{max_rid} {max(vals)/1000:.0f}ms (最高) vs "
                     f"rank_{min_rid} {min(vals)/1000:.0f}ms (最低)")
        L.append("    交叉验证: 方差大的算子通常是通信相关（如 allgatherAicpuKernel），"
                 "检查该算子在不同 rank 上的调用次数和输入 shape 是否一致")
    else:
        L.append("  (无显著跨 rank 方差的算子)")
    L.append("")

    # TransData / format conversion check (Phase 3.3)
    transdata_kws = threshold("multi_rank", "transdata_keywords", ["TransData", "TransForm", "FormatTransfer"])
    transdata_ops = []
    for op in all_ops:
        if any(kw.lower() in op.lower() for kw in transdata_kws):
            vals = [d["ops"].get(op, {}).get("total_us", 0) for _, d in ranks_data]
            counts = [d["ops"].get(op, {}).get("count", 0) for _, d in ranks_data]
            transdata_ops.append((op, sum(vals) / 1000, max(counts)))
    if transdata_ops:
        L.append("## 私有格式转换 (TransData) 检查")
        for op, total_ms, count in sorted(transdata_ops, key=lambda x: -x[1]):
            L.append(f"  {op[:30]:<30} 总耗时 {total_ms:.1f}ms, 最大调用次数 {count}")
        td_total = sum(t[1] for t in transdata_ops)
        if grand_total > 0 and td_total * 1000 / grand_total > 1:
            L.append(f"  [SIGNAL] TransData 总耗时 {td_total:.1f}ms — 私有格式转换开销，"
                     "检查输入 format 是否为 ND 格式")
        L.append("")

    return "\n".join(L)


# === 通信矩阵跨 Rank 对比 ===

def _section_comm_matrix(ranks_data, top_k):
    """Phase 3.4: Cross-rank comm matrix — bandwidth, small packet, byte alignment."""
    L = []
    L.append("# Phase 3.4: 跨 Rank 逐 Link 带宽 + 小包/对齐分析")
    L.append("")

    matrix_ranks = [(rid, d["matrix"]) for rid, d in ranks_data if d.get("matrix")]
    if not matrix_ranks:
        L.append("  (无 communication_matrix.json 数据)")
        L.append("")
        return "\n".join(L)

    # Per-rank bandwidth summary — each rank's matrix contains links FROM that rank.
    # Links "src-dst" only exist in the source rank's matrix, so cross-rank
    # per-link comparison is meaningless. Instead, aggregate per rank and
    # compare bandwidth distributions across ranks by transport type.
    rank_bw = {}  # rid -> {transport: [(bw, size, link, opname, transit_ms), ...]}
    for rid, entries in matrix_ranks:
        by_transport = defaultdict(list)
        for opname, link, bw, size, transport, transit_ms in entries:
            if size > 0:
                by_transport[transport].append((bw, size, link, opname, transit_ms))
        rank_bw[rid] = by_transport

    # Transport type summary
    transport_set = set()
    for by_t in rank_bw.values():
        transport_set.update(by_t.keys())
    L.append(f"  Transport 类型: {', '.join(sorted(transport_set))}")
    L.append("")

    # Per-rank bandwidth by transport type
    rank_ids = [rid for rid, _ in matrix_ranks]
    for transport in sorted(transport_set):
        L.append(f"## {transport} 带宽跨 Rank 对比")
        per_rank = {}
        for rid in rank_ids:
            entries = rank_bw[rid].get(transport, [])
            # Filter bandwidth from tiny transfers or unreliable transit times
            # (< 1MB size or < 0.01ms transit → physically impossible bandwidth values)
            bws_valid = [e[0] for e in entries if e[0] > 0 and e[1] >= 1 and e[4] > 0.01]
            sizes = [e[1] for e in entries]
            total_size = sum(sizes)
            avg_bw = statistics.mean(bws_valid) if bws_valid else 0
            per_rank[rid] = {"bws": bws_valid, "sizes": sizes, "total_size": total_size, "avg_bw": avg_bw,
                             "min_bw": min(bws_valid) if bws_valid else 0,
                             "max_bw": max(bws_valid) if bws_valid else 0,
                             "n_links": len(entries)}

        header = f"  {'Rank':>5} {'Links':>6} {'TotalSize(MB)':>14} {'AvgBW(GB/s)':>12} {'MinBW':>7} {'MaxBW':>7}"
        L.append(header)
        L.append("  " + "-" * (len(header) - 2))
        for rid in rank_ids:
            pr = per_rank[rid]
            L.append(f"  {rid:>5} {pr['n_links']:>6} {pr['total_size']:>14.1f} "
                     f"{pr['avg_bw']:>12.1f} {pr['min_bw']:>7.1f} {pr['max_bw']:>7.1f}")
        L.append("")

        # Cross-rank avg bandwidth comparison
        avg_bws = [per_rank[rid]["avg_bw"] for rid in rank_ids if per_rank[rid]["avg_bw"] > 0]
        if len(avg_bws) >= 2:
            cv = _cv(avg_bws)
            med = _median(avg_bws)
            L.append(f"  AvgBW: min={min(avg_bws):.1f}  median={med:.1f}  max={max(avg_bws):.1f}  CV={cv:.3f}")
            if cv > threshold("multi_rank", "link_bw_cv_signal", 0.30):
                low_rid = rank_ids[[per_rank[r]["avg_bw"] for r in rank_ids].index(min(avg_bws))]
                L.append(f"  [SIGNAL] {transport} 带宽跨 rank 方差大 (CV={cv:.2f}) — "
                         f"rank_{low_rid} 带宽最低 ({min(avg_bws):.1f} GB/s)")
            L.append("")

    # Per-link detail (top by size, across all ranks)
    all_links = []
    for rid, entries in matrix_ranks:
        for opname, link, bw, size, transport, transit_ms in entries:
            if size > 0:
                all_links.append((rid, link, transport, size, bw, opname, transit_ms))
    all_links.sort(key=lambda x: -x[3])

    L.append(f"## 逐 Link 带宽 Top {min(top_k, len(all_links))} (按传输量排序)")
    header = f"  {'Rank':>5} {'Link':<8} {'Transport':<10} {'Size(MB)':>9} {'BW(GB/s)':>9} {'Op':<25}"
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    for rid, link, transport, size, bw, opname, transit_ms in all_links[:top_k]:
        L.append(f"  {rid:>5} {link:<8} {transport:<10} {size:>9.1f} {bw:>9.1f} {opname[:25]:<25}")
    L.append("")

    # Small packet analysis (from Size Distribution in communication.json)
    small_threshold = threshold("multi_rank", "small_packet_mb", 32)
    small_ratio_threshold = threshold("multi_rank", "small_packet_ratio", 0.30)
    size_dist_ranks = [(rid, d.get("size_dist", {})) for rid, d in ranks_data if d.get("size_dist")]
    if size_dist_ranks:
        L.append(f"## 小包分析 (阈值 < {small_threshold}MB)")
        small_ranks = []
        for rid, sd in size_dist_ranks:
            total_count = 0
            small_count = 0
            for link_type, entries in sd.items():
                for size_mb, count, transit_ms in entries:
                    total_count += count
                    if size_mb < small_threshold:
                        small_count += count
            if total_count > 0:
                ratio = small_count / total_count
                L.append(f"  rank_{rid}: 小包 {small_count}/{total_count} ({ratio*100:.0f}%)")
                if ratio > small_ratio_threshold:
                    small_ranks.append(rid)
        if len(small_ranks) > len(size_dist_ranks) / 2:
            L.append(f"  [SIGNAL] {len(small_ranks)}/{len(size_dist_ranks)} 个 rank 小包占比 >{small_ratio_threshold*100:.0f}% "
                     f"(<{small_threshold}MB) — 协议开销占比大")
            L.append("    建议: 增大 Batch Size / 启用梯度融合 (Gradient Fusion) 合并小通信")
        elif small_ranks:
            L.append(f"  少数 rank ({len(small_ranks)}) 有小包问题: {', '.join(f'rank_{r}' for r in small_ranks)}")
        else:
            L.append("  (无严重小包问题)")
        L.append("")

    # Byte alignment check (from matrix transit sizes — skip LOCAL self-loops)
    # Only check links < 100MB where MB precision is adequate for byte-level alignment.
    # Large links (>100MB) have insufficient decimal precision in MB representation.
    align_bytes = threshold("multi_rank", "alignment_bytes", 512)
    unaligned = []
    for rid, link, transport, size, bw, opname, transit_ms in all_links:
        if transport == "LOCAL":
            continue  # LOCAL is self-loop memory copy, alignment not applicable
        if size > 100:
            continue  # MB precision insufficient for byte-level alignment on large transfers
        size_bytes = size * 1024 * 1024
        if size_bytes > 0 and int(size_bytes) % align_bytes != 0:
            unaligned.append((rid, link, transport, size))
    if unaligned:
        n_checked = len([l for l in all_links if l[2] != "LOCAL" and l[3] <= 100])
        L.append(f"## 字节对齐检查 (要求 {align_bytes}B 对齐)")
        L.append(f"  非 {align_bytes}B 对齐的 link: {len(unaligned)} / {n_checked} (仅检查 <100MB 的 link)")
        if len(unaligned) > n_checked * 0.3:
            L.append(f"  [SIGNAL] {len(unaligned)}/{n_checked} 个 link 非 {align_bytes}B 对齐 — "
                     f"带宽可能受影响 (HCCS 要求 512B 对齐)")
            L.append("    交叉验证: 调整张量 shape 或填充 Padding 使传输大小满足对齐要求")
        else:
            for rid, link, transport, size in unaligned[:3]:
                size_bytes = size * 1024 * 1024
                remainder = int(size_bytes) % align_bytes
                L.append(f"  rank_{rid} link {link} ({transport}): {size:.4f}MB (余 {remainder}B)")
        L.append("")

    # RDMA retransmission check (transit time > threshold = potential retransmission)
    retrans_threshold = threshold("multi_rank", "rdma_retransmission_ms", 4000)
    retrans_links = [(rid, link, transport, transit_ms, size)
                     for rid, link, transport, size, bw, opname, transit_ms in all_links
                     if transit_ms > retrans_threshold and transport != "LOCAL"]
    if retrans_links:
        L.append(f"## RDMA 重传检测 (阈值 > {retrans_threshold}ms)")
        for rid, link, transport, transit_ms, size in retrans_links[:5]:
            L.append(f"  [SIGNAL] rank_{rid} link {link} ({transport}): Transit={transit_ms:.0f}ms, "
                     f"Size={size:.1f}MB — 疑似重传 (交换机拥塞/光模块/物理链路)")
        L.append("    交叉验证: 查 RDMA 链路误码率 (BER)、交换机 PFC 配置、UDP 端口冲突")
        L.append("")

    return "\n".join(L)


# === AlltoAll / MoE 负载不均衡检测 (Phase 4) ===

def _section_alltoall_load(ranks_data, top_k):
    """Phase 4: AlltoAll/MoE load imbalance detection via cross-rank shape/duration comparison."""
    L = []
    L.append("# Phase 4: AlltoAll / MoE 负载不均衡检测")
    L.append("")
    L.append("  传统 AllReduce 分析模型不适用于 AlltoAll 和 Expert Parallelism。")
    L.append("  AlltoAll 耗时久时，先查各卡处理的 Token 数量分布是否均衡。")
    L.append("")

    # Check if any rank has alltoall data
    ato_ranks = [(rid, d["alltoall"]) for rid, d in ranks_data if d.get("alltoall")]
    if not ato_ranks:
        L.append("  (无 alltoall kernel 数据 — 非 MoE/EP 场景)")
        L.append("")
        return "\n".join(L)

    # Per-rank alltoall summary
    header = f"  {'Rank':>5} {'AtoA Kernels':>12} {'Total Dur(ms)':>14} {'Avg Dur(us)':>12}"
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    dur_sums = []
    for rid, shapes in ato_ranks:
        n = len(shapes)
        total_dur = sum(s[2] for s in shapes)
        avg_dur = total_dur / n if n > 0 else 0
        dur_sums.append(total_dur)
        L.append(f"  {rid:>5} {n:>12} {total_dur/1000:>14.1f} {avg_dur:>12.0f}")
    L.append("")

    # Cross-rank duration CV — load imbalance signal
    cv_threshold = threshold("multi_rank", "alltoall_cv_signal", 0.20)
    dur_cv = _cv(dur_sums)
    L.append(f"  AlltoAll 总耗时跨 rank CV: {dur_cv:.3f} (阈值 {cv_threshold})")
    if dur_cv > cv_threshold:
        max_rid = ato_ranks[dur_sums.index(max(dur_sums))][0]
        min_rid = ato_ranks[dur_sums.index(min(dur_sums))][0]
        L.append(f"  [DEFINITE] AlltoAll 负载不均衡: CV={dur_cv:.2f} > {cv_threshold}")
        L.append(f"    rank_{max_rid} alltoall 总耗时 {max(dur_sums)/1000:.1f}ms (最高) vs "
                 f"rank_{min_rid} {min(dur_sums)/1000:.1f}ms (最低)")
        L.append("    某张卡处理的 Token 量远超平均 (Expert Imbalance)，"
                 "导致该卡计算时间长，所有卡在 AlltoAll 同步点等待")
        L.append("    优化方向:")
        L.append("      - 调整 MoE 路由算法 (Router)，加入负载均衡损失 (Load Balancing Loss)")
        L.append("      - 调整专家容量因子 (Expert Capacity Factor)，限制单卡最大 Token 数")
    elif dur_cv > 0.10:
        L.append(f"  [SIGNAL] AlltoAll 跨 rank 有一定方差 (CV={dur_cv:.2f}) — 关注 Token 分布")
    else:
        L.append(f"  AlltoAll 跨 rank 均衡 (CV={dur_cv:.3f} < {cv_threshold})")
    L.append("")

    # Show top alltoall instances per rank for shape comparison (skip if all N/A or empty)
    all_shapes_empty = all(
        not s[1].strip() or s[1].strip().upper() in ("N/A", "NA", "NONE")
        for _, shapes in ato_ranks for s in shapes
    )
    if not all_shapes_empty:
        L.append(f"## AlltoAll Input Shapes 跨 Rank 对比 (Top {top_k} per rank)")
        for rid, shapes in ato_ranks[:4]:
            sorted_shapes = sorted(shapes, key=lambda x: -x[2])
            L.append(f"  rank_{rid}:")
            for name, input_shapes, dur in sorted_shapes[:3]:
                clean_shapes = input_shapes.replace("\n", " ").replace(";", " | ").strip()[:80]
                L.append(f"    {name[:25]:<25} dur={dur/1000:.1f}ms  shapes={clean_shapes}")
        L.append("")
    else:
        L.append("  (kernel_details 中 alltoall Input Shapes 为空 — 无法对比 Token 分布，"
                 "以 AlltoAll 耗时 CV 作为负载不均衡的近似指标)")
        L.append("")

    return "\n".join(L)

def _section_signals(text_sections):
    """Extract all signal lines from all sections and present a summary."""
    L = ["# 多卡分析信号汇总", ""]
    tags = ("[DEFINITE]", "[SIGNAL]", "[FUTURE]")
    definite, signal, future = [], [], []

    for label, text in text_sections:
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("- "):
                stripped = stripped[2:]
            for tag in tags:
                if stripped.startswith(tag):
                    level = tag.strip("[]")
                    entry = f"{stripped}  [{label}]"
                    if level == "DEFINITE":
                        definite.append(entry)
                    elif level == "SIGNAL":
                        signal.append(entry)
                    else:
                        future.append(entry)
                    break

    if definite:
        L.append("### [DEFINITE] — 可直接行动")
        for s in definite:
            L.append(f"  {s}")
        L.append("")
    if signal:
        L.append("### [SIGNAL] — 需交叉验证")
        for s in signal:
            L.append(f"  {s}")
        L.append("")
    if future:
        L.append("### [FUTURE] — 算子级诊断")
        for s in future:
            L.append(f"  {s}")
        L.append("")
    if not definite and not signal and not future:
        L.append("  无信号")
        L.append("")

    return "\n".join(L)


# === 主函数 ===

def parse(profiling_dir: str, top_k: int = 15, include_signals: bool = True) -> str:
    """Run all cross-rank comparisons and return a combined report.

    When include_signals=False, the signal summary section is omitted
    (used by run_analysis.py which does its own signal extraction).
    """
    ranks = _discover_ranks(profiling_dir)
    if len(ranks) < 2:
        return (f"[multi_rank] 仅有 {len(ranks)} 个 rank，不支持跨 rank 分析。\n"
                "需要 ≥2 个 rank_N 子目录。")

    # Load data for all ranks
    ranks_data = []
    for rid, ascend_dir in ranks:
        comm = _load_comm_summary(ascend_dir)
        data = {
            "step": _load_step_totals(ascend_dir),
            "step_per_step": _load_step_per_step(ascend_dir),
            "comm": comm,
            "comm_p2p_count": comm["p2p_count"] if comm else 0,
            "ops": _load_op_stats(ascend_dir),
            "matrix": _load_comm_matrix(ascend_dir),
            "size_dist": _load_comm_size_distribution(ascend_dir),
            "alltoall": _load_alltoall_shapes(ascend_dir),
        }
        ranks_data.append((rid, data))

    # Skip ranks with no step_trace data
    ranks_data = [(rid, d) for rid, d in ranks_data if d["step"] is not None]
    if len(ranks_data) < 2:
        return (f"[multi_rank] 仅有 {len(ranks_data)} 个 rank 有 step_trace 数据，无法对比。")

    L = []
    L.append("=" * 70)
    L.append("=== 多卡 Profiling 跨 Rank 分析报告 ===")
    L.append(f"目录: {profiling_dir}")
    L.append(f"Rank 数: {len(ranks_data)} (rank {ranks_data[0][0]} ~ rank {ranks_data[-1][0]})")
    L.append("=" * 70)
    L.append("")

    sections = []

    # Phase 1: 全局扫描与慢卡定位
    sec1 = _section_step_trace(ranks_data, top_k)
    sections.append(("Phase1 StepTrace", sec1))
    L.append(sec1)

    sec2 = _section_parallel_strategy(ranks_data, top_k)
    sections.append(("Phase1 Strategy", sec2))
    L.append(sec2)

    # Phase 2: 时间线分解与并行效率
    sec3 = _section_overlap(ranks_data, top_k)
    sections.append(("Phase2 Overlap", sec3))
    L.append(sec3)

    # Phase 3.4: 通信深度分析
    sec4 = _section_communication(ranks_data, top_k)
    sections.append(("Phase3.4 Comm", sec4))
    L.append(sec4)

    # Phase 3.3: 计算分析
    sec5 = _section_op_statistic(ranks_data, top_k)
    sections.append(("Phase3.3 OpStat", sec5))
    L.append(sec5)

    # Phase 3.4 cont: 带宽/小包/对齐
    sec6 = _section_comm_matrix(ranks_data, top_k)
    sections.append(("Phase3.4 Matrix", sec6))
    L.append(sec6)

    # Phase 4: AlltoAll / MoE
    sec7 = _section_alltoall_load(ranks_data, top_k)
    sections.append(("Phase5 AlltoAll", sec7))
    L.append(sec7)

    # Signal summary
    if include_signals:
        L.append("=" * 70)
        sig_text = _section_signals(sections)
        L.append(sig_text)

    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("profiling_dir",
                        help="多卡 profiling 输出目录（包含 rank_0 ~ rank_N 子目录）")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.profiling_dir, args.top_k)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
