#!/usr/bin/env python3
"""解析 op_statistic.csv — 全局算子耗时分布。

该文件始终较小（约 100 行），提供 device 耗时分布的最高层视图，是识别瓶颈
TYPE 的关键。包含 Core Type 维度（AI_CORE / AI_VECTOR_CORE / AI_CPU），
可在第一步即发现 placement 异常。

用法:
    python parse_op_statistic.py <profiling_dir> [--rank N] [--top-k 30]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output, read_csv_all, safe_float, safe_int


# 预期应在 AI_CORE 上执行的 compute 类算子关键词
_COMPUTE_OP_KEYWORDS = ("MatMul", "Conv", "Gemm", "BatchMatMul", "Addmm")


def _core_abbrev(core_type: str) -> str:
    """Core Type 缩写。"""
    if "AI_CORE" == core_type:
        return "AIC"
    if "AI_VECTOR" in core_type:
        return "AIV"
    if "MIX_AIC" in core_type:
        return "MIX"
    if "MIX_AIV" in core_type:
        return "MIXv"
    if "AI_CPU" in core_type:
        return "CPU"
    return core_type[:4] if core_type else "?"


def parse(profiling_dir: str, rank=None, top_k: int = 30) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "op_statistic.csv"
    rows = read_csv_all(csv_path)

    if not rows:
        return f"[op_statistic] 文件未找到: {csv_path}"

    for row in rows:
        row["_total_us"] = safe_float(row.get("Total Time(us)", 0))
        row["_count"] = safe_int(row.get("Count", 0))
        row["_avg_us"] = row["_total_us"] / row["_count"] if row["_count"] > 0 else 0
        row["_min_us"] = safe_float(row.get("Min Time(us)", 0))
        row["_max_us"] = safe_float(row.get("Max Time(us)", 0))
        row["_core_type"] = row.get("Core Type", "").strip()

    rows_sorted = sorted(rows, key=lambda r: r["_total_us"], reverse=True)
    total_us = sum(r["_total_us"] for r in rows_sorted)
    total_count = sum(r["_count"] for r in rows_sorted)

    # Core Type 聚合
    core_stats = defaultdict(lambda: {"count": 0, "dur_us": 0.0, "types": set()})
    for r in rows_sorted:
        ct = r["_core_type"] or "Unknown"
        core_stats[ct]["count"] += r["_count"]
        core_stats[ct]["dur_us"] += r["_total_us"]
        core_stats[ct]["types"].add(r.get("OP Type", "?"))

    # AI_CPU fallback 检测（排除通信算子）
    COMM_KEYWORDS = tuple(threshold("kernel_details", "comm_keywords",
                         ["broadcast", "allgather", "alltoall", "allreduce",
                          "hcom", "send", "recv", "reducescatter"]))
    aicpu_non_comm = []
    for r in rows_sorted:
        if "AI_CPU" in r["_core_type"]:
            op_type = r.get("OP Type", "")
            if not any(kw in op_type.lower() for kw in COMM_KEYWORDS):
                aicpu_non_comm.append(r)

    lines = []
    lines.append(f"# 算子统计摘要")
    lines.append(f"数据来源: {csv_path}")
    lines.append(f"算子类型总数: {len(rows_sorted)}  |  Kernel 总数: {total_count}")
    lines.append("")

    # Core Type 分布
    has_core_type = any(r["_core_type"] for r in rows_sorted)
    if has_core_type:
        lines.append("## Core Type 分布")
        for ct, info in sorted(core_stats.items(), key=lambda x: -x[1]["dur_us"]):
            pct = info["dur_us"] / total_us * 100 if total_us > 0 else 0
            lines.append(f"  {ct}: {info['count']:,} 次, {info['dur_us']/1000:.1f}ms ({pct:.1f}%)  [{len(info['types'])} 种算子类型]")
        if aicpu_non_comm:
            aicpu_total = sum(r["_total_us"] for r in aicpu_non_comm)
            lines.append(f"  [!] AI_CPU 上有 {len(aicpu_non_comm)} 种非通信算子 ({aicpu_total/1000:.1f}ms) — 详见 E 节")
        lines.append("")

    # 主表：加 Core 列和 Max/Avg 列
    header = (f"{'#':>3} {'OP Type':<32} {'Core':<5} {'Count':>6} "
              f"{'Total(ms)':>9} {'Avg(us)':>8} {'Max/Avg':>7} {'Ratio%':>7} {'Cumul%':>7}")
    lines.append(header)
    lines.append("-" * len(header))

    cumul = 0.0
    for idx, row in enumerate(rows_sorted[:top_k]):
        ratio = (row["_total_us"] / total_us * 100) if total_us > 0 else 0
        cumul += ratio
        core_short = _core_abbrev(row["_core_type"])
        max_avg = row["_max_us"] / row["_avg_us"] if row["_avg_us"] > 0 else 0
        max_avg_str = f"{max_avg:.1f}x" if max_avg > 0 else "-"
        lines.append(
            f"{idx+1:>3} {row.get('OP Type', '?'):<32} {core_short:<5} {row['_count']:>6} "
            f"{row['_total_us']/1000:>9.1f} {row['_avg_us']:>8.1f} "
            f"{max_avg_str:>7} {ratio:>6.1f}% {cumul:>6.1f}%"
        )

    lines.append("")

    # === 可疑信号 ===
    lines.append("## 可疑信号")

    # 1. 集中度
    top3_us = sum(r["_total_us"] for r in rows_sorted[:3])
    top3_ratio = top3_us / total_us * 100 if total_us > 0 else 0
    top3_names = [f"{r.get('OP Type', '?')} {r['_total_us']/total_us*100:.0f}%" for r in rows_sorted[:3]] if total_us > 0 else []
    lines.append(f"- [DEFINITE] Top-3 集中度: {top3_ratio:.1f}% ({', '.join(top3_names)})")
    if top3_ratio > threshold("op_statistic", "top3_concentration", 80):
        lines.append(f"  - 瓶颈高度集中 — 优化 top 算子的杠杆效应显著")

    # 2. AI_CPU fallback（非通信算子在 AI_CPU 上执行）
    if aicpu_non_comm:
        aicpu_total = sum(r["_total_us"] for r in aicpu_non_comm)
        aicpu_names = [r.get("OP Type", "") for r in aicpu_non_comm[:5]]
        lines.append(f"- [SIGNAL] AI_CPU 非通信算子 fallback: {len(aicpu_non_comm)} 种, {aicpu_total/1000:.1f}ms, ops: {', '.join(aicpu_names)}")
        lines.append(f"  - 算子未在 AI Core 上执行。替换为 AI Core 实现或调整 shape/format 避免 fallback")

    # 3. Data movement overhead
    move_keywords = tuple(threshold("op_statistic", "move_keywords",
                                    ["Transpose", "Cast", "Copy", "Contiguous", "Reshape", "MemSet", "Format"]))
    move_us = sum(r["_total_us"] for r in rows_sorted
                  if any(kw.lower() in r.get("OP Type", "").lower() for kw in move_keywords))
    move_ratio = move_us / total_us * 100 if total_us > 0 else 0
    if move_ratio > threshold("op_statistic", "data_movement_ratio", 3):
        move_ops = [r.get("OP Type", "") for r in rows_sorted
                    if any(kw.lower() in r.get("OP Type", "").lower() for kw in move_keywords)
                    and r["_total_us"] > 0]
        lines.append(f"- [SIGNAL] Data movement overhead: {move_ratio:.1f}% ({move_us/1000:.1f}ms), ops: {', '.join(move_ops[:5])}")
        lines.append(f"  - Layout/format 转换开销。交叉验证: 在 kernel_details 中确认 mte 占比，在 operator_details 中定位来源")

    # 4. Core placement 异常：compute 类 op 跑在 AI_VECTOR_CORE 上
    misplaced = []
    for r in rows_sorted:
        op_type = r.get("OP Type", "")
        if any(kw in op_type for kw in _COMPUTE_OP_KEYWORDS):
            if "AI_VECTOR" in r["_core_type"] and r["_total_us"] > 0:
                misplaced.append(r)
    if misplaced:
        misplaced_total = sum(r["_total_us"] for r in misplaced)
        misplaced_names = [f"{r.get('OP Type', '?')}({r['_total_us']/1000:.1f}ms)" for r in misplaced[:3]]
        lines.append(f"- [SIGNAL] Compute op 在 AI_VECTOR_CORE: {len(misplaced)} 种, {misplaced_total/1000:.1f}ms, ops: {', '.join(misplaced_names)}")
        lines.append(f"  - 可能是 shape 不满足 AI_CORE 要求或缺少对应实现。kernel_details --filter <op> 确认 shape + block_dim")

    # 5. 执行离散度异常：Max/Avg > 阈值（同类 op 有异常慢的调用）
    variance_threshold = threshold("op_statistic", "variance_max_avg_ratio", 5.0)
    variance_min_ratio = threshold("op_statistic", "variance_min_total_ratio", 0.01)
    high_variance = []
    for r in rows_sorted:
        if r["_avg_us"] > 0 and r["_max_us"] / r["_avg_us"] > variance_threshold:
            if total_us > 0 and r["_total_us"] / total_us > variance_min_ratio:
                high_variance.append(r)
    if high_variance:
        var_items = [f"{r.get('OP Type', '?')} {r['_max_us']/r['_avg_us']:.1f}x" for r in high_variance[:3]]
        lines.append(f"- [SIGNAL] 执行离散度异常 (Max/Avg > {variance_threshold:.0f}x): {', '.join(var_items)}")
        lines.append(f"  - 部分调用远慢于平均（可能是特定 shape 或冷启动）。kernel_details --filter <op> 下钻 shape-performance 相关性")

    # 6. 高频低耗时算子（fragmentation 信号）— 加总量门控
    frag_multiplier = threshold("op_statistic", "frag_count_multiplier", 3)
    frag_max_avg = threshold("op_statistic", "frag_max_avg_us", 10)
    frag_min_ratio = threshold("op_statistic", "frag_min_total_ratio", 0.01)
    if total_count > 0 and total_us > 0:
        avg_count_per_type = total_count / len(rows_sorted)
        fragmented = [(r.get("OP Type", ""), r["_count"], r["_avg_us"], r["_total_us"])
                      for r in rows_sorted
                      if r["_count"] > avg_count_per_type * frag_multiplier
                      and r["_avg_us"] < frag_max_avg
                      and r["_total_us"] / total_us > frag_min_ratio]
        if fragmented:
            frag_items = [f"{name} {count}x/{avg:.1f}us" for name, count, avg, total in fragmented[:4]]
            lines.append(f"- [SIGNAL] 高频低耗时算子 (fragmentation): {', '.join(frag_items)}")
            lines.append(f"  - kernel_details 的 fusible 序列可确认是否时间上连续")

    # 7. 低频高耗时算子（重型单算子）
    heavy_max_count = threshold("op_statistic", "heavy_max_count", 10)
    heavy_min_avg = threshold("op_statistic", "heavy_min_avg_us", 100)
    heavy_min_ratio = threshold("op_statistic", "heavy_min_ratio", 0.01)
    if total_us > 0:
        heavy = [(r.get("OP Type", ""), r["_count"], r["_avg_us"], r["_total_us"], r["_max_us"])
                 for r in rows_sorted
                 if r["_count"] <= heavy_max_count and r["_avg_us"] > heavy_min_avg
                 and r["_total_us"] / total_us > heavy_min_ratio]
        if heavy:
            heavy_items = [f"{name} avg={avg:.0f}us" for name, count, avg, total, max_us in heavy[:3]]
            lines.append(f"- [SIGNAL] 重型单次调用算子: {', '.join(heavy_items)}")
            lines.append(f"  - kernel_details --filter <op> 下钻逐实例硬件拆分 + shape 相关性")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.profiling_dir, args.rank, args.top_k)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
