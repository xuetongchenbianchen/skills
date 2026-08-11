#!/usr/bin/env python3
"""解析 op_statistic.csv — 全局算子耗时分布。

该文件始终较小（约 100 行），提供 device 耗时分布的最高层视图，是识别瓶颈
TYPE 的关键。包含 Core Type 维度（AI_CORE / AI_VECTOR_CORE / AI_CPU），
可在第一步即发现 AI_CPU fallback。

用法:
    python parse_op_statistic.py <profiling_dir> [--rank N] [--top-k 30]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output, read_csv_all, safe_float, safe_int


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

    # AI_CPU fallback 检测（排除通信算子 — 它们按设计运行在 AI_CPU 上）
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
    lines.append(f"算子类型总数: {len(rows_sorted)}")
    lines.append(f"Device 总耗时: {total_us/1000:.1f} ms")
    lines.append(f"Kernel 总数: {total_count}")
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
            lines.append(f"  [!] AI_CPU 上有 {len(aicpu_non_comm)} 种非通信算子 ({aicpu_total/1000:.1f}ms) — 详见可疑信号")
        lines.append("")

    header = f"{'#':>3} {'OP Type':<32} {'Count':>7} {'Total(ms)':>10} {'Avg(us)':>9} {'Ratio%':>7} {'Cumul%':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    cumul = 0.0
    for idx, row in enumerate(rows_sorted[:top_k]):
        ratio = (row["_total_us"] / total_us * 100) if total_us > 0 else 0
        cumul += ratio
        lines.append(
            f"{idx+1:>3} {row.get('OP Type', '?'):<32} "
            f"{row['_count']:>7} {row['_total_us']/1000:>10.1f} "
            f"{row['_avg_us']:>9.1f} {ratio:>6.1f}% {cumul:>6.1f}%"
        )

    lines.append("")

    # 统计分析
    lines.append("## 可疑信号")
    lines.append("  [DEFINITE]=可直接行动  [SIGNAL]=异常，根因未定 — 需结合其他 profiling 维度交叉验证")

    # 1. 集中度: 瓶颈聚焦程度
    top3_us = sum(r["_total_us"] for r in rows_sorted[:3])
    top3_ratio = top3_us / total_us * 100 if total_us > 0 else 0
    lines.append(f"- [DEFINITE] Top-3 集中度: 占 device 总耗时 {top3_ratio:.1f}%")
    if top3_ratio > threshold("op_statistic", "top3_concentration", 80):
        lines.append(f"  - 瓶颈高度集中 — 优化 top 算子的杠杆效应显著")

    # 2. Data movement overhead（非 compute 算子）
    move_keywords = tuple(threshold("op_statistic", "move_keywords",
                                    ["Transpose", "Cast", "Copy", "Contiguous", "Reshape", "MemSet", "Format"]))
    move_us = sum(r["_total_us"] for r in rows_sorted
                  if any(kw.lower() in r.get("OP Type", "").lower() for kw in move_keywords))
    move_ratio = move_us / total_us * 100 if total_us > 0 else 0
    if move_ratio > threshold("op_statistic", "data_movement_ratio", 3):
        move_ops = [r.get("OP Type", "") for r in rows_sorted
                    if any(kw.lower() in r.get("OP Type", "").lower() for kw in move_keywords)
                    and r["_total_us"] > 0]
        lines.append(f"- [SIGNAL] Data movement overhead: {move_ratio:.1f}% ({move_us/1000:.1f}ms)")
        lines.append(f"  ops: {', '.join(move_ops[:8])}")
        lines.append(f"  - Layout/format 转换开销。交叉验证: 在 kernel_details 中确认 mte 占比，在 operator_details 中定位来源")

    # 3. AI_CPU Fallback [DEFINITE]
    if aicpu_non_comm:
        aicpu_total = sum(r["_total_us"] for r in aicpu_non_comm)
        aicpu_ratio = aicpu_total / total_us * 100 if total_us > 0 else 0
        lines.append(f"- [DEFINITE] AI_CPU Fallback（非通信）: {len(aicpu_non_comm)} 种算子, {aicpu_total/1000:.1f}ms ({aicpu_ratio:.1f}%)")
        for r in aicpu_non_comm[:5]:
            lines.append(f"  {r.get('OP Type', '?')}: count={r['_count']}, total={r['_total_us']/1000:.1f}ms, avg={r['_avg_us']:.1f}us")
        lines.append(f"  - 算子无 AI Core 实现，运行在 AI CPU 上。用 kernel_details --filter <op> 看逐 kernel 详情 + shapes")

    # 4. 高频低耗时算子（fragmentation 信号 — 弱信号）
    frag_multiplier = threshold("op_statistic", "frag_count_multiplier", 3)
    frag_max_avg = threshold("op_statistic", "frag_max_avg_us", 10)
    if total_count > 0:
        avg_count_per_type = total_count / len(rows_sorted)
        fragmented = [(r.get("OP Type", ""), r["_count"], r["_avg_us"])
                      for r in rows_sorted
                      if r["_count"] > avg_count_per_type * frag_multiplier and r["_avg_us"] < frag_max_avg]
        if fragmented:
            lines.append(f"- [SIGNAL] 高频低耗时算子（fragmentation 信号 — 弱信号）:")
            for name, count, avg in fragmented[:5]:
                lines.append(f"  {name}: count={count}, avg={avg:.1f}us")
            lines.append(f"  - 仅表明该算子类型调用频繁且单次耗时短，不保证时间上连续。kernel_details 的 fusible 序列可确认是否连续")

    # 5. 低频高耗时算子（重型单算子）
    heavy_max_count = threshold("op_statistic", "heavy_max_count", 10)
    heavy_min_avg = threshold("op_statistic", "heavy_min_avg_us", 100)
    heavy_min_ratio = threshold("op_statistic", "heavy_min_ratio", 0.01)
    if total_us > 0:
        heavy = [(r.get("OP Type", ""), r["_count"], r["_avg_us"], r["_total_us"])
                 for r in rows_sorted
                 if r["_count"] <= heavy_max_count and r["_avg_us"] > heavy_min_avg
                 and r["_total_us"] / total_us > heavy_min_ratio]
        if heavy:
            lines.append(f"- [SIGNAL] 重型单次调用算子:")
            for name, count, avg, total in heavy[:5]:
                lines.append(f"  {name}: count={count}, avg={avg:.0f}us, total={total/1000:.1f}ms")
            lines.append(f"  - 大 shape 或高开销 kernel，可拆分或换更高效实现。kernel_details --filter <op> 下钻逐实例硬件拆分 + shape 相关性")

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
