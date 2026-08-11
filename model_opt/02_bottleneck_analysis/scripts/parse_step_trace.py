#!/usr/bin/env python3
"""解析 step_trace_time.csv — 每 step 的 device 利用率。

展示每 step 的 Computing / Free / Communication(Not Overlapped) 时间占比。
用于判断 workload 是 Host-Bound、Compute-Bound 还是 Comm-Bound。

字段关系（CANN 官方定义）:
    Total = Stage + Bubble = Computing + Communication(Not Overlapped) + Free
    Communication = Overlapped + Communication(Not Overlapped)
    Overlapped 已含在 Computing 内，不可重复计入 Total

用法:
    python parse_step_trace.py <profiling_dir> [--rank N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output, read_csv_all, safe_float


def _row_totals(row):
    """Extract all step_trace fields and compute the authoritative step total."""
    computing = safe_float(row.get("Computing", 0))
    free = safe_float(row.get("Free", 0))
    comm_raw = safe_float(row.get("Communication", 0))
    comm_not_ovl = safe_float(row.get("Communication(Not Overlapped)", 0))
    overlapped = safe_float(row.get("Overlapped", 0))
    stage = safe_float(row.get("Stage", 0))
    bubble = safe_float(row.get("Bubble", 0))
    comm_no_recv = safe_float(row.get("Communication(Not Overlapped and Exclude Receive)", 0))
    preparing = safe_float(row.get("Preparing", 0))
    total = (stage + bubble) if (stage + bubble) > 0 else (computing + comm_not_ovl + free)
    return {
        "computing": computing, "free": free, "comm_raw": comm_raw,
        "comm_not_ovl": comm_not_ovl, "overlapped": overlapped,
        "stage": stage, "bubble": bubble, "comm_no_recv": comm_no_recv,
        "preparing": preparing, "total": total,
    }


def parse(profiling_dir: str, rank=None) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "step_trace_time.csv"
    rows = read_csv_all(csv_path)

    if not rows:
        return f"[step_trace_time] 文件未找到: {csv_path}"

    lines = []
    lines.append("# Step Trace 耗时摘要")
    lines.append(f"数据来源: {csv_path}")
    lines.append(f"步数: {len(rows)}")
    lines.append("")

    step_data = [_row_totals(r) for r in rows]
    agg_keys = ["computing", "free", "comm_not_ovl", "comm_raw", "overlapped",
                "stage", "bubble", "comm_no_recv", "preparing", "total"]
    tot = {k: sum(s[k] for s in step_data) for k in agg_keys}
    T = tot["total"]

    if T > 0:
        util = tot["computing"] / T * 100
        comm_pct = tot["comm_not_ovl"] / T * 100
        free_pct = tot["free"] / T * 100
        lines.append("## 总体")
        lines.append(f"  迭代总时间: {T/1000:.1f} ms")
        lines.append(f"  Device 利用率: {util:.1f}%  (Computing / Total)")
        lines.append(f"  Computing:            {tot['computing']/1000:.1f} ms ({tot['computing']/T*100:.1f}%)")
        lines.append(f"  Free:                 {tot['free']/1000:.1f} ms ({free_pct:.1f}%)")
        if tot["comm_not_ovl"] > 0:
            lines.append(f"  Comm(Not Overlapped): {tot['comm_not_ovl']/1000:.1f} ms ({comm_pct:.1f}%)")
        if tot["overlapped"] > 0:
            lines.append(f"  Overlapped:           {tot['overlapped']/1000:.1f} ms (已含在 Computing 内)")
        if tot["bubble"] > 0:
            lines.append(f"  Bubble:               {tot['bubble']/1000:.1f} ms ({tot['bubble']/T*100:.1f}%)")
        lines.append("")

        if util < threshold("step_trace", "severe_host_bound_util", 20):
            lines.append("  ** 严重 Host-Bound: device 空闲 >80%，瓶颈在 host 侧 **")
        elif util < threshold("step_trace", "moderate_host_bound_util", 50):
            lines.append("  ** 中度 Host-Bound: device 空闲 >50%，host 侧 overhead 显著 **")
        elif comm_pct > threshold("step_trace", "comm_bound_pct", 20):
            lines.append(f"  ** Comm-Bound: 纯通信占比 {comm_pct:.0f}%，优先 comm-compute overlap / comm 减少 **")
        else:
            lines.append(f"  瓶颈在 device 侧（利用率 {util:.0f}%）")
            lines.append(f"  - 需 kernel 级分析区分 compute-bound 和 memory-bound")
        lines.append("")

        optimizable = (tot["free"] + tot["comm_not_ovl"]) / T * 100
        lines.append(f"  理论下限 (= Computing): {tot['computing']/1000:.1f} ms")
        lines.append(f"  可优化空间: {optimizable:.1f}% ((Free + Comm(Not Overlapped)) / Total)")
        if optimizable > threshold("step_trace", "large_optimizable_space", 30):
            lines.append(f"  - 可优化空间大。非 compute 的 overhead（dispatch/alloc/sync/comm）显著；按此上限而非实现难度对候选排序")
        lines.append("")

        # 优化上限（C4, Amdahl 式，基于本 step 的时间拆分）
        lines.append("## 优化上限（按这些对候选排序）")
        lines.append(f"  Compute 下限（不可低于）:           {tot['computing']/1000:.1f} ms ({tot['computing']/T*100:.1f}%)")
        lines.append(f"  Host/dispatch 上限（可回收 Free）:  {tot['free']/1000:.1f} ms ({free_pct:.1f}%)")
        if tot["comm_not_ovl"] > 0:
            lines.append(f"  Communication 上限（可 overlap/消除）: {tot['comm_not_ovl']/1000:.1f} ms ({comm_pct:.1f}%)")
        if tot["overlapped"] > 0:
            lines.append(f"  Overlapped（已重叠，不需额外优化）:   {tot['overlapped']/1000:.1f} ms")
        if tot["free"] >= tot["comm_not_ovl"] and tot["free"] > 0:
            lines.append(f"  - 最大上限 = host/dispatch (Free)。优先处理 host 侧问题。")
        elif tot["comm_not_ovl"] > 0:
            lines.append(f"  - 最大上限 = communication。优先 comm-compute overlap / comm 减少。")
        lines.append("  子类别上限（更细拆分）:")
        lines.append("    sync vs alloc vs dispatch - operator_details 中 Host Time by Category 部分")
        lines.append("    fusible small-op 节省  - kernel_details 中 Fusible sequences 部分")
        lines.append("")

    # 流水线 Bubble 分析
    if tot["bubble"] > 0:
        lines.append("## 流水线 Bubble 分析")
        bubble_ratio = tot["bubble"] / T * 100 if T > 0 else 0
        lines.append(f"  Bubble 总量: {tot['bubble']/1000:.1f} ms ({bubble_ratio:.1f}% of Total)")
        lines.append(f"  Stage 总量:  {tot['stage']/1000:.1f} ms ({tot['stage']/T*100:.1f}% of Total)" if T > 0 else "")
        if bubble_ratio > threshold("step_trace", "bubble_severe_pct", 20):
            lines.append(f"  [SIGNAL] Bubble 占比 {bubble_ratio:.0f}% — 流水线闲置等待严重，检查 PP stage 数和 micro-batch 数")
        elif bubble_ratio > threshold("step_trace", "bubble_moderate_pct", 5):
            lines.append(f"  [SIGNAL] Bubble 占比 {bubble_ratio:.0f}% — 有一定流水线等待，可考虑增加 micro-batch 或 schedule 优化")
        if tot["comm_no_recv"] > 0:
            lines.append(f"  Comm(Not Overlapped, Exclude Receive): {tot['comm_no_recv']/1000:.1f} ms")
            lines.append(f"    → 区分纯通信耗时 vs 流水线等待耗时（receive 等待 = Bubble）")
        lines.append("")

    if len(step_data) > 1:
        has_prep = any(s["preparing"] > 0 for s in step_data)
        has_bubble = any(s["bubble"] > 0 for s in step_data)
        has_comm = any(s["comm_not_ovl"] > 0 for s in step_data)
        lines.append("## 每 step 拆分")
        header_parts = [f"{'Step':>5}", f"{'Total(ms)':>10}", f"{'Comp(ms)':>10}", f"{'Free(ms)':>10}"]
        if has_comm:
            header_parts.append(f"{'Comm(NO)':>10}")
        if has_bubble:
            header_parts.append(f"{'Bubble(ms)':>11}")
        if has_prep:
            header_parts.append(f"{'Prep(ms)':>10}")
        header_parts.append(f"{'Util%':>7}")
        header = "  " + " ".join(header_parts)
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for idx, s in enumerate(step_data):
            u = s["computing"] / s["total"] * 100 if s["total"] > 0 else 0
            parts = [f"{idx:>5}", f"{s['total']/1000:>10.1f}", f"{s['computing']/1000:>10.1f}", f"{s['free']/1000:>10.1f}"]
            if has_comm:
                parts.append(f"{s['comm_not_ovl']/1000:>10.1f}")
            if has_bubble:
                parts.append(f"{s['bubble']/1000:>11.1f}")
            if has_prep:
                parts.append(f"{s['preparing']/1000:>10.1f}")
            parts.append(f"{u:>6.1f}%")
            lines.append("  " + " ".join(parts))
        lines.append("")

        if has_prep:
            avg_prep = tot["preparing"] / len(step_data)
            avg_comp = tot["computing"] / len(step_data)
            lines.append("## Preparing 分析")
            lines.append(f"  平均每 step Preparing: {avg_prep/1000:.1f} ms")
            lines.append(f"  平均每 step Computing: {avg_comp/1000:.1f} ms")
            if avg_prep > avg_comp:
                lines.append(f"  [SIGNAL] Preparing > Computing: 可能是真实 host 瓶颈或 profiler trace-writing overhead")
                lines.append(f"    Preparing 包含 Level1 profiler trace-writing 开销 — 仅凭此项无法确定。")
                lines.append(f"    交叉验证: 用 L0 重新采集 — 若 L0 下 Preparing 仍高则为真实 host 缺口，否则为 profiler 注入。")
            lines.append("")

    # --- 可疑信号 ---
    step_utils = [s["computing"] / s["total"] * 100 if s["total"] > 0 else 0 for s in step_data]
    step_totals_list = [s["total"] for s in step_data]

    lines.append("## 可疑信号")
    lines.append("  [DEFINITE]=可直接行动  [SIGNAL]=异常，根因未定 — 需结合其他 profiling 维度交叉验证")
    suspects_found = False

    if len(step_data) == 1:
        lines.append(f"  [INFO] 单步推理 profile（{len(step_data)} step）— step variance/spread 信号未启用")
        lines.append(f"    使用上方的总体利用率与可优化空间；通过 trace_view / operator_details 交叉验证 host-bound 成因")
        suspects_found = True

    if len(step_utils) > 1:
        util_min = min(step_utils)
        util_max = max(step_utils)
        if util_max - util_min > threshold("step_trace", "step_util_variance", 20):
            lines.append(f"  [SIGNAL] step 利用率 variance: {util_min:.1f}% ~ {util_max:.1f}%")
            lines.append(f"    某些 step 效率明显偏低 — 交叉验证: 检查 warmup/compilation/dynamic shape")
            suspects_found = True

    if len(step_totals_list) > 1:
        total_min = min(step_totals_list)
        total_max = max(step_totals_list)
        if total_max > total_min * threshold("step_trace", "step_duration_spread", 2.0):
            lines.append(f"  [SIGNAL] step duration spread: {total_min/1000:.1f}ms ~ {total_max/1000:.1f}ms ({total_max/total_min:.1f}x)")
            lines.append(f"    波动较大 — 交叉验证: 检查 trace_view 中 compile 事件是否集中在异常 step")
            suspects_found = True

    if not suspects_found:
        lines.append("  无 — 各 step 表现一致")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.profiling_dir, args.rank)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
