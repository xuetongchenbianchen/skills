#!/usr/bin/env python3
"""Profiling 统一分析入口。

按顺序执行所有 parse 脚本，输出一份完整的分析报告。
报告开头生成 Executive Summary（瓶颈判定 + 关键数字 + 信号汇总）。

用法:
    python run_analysis.py <L1 profiling 目录> [--l0-dir <L0目录>] [--rank N] [--output 报告路径]

报告结构:
    Executive Summary（自动生成）
    A. 全局视角 (step_trace, + L0 交叉验证)
    B. 设备侧：算子分布 (op_statistic)
    C. 设备侧：Kernel 级详情 (kernel_details)
    D. Host-Device 交互 (trace_view)
    E. 源码定位 (operator_details)
    F. 内存 (memory_record, operator_memory)
    G. CANN 运行时 (api_statistic)
    H. 通信 (communication, 仅多卡时存在)

默认行为: 报告自动保存到 L1 profiling 目录下的 analysis_report.txt。
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import find_ascend_profiler_output, read_csv_all, safe_float

import parse_step_trace
import parse_op_statistic
import parse_kernel_details
import parse_trace_view
import parse_operator_details
import parse_memory_record
import parse_operator_memory
import parse_api_statistic
import parse_communication


DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70


def _extract_step_trace_summary(profiling_dir: str, rank=None):
    """直接从 CSV 提取 step_trace 关键数字（不依赖文本解析）。"""
    try:
        ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
        csv_path = ascend_dir / "step_trace_time.csv"
        rows = read_csv_all(csv_path)
        if not rows:
            return None
        step_data = [parse_step_trace._row_totals(r) for r in rows]
        agg_keys = ["computing", "free", "comm_not_ovl", "total"]
        tot = {k: sum(s[k] for s in step_data) for k in agg_keys}
        T = tot["total"]
        if T <= 0:
            return None
        return {
            "total_ms": T / 1000,
            "computing_ms": tot["computing"] / 1000,
            "free_ms": tot["free"] / 1000,
            "comm_not_ovl_ms": tot["comm_not_ovl"] / 1000,
            "utilization": tot["computing"] / T * 100,
            "optimizable": (tot["free"] + tot["comm_not_ovl"]) / T * 100,
        }
    except Exception:
        return None


def _build_executive_summary(l1_summary, l0_summary) -> str:
    """生成数值概览 — 仅包含各节独立做不到的跨脚本信息。"""
    lines = [SUB_DIVIDER, "--- 数值概览 ---", ""]

    # 时间分解
    lines.append("## 时间分解")
    if l0_summary:
        lines.append(f"  L0 (真实性能): Total {l0_summary['total_ms']:.1f}ms = "
                     f"Computing {l0_summary['computing_ms']:.1f}ms ({l0_summary['utilization']:.0f}%) + "
                     f"Free {l0_summary['free_ms']:.1f}ms ({100-l0_summary['utilization']:.0f}%)"
                     + (f" + Comm {l0_summary['comm_not_ovl_ms']:.1f}ms" if l0_summary['comm_not_ovl_ms'] > 0 else ""))
    if l1_summary:
        if l0_summary:
            delta_pp = l0_summary["utilization"] - l1_summary["utilization"]
            if delta_pp > 20:
                lines.append(f"  L1 (含 profiler): Total {l1_summary['total_ms']:.1f}ms — "
                             f"profiler 伪影严重 (Utilization 差 {delta_pp:.0f}pp)，瓶颈判定以 L0 为准")
            else:
                lines.append(f"  L1: Total {l1_summary['total_ms']:.1f}ms, Utilization {l1_summary['utilization']:.1f}% "
                             f"(与 L0 差 {delta_pp:.0f}pp，profiler 影响可接受)")
        else:
            lines.append(f"  L1: Total {l1_summary['total_ms']:.1f}ms = "
                         f"Computing {l1_summary['computing_ms']:.1f}ms ({l1_summary['utilization']:.0f}%) + "
                         f"Free {l1_summary['free_ms']:.1f}ms ({100-l1_summary['utilization']:.0f}%)"
                         + (f" + Comm {l1_summary['comm_not_ovl_ms']:.1f}ms" if l1_summary['comm_not_ovl_ms'] > 0 else ""))
            lines.append(f"  (无 L0 交叉验证，L1 数字可能含 profiler 伪影)")
    lines.append("")

    # 信号图例（全局声明一次）
    lines.append("## 信号标记说明")
    lines.append("  [DEFINITE] = 可直接行动  |  [SIGNAL] = 异常，需结合其他维度交叉验证")
    lines.append("")

    return "\n".join(lines)


def run_section(title: str, func, *args, **kwargs) -> str:
    """执行单个 parse 函数，用章节标题包裹输出。"""
    lines = [SUB_DIVIDER, f"--- {title} ---", ""]
    try:
        result = func(*args, **kwargs)
        lines.append(result.rstrip())
    except Exception as e:
        lines.append(f"[ERROR] {func.__module__}.{func.__name__} failed: {e}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("profiling_dir", help="L1 profiling 输出目录")
    parser.add_argument("--l0-dir", default=None,
                        help="L0 profiling 目录（用于 L0/L1 交叉验证）")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--output", "-o", default=None,
                        help="报告输出路径（默认: 保存到 L1 目录下 analysis_report.txt）")
    args = parser.parse_args()

    l1_dir = args.profiling_dir
    rank = args.rank

    # 提取 step_trace 关键数字（用于 Executive Summary）
    l1_summary = _extract_step_trace_summary(l1_dir, rank)
    l0_summary = _extract_step_trace_summary(args.l0_dir, rank) if args.l0_dir else None

    sections = []
    sections.append(DIVIDER)
    sections.append("=== Phase 2 Profiling 分析报告 ===")
    sections.append(f"L1 目录: {l1_dir}")
    if args.l0_dir:
        sections.append(f"L0 参考目录: {args.l0_dir}")
    else:
        sections.append("L0 参考目录: 未提供（L1 结论未经交叉验证，须谨慎）")
    if rank is not None:
        sections.append(f"Rank: {rank}")
    sections.append("")

    # --- 各节内容生成 ---

    # --- A. 全局视角 ---
    section_a = [SUB_DIVIDER, "--- A. 全局视角 ---", ""]

    section_a.append("[L1] " + parse_step_trace.parse(l1_dir, rank).rstrip())

    if args.l0_dir:
        section_a.append("")
        section_a.append("[L0] " + parse_step_trace.parse(args.l0_dir, rank).rstrip())
        section_a.append("")
        # 结构化 L0/L1 交叉验证对比
        section_a.append("## L0/L1 交叉验证")
        section_a.append("  对比 L0（无 profiler 开销的真实性能）和 L1（含 profiler barrier 的详细数据）：")
        section_a.append("  若 L1 Utilization 显著低于 L0（差 >20pp），瓶颈类型判定以 L0 为准；")
        section_a.append("  L1 的算子级数据（op_statistic、kernel_details 等）仍然有效。")

    section_a.append("")
    sections.append("\n".join(section_a))

    # --- B. 设备侧：算子分布 ---
    sections.append(run_section(
        "B. 设备侧：算子分布",
        parse_op_statistic.parse, l1_dir, rank,
    ))

    # --- C. 设备侧：Kernel 级详情 ---
    sections.append(run_section(
        "C. 设备侧：Kernel 级详情",
        parse_kernel_details.parse, l1_dir, rank,
    ))

    # --- D. Host-Device 交互 ---
    ascend_dir = find_ascend_profiler_output(l1_dir, rank)
    trace_view_path = ascend_dir / "trace_view.json"
    if trace_view_path.exists():
        sections.append(run_section(
            "D. Host-Device 交互",
            parse_trace_view.parse, trace_view_path, 15, 50.0,
        ))
    else:
        sections.append(f"{SUB_DIVIDER}\n--- D. Host-Device 交互 ---\n\n[trace_view] File not found: {trace_view_path}\n")

    # --- E. 源码定位 ---
    sections.append(run_section(
        "E. 源码定位",
        parse_operator_details.parse_overview, l1_dir, rank,
    ))

    # --- F. 内存 ---
    section_f = [SUB_DIVIDER, "--- F. 内存 ---", ""]
    try:
        section_f.append(parse_memory_record.parse(l1_dir, rank).rstrip())
    except Exception as e:
        section_f.append(f"[ERROR] parse_memory_record failed: {e}")
    section_f.append("")
    try:
        section_f.append(parse_operator_memory.parse(l1_dir, rank).rstrip())
    except Exception as e:
        section_f.append(f"[ERROR] parse_operator_memory failed: {e}")
    section_f.append("")
    sections.append("\n".join(section_f))

    # --- G. CANN 运行时 ---
    sections.append(run_section(
        "G. CANN 运行时",
        parse_api_statistic.parse, l1_dir, rank,
    ))

    # --- H. 通信（仅多卡）---
    comm_path = ascend_dir / "communication.json"
    matrix_path = ascend_dir / "communication_matrix.json"
    if comm_path.exists():
        sections.append(run_section(
            "H. 通信（多卡）",
            parse_communication.parse, comm_path, matrix_path, 15,
        ))

    sections.append(DIVIDER)

    # 生成数值概览
    exec_summary = _build_executive_summary(l1_summary, l0_summary)

    # 最终报告: 数值概览在报告头之后
    report = "\n".join(sections[:5]) + "\n" + exec_summary + "\n" + "\n".join(sections[5:])

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(l1_dir) / "analysis_report.txt"

    output_path.write_text(report, encoding="utf-8")
    print(f"分析报告已保存: {output_path}")
    print(report)


if __name__ == "__main__":
    main()
