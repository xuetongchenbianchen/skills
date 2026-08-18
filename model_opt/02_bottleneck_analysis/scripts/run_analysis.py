#!/usr/bin/env python3
"""Profiling 统一分析入口。

按顺序执行所有 parse 脚本，输出一份两段式分析报告：
  1. 总章：全局优化空间 + 所有 signal 汇总（渐进披露，首读只需看此章）
  2. 细节章：各 parse 脚本的完整 statistics 输出（供下钻时参考）

用法:
    python run_analysis.py <L1 profiling 目录> [--l0-dir <L0目录>] [--rank N] [--output 报告路径]

报告结构:
    总章：优化空间与信号汇总
      1. 全局优化空间 (step_trace + L0 交叉验证)
      2. 信号清单 ([DEFINITE] / [SIGNAL] / [FUTURE])
    --- 以下为细节章，供下钻时参考（首读可跳过）---
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
import re
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

_SIGNAL_TAGS = ("[DEFINITE]", "[SIGNAL]", "[FUTURE]")


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


def _extract_signals(text: str, section_label: str) -> list:
    """从 parse 输出文本中提取 signal 行。

    返回 [(level, section_label, signal_text), ...]
    level 为 "DEFINITE" / "SIGNAL" / "FUTURE"
    """
    signals = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- "):
            stripped = stripped[2:]
        for tag in _SIGNAL_TAGS:
            if stripped.startswith(tag):
                level = tag.strip("[]")
                signals.append((level, section_label, stripped))
                break
    return signals


def _build_global_optimization_space(l1_summary, l0_summary) -> str:
    """总章第 1 节：全局优化空间。"""
    lines = ["## 1. 全局优化空间"]

    if l0_summary:
        s = l0_summary
        lines.append(f"  L0 (真实性能): Total {s['total_ms']:.1f}ms = "
                     f"Computing {s['computing_ms']:.1f}ms ({s['utilization']:.0f}%) + "
                     f"Free {s['free_ms']:.1f}ms ({100-s['utilization']:.0f}%)"
                     + (f" + Comm {s['comm_not_ovl_ms']:.1f}ms" if s['comm_not_ovl_ms'] > 0 else ""))

    if l1_summary:
        s = l1_summary
        if l0_summary:
            delta_pp = l0_summary["utilization"] - s["utilization"]
            if delta_pp > 20:
                lines.append(f"  L1 (含 profiler): Total {s['total_ms']:.1f}ms — "
                             f"profiler 伪影严重 (Utilization 差 {delta_pp:.0f}pp)，瓶颈判定以 L0 为准")
            else:
                lines.append(f"  L1: Total {s['total_ms']:.1f}ms, Utilization {s['utilization']:.1f}% "
                             f"(与 L0 差 {delta_pp:.0f}pp，profiler 影响可接受)")
        else:
            lines.append(f"  L1: Total {s['total_ms']:.1f}ms = "
                         f"Computing {s['computing_ms']:.1f}ms ({s['utilization']:.0f}%) + "
                         f"Free {s['free_ms']:.1f}ms ({100-s['utilization']:.0f}%)"
                         + (f" + Comm {s['comm_not_ovl_ms']:.1f}ms" if s['comm_not_ovl_ms'] > 0 else ""))
            lines.append(f"  (无 L0 交叉验证，L1 数字可能含 profiler 伪影)")

    if not l1_summary and not l0_summary:
        lines.append("  (无法提取 step_trace 数据)")

    lines.append("")
    return "\n".join(lines)


def _build_signal_summary(all_signals: list) -> str:
    """总章第 2 节：信号清单，按 DEFINITE / SIGNAL / FUTURE 分组。"""
    lines = ["## 2. 信号清单", ""]

    definite = [(sec, txt) for lvl, sec, txt in all_signals if lvl == "DEFINITE"]
    signal = [(sec, txt) for lvl, sec, txt in all_signals if lvl == "SIGNAL"]
    future = [(sec, txt) for lvl, sec, txt in all_signals if lvl == "FUTURE"]

    if definite:
        lines.append("### [DEFINITE] — 可直接行动")
        for sec, txt in definite:
            lines.append(f"  {txt}  [{sec}]")
        lines.append("")

    if signal:
        lines.append("### [SIGNAL] — 需交叉验证")
        for sec, txt in signal:
            lines.append(f"  {txt}  [{sec}]")
        lines.append("")

    if future:
        lines.append("### [FUTURE] — 算子级诊断（暂不关注，等功能完善后启用）")
        for sec, txt in future:
            lines.append(f"  {txt}  [{sec}]")
        lines.append("")

    if not definite and not signal and not future:
        lines.append("  无信号")
        lines.append("")

    return "\n".join(lines)


def run_section(title: str, func, *args, **kwargs):
    """执行单个 parse 函数，返回 (章节文本, signal列表)。"""
    lines = [SUB_DIVIDER, f"--- {title} ---", ""]
    try:
        result = func(*args, **kwargs)
        lines.append(result.rstrip())
    except Exception as e:
        result = f"[ERROR] {func.__module__}.{func.__name__} failed: {e}"
        lines.append(result)
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

    # 提取 step_trace 关键数字（用于总章全局优化空间）
    l1_summary = _extract_step_trace_summary(l1_dir, rank)
    l0_summary = _extract_step_trace_summary(args.l0_dir, rank) if args.l0_dir else None

    # --- 执行各 parse 脚本，收集输出和 signal ---
    all_signals = []
    detail_sections = []
    section_labels = {}

    # A. 全局视角
    section_a_text = "[L1] " + parse_step_trace.parse(l1_dir, rank).rstrip()
    if args.l0_dir:
        section_a_text += "\n\n[L0] " + parse_step_trace.parse(args.l0_dir, rank).rstrip()
        section_a_text += "\n\n## L0/L1 交叉验证\n"
        section_a_text += "  对比 L0（无 profiler 开销的真实性能）和 L1（含 profiler barrier 的详细数据）：\n"
        section_a_text += "  若 L1 Utilization 显著低于 L0（差 >20pp），瓶颈类型判定以 L0 为准；\n"
        section_a_text += "  L1 的算子级数据（op_statistic、kernel_details 等）仍然有效。"
    all_signals.extend(_extract_signals(section_a_text, "A"))
    detail_sections.append(f"{SUB_DIVIDER}\n--- A. 全局视角 ---\n\n{section_a_text}\n")

    # B. 设备侧：算子分布
    sec_b = run_section("B. 设备侧：算子分布", parse_op_statistic.parse, l1_dir, rank)
    all_signals.extend(_extract_signals(sec_b, "B"))
    detail_sections.append(sec_b)

    # C. 设备侧：Kernel 级详情
    sec_c = run_section("C. 设备侧：Kernel 级详情", parse_kernel_details.parse, l1_dir, rank)
    all_signals.extend(_extract_signals(sec_c, "C"))
    detail_sections.append(sec_c)

    # D. Host-Device 交互
    ascend_dir = find_ascend_profiler_output(l1_dir, rank)
    trace_view_path = ascend_dir / "trace_view.json"
    if trace_view_path.exists():
        sec_d = run_section("D. Host-Device 交互", parse_trace_view.parse, trace_view_path, 15, 50.0)
    else:
        sec_d = f"{SUB_DIVIDER}\n--- D. Host-Device 交互 ---\n\n[trace_view] File not found: {trace_view_path}\n"
    all_signals.extend(_extract_signals(sec_d, "D"))
    detail_sections.append(sec_d)

    # E. 源码定位
    sec_e = run_section("E. 源码定位", parse_operator_details.parse_overview, l1_dir, rank)
    all_signals.extend(_extract_signals(sec_e, "E"))
    detail_sections.append(sec_e)

    # F. 内存
    sec_f_lines = [SUB_DIVIDER, "--- F. 内存 ---", ""]
    try:
        sec_f_lines.append(parse_memory_record.parse(l1_dir, rank).rstrip())
    except Exception as e:
        sec_f_lines.append(f"[ERROR] parse_memory_record failed: {e}")
    sec_f_lines.append("")
    try:
        sec_f_lines.append(parse_operator_memory.parse(l1_dir, rank).rstrip())
    except Exception as e:
        sec_f_lines.append(f"[ERROR] parse_operator_memory failed: {e}")
    sec_f_lines.append("")
    sec_f = "\n".join(sec_f_lines)
    all_signals.extend(_extract_signals(sec_f, "F"))
    detail_sections.append(sec_f)

    # G. CANN 运行时
    sec_g = run_section("G. CANN 运行时", parse_api_statistic.parse, l1_dir, rank)
    all_signals.extend(_extract_signals(sec_g, "G"))
    detail_sections.append(sec_g)

    # H. 通信（仅多卡）
    comm_path = ascend_dir / "communication.json"
    matrix_path = ascend_dir / "communication_matrix.json"
    if comm_path.exists():
        sec_h = run_section("H. 通信（多卡）", parse_communication.parse, comm_path, matrix_path, 15)
        all_signals.extend(_extract_signals(sec_h, "H"))
        detail_sections.append(sec_h)

    # --- 组装报告 ---
    report_parts = []
    report_parts.append(DIVIDER)
    report_parts.append("=== Phase 2 Profiling 分析报告 ===")
    report_parts.append(f"L1 目录: {l1_dir}")
    if args.l0_dir:
        report_parts.append(f"L0 参考目录: {args.l0_dir}")
    else:
        report_parts.append("L0 参考目录: 未提供（L1 结论未经交叉验证，须谨慎）")
    if rank is not None:
        report_parts.append(f"Rank: {rank}")
    report_parts.append("")

    # 总章
    report_parts.append(DIVIDER)
    report_parts.append("=== 总章：优化空间与信号汇总 ===")
    report_parts.append(DIVIDER)
    report_parts.append("")
    report_parts.append(_build_global_optimization_space(l1_summary, l0_summary))
    report_parts.append(_build_signal_summary(all_signals))

    # 细节章
    report_parts.append(DIVIDER)
    report_parts.append("=== 以下为细节章，供下钻时参考（首读可跳过）===")
    report_parts.append(DIVIDER)
    report_parts.append("")
    for sec in detail_sections:
        report_parts.append(sec)

    report_parts.append(DIVIDER)

    report = "\n".join(report_parts)

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
