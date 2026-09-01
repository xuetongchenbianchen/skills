#!/usr/bin/env python3
"""通信分析统一入口 — M1 溯源 + M2 Matrix + M3 跨 Rank 对比 + M4 带宽自基准。

模式:
  单 rank（默认）:
    M1 源码溯源 — hcom 算子 → Hccl host 调用 → Call Stack → 源码位置
         （trace 主路径 [ALIGNED] / CSV 序号对齐 [RISKY] / 聚合栈 [AGGREGATED]）
    M1 瞬态污染防护 — Top-K 疑点头部位置标注 + wait 头部集中度
    M2 Matrix 深度分析 — rank-pair 倾斜 / transport 合理性 / 慢链路（size 条件化自基准）/
         链路矩阵渲染 / 组↔算子回链 / 字节对齐 / RDMA 重传 / 延迟主导小包（L̂ 数据内估计）
  跨 Rank（多 rank 目录时自动启用）:
    M3 跨 Rank 对比 — per-op wait join 慢卡（straggler）定位 / per-rank 汇总（R_wait 仅展示）/
         重叠对比（归因二分：串行型 vs 依赖性串行）/ alltoall 负载不均判读
  --trace-source <op> — 单算子深挖溯源（输出完整调用栈）

数据源: communication.json + communication_matrix.json（多卡、profiler_level >= Level1），
        step_trace_time.csv 重叠三列，operator_details.csv / trace_view.json（溯源）。

用法:
    python parse_communication.py <profiling_dir> [--rank N] [--top-k 15]
        [--trace-source <opname>] [--no-trace] [--output out.txt]
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    find_ascend_profiler_output, read_csv_all, stream_csv, safe_float, threshold,
)

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def extract_op_type(opname: str) -> str:
    """hcom_allGather__612_0_1@xxx -> allGather"""
    base = opname.split("@")[0].split("__")[0]
    return base.replace("hcom_", "")


def extract_domain(opname: str) -> str:
    """hcom_allGather__612_0_1@5862276093215481612 -> 5862276093215481612"""
    if "@" in opname:
        return opname.split("@", 1)[1]
    return ""


def _pct(part, whole):
    return part / whole * 100 if whole > 0 else 0.0


def _quantile(sorted_vals, q):
    """Quantile of pre-sorted list. q in [0,1]."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    if ms >= 1:
        return f"{ms:.1f}ms"
    return f"{ms * 1000:.0f}us"


def _condense_stack(stack: str, max_frames: int = 3) -> str:
    """压缩调用栈：过滤框架帧，保留项目帧（栈为自底向上，取最靠近通信调用的前 N 帧）。"""
    if not stack:
        return "(no stack)"
    frames = [f.strip() for f in stack.replace("\r\n", "\n").split("\n") if f.strip()]
    markers = ("site-packages", "dist-packages", "/lib/python", "torch/distributed",
               "torch/nn", "torch/_ops", "torch_npu", "autograd")
    project = [f for f in frames if not any(m in f for m in markers)]
    chosen = project if project else frames
    return " ← ".join(chosen[:max_frames])


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _discover_ranks(profiling_dir: str) -> list:
    """Return sorted [(rank_id, ascend_dir)] for all rank_N dirs."""
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


def _load_comm_ops(ascend_dir: Path) -> dict:
    """解析 communication.json。

    Returns:
        {"ops": [ {name, type, domain, step, start_us, elapse, transit, wait, sync, idle} ],
         "total": {elapse, transit, wait, sync, idle},
         "p2p": [ {name, elapse, transit, wait} ],
         "p2p_count": int}
    ops 按算子名聚合（跨 step 求和，start 取最早）；total 来自 Total Op Info（防重复计数）。
    """
    comm_path = ascend_dir / "communication.json"
    if not comm_path.exists():
        return {}
    data = json.loads(comm_path.read_text(encoding="utf-8"))

    ops_by_name = {}
    total = {"elapse": 0.0, "transit": 0.0, "wait": 0.0, "sync": 0.0, "idle": 0.0}
    p2p = []
    for step, sd in data.items():
        collective = sd.get("collective", {})
        total_info = collective.get("Total Op Info", {}).get("Communication Time Info", {})
        total["elapse"] += safe_float(total_info.get("Elapse Time(ms)", 0))
        total["transit"] += safe_float(total_info.get("Transit Time(ms)", 0))
        total["wait"] += safe_float(total_info.get("Wait Time(ms)", 0))
        total["sync"] += safe_float(total_info.get("Synchronization Time(ms)", 0))
        total["idle"] += safe_float(total_info.get("Idle Time(ms)", 0))
        for opname, info in collective.items():
            if opname == "Total Op Info":
                continue
            ti = info.get("Communication Time Info", {})
            rec = ops_by_name.get(opname)
            if rec is None:
                rec = {
                    "name": opname,
                    "type": extract_op_type(opname),
                    "domain": extract_domain(opname),
                    "start_us": safe_float(ti.get("Start Timestamp(us)", 0)),
                    "elapse": 0.0, "transit": 0.0, "wait": 0.0, "sync": 0.0, "idle": 0.0,
                    "steps": 0,
                }
                ops_by_name[opname] = rec
            rec["elapse"] += safe_float(ti.get("Elapse Time(ms)", 0))
            rec["transit"] += safe_float(ti.get("Transit Time(ms)", 0))
            rec["wait"] += safe_float(ti.get("Wait Time(ms)", 0))
            rec["sync"] += safe_float(ti.get("Synchronization Time(ms)", 0))
            rec["idle"] += safe_float(ti.get("Idle Time(ms)", 0))
            rec["steps"] += 1
        for opname, info in sd.get("p2p", {}).items():
            ti = info.get("Communication Time Info", {}) if isinstance(info, dict) else {}
            p2p.append({
                "name": opname,
                "elapse": safe_float(ti.get("Elapse Time(ms)", 0)),
                "transit": safe_float(ti.get("Transit Time(ms)", 0)),
                "wait": safe_float(ti.get("Wait Time(ms)", 0)),
            })
    return {
        "ops": sorted(ops_by_name.values(), key=lambda o: o["start_us"]),
        "total": total,
        "p2p": p2p,
        "p2p_count": len(p2p),
    }


def _load_matrix(ascend_dir: Path) -> list:
    """解析 communication_matrix.json → [{group, link, transport, size_mb, transit_ms, bw, op_name}]"""
    mat_path = ascend_dir / "communication_matrix.json"
    if not mat_path.exists():
        return []
    data = json.loads(mat_path.read_text(encoding="utf-8"))
    entries = []
    for step, sd in data.items():
        for group, links in sd.get("collective", {}).items():
            for link, lv in links.items():
                entries.append({
                    "group": group,
                    "link": link,
                    "transport": lv.get("Transport Type", "?"),
                    "size_mb": safe_float(lv.get("Transit Size(MB)", 0)),
                    "transit_ms": safe_float(lv.get("Transit Time(ms)", 0)),
                    "bw": safe_float(lv.get("Bandwidth(GB/s)", 0)),
                    "op_name": lv.get("Op Name", ""),
                })
    return entries


def _load_step_overlap(ascend_dir: Path) -> dict:
    """step_trace_time.csv 的重叠三列（us → ms 聚合，total 口径与 parse_step_trace 一致）。"""
    csv_path = ascend_dir / "step_trace_time.csv"
    rows = read_csv_all(csv_path)
    if not rows:
        return {}
    totals = {"computing": 0.0, "free": 0.0, "comm_raw": 0.0, "comm_not_ovl": 0.0,
              "overlapped": 0.0, "stage": 0.0, "bubble": 0.0, "total": 0.0}
    for r in rows:
        computing = safe_float(r.get("Computing", 0))
        free = safe_float(r.get("Free", 0))
        comm_raw = safe_float(r.get("Communication", 0))
        comm_not_ovl = safe_float(r.get("Communication(Not Overlapped)", 0))
        overlapped = safe_float(r.get("Overlapped", 0))
        stage = safe_float(r.get("Stage", 0))
        bubble = safe_float(r.get("Bubble", 0))
        total = (stage + bubble) if (stage + bubble) > 0 else (computing + comm_not_ovl + free)
        totals["computing"] += computing
        totals["free"] += free
        totals["comm_raw"] += comm_raw
        totals["comm_not_ovl"] += comm_not_ovl
        totals["overlapped"] += overlapped
        totals["stage"] += stage
        totals["bubble"] += bubble
        totals["total"] += total
    return {k: v / 1000 for k, v in totals.items()}

# ---------------------------------------------------------------------------
# M1 溯源：host 侧 Hccl 调用加载与对齐
# ---------------------------------------------------------------------------

_TRACE_EVENT_RE = re.compile(r'\{[^{}]*?"name": "(Hccl[^"]*)"[^{}]*?\{[^{}]*\}[^{}]*?\}')


def _load_host_ops_csv(ascend_dir: Path) -> dict:
    """流式扫描 operator_details.csv，按 Name 收集 Hccl 行。

    Returns {host_name: [{"call_stack": str, "row_idx": int}, ...]}（文件序 = 时间序）。
    """
    csv_path = ascend_dir / "operator_details.csv"
    if not csv_path.exists():
        return {}
    result = defaultdict(list)
    for idx, row in enumerate(stream_csv(csv_path)):
        name = row.get("Name", "")
        if name.startswith("Hccl"):
            result[name].append({
                "call_stack": row.get("Call Stack", ""),
                "row_idx": idx,
            })
    return dict(result)


def _load_host_ops_trace(ascend_dir: Path) -> dict:
    """分块扫描 trace_view.json，抽取 Hccl* cpu_op 事件（ts + Call stack）。

    事件为单层嵌套 JSON（args 内为平铺字段），正则匹配后逐个 json.loads 解析。
    Returns {host_name: [{"ts": float_us, "call_stack": str}, ...] 按 ts 排序}。
    """
    trace_path = ascend_dir / "trace_view.json"
    if not trace_path.exists():
        return {}
    events = defaultdict(list)
    overlap = 65536
    tail = ""
    with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            buf = tail + chunk
            last_brace = buf.rfind("}")
            scan_region = buf if last_brace < 0 else buf[:last_brace + 1]
            for m in _TRACE_EVENT_RE.finditer(scan_region):
                try:
                    ev = json.loads(m.group(0))
                except (json.JSONDecodeError, ValueError):
                    continue
                name = ev.get("name", "")
                if not name.startswith("Hccl"):
                    continue
                args = ev.get("args", {}) or {}
                events[name].append({
                    "ts": safe_float(ev.get("ts", 0)),
                    "call_stack": args.get("Call stack", "") or args.get("Call Stack", ""),
                })
            tail = buf[-overlap:] if last_brace < 0 else buf[max(0, len(buf) - overlap):]
    for name in events:
        # 去除分块边界重复匹配（同一事件在 overlap 区被扫描两次）与完全同 ts 的重复事件
        rows = sorted(events[name], key=lambda e: e["ts"])
        deduped = []
        for r in rows:
            if deduped and r["ts"] == deduped[-1]["ts"]:
                continue
            deduped.append(r)
        events[name] = deduped
    return dict(events)


def _device_overlap_ratio(ops: list) -> float:
    """同 (type, domain) 内通信算子时间区间的相邻重叠率（序号对齐风险检测）。"""
    if len(ops) < 2:
        return 0.0
    srt = sorted(ops, key=lambda o: o["start_us"])
    overlapping = 0
    for prev, cur in zip(srt, srt[1:]):
        prev_end = prev["start_us"] + prev["elapse"] * 1000
        if cur["start_us"] < prev_end:
            overlapping += 1
    return overlapping / (len(srt) - 1)


def _align_type(device_ops: list, csv_host_ops: dict, trace_host_ops: dict, no_trace: bool):
    """对某一通信类型做溯源对齐。

    Returns (method, host_rows, note)：
      method ∈ {"trace", "csv", "aggregate"}；host_rows 与 device_ops（按 start 排序）按序对应。
    """
    if not device_ops:
        return "aggregate", [], "无通信算子"
    dev_sorted = sorted(device_ops, key=lambda o: o["start_us"])
    op_type = dev_sorted[0]["type"]

    def _pick_host_name(host_ops: dict):
        """候选 host 名中取计数与 device 数一致者（精确 1:1 才可序号对齐）。"""
        candidates = threshold("communication", "comm_host_op_map", {}).get(f"hcom_{op_type}", [])
        for cand in candidates:
            if len(host_ops.get(cand, [])) == len(dev_sorted):
                return cand
        return None

    # --- trace 主路径（唯一可自校验：host_ts <= device_start） ---
    if not no_trace and trace_host_ops:
        name = _pick_host_name(trace_host_ops)
        if name:
            rows = trace_host_ops[name]
            lags = [d["start_us"] - h["ts"] for d, h in zip(dev_sorted, rows)]
            tol = threshold("communication", "host_ts_lag_tolerance_us", 1000)
            valid = sum(1 for lag in lags if lag >= -tol)
            if valid >= len(lags) * 0.95:
                p95 = _quantile(sorted(lags), 0.95)
                note = f"[ALIGNED·trace] host_ts 早于 device_start（P95 lag {p95/1000:.1f}ms）"
                return "trace", rows, note
            return "aggregate", [], f"[AGGREGATED] trace 序号对齐 ts 校验未过（{valid}/{len(lags)}），降级"
        # trace 中无计数匹配的候选名 → 落到 CSV 判定

    # --- CSV 序号对齐 fallback（文件序 = 时间序，无时间戳无法自证 → RISKY） ---
    if csv_host_ops:
        name = _pick_host_name(csv_host_ops)
        if name:
            overlap_ratio = _device_overlap_ratio(dev_sorted)
            if overlap_ratio > threshold("communication", "overlap_risk_ratio", 0.02):
                return "aggregate", [], (
                    f"[AGGREGATED] 同域时间区间重叠率 {overlap_ratio:.1%} 超阈值，"
                    "序号对齐不可靠，降级为聚合栈")
            return "csv", csv_host_ops[name], "[RISKY·csv] 文件序对齐（无时间戳自证）"

    return "aggregate", [], "[AGGREGATED] host 侧无计数匹配的 Hccl 记录"


def _aggregate_stacks(host_rows: list, top_n: int = 3) -> list:
    """聚合栈分布兜底：distinct Call Stack + 次数。"""
    counts = defaultdict(int)
    for r in host_rows:
        stack = r.get("call_stack", "").strip()
        if stack:
            counts[_condense_stack(stack, 6)] += 1
    return sorted(counts.items(), key=lambda x: -x[1])[:top_n]

# ---------------------------------------------------------------------------
# M1 溯源报告（H3 Top Ops + 瞬态防护 + 疑点块）
# ---------------------------------------------------------------------------

def _head_region(comm_ops: list):
    """通信总时窗与头部区段边界（us）。Returns (span_start, span_end, head_end) or None。"""
    if not comm_ops:
        return None
    starts = [o["start_us"] for o in comm_ops]
    ends = [o["start_us"] + o["elapse"] * 1000 for o in comm_ops]
    span_start, span_end = min(starts), max(ends)
    frac = threshold("communication", "head_window_frac", 0.1)
    return span_start, span_end, span_start + (span_end - span_start) * frac


def _transient_guard(comm_ops: list) -> list:
    """M1 瞬态污染防护：wait 头部集中度（H1 一行 + 可选 SIGNAL）。"""
    region = _head_region(comm_ops)
    if not region:
        return []
    _, _, head_end = region
    total_wait = sum(o["wait"] for o in comm_ops)
    head_wait = sum(o["wait"] for o in comm_ops if o["start_us"] < head_end)
    lines = [f"  Wait 头部集中度: 头部区段（前 {threshold('communication', 'head_window_frac', 0.1):.0%} 时窗）"
             f"占 wait 总量 {_pct(head_wait, total_wait):.1f}%"]
    if total_wait > 0 and head_wait / total_wait > threshold("communication", "head_wait_ratio", 0.6):
        lines.append(f"  [SIGNAL] wait 集中于时间轴头部（{_pct(head_wait, total_wait):.0f}%）— 疑瞬态污染"
                     "（warmup/编译期），验证路径: warmup 是否生效 / 多 step 是否复现 / L0 对照")
    return lines


def _build_alignment(ascend_dir: Path, comm_ops: list, no_trace: bool) -> dict:
    """按通信类型构建溯源对齐。Returns:
       {type: {"method", "rows", "note", "index": {opname: row_idx}}}
    """
    by_type = defaultdict(list)
    for op in comm_ops:
        by_type[op["type"]].append(op)

    trace_host_ops = {}
    if not no_trace and (ascend_dir / "trace_view.json").exists():
        trace_host_ops = _load_host_ops_trace(ascend_dir)

    csv_host_ops = None  # 懒加载
    align = {}
    for op_type, ops in by_type.items():
        method, rows, note = _align_type(ops, {}, trace_host_ops, no_trace)
        if method == "aggregate" and csv_host_ops is None:
            csv_host_ops = _load_host_ops_csv(ascend_dir)
        if method == "aggregate" and csv_host_ops:
            method, rows, note = _align_type(ops, csv_host_ops, trace_host_ops, no_trace)
        dev_sorted = sorted(ops, key=lambda o: o["start_us"])
        index = {op["name"]: i for i, op in enumerate(dev_sorted)}
        align[op_type] = {"method": method, "rows": rows, "note": note, "index": index}
    return align


def _op_source(op: dict, align: dict):
    """单个通信算子的源码位置。Returns (source_str, confidence_str) 或 (None, None)。"""
    a = align.get(op["type"])
    if not a or a["method"] == "aggregate":
        return None, None
    idx = a["index"].get(op["name"])
    if idx is None or idx >= len(a["rows"]):
        return None, None
    stack = a["rows"][idx].get("call_stack", "")
    if not stack.strip():
        return None, None
    conf = "ALIGNED" if a["method"] == "trace" else "RISKY"
    return _condense_stack(stack), conf


def _section_trace_report(ascend_dir: Path, comm_data: dict, top_k: int, no_trace: bool):
    """H3 Top Ops（含源码列 + 头部标注）+ 疑点块。

    Returns (h3_text, suspects_text, h1_extra_lines, align)
    """
    comm_ops = comm_data["ops"]
    region = _head_region(comm_ops)
    head_end = region[2] if region else float("inf")

    # 疑点：wait 主导且 elapse 达到门槛的 Top-K（成本控制：有疑点才加载 host 侧数据）
    min_elapse = threshold("communication", "suspect_min_elapse_ms", 100)
    candidates = [o for o in comm_ops if o["elapse"] >= min_elapse and o["wait"] > 0]
    suspects = sorted(candidates, key=lambda o: -o["wait"])[
        :threshold("communication", "trace_top_k", 3)]

    align = {}
    h1_extra = []
    if suspects:
        align = _build_alignment(ascend_dir, comm_ops, no_trace)
        methods = {a["method"] for a in align.values()}
        if "trace" in methods:
            h1_extra.append("  溯源方式: trace ts 对齐 [ALIGNED]")
        elif "csv" in methods:
            h1_extra.append("  溯源方式: CSV 序号对齐 [RISKY]（trace 不可用）")
        elif "aggregate" in methods:
            h1_extra.append("  溯源方式: 聚合栈 [AGGREGATED]（对齐不可靠，宁降级不硬对）")

    # --- H3: Top Ops by Elapse ---
    L = [f"## H3. Top Ops by Elapse（源码列仅疑点自动溯源；深挖用 --trace-source）"]
    top_ops = sorted(comm_ops, key=lambda o: -o["elapse"])[:top_k]
    L.append(f"  {'Op':<44} {'Type':<12} {'Elapse':>9} {'Wait%':>6}  源码位置 / 置信度")
    L.append("  " + "-" * 110)
    for op in top_ops:
        src, conf = _op_source(op, align)
        head_flag = " [头部]" if op["start_us"] < head_end else ""
        if src:
            src_txt = f"{src} [{conf}]{head_flag}"
        elif op in suspects:
            src_txt = "(聚合栈见疑点块)"
        else:
            src_txt = "—"
        L.append(f"  {op['name'][:44]:<44} {op['type']:<12} {_fmt_ms(op['elapse']):>9} "
                 f"{_pct(op['wait'], op['elapse']):>5.0f}%  {src_txt}")

    # --- 疑点块（COMM_SUSPECTS，供 run_analysis 切块上浮总章） ---
    S = []
    if suspects:
        for i, op in enumerate(suspects, 1):
            src, conf = _op_source(op, align)
            idx = align.get(op["type"], {}).get("index", {}).get(op["name"])
            n_type = len(align.get(op["type"], {}).get("index", {}))
            head_flag = "头部" if op["start_us"] < head_end else ""
            S.append(f"{i}. {op['name']}  |  elapse {_fmt_ms(op['elapse'])}  |  "
                     f"wait {_pct(op['wait'], op['elapse']):.1f}%  |  序号 {idx}/{n_type} {head_flag}".rstrip())
            a = align.get(op["type"], {})
            if src:
                S.append(f"   源码: {src}")
                S.append(f"   置信度: {a['note'] if a.get('note') else conf}")
                S.append(f"   归因线索: 检查该调用点通信是否冗余/原语不当/可掩盖（见 profiling_to_action 归因层 #8）")
            elif a.get("method") == "aggregate":
                stacks = _aggregate_stacks(a.get("rows", []))
                S.append(f"   源码: (对齐降级) {a.get('note', '')}")
                for stack, cnt in stacks:
                    S.append(f"   聚合栈({cnt}x): {stack}")
    return "\n".join(L), "\n".join(S), h1_extra, align


def _trace_source_mode(ascend_dir: Path, comm_data: dict, opname: str, no_trace: bool) -> str:
    """--trace-source 深挖模式：单算子完整调用栈。"""
    comm_ops = comm_data["ops"]
    target = next((o for o in comm_ops if o["name"] == opname or o["name"].split("@")[0] == opname), None)
    L = ["# Communication 单算子溯源", f"目标: {opname}"]
    if not target:
        avail = "\n".join(f"  {o['name']}" for o in sorted(comm_ops, key=lambda o: -o['elapse'])[:10])
        return "\n".join(L + ["  (未找到该算子。Top elapse 算子供参考:)", avail])

    ti_fmt = (f"elapse {_fmt_ms(target['elapse'])} | transit {_fmt_ms(target['transit'])} | "
              f"wait {_fmt_ms(target['wait'])} | start_us {target['start_us']:.0f}")
    L.append(f"匹配: {target['name']}")
    L.append(f"耗时: {ti_fmt}")
    L.append("")

    if not (ascend_dir / "trace_view.json").exists() and not (ascend_dir / "operator_details.csv").exists():
        return "\n".join(L + ["  (无溯源数据源：trace_view.json 与 operator_details.csv 均不存在；"
                             "Call Stack 桥需 with_stack=True 采集)"])

    align = _build_alignment(ascend_dir, comm_ops, no_trace)
    a = align.get(target["type"], {})
    idx = a.get("index", {}).get(target["name"])
    L.append(f"对齐方式: {a.get('note', '(无)')}")
    if a.get("method") == "aggregate":
        for stack, cnt in _aggregate_stacks(a.get("rows", [])):
            L.append(f"聚合栈({cnt}x):")
            for frame in stack.split(" ← "):
                L.append(f"  {frame}")
        return "\n".join(L)
    if idx is None or idx >= len(a["rows"]):
        return "\n".join(L + ["  (序号越界，无法对齐)"])
    stack = a["rows"][idx].get("call_stack", "")
    L.append("完整调用栈（自底向上）:")
    for frame in stack.replace("\r\n", "\n").split("\n"):
        if frame.strip():
            L.append(f"  {frame.strip()}")
    return "\n".join(L)

# ---------------------------------------------------------------------------
# M2 Matrix 深度分析（H5）：倾斜 / transport 合理性 / 慢链路自基准 / 渲染 / 回链 /
#                          字节对齐 / RDMA 重传 / 延迟主导小包
# ---------------------------------------------------------------------------

def _bucket_of(size_mb: float, buckets: list) -> str:
    """size 对数分档标签。"""
    bounds = sorted(buckets)
    if size_mb < bounds[0]:
        return f"<{bounds[0]}MB"
    for lo, hi in zip(bounds, bounds[1:] + [None]):
        if hi is None:
            return f">={lo}MB"
        if lo <= size_mb < hi:
            return f"{lo}-{hi}MB"
    return f">={bounds[-1]}MB"


def _bucket_min_mb(bkt_label: str) -> float:
    """分档标签 → 档下界（MB）。'<1MB'→0, '1-10MB'→1, '>=100MB'→100。"""
    num = bkt_label.lstrip("<>=").split("MB")[0].split("-")[0]
    return safe_float(num) if not bkt_label.startswith("<") else 0.0


def _section_matrix(matrix: list) -> str:
    """H5 Matrix 深度分析。matrix 为 [{group, link, transport, size_mb, transit_ms, bw, op_name}]。"""
    L = ["## H5. Matrix 深度分析"]
    if not matrix:
        L.append("  (无 communication_matrix.json 数据)")
        return "\n".join(L)
    entries = [e for e in matrix if e["size_mb"] > 0]
    transports = sorted({e["transport"] for e in entries})
    L.append(f"  逻辑组 {len({e['group'] for e in entries})} 个，link {len(entries)} 条，"
             f"transport: {', '.join(transports)}")
    L.append("")

    skew_ratio = threshold("communication", "skew_ratio", 3.0)
    skew_min_size = threshold("communication", "skew_min_size_mb", 10)
    groups = defaultdict(list)
    for e in entries:
        groups[e["group"]].append(e)

    # --- rank-pair 倾斜（同 transport 内比较、排除小消息、按类型解读） ---
    L.append("### rank-pair 数据倾斜")
    skew_hits = []
    for gname, gents in sorted(groups.items(), key=lambda x: -sum(e["size_mb"] for e in x[1])):
        remote = [e for e in gents if e["transport"] != "LOCAL" and e["size_mb"] >= skew_min_size]
        by_transport = defaultdict(list)
        for e in remote:
            by_transport[e["transport"]].append(e)
        for tr, ents in by_transport.items():
            sizes = [e["size_mb"] for e in ents]
            if len(sizes) < 2 or min(sizes) <= 0:
                continue
            ratio = max(sizes) / min(sizes)
            if ratio > skew_ratio:
                busiest = max(ents, key=lambda e: e["size_mb"])
                idlest = min(ents, key=lambda e: e["size_mb"])
                op_type = busiest["group"].split("-")[0]
                hint = ("负载分布不均（alltoall 倾斜常是目标发现；MoE/手工切分均可能）"
                        if op_type.startswith("alltoall") else "实现问题（集合通信理想形态各 rank 收发均衡）")
                skew_hits.append(gname)
                L.append(f"  [SIGNAL] {gname}（{tr}）: 倾斜比 {ratio:.1f}x — "
                         f"link {busiest['link']} {busiest['size_mb']:.0f}MB vs link {idlest['link']} "
                         f"{idlest['size_mb']:.0f}MB → 判读: {hint}")
                L.append(f"    回链: {busiest['op_name']}（可 --trace-source 深挖）")
    if not skew_hits:
        L.append(f"  (无超 {skew_ratio}x 的倾斜（已排除 <{skew_min_size}MB 小消息与 LOCAL）)")
    L.append("")

    # --- transport 合理性（只用数据内证据，不断言拓扑） ---
    L.append("### Transport 合理性")
    pair_transports = defaultdict(set)
    mixed_groups = []
    for gname, gents in groups.items():
        remote = [e for e in gents if e["transport"] != "LOCAL" and e["size_mb"] >= skew_min_size]
        trs = {e["transport"] for e in remote}
        for e in remote:
            pair_transports[e["link"]].add(e["transport"])
        if len(trs) > 1:
            desc = ", ".join(f"{e['link']}→{e['transport']}({e['size_mb']:.0f}MB)" for e in remote)
            mixed_groups.append(gname)
            L.append(f"  [SIGNAL] {gname}: 同组同量级流量 transport 混杂 — {desc}")
            L.append("    需对照部署拓扑确认 rank-设备映射（文件不含物理拓扑，不断言应走什么）")
    inconsistent_pairs = {p: trs for p, trs in pair_transports.items() if len(trs) > 1}
    if inconsistent_pairs:
        for p, trs in list(inconsistent_pairs.items())[:5]:
            L.append(f"  [SIGNAL] link {p} 跨组 transport 不一致: {', '.join(sorted(trs))} — 硬信号")
    if not mixed_groups and not inconsistent_pairs:
        L.append("  (组内 transport 一致，无跨组不一致)")
    L.append("")

    # --- 慢链路定位（M4 自基准：size 条件化 + 档内分位数） ---
    L.append("### 慢链路定位（带宽自基准）")
    buckets = threshold("communication", "slow_link_size_buckets", [1, 10, 100])
    bw_p50_ratio = threshold("communication", "bw_p50_ratio", 0.5)
    by_tb = defaultdict(list)  # (transport, bucket) -> [entries]
    for e in entries:
        if e["transport"] != "LOCAL" and e["bw"] > 0:
            by_tb[(e["transport"], _bucket_of(e["size_mb"], buckets))].append(e)
    slow_found = False
    envelope = {}
    large_bucket_min = buckets[1] if len(buckets) > 1 else 10
    for (tr, bkt), ents in sorted(by_tb.items()):
        bws = sorted(e["bw"] for e in ents)
        p50 = _quantile(bws, 0.5)
        envelope[(tr, bkt)] = max(bws)
        # 仅大 size 档报慢链路（小消息低带宽是固定延迟的物理必然）
        if _bucket_min_mb(bkt) < large_bucket_min:
            continue
        slow = [e for e in ents if e["bw"] < p50 * bw_p50_ratio]
        if slow:
            slow_found = True
            L.append(f"  [SIGNAL] {tr} {bkt}: {len(slow)}/{len(ents)} 条 link 带宽 < 档内 P50"
                     f"({p50:.1f})×{bw_p50_ratio} — 慢链路候选:")
            for e in sorted(slow, key=lambda x: x["bw"])[:3]:
                L.append(f"    {e['link']:<6} {e['bw']:>7.1f}GB/s  {e['size_mb']:>8.1f}MB  {e['group']}")
    if not slow_found:
        L.append("  (大 size 档内无低于 P50×%s 的慢链路)" % bw_p50_ratio)
    # 带宽上包络（该环境实测峰值估计）+ 可选绝对基准
    large_bkts = {k: v for k, v in envelope.items() if _bucket_min_mb(k[1]) >= large_bucket_min}
    if large_bkts:
        peak = max(large_bkts.values())
        L.append(f"  带宽上包络（大 size 档最大值，本环境实测峰值估计）: {peak:.1f} GB/s")
    soc_table = threshold("communication", "soc_bw_table", {})
    if soc_table:
        L.append(f"  (soc_bw_table 已配置 {len(soc_table)} 项 — 绝对基准判据可用)")
    else:
        L.append("  (soc_bw_table 默认空：绝对基准未启用，仅自基准生效——所有链路一起慢的系统性问题检测不出)")
    L.append("")

    # --- 链路矩阵渲染（流量最大逻辑组） ---
    biggest = max(groups.values(), key=lambda g: sum(e["size_mb"] for e in g))
    gname = biggest[0]["group"]
    L.append(f"### 链路矩阵渲染（{gname}，按传输量 MB，. = <1）")
    ranks = sorted({int(e["link"].split("-")[0]) for e in biggest} |
                   {int(e["link"].split("-")[1]) for e in biggest})
    size_map = {e["link"]: e["size_mb"] for e in biggest}
    header = "       " + " ".join(f"r{r:>4}" for r in ranks)
    L.append(header)
    for src in ranks:
        row = []
        for dst in ranks:
            sz = size_map.get(f"{src}-{dst}", 0)
            row.append(f"{sz:>5.0f}" if sz >= 1 else ("  .  " if sz > 0 else "    0"))
        L.append(f"  r{src:>3}  " + " ".join(row))
    L.append("")

    # --- 字节对齐（跳过 LOCAL；仅 <100MB，MB 小数精度限制） ---
    align_bytes = threshold("communication", "alignment_bytes", 512)
    unaligned_ratio_th = threshold("communication", "unaligned_link_ratio", 0.3)
    checkable = [e for e in entries if e["transport"] != "LOCAL" and 0 < e["size_mb"] <= 100]
    unaligned = [e for e in checkable if int(e["size_mb"] * 1024 * 1024) % align_bytes != 0]
    L.append("### 字节对齐")
    if checkable:
        ratio = len(unaligned) / len(checkable)
        L.append(f"  非 {align_bytes}B 对齐 link: {len(unaligned)}/{len(checkable)}（仅检查 <100MB 非 LOCAL）")
        if ratio > unaligned_ratio_th:
            L.append(f"  [SIGNAL] 非 {align_bytes}B 对齐占比 {ratio:.0%} > {unaligned_ratio_th:.0%} — "
                     "带宽可能受影响（HCCS 对齐要求），调整张量 shape 或 Padding")
        elif unaligned:
            for e in unaligned[:3]:
                rem = int(e["size_mb"] * 1024 * 1024) % align_bytes
                L.append(f"  {e['link']:<6} {e['size_mb']:.4f}MB（余 {rem}B）")
    else:
        L.append("  (无可检查的 link)")
    L.append("")

    # --- RDMA 重传疑似 ---
    retrans_th = threshold("communication", "rdma_retransmission_ms", 4000)
    retrans = [e for e in entries if e["transport"] != "LOCAL" and e["transit_ms"] > retrans_th]
    L.append("### RDMA 重传疑似")
    if retrans:
        for e in sorted(retrans, key=lambda x: -x["transit_ms"])[:5]:
            L.append(f"  [SIGNAL] {e['link']:<6} ({e['transport']}): Transit={e['transit_ms']:.0f}ms, "
                     f"Size={e['size_mb']:.1f}MB — 疑似重传")
        L.append("    交叉验证: 链路误码率 BER / 交换机 PFC 配置 / UDP 端口冲突")
    else:
        L.append(f"  (无 Transit > {retrans_th}ms 的 link)")
    L.append("")

    # --- 延迟主导小包（L̂ 数据内估计） ---
    L.append("### 延迟主导小包（L̂ 数据内估计）")
    lat_multiple = threshold("communication", "latency_dominated_multiple", 3.0)
    min_samples = threshold("communication", "latency_est_min_samples", 10)
    small_ratio_th = threshold("communication", "small_packet_ratio", 0.3)
    default_mode = False
    for tr in transports:
        ents = [e for e in entries if e["transport"] == tr]
        if not ents:
            continue
        smallest = [e for e in ents if e["size_mb"] < buckets[0]]
        if len(smallest) >= min_samples:
            l_hat = _quantile(sorted(e["transit_ms"] for e in smallest), 0.5)
            mode = f"L̂={l_hat * 1000:.0f}us（最小档 P50, n={len(smallest)}）"
        else:
            l_hat = None
            mode = f"[DEFAULT] 样本不足({len(smallest)}<{min_samples})，退回固定 1MB 阈值"
            default_mode = True
        if l_hat is not None:
            dominated = [e for e in ents if e["transit_ms"] < lat_multiple * l_hat]
        else:
            dominated = [e for e in ents if e["size_mb"] < buckets[0]]
        ratio = len(dominated) / len(ents) if ents else 0
        L.append(f"  {tr}: 延迟主导 {len(dominated)}/{len(ents)}（{ratio:.0%}）— {mode}")
        if ratio > small_ratio_th:
            if l_hat is not None:
                L.append(f"    [SIGNAL] 延迟主导占比 > {small_ratio_th:.0%} — 建议合并通信/batching；"
                         f"收益上限估算 = 消息次数 × L̂ ≈ {_fmt_ms(len(dominated) * l_hat)}")
            else:
                L.append(f"    [SIGNAL] 延迟主导占比 > {small_ratio_th:.0%} — 建议合并通信/batching"
                         "（L̂ 未估计，收益估算待 DEFAULT 模式解除后可用）")
    L.append("")

    return "\n".join(L)

# ---------------------------------------------------------------------------
# M3 跨 Rank 通信对比（H7，--all-ranks）
# ---------------------------------------------------------------------------

def _load_alltoall_durations(ascend_dir: Path) -> float:
    """kernel_details.csv 中 alltoall kernel 的总耗时（us，负载不均交叉验证指标）。"""
    csv_path = ascend_dir / "kernel_details.csv"
    if not csv_path.exists():
        return 0.0
    total = 0.0
    for row in stream_csv(csv_path):
        name = row.get("Name", "")
        if "alltoall" in name.lower() or "all2all" in name.lower():
            total += safe_float(row.get("Duration(us)", 0))
    return total


def _section_cross_rank(ranks_data: list, top_k: int) -> str:
    """H7 跨 Rank 对比。ranks_data = [(rid, ascend_dir, comm_data, step_overlap), ...]"""
    L = ["## H7. 跨 Rank 对比（--all-ranks）"]
    rids = [rid for rid, _, _, _ in ranks_data]
    L.append(f"  Rank: {', '.join(f'rank_{r}' for r in rids)}（join 按 hcom 全名，@域ID 天然分组）")
    L.append("")

    # --- M3.1 per-op wait join（慢卡定位在此得出） ---
    L.append("### M3.1 per-op wait join（straggler 定位）")
    ops_by_name = {}
    for rid, _, comm, _ in ranks_data:
        for op in comm["ops"]:
            ops_by_name.setdefault(op["name"], {})[rid] = op
    joined = []
    for name, per_rank in ops_by_name.items():
        if len(per_rank) < 2:
            continue
        waits = {rid: op["wait"] for rid, op in per_rank.items()}
        joined.append({"name": name, "waits": waits,
                       "max_wait": max(waits.values()), "min_wait": min(waits.values()),
                       "elapse": max(o["elapse"] for o in per_rank.values())})
    min_elapse = threshold("communication", "suspect_min_elapse_ms", 100)
    significant = [j for j in joined if j["elapse"] >= min_elapse]
    top_wait = sorted(significant, key=lambda j: -j["max_wait"])[:top_k]

    gap = threshold("communication", "straggler_min_wait_gap", 0.5)
    votes = defaultdict(int)
    vote_ops = 0
    if top_wait:
        header = f"  {'Op':<44} " + " ".join(f"r{r:>6}" for r in rids) + "   max/min"
        L.append(header)
        L.append("  " + "-" * (len(header) - 2))
        for j in top_wait:
            row = " ".join(f"{j['waits'].get(r, 0):>7.0f}" for r in rids)
            ratio = j["max_wait"] / j["min_wait"] if j["min_wait"] > 0 else float("inf")
            L.append(f"  {j['name'][:44]:<44} {row}   {ratio:>6.1f}x")
            if ratio > 1 + gap:
                vote_ops += 1
                votes[min(j["waits"], key=j["waits"].get)] += 1
        L.append("")
        if votes and vote_ops > 0:
            straggler, cnt = max(votes.items(), key=lambda x: x[1])
            if cnt >= vote_ops * 0.5 and cnt >= 2:
                L.append(f"  [DEFINITE] straggler = rank_{straggler}：在 {cnt}/{vote_ops} 个高 wait 算子上"
                         f" wait 最小（最后到达的被等者）→ 其计算侧慢，导致其余 rank 等待")
                L.append(f"    归因: 等待型。后续: 对 rank_{straggler} 跑单卡脚本（--rank {straggler}）"
                         "定位其计算侧根因")
            else:
                L.append("  (高 wait 算子上无一致的最小 wait rank — 无显著 straggler)")
    else:
        L.append(f"  (无 elapse >= {min_elapse}ms 的可对比算子)")
    L.append("")

    # --- M3.2 per-rank 汇总（R_wait 仅画像/排序，无阈值判定） ---
    L.append("### M3.2 per-rank 汇总（R_wait 仅画像展示）")
    header = f"  {'Rank':>5} {'Elapse(ms)':>11} {'Transit(ms)':>12} {'Wait(ms)':>10} {'Wait%':>6}"
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    for rid, _, comm, _ in ranks_data:
        t = comm["total"]
        L.append(f"  {rid:>5} {t['elapse']:>11.1f} {t['transit']:>12.1f} {t['wait']:>10.1f} "
                 f"{_pct(t['wait'], t['elapse']):>5.1f}%")
    L.append("")
    type_totals = defaultdict(lambda: defaultdict(float))
    for rid, _, comm, _ in ranks_data:
        for op in comm["ops"]:
            type_totals[op["type"]][rid] += op["elapse"]
    r_waits = []
    for op_type, per_rank in type_totals.items():
        vals = list(per_rank.values())
        if len(vals) >= 2 and max(vals) > 0:
            r_waits.append((op_type, 1 - statistics.mean(vals) / max(vals), vals))
    r_waits.sort(key=lambda x: -x[1])
    top_n = threshold("communication", "r_wait_sort_top_k", 5)
    if r_waits:
        L.append(f"  按类型 R_wait 排序（Top {min(top_n, len(r_waits))}，画像指标，判定见 M3.1 / wait-transit 分解）:")
        for op_type, rw, vals in r_waits[:top_n]:
            L.append(f"    {op_type:<14} R_wait={rw:.2f}  (T: {'/'.join(f'{v:.0f}' for v in vals)})")
    L.append("")

    # --- M3.3 重叠对比（现象测量；归因二分：串行型 vs 依赖性串行） ---
    L.append("### M3.3 重叠对比（现象测量，归因二分）")
    comm_bound = threshold("step_trace", "comm_bound_pct", 20)
    overlap_rows = []
    for rid, _, _, step in ranks_data:
        if not step or step.get("total", 0) <= 0:
            continue
        overlap_rows.append({
            "rid": rid,
            "total": step["total"],
            "ovl": step["overlapped"],
            "novl": step["comm_not_ovl"],
            "util": step["computing"] / step["total"] * 100,
        })
    if overlap_rows:
        header = f"  {'Rank':>5} {'Total(ms)':>10} {'Ovl(ms)':>10} {'CommNO(ms)':>11} {'Ovl%':>6} {'CommNO%':>8} {'Util%':>6}"
        L.append(header)
        L.append("  " + "-" * (len(header) - 2))
        for r in overlap_rows:
            L.append(f"  {r['rid']:>5} {r['total']:>10.0f} {r['ovl']:>10.1f} {r['novl']:>11.1f} "
                     f"{_pct(r['ovl'], r['total']):>5.1f}% {_pct(r['novl'], r['total']):>7.1f}% {r['util']:>5.1f}%")
        L.append("")
        avg_novl_pct = statistics.mean(_pct(r["novl"], r["total"]) for r in overlap_rows)
        avg_ovl_pct = statistics.mean(_pct(r["ovl"], r["total"]) for r in overlap_rows)
        avg_util = statistics.mean(r["util"] for r in overlap_rows)
        if avg_novl_pct > comm_bound or (avg_util < 50 and avg_novl_pct > 0):
            L.append(f"  [SIGNAL] Comm(NotOvl) 占比 {avg_novl_pct:.1f}%（阈值 {comm_bound}%）— 掩盖不足疑点，归因二分:")
            L.append("    - 串行型（可掩盖未掩盖）: 通信调用点上下游存在独立计算但调度串行 → 修异步通信流/调度重排。")
            L.append("      数据内提示: 通信 kernel 前后 idle gap、多流并发占比；确认需 Line A 源码分析")
            L.append("    - 依赖性串行（不可掩盖）: 通信是下一步计算的直接输入（如 TP 逐层 allgather）→ overlap 上限为 0，")
            L.append("      正路: 减通信量（冗余/原语/拓扑）或改变依赖结构（流水线/分块，策略性，需 ★A 确认）")
        else:
            L.append(f"  (Comm(NotOvl) 占比 {avg_novl_pct:.1f}% 未超 {comm_bound}% 阈值，非掩盖不足)")
        # 假性重叠
        ovl_severe = threshold("communication", "overlap_severe_pct", 5)
        false_th = threshold("communication", "false_overlap_comm_pct", 20)
        if avg_ovl_pct > ovl_severe and avg_novl_pct > false_th:
            L.append(f"  [SIGNAL] 疑假性重叠: 重叠 {avg_ovl_pct:.0f}% 但未重叠通信仍占 {avg_novl_pct:.0f}% — "
                     "通信 DMA 与计算争抢 HBM 嫌疑")
            L.append("    交叉验证: 开/关重叠时同一计算算子耗时是否增加 > 10-20%")
    else:
        L.append("  (无 step_trace 重叠数据)")
    L.append("")

    # --- alltoall 负载不均判读（wait 主导 × M2 倾斜；kernel 耗时 CV 交叉验证） ---
    L.append("### alltoall 负载不均判读")
    a2a_type = type_totals.get("alltoall", {})
    if a2a_type:
        vals = list(a2a_type.values())
        cv = statistics.stdev(vals) / statistics.mean(vals) if len(vals) >= 2 and statistics.mean(vals) > 0 else 0
        wait_pct = statistics.mean(
            _pct(comm["total"]["wait"], comm["total"]["elapse"]) for _, _, comm, _ in ranks_data)
        kernel_durs = {rid: _load_alltoall_durations(ad) for rid, ad, _, _ in ranks_data}
        k_vals = [v / 1000 for v in kernel_durs.values() if v > 0]
        k_cv = (statistics.stdev(k_vals) / statistics.mean(k_vals)
                if len(k_vals) >= 2 and statistics.mean(k_vals) > 0 else 0.0)
        cv_th = threshold("communication", "alltoall_cv_signal", 0.20)
        L.append(f"  alltoall 通信量跨 rank CV = {cv:.3f}；kernel 耗时跨 rank CV = {k_cv:.3f}"
                 f"（阈值 {cv_th}）；全局 wait 占比 {wait_pct:.0f}%")
        if cv > cv_th and wait_pct > 50:
            L.append(f"  [SIGNAL] alltoall 通信量不均（CV {cv:.2f}）+ wait 主导 → Token/数据分布不均"
                     "（MoE 为 LLM 成因之一，手工切分同样触发）")
            cross = "一致" if k_cv > cv_th else "不一致（需复核）"
            L.append(f"    kernel 耗时 CV = {k_cv:.2f}，交叉验证{cross}；与 H5 rank-pair 倾斜互为印证")
            L.append("    优化方向: 负载再平衡（容量因子/切分配置），不预设训练侧手段")
        elif k_cv > cv_th:
            L.append(f"  [SIGNAL] alltoall kernel 耗时不均（CV {k_cv:.2f}）但通信量均衡（CV {cv:.2f}）→ "
                     "非负载不均，指向链路/带宽差异（见 H5 慢链路与 transport 混杂）")
        else:
            L.append("  (alltoall 量与 kernel 耗时跨 rank 基本均衡，或非 wait 主导)")
    else:
        L.append("  (无 alltoall 通信)")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 报告组装
# ---------------------------------------------------------------------------

def parse(profiling_dir: str, rank=None, top_k: int = 15,
          trace_source: str = None, no_trace: bool = False) -> str:
    """主入口。run_analysis 调用: parse(profiling_dir, rank, 15)（多 rank 目录自动 --all-ranks）。"""
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    comm_path = ascend_dir / "communication.json"
    if not comm_path.exists():
        return (f"[communication] 文件未找到: {comm_path}\n"
                "Communication 数据需要多卡场景下 profiler_level >= Level1。")

    comm_data = _load_comm_ops(ascend_dir)
    if not comm_data or not comm_data["ops"]:
        return f"[communication] 在 {comm_path} 中未发现 collective op"

    # --trace-source 深挖模式
    if trace_source:
        return _trace_source_mode(ascend_dir, comm_data, trace_source, no_trace)

    # 跨 rank 自动启用（H 节检测到多个 rank_N 目录即进入，rank 参数只决定单 rank 细节的主视角）
    ranks = _discover_ranks(profiling_dir)
    cross_rank_text = ""
    ranks_data = []
    if len(ranks) >= 2:
        for rid, rdir in ranks:
            rcomm = _load_comm_ops(rdir)
            if rcomm and rcomm["ops"]:
                ranks_data.append((rid, rdir, rcomm, _load_step_overlap(rdir)))
        if len(ranks_data) >= 2:
            cross_rank_text = _section_cross_rank(ranks_data, top_k)

    L = []
    L.append("# Communication 分析")
    L.append(f"数据来源: {comm_path}")
    L.append("")

    # --- H1 摘要 ---
    L.append("## H1. 摘要")
    total = comm_data["total"]
    L.append(f"  通信算子: {len(comm_data['ops'])}  |  P2P op: {comm_data['p2p_count']}")
    L.append(f"  总 Elapse {_fmt_ms(total['elapse'])} = Transit {_fmt_ms(total['transit'])}"
             f"（{_pct(total['transit'], total['elapse']):.1f}%）+ Wait {_fmt_ms(total['wait'])}"
             f"（{_pct(total['wait'], total['elapse']):.1f}%）+ Sync {_fmt_ms(total['sync'])}"
             f" + Idle {_fmt_ms(total['idle'])}")
    # 一致性检查：step_trace Communication 列 vs Total Op Info（两份文件是同一测量的两个视角）
    step_primary = _load_step_overlap(ascend_dir)
    if step_primary and step_primary.get("comm_raw", 0) > 0 and total["elapse"] > 0:
        delta = abs(step_primary["comm_raw"] - total["elapse"]) / total["elapse"]
        if delta < 0.02:
            L.append(f"  一致性检查通过: step_trace Communication {step_primary['comm_raw']:.1f}ms "
                     f"≈ Total Op Info {total['elapse']:.1f}ms（差 {delta:.1%}）")
        else:
            L.append(f"  [SIGNAL] step_trace Communication {step_primary['comm_raw']:.1f}ms 与 "
                     f"Total Op Info {total['elapse']:.1f}ms 偏差 {delta:.0%} — 两视角不一致，数据可疑")
    if total["elapse"] > 0 and total["wait"] / total["elapse"] > threshold("communication", "wait_dominant_ratio", 0.8):
        L.append(f"  [DEFINITE] Wait 占主导（{_pct(total['wait'], total['elapse']):.0f}%）— "
                 "synchronization-bound 而非 bandwidth-bound")
        L.append("    归因: 等待型（谁在等谁见 H7 per-op join；带宽问题以 wait/transit 分解为准）")
    L.extend(_transient_guard(comm_data["ops"]))

    # H3（含溯源，成本控制：有疑点才加载 host 侧）
    h3_text, suspects_text, h1_extra, _ = _section_trace_report(ascend_dir, comm_data, top_k, no_trace)
    L.extend(h1_extra)
    if cross_rank_text and ranks_data:
        waits = {rid: c["total"]["wait"] for rid, _, c, _ in ranks_data}
        fastest = min(waits, key=waits.get)
        L.append(f"  慢卡提示: rank_{fastest} 通信 wait 最小 → straggler 嫌疑（详见 H7 M3.1）")
    L.append("")

    # --- H2 按算子类型 ---
    L.append("## H2. 按算子类型")
    by_type = defaultdict(lambda: {"count": 0, "elapse": 0.0, "transit": 0.0, "wait": 0.0})
    for op in comm_data["ops"]:
        a = by_type[op["type"]]
        a["count"] += 1
        a["elapse"] += op["elapse"]
        a["transit"] += op["transit"]
        a["wait"] += op["wait"]
    L.append(f"  {'Type':<15} {'Count':>6} {'Elapse(ms)':>11} {'Transit(ms)':>12} {'Wait(ms)':>10} {'Wait%':>6}")
    L.append("  " + "-" * 66)
    per_type_th = threshold("communication", "per_type_wait_ratio", 0.9)
    per_type_min = threshold("communication", "per_type_min_count", 10)
    for t, agg in sorted(by_type.items(), key=lambda x: -x[1]["elapse"]):
        L.append(f"  {t:<15} {agg['count']:>6} {agg['elapse']:>11.1f} {agg['transit']:>12.1f} "
                 f"{agg['wait']:>10.1f} {_pct(agg['wait'], agg['elapse']):>5.1f}%")
        if agg["elapse"] > 0 and agg["wait"] / agg["elapse"] > per_type_th and agg["count"] > per_type_min:
            L.append(f"    [SIGNAL] {t}: {agg['count']} 个 op，{agg['wait']/agg['elapse']*100:.0f}% wait — "
                     "交叉验证: H7 per-op join 查 straggler")
    L.append("")

    # --- H3 / H4 / H5 ---
    L.append(h3_text)
    L.append("")
    if comm_data["p2p"]:
        L.append(f"## H4. P2P Ops（send/recv）— {len(comm_data['p2p'])} 个")
        p2p_sorted = sorted(comm_data["p2p"], key=lambda p: -p["elapse"])[:top_k]
        L.append(f"  {'Op':<50} {'Elapse(ms)':>10} {'Transit(ms)':>12} {'Wait(ms)':>9}")
        for p in p2p_sorted:
            L.append(f"  {p['name'][:50]:<50} {p['elapse']:>10.2f} {p['transit']:>12.2f} {p['wait']:>9.2f}")
        p2p_total = sum(p["elapse"] for p in comm_data["p2p"])
        p2p_wait = sum(p["wait"] for p in comm_data["p2p"])
        if p2p_total > 0 and p2p_wait / p2p_total > 0.5:
            L.append(f"  - P2P 以 wait 为主（{p2p_wait/p2p_total*100:.0f}%）— 交叉验证 pipeline 气泡")
        L.append("")

    L.append(_section_matrix(_load_matrix(ascend_dir)))
    L.append("")

    # --- H6 可疑信号（汇总本节所有 [SIGNAL]/[DEFINITE] 行） ---
    L.append("## H6. 可疑信号汇总")
    body = "\n".join(L[1:]) + "\n" + cross_rank_text + "\n" + suspects_text
    sig_lines = [ln.strip() for ln in body.split("\n")
                 if ln.strip().startswith(("[DEFINITE]", "[SIGNAL]", "[FUTURE]"))]
    if sig_lines:
        for s in sig_lines:
            L.append(f"  {s}")
    else:
        L.append("  无")
    L.append("")

    # --- H7 跨 Rank ---
    if cross_rank_text:
        L.append(cross_rank_text)

    # --- COMM_SUSPECTS 定界块（run_analysis 切块上浮总章） ---
    if suspects_text:
        L.append("<<<COMM_SUSPECTS>>>")
        L.append(suspects_text)
        L.append("<<<END_COMM_SUSPECTS>>>")

    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--trace-source", default=None, metavar="OP",
                        help="单算子深挖溯源（输出完整调用栈）")
    parser.add_argument("--no-trace", action="store_true",
                        help="跳过 trace_view.json 扫描（只用 CSV 序号对齐）")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    result = parse(args.profiling_dir, args.rank, args.top_k,
                   trace_source=args.trace_source, no_trace=args.no_trace)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
