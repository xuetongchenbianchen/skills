#!/usr/bin/env python3
"""解析 api_statistic.csv — CANN runtime (ACL) API 调用耗时统计。

该文件在 L1 生成（profiler_level=Level1）。按三个层级聚合 CANN runtime API 调用的
wall-clock 耗时：
- acl: AscendCL runtime API（以 *_Tiling 为主 — 每个 aclnn op 的 host 侧 tiling）
- communication: HCCL communication API（Notify_* 等）
- node: node 级 launch（count == trace_view 中的 Node@launch）

相比其他文件的独特价值：将 host dispatch overhead 拆解到
**CANN runtime API 层**（tiling / launch / comm notify），补充
operator_details（op 名层级）与 trace_view（逐实例时间线）。
回答"host 时间中 tiling、launch、comm-notify 各占多少"。

用法:
    python parse_api_statistic.py <profiling_dir> [--rank N] [--top-k 20]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import threshold, find_ascend_profiler_output, read_csv_all, safe_float, safe_int


def parse(profiling_dir: str, rank=None, top_k: int = 20) -> str:
    ascend_dir = find_ascend_profiler_output(profiling_dir, rank)
    csv_path = ascend_dir / "api_statistic.csv"
    rows = read_csv_all(csv_path)

    if not rows:
        return f"[api_statistic] 文件未找到或为空: {csv_path}\n(api_statistic.csv 在 L1 生成；L0 下不存在。)"

    # Level 名称规范化（兼容不同 CANN 版本）
    _LEVEL_MAP = {"ascendcl": "acl", "hccl": "communication"}
    # 按 Level 聚合
    by_level = defaultdict(lambda: {"count": 0, "time_us": 0.0, "apis": []})
    total_time = 0.0
    for r in rows:
        lv_raw = r.get("Level", "?").strip()
        lv = _LEVEL_MAP.get(lv_raw.lower(), lv_raw.lower())
        cnt = safe_int(r.get("Count", 0))
        t = safe_float(r.get("Time(us)", 0))
        avg = safe_float(r.get("Avg(us)", 0))
        mx = safe_float(r.get("Max(us)", 0))
        name = r.get("API Name", "?")
        by_level[lv]["count"] += cnt
        by_level[lv]["time_us"] += t
        by_level[lv]["apis"].append((t, name, cnt, avg, mx))
        total_time += t

    lines = []
    lines.append("# API Statistic 摘要 (CANN runtime API overhead)")
    lines.append(f"数据来源: {csv_path}")
    lines.append(f"API 总耗时: {total_time/1000:.1f} ms")
    lines.append("")

    # 各 Level 摘要
    lines.append("## 按 Level")
    for lv, info in sorted(by_level.items()):
        pct = info["time_us"] / total_time * 100 if total_time > 0 else 0
        lines.append(f"  {lv:<14} calls={info['count']:>7,}  time={info['time_us']/1000:>9.1f} ms ({pct:>5.1f}%)")
    lines.append("")

    # acl 层级: tiling 拆解（每个 op 类型的 host 侧 dispatch 开销）
    acl = by_level.get("acl")
    if acl:
        lines.append(f"## ACL API Top {min(top_k, len(acl['apis']))} (host runtime，按总耗时)")
        lines.append("  Tiling = 每个 aclnn op 的 host 侧参数计算（可缓存；dynamic shape 重复 tiling 是浪费）。")
        header = f"  {'API Name':<32} {'Count':>7} {'Total(ms)':>10} {'Avg(us)':>9} {'Max(us)':>9}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for t, name, cnt, avg, mx in sorted(acl["apis"], key=lambda x: -x[0])[:top_k]:
            lines.append(f"  {name:<32} {cnt:>7} {t/1000:>10.2f} {avg:>9.2f} {mx:>9.2f}")
        # host 侧预计算小计（tiling + workspace）
        tiling_us = sum(t for t, n, *_ in acl["apis"] if n.lower().endswith("_tiling"))
        workspace_us = sum(t for t, n, *_ in acl["apis"] if "getworkspacesize" in n.lower())
        if tiling_us > 0 or workspace_us > 0:
            lines.append("")
            host_pre = tiling_us + workspace_us
            lines.append(f"  Host 侧预计算: tiling={tiling_us/1000:.1f}ms  workspace={workspace_us/1000:.1f}ms  合计={host_pre/1000:.1f}ms ({host_pre/total_time*100:.1f}%)")
            if host_pre / total_time > threshold("api_statistic", "host_precompute_ratio", 0.2):
                lines.append("  - 每次 aclnn 调用前的 host 侧计算。shape 不变时可缓存；图模式自动缓存。")
        lines.append("")

    # node launch（count 应与 trace_view Node@launch 一致）
    node = by_level.get("node")
    if node:
        lines.append("## Node 级 Launch")
        for t, name, cnt, avg, mx in node["apis"]:
            lines.append(f"  {name}: count={cnt:,}  total={t/1000:.1f}ms  avg={avg:.1f}us")
        lines.append(f"  (count 应与 trace_view Node@launch 事件一致)")
        lines.append("")

    # communication API
    comm = by_level.get("communication")
    if comm:
        lines.append(f"## Communication API Top {min(top_k, len(comm['apis']))}")
        for t, name, cnt, avg, mx in sorted(comm["apis"], key=lambda x: -x[0])[:top_k]:
            lines.append(f"  {name:<24} count={cnt:>6,}  total={t/1000:>8.1f}ms  avg={avg:.2f}us")
        lines.append("")

    lines.append("## 可疑信号")
    # 对 acl API 分类以找出主导的 host-API 开销来源
    if acl:
        cat_us = defaultdict(float)
        for t, name, *_ in acl["apis"]:
            nl = name.lower()
            if any(k in nl for k in ("free", "malloc", "mapphysical", "unmapmem", "alloc", "freephysical")):
                cat_us["memory mgmt"] += t
            elif "sync" in nl:
                cat_us["stream/device sync"] += t
            elif "compile" in nl:
                cat_us["compile"] += t
            elif "getworkspacesize" in nl:
                cat_us["workspace"] += t
            elif nl.endswith("_tiling"):
                cat_us["tiling"] += t
            elif "launch" in nl:
                cat_us["launch"] += t
            else:
                cat_us["other"] += t
        if cat_us:
            dom_cat, dom_us = max(cat_us.items(), key=lambda x: x[1])
            dom_pct = dom_us / total_time * 100 if total_time > 0 else 0
            for c, u in sorted(cat_us.items(), key=lambda x: -x[1]):
                lines.append(f"  {c:<16} {u/1000:>9.1f} ms ({u/total_time*100 if total_time>0 else 0:>5.1f}%)")
            if dom_pct > threshold("api_statistic", "dominant_category_pct", 20):
                hint = {
                    "memory mgmt": "- Memory mgmt 主导（Free/Malloc/Unmap）。高 churn = 频繁 alloc/free；buffer 复用可减少。operator_memory 可确认重复 alloc。",
                    "stream/device sync": "- Sync 主导（SynchronizeStream/Device）。显式 sync 或 .item() 强制 D-H；避免循环中隐式同步。operator_details 的 sync category 可定位触发源。",
                    "compile": "- JIT 编译主导（aclopCompileAndExecute）。推理前 warmup 确保所有 shape 编译过；频繁新 shape 会反复触发。",
                    "workspace": "- GetWorkspaceSize 主导。每次 aclnn 调用前 CANN 计算 workspace 需求，shape 不变时可缓存；图模式自动缓存。",
                    "tiling": "- Tiling 主导；为静态/重复 shape 缓存 tiling (graph compile / op cache)。",
                    "launch": "- Launch overhead；减少 op 数量 (fusion / graph compile)。",
                }.get(dom_cat, "")
                if hint:
                    lines.append(f"  [SIGNAL] {hint}")
    else:
        lines.append("  无")
    lines.append("")
    lines.append("## 关联")
    lines.append("  - operator_details: 此处的 tiling/launch 是该处 host 时间的 CANN-API 层（按 op 名）")
    lines.append("  - trace_view: node launch count 与 Node@launch 一致；tiling 在 host thread 上先于每次 launch")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiling_dir")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()
    result = parse(args.profiling_dir, args.rank, args.top_k)
    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
