#!/usr/bin/env python3
"""切分方案估算器（第三步：通信代价与可行性）。

★ 原地运行，无需 copy 到项目：本脚本只做计算，产物是估算报告（stdout），
  报告数值供方案确认与 evidence_db 记录使用。
★ 输入 spec.json 由 agent 在第一、二步（读源码提取状态尺寸、定性分类）完成后填写。
★ 实现的模型与 analysis_workflow.md 第三步一致：
  - 显存: M = (M_param − M_repl)/p_param + M_repl + M_state/p_state
            + (M_act − M_act_intra)/p_act + M_act_intra/(tp·sp·cp) + M_overhead，
    阈值 HBM × 0.8（M_repl 不除并行度；M_act_intra = 峰值激活中的单层内瞬态，
    pp 不削减——OOM 根因为单层组件时必填，否则 PP 方案显存被低估）
  - 并行度映射: p_param = tp×pp×ep；p_state = tp×pp×cp；p_act = tp×pp×cp×sp
  - 通信: α-β 模型 T = S·α + D/带宽；D 为每卡发送量（Ring 带宽项 (P-1)/P）
  - 收益: 计算耗时由 FLOPs 解析估算（线性项 2·N_param·tokens + extra_flops 架构特有项，
    峰值×利用率），估算延迟 = compute/(tp×sp×cp) + 通信合计；pp 在推理 m=1 时
    计算不并行（stage 串行），净收益由报告直接给出，排序键 = 估算延迟
  - η: PP 气泡率 = (P-1)/(P+m-1)，推理 m=1

符号: b/s/h = batch/序列长/hidden dim，L = 层数，L_moe = MoE 层数，token = 每 MoE 层
token 数（默认 b×s），P = 该维度的切分并行度，D = 每卡发送量（Ring 带宽项：
各卡互发自己独有的部分，系数 (P-1)/P），S = 通信步数（串行同步次数），
α = 启动延迟，带宽 = 单链路带宽。消息量 bsh、sh 为维度乘积，字节数按 dtype 折算。

用法:
  python estimate_split.py --spec spec.json             # 估算 spec 内候选
  python estimate_split.py --spec spec.json --cards 8   # 枚举 8 卡全部切分组合

spec.json 示例（完整字段定义——必填/默认/约束——见下方 load_spec 的 docstring；
states 尺寸由 agent 从源码提取；hardware 可省略用典型值）:
  {
    "dims":     {"b": 1, "s": 4096, "h": 4096, "L": 32, "L_moe": 0},
    "bytes":    {"param": 2, "act": 2},
    "states":   {"param": 1.4e10, "state": 6.0e9, "act": 1.2e9, "overhead": 2.0e9,
                 "replicated_param_bytes": 0},
    "hardware": {"hbm": 6.4e10, "alpha_us": 20, "bandwidth_GBs": 392},
    "candidates": [{"tp": 2, "pp": 1, "sp": 1, "cp": 1, "ep": 1}]
  }

注意: 估算为量级判断（α 校准后整体误差 10-30%），用于候选间**相对比较**与剪枝，
绝对耗时以回归 Phase 1 后的实测为准：
  - hardware 典型值（α=20µs、392GB/s，910B 单机 HCCS 量级参考）不含各卡时长不齐的
    同步等待（straggler）——未校准时通信项可低估约 3 倍；真实环境先跑
    `comm_self_test.py --timing` 实测等效 α/带宽，填入 spec 的 hardware 覆盖
  - 计算耗时按 FLOPs 解析估算（峰值 313T × 利用率 0.35 默认，hardware 可覆盖）：
    线性项 2·N_param·tokens 通用（matmul 主导的计算均满足 FLOPs ≈ 2×参数量×输出
    位置数——序列模型 tokens=b·s、CNN=空间位置数、GNN=节点数）；架构特有项填
    extra_flops（attention = 4·L·b·s²·h，GNN 边聚合量级 ≈ 2·E·h·L，其他按源码推导）。
    利用率偏离时误差增大，但误差为候选间共享（同乘系数），不影响相对排序
  - 动态 shape 按上界估；spec 的 dims/states 对应单一 regime，多 regime 负载
    （短/长/批量）应按各主导 regime 分别填 spec 各跑一次
  - 跨维重分布（row↔col）依赖具体模型结构，不在枚举/估算范围内，需要时在报告
    基础上手工追加；枚举不含 sp 维（SP 候选直接写进 candidates）
  - ep 的计算收益未建模（取决于 MoE 负载均衡），报告只给通信成本
"""

import argparse
import json
import math
from pathlib import Path

GB = 1e9
SMALL_MSG = 256 * 1024  # Ring/Tree 交叉点量级：小于此值延迟项主导（Tree，步数少）
TYPICAL_HW = {"alpha_us": 20.0, "bandwidth_GBs": 392.0, "flops_tflops": 313.0, "compute_util": 0.35}  # 910B 量级参考；α/带宽建议实测校准（comm_self_test.py --timing）


def load_spec(path):
    """读取并严格校验 spec.json。

    字段定义（✅ = 必填；右列 = 默认值；字节数除注明外）:
      dims.b / dims.s / dims.h / dims.L      ✅    batch / 序列长 / hidden dim / 层数（第一步读源码）
      dims.L_moe                             0     MoE 层数；为 0 时忽略所有 ep 候选
      dims.token                             b×s   每 MoE 层 token 数
      bytes.param / bytes.act                2     每参数 / 每激活元素字节数（BF16=2，FP32=4）
      states.param                           ✅    参数总字节数（第一步）
      states.replicated_param_bytes          0     各 rank 复制的参数字节数（TP 下 LayerNorm 权重、
                                                  embedding/lm_head 等不被切分整份加载的部分），
                                                  显存计算中该项不除并行度；须 < states.param
      states.state                           0     持久状态总字节数（KV cache 等，第一步）
      states.act                             ✅    峰值激活字节数（内存时间线峰值 ≈ 跨层常驻 + 单层内峰值）
      states.act_intra                       0     峰值激活中的单层内瞬态部分（attention score、
                                                  MLP 中间量等；pp 不削减它，只被 tp/sp/cp 切）。
                                                  OOM 根因为单层组件时必填——缺省时 PP 方案
                                                  显存被低估（全局达标但根因组件仍超限）
      states.overhead                        2e9   框架与临时缓冲开销
      extra_flops                           0     架构特有计算量（FLOPs/步，与参数量不成比例的部分，
                                                  agent 依源码/论文推导）：attention 模型填
                                                  4·L·b·s²·h（QK^T+AV 得分计算）；GNN 消息传递
                                                  量级 ≈ 2·E·h·L（E=边数）；CNN/MLP 等纯
                                                  matmul 主导架构不填（线性项已覆盖）
      （计算耗时不是 spec 字段——由脚本按 FLOPs 解析估算，hardware.flops_tflops /
       compute_util 可调；解析估算不偏向任何候选，PP 仍按估算延迟正确垫底）
      hardware.hbm                           64e9  单卡 HBM 字节数
      hardware.alpha_us                      20    通信启动延迟 µs（典型值，可覆盖；建议实测校准）
      hardware.bandwidth_GBs                 392   单链路带宽 GB/s（典型值，可覆盖；建议实测校准）
      hardware.flops_tflops                  313   单卡峰值算力 TFLOPS（910B 量级参考，可覆盖）
      hardware.compute_util                  0.35  计算利用率（经验值，可覆盖；解析估算用）
      candidates                             —     显式候选列表；缺省时必须给 --cards
      candidates[].tp/pp/sp/cp/ep         1     各维度切分并行度（tp 建议 2 的幂）

    缺必填字段、非法值、未知字段（顶层与嵌套均查）一律报错退出并指明字段名。
    """
    try:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"spec.json 不是合法 JSON: {e}")
    if not isinstance(spec, dict):
        raise SystemExit("spec.json 顶层必须是 JSON 对象")
    dims, states = spec.get("dims"), spec.get("states")
    if not isinstance(dims, dict):
        raise SystemExit("缺少 dims 对象（必填: dims.b / dims.s / dims.h / dims.L）")
    if not isinstance(states, dict):
        raise SystemExit("缺少 states 对象（必填: states.param / states.act）")
    for key in ("b", "s", "h", "L"):
        if not isinstance(dims.get(key), (int, float)) or dims[key] <= 0:
            raise SystemExit(f"dims.{key} 必填且必须为正数")
    extra_flops = spec.get("extra_flops", 0.0)
    if not isinstance(extra_flops, (int, float)) or extra_flops < 0:
        raise SystemExit("extra_flops 必须为非负数（架构特有计算量，FLOPs/步）")
    for key in ("param", "act"):
        if not isinstance(states.get(key), (int, float)) or states[key] <= 0:
            raise SystemExit(f"states.{key} 必填且必须为正数（字节数，读源码提取）")
    act_intra = states.get("act_intra", 0.0)
    if not isinstance(act_intra, (int, float)) or act_intra < 0:
        raise SystemExit("states.act_intra 必须为非负数（字节数）")
    if act_intra > states["act"]:
        raise SystemExit("states.act_intra 必须小于等于 states.act（是峰值的组成部分）")
    allowed_top = {"dims", "bytes", "states", "hardware", "candidates", "extra_flops"}
    unknown = set(spec) - allowed_top
    if unknown:
        raise SystemExit(f"spec 含未知顶层字段 {sorted(unknown)}（只允许 {sorted(allowed_top)}）")
    allowed_nested = {
        "dims": {"b", "s", "h", "L", "L_moe", "token"},
        "states": {"param", "state", "act", "act_intra", "overhead", "replicated_param_bytes"},
        "bytes": {"param", "act"},
        "hardware": {"hbm", "alpha_us", "bandwidth_GBs", "flops_tflops", "compute_util"},
    }
    for section, keys in allowed_nested.items():
        obj = spec.get(section, {})
        if isinstance(obj, dict):
            unknown = set(obj) - keys
            if unknown:
                raise SystemExit(f"{section} 含未知字段 {sorted(unknown)}（只允许 {sorted(keys)}）")
    for key in ("flops_tflops", "compute_util"):
        v = spec.get("hardware", {}).get(key)
        if v is not None and (not isinstance(v, (int, float)) or v <= 0):
            raise SystemExit(f"hardware.{key} 必须为正数")
    bytes_, hw = spec.get("bytes", {}), spec.get("hardware", {})
    replicated = states.get("replicated_param_bytes", 0.0)
    if not isinstance(replicated, (int, float)) or replicated < 0:
        raise SystemExit("states.replicated_param_bytes 必须为非负数（字节数）")
    if replicated >= states["param"]:
        raise SystemExit("states.replicated_param_bytes 必须小于 states.param（复制参数是总参数的子集）")
    d = {
        "b": dims["b"], "s": dims["s"], "h": dims["h"], "L": dims["L"],
        "L_moe": dims.get("L_moe", 0), "token": dims.get("token", dims["b"] * dims["s"]),
        "param_bytes": states["param"], "replicated_param_bytes": replicated,
        "state_bytes": states.get("state", 0.0),
        "act_bytes": states["act"], "act_intra": act_intra,
        "overhead": states.get("overhead", 2.0 * GB),
        "elem_param": bytes_.get("param", 2), "elem_act": bytes_.get("act", 2),
        "hbm": hw.get("hbm", 64.0 * GB),
        "alpha": hw.get("alpha_us", TYPICAL_HW["alpha_us"]) * 1e-6,
        "bw": hw.get("bandwidth_GBs", TYPICAL_HW["bandwidth_GBs"]) * 1e9,
        "flops_tflops": hw.get("flops_tflops", TYPICAL_HW["flops_tflops"]),
        "compute_util": hw.get("compute_util", TYPICAL_HW["compute_util"]),
        "extra_flops": extra_flops,
    }
    return d, spec


def enumerate_candidates(cards, has_moe):
    """枚举 tp×pp×cp×ep 恰好用满 cards 的组合（tp 取 2 的幂，sp 固定 1）。

    SP/混合维候选不自动枚举——直接写进 spec 的 candidates 数组。
    """
    def divisors(n):
        return [i for i in range(1, n + 1) if n % i == 0]

    out = []
    tp_options = [p for p in divisors(cards) if p & (p - 1) == 0]
    for tp in tp_options:
        for pp in divisors(cards // tp):
            for cp in divisors(cards // tp // pp):
                eps = divisors(cards // tp // pp // cp) if has_moe else [1]
                for ep in eps:
                    if tp * pp * cp * ep == cards:
                        out.append({"tp": tp, "pp": pp, "sp": 1, "cp": cp, "ep": ep})
    return out


def memory(d, c):
    """每卡显存分解——可行性硬约束 M < HBM × 0.8。

    公式: M = (P_param − P_repl)/p_param + P_repl + S/p_state
              + (A − A_intra)/p_act + A_intra/(tp·sp·cp) + O

    并行度映射（= analysis_workflow 第三步的规则）：
    p_param = tp×pp×ep —— 切权重维三者同切 / 按层分布随层分布 / 切分支只切分支参数
    p_state = tp×pp×cp —— 切状态增长维（CP）不切参数，state 只被 tp/pp/cp 切
    p_act   = tp×pp×cp×sp —— 激活额外被切归约维（SP）切
    P_repl 不除并行度（各卡整份加载的 LayerNorm 权重 / embedding / lm_head 等）。
    A_intra（峰值激活中的单层内瞬态）不除 pp——它整份驻留在拥有该层的那张卡上，
    只被 tp/sp/cp 切；根因为单层组件 OOM 时不填会低估 PP 方案显存。
    """
    p_param = c["tp"] * c["pp"] * c["ep"]
    p_state = c["tp"] * c["pp"] * c["cp"]
    p_act = c["tp"] * c["pp"] * c["cp"] * c["sp"]
    p_act_nopp = c["tp"] * c["sp"] * c["cp"]
    parts = {
        "param": (d["param_bytes"] - d["replicated_param_bytes"]) / p_param
                 + d["replicated_param_bytes"],
        "state": d["state_bytes"] / p_state,
        "act": (d["act_bytes"] - d["act_intra"]) / p_act
               + d["act_intra"] / p_act_nopp,
        "overhead": d["overhead"],
    }
    return parts, p_param, p_state, p_act


def comm_items(d, c):
    """逐切分动作生成通信事件表 [(名称, 次数/步, 每卡发送量D, 并行度P, 类型)]。

    消息量基准 msg = b·s·h·elem_act（整份激活字节数）；D 为每卡发送量，
    系数 (P−1)/P 为 Ring 带宽项下界（每卡只发自己独有的部分）。

    与 analysis_workflow 第三步通信表逐行对应（推理仅前向，频率为训练的一半）：
    - 切权重维（TP 标准配对，列+行配对仅一处）: 每层 1 次 AllReduce，D = (P-1)/P × bsh
      注意: 常见实现每层 2 次 AR（attn.out 与 FFN.down 各一次，未做进一步配对合并）——
      若计划如此实现，通信估算按 ×2 理解（或 dims.L 填 2L），以实测为准
    - 切归约维（SP）: 每层 1 次 AG+RS（拆开的 AllReduce），D 同 TP
    - 切长归约轴（CP）: 环形 P 步，每步 D = sh/P（总发送量 = sh）
    - 切层维（PP）: 每 stage 边界 1 次 P2P，D = bsh；另有气泡率计入 η（见 render）
    - 切分支维（EP）: 每 MoE 层 1 次 AllToAll，D = token×h
    - 跨维重分布（row↔col）: 依赖具体模型结构，不在本表——需手工追加
    """
    msg = d["b"] * d["s"] * d["h"] * d["elem_act"]
    items = []
    if c["tp"] > 1:
        items.append(("切权重维 AllReduce", d["L"], (c["tp"] - 1) / c["tp"] * msg, c["tp"], "coll"))
    if c["sp"] > 1:
        items.append(("切归约维 AG+RS", d["L"], (c["sp"] - 1) / c["sp"] * msg, c["sp"], "coll"))
    if c["cp"] > 1:
        items.append(("切长归约轴 环形P2P", c["cp"], d["s"] * d["h"] * d["elem_act"] / c["cp"], c["cp"], "p2p"))
    if c["pp"] > 1:
        items.append(("切层维 P2P", c["pp"], msg, c["pp"], "p2p"))
    if c["ep"] > 1 and d["L_moe"] > 0:
        items.append(("切分支维 AllToAll", d["L_moe"], d["token"] * d["h"] * d["elem_act"], c["ep"], "a2a"))
    return items


def steps(op, p, nbytes):
    """通信步数 S（α-β 模型的延迟项系数，串行同步次数）。

    p2p: 单跳 1 步；AllToAll: log2(P) 步。
    集合通信（AR/AG/RS）按消息大小二选一——延迟下界 Ω(log P) 与带宽下界
    Ω((P-1)/P)·M 不可同时达到，小消息走 Tree（2·log2(P) 步，延迟最优），
    大消息走 Ring（2(P-1) 步，带宽最优；D 已按 (P-1)/P 取带宽项，即下界）。
    交叉点量级为几十~几百 KB，此处取 256KB；通信库（HCCL）实际自动选择，
    本函数的 S 仅用于估算。
    """
    if op == "p2p":
        return 1
    if op == "a2a":
        return max(1, math.ceil(math.log2(p)))
    if nbytes < SMALL_MSG:  # 小消息：Tree（延迟最优）
        return 2 * max(1, math.ceil(math.log2(p)))
    return 2 * (p - 1)      # 大消息：Ring（带宽最优）


def _analytic_compute_ms(d):
    """单卡计算耗时解析估算。

    公式: FLOPs   = 2·N_param·tokens    —— 线性项（通用：matmul 主导的计算均满足
                                          FLOPs ≈ 2×参数量×输出位置数。序列模型 tokens=b·s，
                                          CNN tokens=空间位置数 H×W，GNN tokens=节点数）
                  + extra_flops         —— 架构特有项（与参数量不成比例的计算，agent 依
                                          源码/论文推导：attention = 4·L·b·s²·h，
                                          GNN 消息传递量级 ≈ 2·E·h·L，E=边数）
           compute = FLOPs / (峰值 TFLOPS × 1e12 × 利用率) × 1e3   [ms]

    误差为候选间共享（同乘系数），不影响相对排序；利用率偏离时经
    hardware.flops_tflops / compute_util 校正。
    """
    n_param = d["param_bytes"] / d["elem_param"]
    flops = 2 * n_param * d["b"] * d["s"] + d["extra_flops"]
    return flops / (d["flops_tflops"] * 1e12 * d["compute_util"]) * 1e3


def evaluate(d, c):
    parts, p_param, p_state, p_act = memory(d, c)
    m_total = sum(parts.values())
    rows, comm_total = [], 0.0
    for name, count, dd, p, op in comm_items(d, c):
        s = steps(op, p, dd)
        t = s * d["alpha"] + dd / d["bw"]
        rows.append((name, count, dd, s, t, count * t))
        comm_total += count * t
    # 收益侧（目标函数）: est_latency = compute/(tp·sp·cp) + Σ(次数 × t)，
    # 其中 t = S·α + D/带宽（comm_items / steps 生成，α-β 模型）。
    # 计算被 tp×sp×cp 并行（compute_ms 为 FLOPs 解析估算，main 中已解析）；
    # pp 不进分母——m=1 推理 stage 串行、计算不并行（仅引入 P2P 开销与气泡）。
    # 单位：comm_total 为秒（×1e3 转 ms），compute_ms 为毫秒。
    compute_p = c["tp"] * c["sp"] * c["cp"]
    est_latency = d["compute_ms"] / compute_p + comm_total * 1e3
    return {"cand": c, "parts": parts, "p": (p_param, p_state, p_act),
            "m_total": m_total, "rows": rows, "comm_total": comm_total,
            "est_latency": est_latency}


def fmt_gb(x):
    return f"{x / GB:.2f}GB"


def render(d, r, idx):
    c = r["cand"]
    p_param, p_state, p_act = r["p"]
    limit = d["hbm"] * 0.8
    ok = r["m_total"] < limit
    tag = " ".join(f"{k}={v}" for k, v in c.items() if v > 1) or "单卡(基准)"
    act_disp = (f"act/{p_act}" if d["act_intra"] == 0 else
                f"act[常驻/{p_act}+层内/{c['tp'] * c['sp'] * c['cp']}]")
    lines = [f"[候选 {idx}] {tag}",
             f"  显存: param/{p_param} {fmt_gb(r['parts']['param'])} + state/{p_state} "
             f"{fmt_gb(r['parts']['state'])} + {act_disp} {fmt_gb(r['parts']['act'])} "
             f"+ overhead {fmt_gb(r['parts']['overhead'])} = {fmt_gb(r['m_total'])} "
             f"(限值 {fmt_gb(limit)}) -> {'可行' if ok else 'OOM 不可行'}"]
    if r["rows"]:
        lines.append("  通信(每步):")
        for name, count, dd, s, t, tt in r["rows"]:
            lines.append(f"    {name}: D={dd / 1e6:.1f}MB 步数S={s} "
                         f"单次{t * 1e3:.3f}ms ×{count}次 = {tt * 1e3:.3f}ms")
        lines.append(f"    合计 ≈ {r['comm_total'] * 1e3:.3f} ms/步")
    else:
        lines.append("  通信: 零通信（单卡基准）")
    gain = 1.0 - r["est_latency"] / d["compute_ms"]
    lines.append(f"  收益: 估算延迟 ≈ {r['est_latency']:.1f} ms/步 "
                 f"(计算 {d['compute_ms']:.1f}/{c['tp'] * c['sp'] * c['cp']} + "
                 f"通信 {r['comm_total'] * 1e3:.1f})，净收益 ≈ {gain:+.0%}")
    if c["pp"] > 1:
        lines.append(f"  η: PP 气泡率 = (P-1)/P = {(c['pp'] - 1) / c['pp']:.0%}"
                     f"（m=1，单请求/单 microbatch）——计算不并行（stage 串行），"
                     f"推理单请求通常零收益，仅 m>>P（批量/服务化并发）时考虑")
    if c["tp"] & (c["tp"] - 1):
        lines.append("  警告: tp 非 2 的幂，HCCL 集合通信优化差")
    if d["state_bytes"] > 0.4 * r["m_total"] and p_state == 1 and c["cp"] == 1:
        lines.append("  提示: 持久状态占比高但未切（p_state=1），考虑 cp")
    if d["act_intra"] > 0.4 * r["m_total"] and c["pp"] > 1 and c["tp"] * c["sp"] * c["cp"] == 1:
        lines.append("  提示: 单层内峰值占比高但仅用 PP——层内峰值不被 pp 削减，考虑 tp/sp/cp")
    return ok, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="spec.json 路径")
    parser.add_argument("--cards", type=int, default=None,
                        help="总卡数；不给则只估算 spec.candidates")
    args = parser.parse_args()

    d, spec = load_spec(args.spec)
    d["compute_ms"] = _analytic_compute_ms(d)
    cand_keys = {"tp", "pp", "sp", "cp", "ep"}
    for c in spec.get("candidates", []):
        unknown = set(c) - cand_keys
        if unknown:
            raise SystemExit(f"candidates 含未知字段 {sorted(unknown)}（只允许 {sorted(cand_keys)}）")
        for k, v in c.items():
            if not isinstance(v, int) or v < 1:
                raise SystemExit(f"candidates[].{k} 必须为 >=1 的整数")
    cands = [dict({"tp": 1, "pp": 1, "sp": 1, "cp": 1, "ep": 1}, **c)
             for c in spec.get("candidates", [])]
    if args.cards:
        cands = enumerate_candidates(args.cards, d["L_moe"] > 0)
        print(f"注: --cards 枚举 {args.cards} 卡组合，忽略 spec 中的 candidates\n")
    if not cands:
        raise SystemExit("无候选：spec 未给 candidates 且未指定 --cards")

    print(f"硬件: HBM={fmt_gb(d['hbm'])} α={d['alpha'] * 1e6:.0f}µs 带宽={d['bw'] / GB:.0f}GB/s"
          + ("" if "hardware" in spec else " (典型值，spec 可覆盖)"))
    print(f"模型: L={d['L']} b={d['b']} s={d['s']} h={d['h']}"
          + (f" L_moe={d['L_moe']}" if d["L_moe"] else ""))
    extra_note = f" + extra_flops {d['extra_flops'] / 1e12:.1f}T" if d["extra_flops"] else ""
    print(f"计算: {d['compute_ms']:.1f} ms/步（FLOPs 解析估算：线性项{extra_note}，峰值 "
          f"{d['flops_tflops']:.0f}T × 利用率 {d['compute_util']:.2f}，hardware 可覆盖）\n")

    results = [evaluate(d, c) for c in cands]
    feasible = []
    for i, r in enumerate(results, 1):
        ok, lines = render(d, r, i)
        if ok:
            feasible.append((r, i))
        print("\n".join(lines) + "\n")

    print("=" * 60)
    if not feasible:
        print("无可行候选：全部 OOM。回第二步换切分维度，或检查状态尺寸估计。")
        return
    print(f"可行候选按估算延迟排序（净收益 = 1 − 估算延迟/{d['compute_ms']:.1f}ms，"
          f"估算为相对比较，绝对值以实测为准）:")
    ranked = sorted(feasible, key=lambda x: x[0]["est_latency"])
    best = ranked[0][0]["est_latency"]
    for rank, (r, i) in enumerate(ranked, 1):
        c = r["cand"]
        tag = " ".join(f"{k}={v}" for k, v in c.items() if v > 1) or "单卡"
        gain = 1 - r["est_latency"] / d["compute_ms"]
        notes = []
        if c["pp"] > 1:
            notes.append("PP 计算零并行（m=1）")
        if gain <= 0:
            notes.append("净收益≤0，剪掉")
        elif rank > 1 and r["est_latency"] <= best * 1.3:
            notes.append("≈同档（差异<30% 误差带，可用工程量决胜）")
        print(f"  {rank}. [候选 {i}] {tag}: 显存 {fmt_gb(r['m_total'])}, "
              f"估算延迟 {r['est_latency']:.1f}ms (净收益 {gain:+.0%})"
              + (f"——{'；'.join(notes)}" if notes else ""))
    top_r, top_i = ranked[0]
    top_tag = " ".join(f"{k}={v}" for k, v in top_r["cand"].items() if v > 1) or "单卡"
    top_gain = 1 - top_r["est_latency"] / d["compute_ms"]
    if top_gain <= 0:
        print(f"\n结论: 全部可行候选净收益 ≤ 0——仅当显存容量是唯一目标时可选延迟损失最小者"
              f"（[候选 {top_i}] {top_tag}）；延迟目标下回第二步换切分维度。")
    else:
        print(f"\n推荐: [候选 {top_i}] {top_tag}（净收益 {top_gain:+.0%}）。"
              "差异 <30%（估算误差带）的 ≈同档 候选可用实现工作量决胜；"
              "以工程量为由选择差距超误差带的候选属于误用——工程量是并列决胜项，不是方案选择依据。")
    print("\nschema 记录: actual_comm_bytes 取各动作 D×次数之和；本估算的 D 即 Ring 带宽项 "
          "(P-1)/P×消息量，ratio = actual/lower_bound = 1.00。")


if __name__ == "__main__":
    main()
