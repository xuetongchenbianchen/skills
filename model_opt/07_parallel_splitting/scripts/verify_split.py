#!/usr/bin/env python3
"""切分正确性验证模板脚本（第五步）。

★ 本脚本是模板：AGENT 必须基于本脚本填写模型相关代码，禁止重写比较框架。
★ 防作弊设计：
  - 输入确定性：sample 由固定逻辑/种子生成，单卡/多卡用同一份输入
  - 基线落盘：单卡输出保存到磁盘，多卡加载后对比（不重新生成基线）
  - 分层强制：Tier 1 → 2a → 2b 顺序执行，前一层不过不进下一层
  - 阈值锁定：atol/rtol 由切分类型决定，运行时不可放宽
  - bit-exact 自动触发：tolerance 未通过时自动定位首个差异元素
★ 样本纪律（耗时控制）：固定 1 条代表性真实样本 + 1 条最小冒烟输入。
  切分验证只回答"切分实施是否正确"（张量分布 / 通信 / 浮点容差），不承担
  全量精度回归——切分通过后回归 Phase 0 的「0.5 精度验证」（golden 对齐）
  承担；Phase 4 门禁服务的是其后的优化循环，与切分验证无关。代表性样本的选择标准：
  1) 覆盖被切维度的敏感特征：切 batch / 含 padding → 须含 padding 边界；
     切 seq → 生产代表性长度；切权重 → 任一常规样本即可
  2) 规模取生产主流 regime 的代表（中位），不取最大——切分 bug 靠
     维覆盖（padding/长度/被切维边界）暴露，不靠样本数量

用法:
  # 1. 单卡采集基线（切分前，原始模型）
  python verify_split.py --mode baseline --split-type order_preserved

  # 2. 多卡验证（切分后，并行模型）
  torchrun --nproc_per_node=N verify_split.py --mode verify --split-type order_preserved

split-type 决定 atol:
  order_preserved (切batch / fp32切seq / 切层)     → atol=1e-6
  order_changed    (bf16切seq / 切权重 / 切专家)   → atol=1e-3

依赖: comm/comm_primitives.py（init_distributed, ParallelConfig）
"""

import argparse, json, sys
from pathlib import Path
import torch

# ═══════════════════════════════════════════════════════════
# AGENT 填写区（填写以下 4 个函数，禁止修改比较框架）
# ═══════════════════════════════════════════════════════════

def build_model():
    """构建模型并加载权重，返回 model（.eval() 由框架调用）。
    baseline 模式: 构建原始未切分模型。
    verify 模式:   构建切分后模型（框架已 init_distributed；enable_parallel()
                  由框架在 build_model 之后自动调用）。
    """
    raise NotImplementedError("agent 填写：模型构建 + 权重加载")

def build_sample(sample_id: str):
    """构建确定性输入样本。同一 sample_id 必须返回完全相同的输入。
    ★ 禁止使用未设种子的随机数；如需随机，用 torch.manual_seed(42) 等固定种子。
    ★ real_1 是唯一一条代表性真实样本，按文件头「样本纪律」选择：
      覆盖被切维度的敏感特征（padding 边界 / 代表性长度），生产中位规模，
      不取最大。禁止为"测得全"而追加样本——全量精度回归属回归后的 Phase 0「0.5 精度验证」。
    """
    raise NotImplementedError("agent 填写：确定性输入构建")

def run_inference(model, sample):
    """执行一次前向推理，返回模型输出（Tensor / dict / list 均可）。
    verify 模式下需保证输出已 gather 到完整结果（与单卡输出可对齐）。
    """
    raise NotImplementedError("agent 填写：前向推理调用")

def enable_parallel():
    """启用并行切分（verify 模式下框架自动调用，baseline 模式不调用）。
    通常调用项目中的 enable_parallel() 或注册 monkey-patch。
    ★ agent 必须实现此函数，禁止 no-op——否则切分未生效，验证无意义。
    """
    raise NotImplementedError("agent 填写：启用并行逻辑（如注册 monkey-patch / 切 flag）")

# ═══════════════════════════════════════════════════════════
# 比较框架（★禁止修改★）
# ═══════════════════════════════════════════════════════════

THRESHOLDS = {
    "order_preserved": {"atol": 1e-6, "rtol": 1e-5},
    "order_changed":   {"atol": 1e-3, "rtol": 1e-3},
}

def _detach(obj):
    """将输出中的 Tensor detach 到 CPU float。"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().float()
    return obj

def _is_scalar(x):
    return isinstance(x, (int, float)) or (isinstance(x, torch.Tensor) and x.numel() == 1)

def compare_outputs(a, b, atol, rtol, path="root"):
    """递归比较（Tensor / dict / list / scalar），返回 (passed, details[])。"""
    a, b = _detach(a), _detach(b)
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return False, [f"{path}: shape {tuple(a.shape)} vs {tuple(b.shape)}"]
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        diff = (a - b).abs().max().item() if a.numel() else 0.0
        return ok, [f"{path}: allclose {'✓' if ok else '✗'} (max_diff={diff:.2e}, atol={atol})"]
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False, [f"{path}: keys {set(a)} vs {set(b)}"]
        ok, det = True, []
        for k in a:
            r, d = compare_outputs(a[k], b[k], atol, rtol, f"{path}.{k}")
            ok = ok and r
            det.extend(d)
        return ok, det
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False, [f"{path}: len {len(a)} vs {len(b)}"]
        ok, det = True, []
        for i, (x, y) in enumerate(zip(a, b)):
            r, d = compare_outputs(x, y, atol, rtol, f"{path}[{i}]")
            ok = ok and r
            det.extend(d)
        return ok, det
    ok = (abs(float(a) - float(b)) <= atol + rtol * abs(float(b))
          if _is_scalar(a) and _is_scalar(b) else a == b)
    return ok, [f"{path}: {'✓' if ok else '✗'} ({a} vs {b})"]

def bit_exact_diff(a, b, atol, rtol, path="root"):
    """tolerance 失败时，定位首个超容差元素（与 compare_outputs 同阈值）。返回差异描述或 None。"""
    a, b = _detach(a), _detach(b)
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return f"{path}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
        mask = ~torch.isclose(a, b, atol=atol, rtol=rtol)
        if not mask.any():
            return None
        idx = tuple(int(i) for i in mask.nonzero()[0])
        return f"{path}: @{idx} a={a[idx].item():.6e} b={b[idx].item():.6e}"
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {set(a)} vs {set(b)}"
        for k in a:
            r = bit_exact_diff(a[k], b[k], atol, rtol, f"{path}.{k}")
            if r:
                return r
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = bit_exact_diff(x, y, atol, rtol, f"{path}[{i}]")
            if r:
                return r
    elif a != b:
        return f"{path}: 类型或值不等 ({type(a).__name__}={a} vs {type(b).__name__}={b})"
    return None

# ═══════════════════════════════════════════════════════════
# 验证分层（★禁止修改★ — 前层不过不进下一层）
# ═══════════════════════════════════════════════════════════

SMOKE_ID = "smoke"              # 最小输入：管线 sanity + shape 断言（agent 在 build_sample 中实现）
REAL_IDS = ["real_1"]           # 唯一一条代表性真实样本（选择标准见文件头「样本纪律」）

def _collect_baseline(baseline_dir):
    """单卡采集：对每个 sample 跑推理，保存输出。"""
    from comm.comm_primitives import init_distributed, ParallelConfig
    init_distributed()
    cfg = ParallelConfig.get()
    assert not cfg.is_parallel, "baseline 必须单卡运行（world_size=1）"
    model = build_model().to(cfg.device).eval()
    results = {}
    with torch.no_grad():
        for sid in [SMOKE_ID] + REAL_IDS:
            results[sid] = _detach(run_inference(model, build_sample(sid)))
    out = Path(baseline_dir) / "baseline_outputs.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, out)
    print(f"[baseline] 保存 {len(results)} 样本 → {out}")

def _verify(baseline, threshold, baseline_dir, report_path):
    """多卡验证：加载基线，跑切分后推理，逐层对比。仅 rank 0 比较+写报告，
    结果广播到所有 rank 确保同步退出。"""
    from comm.comm_primitives import init_distributed, ParallelConfig
    import torch.distributed as dist
    init_distributed()
    cfg = ParallelConfig.get()
    assert cfg.is_parallel, "verify 必须 torchrun --nproc_per_node>1"
    model = build_model().to(cfg.device).eval()
    enable_parallel()  # 框架强制调用，防止 agent 跳过切分
    atol, rtol = threshold["atol"], threshold["rtol"]
    report = {"threshold": threshold, "tiers": []}

    def _compare(sid, tier):
        sample = build_sample(sid)
        with torch.no_grad():
            out = run_inference(model, sample)
        if cfg.rank == 0:
            passed, details = compare_outputs(out, baseline[sid], atol, rtol)
            entry = {"tier": tier, "sample": sid, "passed": passed, "details": details}
            if not passed:
                entry["bit_exact"] = bit_exact_diff(_detach(out), baseline[sid],
                                                    atol, rtol) or "未定位到"
            report["tiers"].append(entry)
        else:
            passed = None
        # 广播结果到所有 rank，确保同步退出（防止 rank 0 失败后其他 rank 挂起）
        result = [passed]
        dist.broadcast_object_list(result, src=0)
        return result[0]

    # Tier 1 + 2a 合并: 最小输入跑通 + tolerance
    t1 = _compare(SMOKE_ID, "Tier1-smoke+2a")
    if not t1:
        if cfg.rank == 0:
            report["overall"] = False
            _write_report(report, report_path)
        print("[FAIL] Tier 1 冒烟未通过，终止")
        sys.exit(1)

    # Tier 2b: 单条代表性真实样本（见文件头「样本纪律」）
    t2b = True
    for sid in REAL_IDS:
        r = _compare(sid, "Tier2b-real")
        if not r:
            t2b = False
            break

    overall = bool(t1) and t2b
    if cfg.rank == 0:
        report["overall"] = overall
        _write_report(report, report_path)
    print(f"[{'PASS' if overall else 'FAIL'}] "
          f"{'全部通过 ✓' if overall else '验证未通过，详见 report'}")
    sys.exit(0 if overall else 1)

def _write_report(report, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[report] → {p}")

# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="切分正确性验证模板（第五步）")
    ap.add_argument("--mode", choices=["baseline", "verify"], required=True)
    ap.add_argument("--split-type", choices=list(THRESHOLDS), required=True,
                    help="order_preserved=1e-6 / order_changed=1e-3（必填，防止静默落入宽松档）")
    ap.add_argument("--baseline-dir", default="./verify_baseline")
    ap.add_argument("--report", default="verify_report.json")
    args = ap.parse_args()
    threshold = THRESHOLDS[args.split_type]
    if args.mode == "baseline":
        _collect_baseline(args.baseline_dir)
    else:
        baseline = torch.load(Path(args.baseline_dir) / "baseline_outputs.pt",
                              map_location="cpu", weights_only=False)
        _verify(baseline, threshold, args.baseline_dir, args.report)

if __name__ == "__main__":
    main()
