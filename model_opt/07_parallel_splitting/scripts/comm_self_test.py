#!/usr/bin/env python3
"""通信原语环境自检——原地运行，不 copy 到项目。

验证多卡通信环境本身是否正常（HCCL/NCCL 初始化、基础原语、组合函数往返）。
调试排查第一步: 本脚本通过 -> 问题在切分实现逻辑；本脚本失败 -> 环境/部署问题。

用法:
  torchrun --nproc_per_node=N comm_self_test.py            # 环境自检（默认）
  torchrun --nproc_per_node=N comm_self_test.py --timing   # 通信参数校准（α/带宽）

--timing 模式测量真实环境下的等效通信参数（含各卡时长不齐的同步等待），
输出可直接填入 estimate_split.py 的 spec.hardware 覆盖——未校准时估算器的
典型值（α=20µs）可低估实际 AllReduce 耗时约 3 倍（straggler 等待为主因）。
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.distributed as dist

from comm_primitives import (init_distributed, ParallelConfig, all_gather,
                             all_gather_into_tensor, all_reduce, reduce,
                             reduce_scatter, broadcast, gather, scatter,
                             all_to_all)
from comm_recipes import (local_chunk, allgather_along_dim, row_to_col,
                          col_to_row)


def _bench_all_reduce(nbytes, iters=20, warmup=5):
    """计时一次 AllReduce（fp32），返回每轮秒数（含同步等待，取中位数）。"""
    cfg = ParallelConfig.get()
    t = torch.ones(max(1, nbytes // 4), dtype=torch.float32, device=cfg.device)
    for _ in range(warmup):
        all_reduce(t)
    dist.barrier()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        all_reduce(t)
        dist.barrier()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def timing_test():
    """校准等效通信参数：小消息定 α（延迟+等待项），大消息定带宽。"""
    cfg = ParallelConfig.get()
    rank, ws = cfg.rank, cfg.world_size
    small_bytes, large_bytes = 64 * 1024, 64 * 1024 * 1024
    t_small = _bench_all_reduce(small_bytes)
    t_large = _bench_all_reduce(large_bytes)
    if rank != 0:
        return True
    # Ring AllReduce 每卡实际搬运量 ≈ 2·(P-1)/P × 消息量；大消息耗时以此反推带宽
    moved = 2 * (ws - 1) / ws * large_bytes
    eff_bw = moved / t_large / 1e9
    # 小消息耗时 ≈ S·α（Tree S=2·log2(P)），反推等效 α；直接取单次耗时作保守值亦可
    import math
    alpha_us = t_small / (2 * max(1, math.ceil(math.log2(ws)))) * 1e6
    print(f"[timing] world_size={ws}")
    print(f"  小消息 AllReduce ({small_bytes // 1024}KB): {t_small * 1e6:.1f} µs/次")
    print(f"  大消息 AllReduce ({large_bytes // 1024 // 1024}MB): {t_large * 1e3:.3f} ms/次")
    print(f"  等效 α ≈ {alpha_us:.1f} µs（含同步等待）；等效带宽 ≈ {eff_bw:.0f} GB/s")
    print("  建议填入 spec 覆盖典型值:")
    print(f'    "hardware": {{"alpha_us": {round(alpha_us, 1)}, '
          f'"bandwidth_GBs": {round(eff_bw)}}}')
    return True


def self_test():
    """验证基础原语在当前环境下工作正常。

    经 comm_self_test.py 直接运行（torchrun --nproc_per_node=N comm_self_test.py），
    也可在项目调试时单独调用本函数。
    """
    cfg = ParallelConfig.get()
    rank, ws, device = cfg.rank, cfg.world_size, cfg.device
    failures = 0

    def check(name, cond):
        nonlocal failures
        if not cond:
            failures += 1
        print(f"[{'PASS' if cond else 'FAIL'}] rank{rank} {name}")

    # 1. all_gather
    t = torch.full((4,), float(rank + 1), device=device)
    out_list = [torch.empty_like(t) for _ in range(ws)]
    all_gather(out_list, t)
    check("all_gather",
          [x[0].item() for x in out_list] == [float(i + 1) for i in range(ws)])

    # 2. all_gather_into_tensor
    out_tensor = torch.empty(ws * 4, device=device)
    all_gather_into_tensor(out_tensor, t)
    check("all_gather_into_tensor",
          out_tensor.tolist() == [float(i + 1) for i in range(ws) for _ in range(4)])

    # 3. all_reduce
    t = torch.full((4,), float(rank + 1), device=device)
    all_reduce(t)
    expected = sum(float(i + 1) for i in range(ws))
    check("all_reduce", torch.allclose(t, torch.full((4,), expected, device=device)))

    # 4. reduce (结果仅在 dst)
    t = torch.full((4,), float(rank + 1), device=device)
    reduce(t, dst=0)
    if rank == 0:
        check("reduce", torch.allclose(t, torch.full((4,), expected, device=device)))
    else:
        check("reduce", torch.allclose(t, torch.full((4,), float(rank + 1), device=device)))

    # 5. reduce_scatter
    input_list = [torch.full((4,), float(i + 1), device=device) for i in range(ws)]
    output = torch.empty(4, device=device)
    reduce_scatter(output, input_list)
    check("reduce_scatter", torch.allclose(output, torch.full((4,), expected, device=device)))

    # 6. broadcast
    t = torch.zeros(4, device=device)
    if rank == 0:
        t.fill_(42.0)
    broadcast(t, src=0)
    check("broadcast", torch.allclose(t, torch.full((4,), 42.0, device=device)))

    # 7. gather (结果仅在 dst)
    t = torch.full((4,), float(rank), device=device)
    gl = [torch.empty_like(t) for _ in range(ws)] if rank == 0 else None
    gather(t, gl, dst=0)
    if rank == 0:
        expected = [float(i) for i in range(ws) for _ in range(4)]
        check("gather", torch.cat(gl).tolist() == expected)
    else:
        check("gather", gl is None)

    # 8. scatter (src 分发)
    t = torch.empty(4, device=device)
    sl = [torch.full((4,), float(i * 10), device=device) for i in range(ws)] \
        if rank == 0 else None
    scatter(sl, t, src=0)
    check("scatter", torch.allclose(t, torch.full((4,), float(rank * 10), device=device)))

    # 9. all_to_all
    t = torch.full((4,), float(rank), device=device)
    input_list = [t.clone() for _ in range(ws)]
    output_list = [torch.empty_like(t) for _ in range(ws)]
    all_to_all(output_list, input_list)
    check("all_to_all",
          all(output_list[i][0].item() == float(i) for i in range(ws)))

    # 10. 便捷函数: allgather_along_dim + local_chunk 往返
    full = torch.arange(ws * 4, dtype=torch.float, device=device)
    broadcast(full, src=0)
    local = local_chunk(full, dim=0)
    recovered = allgather_along_dim(local, dim=0)
    check("allgather/local_chunk roundtrip", torch.equal(full, recovered))

    # 11. 便捷函数: row_to_col / col_to_row 往返
    # 往返恒等要求各 rank 输入张量一致（row_to_col 后各块来自各 rank）
    t = torch.zeros(4, ws * 2, 3, device=device)
    if rank == 0:
        t = torch.arange(4 * ws * 2 * 3, dtype=torch.float,
                         device=device).view(4, ws * 2, 3)
    broadcast(t, src=0)
    swapped = row_to_col(t)
    back = col_to_row(swapped)
    check("row_to_col/col_to_row roundtrip", torch.allclose(t, back, atol=1e-6))

    print(f"rank{rank}: {'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return failures == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="通信原语环境自检 / 通信参数校准")
    ap.add_argument("--timing", action="store_true",
                    help="校准模式：测量等效 α 与带宽，输出 spec 覆盖建议（跳过功能自检）")
    args = ap.parse_args()
    init_distributed()
    ok = timing_test() if args.timing else self_test()
    raise SystemExit(0 if ok else 1)
