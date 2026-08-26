#!/usr/bin/env python3
"""通信原语环境自检——原地运行，不 copy 到项目。

验证多卡通信环境本身是否正常（HCCL/NCCL 初始化、基础原语、组合函数往返）。
调试排查第一步: 本脚本通过 -> 问题在切分实现逻辑；本脚本失败 -> 环境/部署问题。

用法:
  torchrun --nproc_per_node=N comm_self_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from comm_primitives import (init_distributed, ParallelConfig, all_gather,
                             all_gather_into_tensor, all_reduce, reduce,
                             reduce_scatter, broadcast, gather, scatter,
                             all_to_all)
from comm_recipes import (local_chunk, allgather_along_dim, row_to_col,
                          col_to_row)


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
    init_distributed()
    ok = self_test()
    raise SystemExit(0 if ok else 1)
