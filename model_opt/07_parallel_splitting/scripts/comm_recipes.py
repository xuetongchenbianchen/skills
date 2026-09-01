#!/usr/bin/env python3
"""切分场景的通信组合函数——菜单式取用，不整文件 copy。

按分析的张量分布约定表挑选需要的函数，copy 进项目的 comm/ 模块，
并把 import 行改为:
    from comm.comm_primitives import _ws, _rank, all_gather, ...

菜单:
  - allgather_along_dim / local_chunk       沿维还原完整张量 / 本地取片（切分入口与出口）
  - reduce_scatter_along_dim                归约后取本 rank 分片（任意维）
  - alltoall_along_dim                      分片交换后拼接（跨维重分布）
  - row_to_col / col_to_row                 分片维 dim=1 <-> dim=0 切换（TP x SP 混合）
  - issue_on_comm_stream / wait_comm        通信-计算重叠工具（供主流程 Phase 3 hide_latency
                                             取用；切分实施阶段不引入）

依赖: 同目录 comm_primitives.py 的基础原语（须先 copy 基座）。
"""

import torch
import torch.distributed as dist

from comm_primitives import (_ws, _rank, ParallelConfig, all_gather,
                             all_gather_into_tensor, reduce_scatter,
                             all_to_all)


def allgather_along_dim(t: torch.Tensor, dim: int = 0,
                        group=None) -> torch.Tensor:
    """AllGather 后沿 dim 拼接: [L/P, ...] → [L, ...]

    dim=0 用 all_gather_into_tensor（单 buffer，峰值减半）。
    """
    ws = _ws(group)
    if ws <= 1:
        return t

    if dim == 0:
        output = torch.empty(ws * t.shape[0], *t.shape[1:],
                             dtype=t.dtype, device=t.device)
        all_gather_into_tensor(output, t, group=group)
        return output
    else:
        tensor_list = [torch.empty_like(t) for _ in range(ws)]
        all_gather(tensor_list, t, group=group)
        return torch.cat(tensor_list, dim=dim)


def local_chunk(t: torch.Tensor, dim: int = 0,
                group=None) -> torch.Tensor:
    """本地取片: [L, ...] → [L/P, ...]（无通信，按 rank 取对应 chunk）。"""
    ws = _ws(group)
    if ws <= 1:
        return t
    chunk = t.chunk(ws, dim=dim)[_rank(group)]
    return chunk.clone() if chunk.is_contiguous() else chunk.contiguous()


def reduce_scatter_along_dim(t: torch.Tensor, dim: int = 0,
                             op=dist.ReduceOp.SUM,
                             group=None) -> torch.Tensor:
    """ReduceScatter 沿任意维度: 归约后取本 rank 对应分片。

    dim=0 直接用 reduce_scatter。其他维度需 permute。
    """
    ws = _ws(group)
    if ws <= 1:
        return t

    if dim == 0:
        output = torch.empty(t.shape[0] // ws, *t.shape[1:],
                             dtype=t.dtype, device=t.device)
        reduce_scatter(output, list(t.chunk(ws, dim=0)), op=op, group=group)
        return output
    else:
        perm = [dim] + [i for i in range(t.ndim) if i != dim]
        inv_perm = [0] * t.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        t_perm = t.permute(perm).contiguous()
        output = torch.empty(t_perm.shape[0] // ws, *t_perm.shape[1:],
                             dtype=t.dtype, device=t.device)
        reduce_scatter(output, list(t_perm.chunk(ws, dim=0)),
                       op=op, group=group)
        return output.permute(inv_perm).contiguous()


def alltoall_along_dim(t: torch.Tensor, dim: int,
                       group=None) -> torch.Tensor:
    """AllToAll 后沿 dim=0 拼接: 各 rank 沿 dim 的分片交换后拼接。

    ★NPU 要求 chunk 后 .contiguous()。
    """
    ws = _ws(group)
    if ws <= 1:
        return t

    input_list = [c.contiguous() for c in t.chunk(ws, dim=dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(ws)]
    all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=0)


def row_to_col(t: torch.Tensor, group=None) -> torch.Tensor:
    """[N/P, N, c] → [N, N/P, c]：分片从 dim=1 移到 dim=0。"""
    return alltoall_along_dim(t, dim=1, group=group)


def col_to_row(t: torch.Tensor, group=None) -> torch.Tensor:
    """[N, N/P, c] → [N/P, N, c]：分片从 dim=0 移到 dim=1。"""
    ws = _ws(group)
    if ws <= 1:
        return t
    input_list = [c.contiguous() for c in t.chunk(ws, dim=0)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(ws)]
    all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=1)


def issue_on_comm_stream(comm_fn):
    """在独立通信流上执行 comm_fn，返回 event 供计算流等待。"""
    cfg = ParallelConfig.get()
    if cfg.comm_stream is None:
        comm_fn()
        return None

    dev = torch.npu if cfg.device == "npu" else torch.cuda
    current = dev.current_stream()
    cfg.comm_stream.wait_stream(current)

    with dev.stream(cfg.comm_stream):
        comm_fn()

    event = dev.Event()
    event.record(cfg.comm_stream)
    return event


def wait_comm(event):
    """等待通信完成。event 为 None 时 no-op。"""
    if event is not None:
        cfg = ParallelConfig.get()
        dev = torch.npu if cfg.device == "npu" else torch.cuda
        dev.current_stream().wait_event(event)
