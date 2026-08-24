#!/usr/bin/env python3
"""并行切分通信原语模块。

分层结构:
  1. 基础原语（1:1 对应 torch.distributed，world_size=1 → no-op，NPU → contiguous）
  2. 便捷函数（组合基础原语，面向并行切分场景）
  3. 通信-计算重叠（NPU 专用）
  4. 自检

依赖: torch, torch.distributed; NPU 环境还需 torch_npu（自动检测）。
用法: torchrun --nproc_per_node=N your_script.py
"""

import os
import torch
import torch.distributed as dist


class ParallelConfig:
    """全局配置单例。"""
    _instance = None

    def __init__(self):
        self.is_parallel = False
        self.rank = 0
        self.world_size = 1
        self.device = "cpu"
        self.backend = None
        self.comm_stream = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None


def init_distributed(backend: str = "hccl"):
    """初始化分布式环境。单卡时 no-op。

    ★必须先 import torch_npu 再探测，否则 torch.npu.is_available() 恒 False。
    """
    cfg = ParallelConfig.get()

    try:
        import torch_npu  # noqa: F401
        has_npu = torch.npu.is_available()
    except ImportError:
        has_npu = False
    has_cuda = torch.cuda.is_available()

    world_size_env = os.environ.get("WORLD_SIZE", "1")
    if world_size_env == "1" or int(world_size_env) <= 1:
        if has_npu:
            torch.npu.set_device(0)
        elif has_cuda:
            torch.cuda.set_device(0)
        cfg.is_parallel = False
        cfg.device = "npu" if has_npu else ("cuda" if has_cuda else "cpu")
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if has_npu:
        torch.npu.set_device(local_rank)
        cfg.device = "npu"
    elif has_cuda:
        torch.cuda.set_device(local_rank)
        cfg.device = "cuda"
        backend = "nccl"
    else:
        cfg.device = "cpu"
        backend = "gloo"

    dist.init_process_group(backend)
    cfg.rank = dist.get_rank()
    cfg.world_size = dist.get_world_size()
    cfg.is_parallel = cfg.world_size > 1
    cfg.backend = backend

    if cfg.is_parallel and cfg.device == "npu":
        cfg.comm_stream = torch.npu.Stream()
    elif cfg.is_parallel and cfg.device == "cuda":
        cfg.comm_stream = torch.cuda.Stream()


def _ws(group=None) -> int:
    cfg = ParallelConfig.get()
    return dist.get_world_size(group) if cfg.is_parallel else 1


def _rank(group=None) -> int:
    cfg = ParallelConfig.get()
    return dist.get_rank(group) if cfg.is_parallel else 0


def _global_rank(group, local_rank: int) -> int:
    return dist.get_global_rank(group, local_rank) if group else local_rank


# ──────────────────────────────────────────────────────────
# 1. 基础原语（1:1 对应 torch.distributed API）
#    world_size=1 → no-op; NPU → 自动 .contiguous()
# ──────────────────────────────────────────────────────────

def all_gather(output_list, input_tensor, group=None):
    """AllGather: 每个 rank 获得所有 rank 的张量副本。

    output_list: 预分配的 list[Tensor]，长度 = world_size
    """
    if _ws(group) <= 1:
        output_list[0] = input_tensor
        return
    dist.all_gather(output_list, input_tensor.contiguous(), group=group)


def all_gather_into_tensor(output_tensor, input_tensor, group=None):
    """AllGather 到单 buffer（更高效，峰值内存减半）。"""
    if _ws(group) <= 1:
        output_tensor.copy_(input_tensor)
        return
    dist.all_gather_into_tensor(output_tensor, input_tensor.contiguous(),
                                group=group)


def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None):
    """AllReduce: 跨 rank 归约，结果在所有 rank 上。in-place。"""
    if _ws(group) <= 1:
        return
    dist.all_reduce(tensor.contiguous(), op=op, group=group)


def reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=None):
    """Reduce: 跨 rank 归约，结果仅在 dst rank 上。in-place。

    与 all_reduce 的区别：结果只在 dst，其他 rank 的 tensor 不变。
    """
    if _ws(group) <= 1:
        return
    dist.reduce(tensor.contiguous(), dst=_global_rank(group, dst),
                op=op, group=group)


def reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=None):
    """ReduceScatter: 归约后分片，每个 rank 获得结果的 1/P。

    output: 本 rank 的输出 Tensor
    input_list: 本 rank 的输入 list[Tensor]，长度 = world_size
    """
    if _ws(group) <= 1:
        output.copy_(input_list[0])
        return
    dist.reduce_scatter(output, [t.contiguous() for t in input_list],
                        op=op, group=group)


def broadcast(tensor, src=0, group=None):
    """Broadcast: 从 src rank 广播到所有 rank。in-place。"""
    if _ws(group) <= 1:
        return
    dist.broadcast(tensor.contiguous(), src=_global_rank(group, src),
                   group=group)


def gather(tensor, gather_list, dst=0, group=None):
    """Gather: 收集所有 rank 的张量到 dst rank。

    dst rank: gather_list 被填充
    其他 rank: gather_list 为 None
    """
    if _ws(group) <= 1:
        if gather_list is not None:
            gather_list[0] = tensor
        return
    gl = gather_list if _rank(group) == dst else None
    dist.gather(tensor.contiguous(), gather_list=gl,
                dst=_global_rank(group, dst), group=group)


def scatter(scatter_list, tensor, src=0, group=None):
    """Scatter: src rank 将 scatter_list 中的不同块分发到各 rank。

    src rank: scatter_list 为 list[Tensor]，长度 = world_size
    所有 rank: tensor 被填充为本 rank 收到的块
    """
    if _ws(group) <= 1:
        tensor.copy_(scatter_list[0])
        return
    sl = scatter_list if _rank(group) == src else None
    dist.scatter(tensor, sl, src=_global_rank(group, src), group=group)


def all_to_all(output_list, input_list, group=None):
    """AllToAll: 所有 rank 两两交换数据。

    output_list / input_list: 预分配的 list[Tensor]，长度 = world_size
    ★NPU 要求 input_list 中每个 tensor 是 contiguous 的。
    """
    if _ws(group) <= 1:
        output_list[0] = input_list[0]
        return
    dist.all_to_all(output_list, [t.contiguous() for t in input_list],
                    group=group)


# ──────────────────────────────────────────────────────────
# 2. 便捷函数（组合基础原语，面向并行切分场景）
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
# 3. 通信-计算重叠（NPU 专用）
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
# 4. 自检
# ──────────────────────────────────────────────────────────

def self_test():
    """验证基础原语在当前环境下工作正常。

    用法:
      torchrun --nproc_per_node=4 -c \\
        "from comm.comm_primitives import init_distributed, self_test; \\
         init_distributed(); self_test()"
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
          all(output_list[i][0].item() == float(rank) for i in range(ws)))

    # 10. 便捷函数: allgather_along_dim + local_chunk 往返
    full = torch.arange(ws * 4, dtype=torch.float, device=device)
    broadcast(full, src=0)
    local = local_chunk(full, dim=0)
    recovered = allgather_along_dim(local, dim=0)
    check("allgather/local_chunk roundtrip", torch.equal(full, recovered))

    # 11. 便捷函数: row_to_col / col_to_row 往返
    t = torch.full((4, ws * 2, 3), float(rank), device=device)
    swapped = row_to_col(t)
    back = col_to_row(swapped)
    check("row_to_col/col_to_row roundtrip", torch.allclose(t, back, atol=1e-6))

    print(f"rank{rank}: {'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return failures == 0
