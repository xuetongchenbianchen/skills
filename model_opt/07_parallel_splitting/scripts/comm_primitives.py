#!/usr/bin/env python3
"""并行切分通信基座——模板，整文件 copy 到项目。

只含与 torch.distributed 1:1 对应的基础原语:
  world_size=1 → no-op（单卡天然兼容）; NPU → 自动 .contiguous()。

切分场景的组合函数（沿维聚合/取片/分片维切换、通信-计算重叠）在
comm_recipes.py——按需取用，不整文件 copy；环境自检在 comm_self_test.py
——原地运行，不 copy。

依赖: torch, torch.distributed; NPU 环境还需 torch_npu（自动检测）。
用法: copy 到项目 comm/comm_primitives.py 后,
      from comm.comm_primitives import init_distributed, ParallelConfig, ...
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

