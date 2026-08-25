"""Centralized thresholds for all parse scripts.

Edit values here instead of hunting through individual scripts.
Loaded via common.threshold(script, key).

NOTE: All numeric thresholds are workload-dependent defaults, not universal
judgments. Tune per model/framework/chip — e.g., fusible_small_us for LLM decode
(short kernels) differs from prefill; suspect_mac_ratio baseline differs across
chip generations. Treat values as starting points, validate against your workload.
"""

THRESHOLDS = {
    "step_trace": {
        "severe_host_bound_util": 20,       # % — below this = severe host-bound
        "moderate_host_bound_util": 50,     # % — below this = moderate host-bound
        "comm_bound_pct": 20,              # % — Comm(Not Overlapped) above this = Comm-Bound
        "bubble_severe_pct": 20,           # % — Bubble/Total above this = severe pipeline stall
        "bubble_moderate_pct": 5,          # % — Bubble/Total above this = moderate pipeline stall
        "step_util_variance": 20,           # % — max-min util difference across steps
        "step_duration_spread": 2.0,        # max/min ratio for step duration outlier
        "large_optimizable_space": 30,      # % — Free/Total above this = large optimizable space
    },

    "op_statistic": {
        "top3_concentration": 80,           # % — top-3 ops > this = concentrated bottleneck
        "move_keywords": [                  # op types classified as data movement
            "Transpose", "Cast", "Copy", "Contiguous", "Reshape", "MemSet", "Format",
        ],
        "data_movement_ratio": 3,           # % — data movement > this = signal
        "frag_count_multiplier": 3,         # x avg count — fragmentation signal
        "frag_max_avg_us": 10,              # us — avg below this + high count = fragmented
        "heavy_max_count": 10,              # count <= this = heavy single-invocation
        "heavy_min_avg_us": 100,            # us — avg above this = heavy
        "heavy_min_ratio": 0.01,            # total/total ratio above this = heavy
    },

    "kernel_details": {
        "suspect_min_duration_us": 10,      # us — kernels above this considered for suspect
        "suspect_mac_ratio": 0.2,           # mac_ratio below this = low compute (AI_CORE)
        "suspect_vec_ratio": 0.05,          # vec_ratio below this = low compute (AI_VECTOR)
        "block_dim_buckets": [8, 28],       # boundaries: 1, 2-8, 9-28, 29+
        "wait_buckets_us": [100, 500, 2000],# boundaries for wait time distribution
        "cube_low_util": 50,                # % — cube utilization below this = low
        "low_parallelism_ratio": 0.1,       # ratio — block_dim=1 duration share above this = signal
        "hw_dominance_ratio": 1.5,          # x — mte>mac*1.5 = memory-dominated, vice versa
        "fusible_small_us": 10.0,           # us — kernels below this = fusible candidate
        "fusible_min_length": 5,            # min consecutive small kernels for a sequence
        "fusible_min_total_us": 100,        # us — min cumulative duration for a sequence
        "compute_bound_mac_ratio": 0.5,     # mac_ratio above this + high dur = true compute-bound (replace/quant target)
        "comm_keywords": [                  # AI_CPU ops that are communication (excluded from fallback)
            "broadcast", "allgather", "alltoall", "allreduce", "hcom", "send", "recv", "reducescatter",
        ],
        "short_kernel_dominant": 60,        # % — short kernel (<20us) ratio above this = dominant
        "median_wait_threshold_us": 100,   # us — median wait above this = universally high wait
        "non_nd_format_ratio": 0.1,        # ratio — non-ND input format above this = layout conversion signal
        "filter_high_wait_multiplier": 3,  # x — wait > avg * this in filter mode = high-wait instance
        "filter_high_wait_min_us": 200,    # us — minimum wait for filter mode high-wait context
    },

    "trace_view": {
        "gap_buckets_us": [10, 50, 200],    # boundaries for gap distribution
        "compute_task_types": [             # task types classified as compute
            "AI_CORE", "AI_VECTOR", "AICORE", "AIVEC", "MIX", "VECTOR",
        ],
        "compile_early_window": 0.2,        # fraction of timeline considered "early"
        "compile_early_frac": 0.8,          # % of compile in early window = Type A (warmup)
        "freq_decrease_ratio": 0.05,        # % — frequency degradation above this = signal
        "sync_co_ratio": 10,                # % — sync/launch ratio above this = signal
        "prefetch_keywords": [              # op names classified as prefetch/prealloc candidates
            "aten::to", "copy_", "aten::copy", "::empty", "empty_",
            "aten::empty", "memcpy", "to_copy", "_to_copy", "pin_memory",
        ],
        "stack_lib_markers": [              # call stack frames matching these are filtered out
            "site-packages", "dist-packages", "/lib/python",
            "torch/nn/modules", "torch/_ops", "autograd/profiler", "torch_npu/profiler",
        ],
        "stack_max_frames": 6,              # max project frames shown in condensed call stack
        "disp_lat_sample_cap": 200000,      # max dispatch latency samples stored
        "compile_ts_cap": 500000,           # max compile timestamps stored
        "dispatch_kernel_ratio": 50,        # % — dispatch/kernel-active above this = significant
        "h2d_gap_threshold_us": 50,         # us — device starts within this of launch = host-bound op
        "h2d_min_run_len": 3,               # min consecutive host-bound ops to form a region
        "h2d_max_runs": 15,                 # max regions reported (sorted by device idle time)
        "h2d_callstack_per_run": 2,         # max distinct op call stacks shown per region
    },

    "memory_record": {
        "frag_gap_mb": 1000,                # MB — large fragmentation gap above this = signal
        "growth_min_records": 20,           # min records for growth trend analysis
        "growth_mb": 100,                   # MB — growth above this between early/late = signal
        "churn_jump_mb": 50,                # MB — jumps above this = large
        "churn_count": 20,                  # count of large jumps above this = high churn
        "frag_growth_mb": 50,               # MB — fragmentation growth above this = signal
        "oom_risk_mb": 60000,               # MB — reserved above this = OOM risk
    },

    "operator_memory": {
        "short_life_us": 1000000,           # 1s — tensors alive < this = short-lived (waste)
        "short_lived_min_kb": 100,          # KB — size above this for short-lived tracking
        "short_lived_max_life_us": 1000,    # us — lifetime below this for short-lived tracking
        "size_track_min_kb": 10,            # KB — size above this for repeated alloc tracking
        "repeated_count": 10,               # count above this = repeated alloc signal
        "churn_total_kb": 10000,            # KB — short-lived total above this = churn signal
        "dominate_ratio": 0.5,              # ratio above this = single op dominates
        "parallelism_ratio": 0.8,           # projected peak / HBM above this = parallelism trigger
    },

    "operator_details": {
        "pure_host_pct": 50,                # % — pure host ops above this = DEFINITE signal
        "extreme_hd_ratio": 10,             # x — host > device * this = extreme ratio
        "extreme_host_us": 5000,            # us — host above this for extreme ratio
        "hd_ratio_display_cap": 10000,      # display "∞" above this
        "aicpu_fallback_min_device_us": 1000, # us — device_us above this for AI_CPU fallback detection
        "aicpu_fallback_aicore_ratio": 0.5, # ratio — AICore/device below this = AI_CPU fallback
        "sync_dominance_pct": 20,           # % — sync category above this of total host = dominant
        "other_breakdown_pct": 10,          # % — other category above this = auto-breakdown
        # Host category classification rules (C1). Classify by op ROLE, not name —
        # these patterns are framework defaults, adjust per model/framework. Order
        # matters: first match wins (sync > alloc > H2D > dispatch-cann > dispatch-aten > framework > compile).
        # Key: alloc and H2D must come BEFORE dispatch(aten) so that aten::slice
        # matches alloc (not dispatch) and aten::to matches H2D (not dispatch).
        "host_category_rules": {
            "sync (D-to-H)": ["_local_scalar", "::item", ".item", "numpy"],
            "alloc/metadata": ["empty", "as_strided", "view", "reshape", "clone",
                               "contiguous", "detach", "expand", "squeeze", "unsqueeze",
                               "slice", "select", "resize_"],
            "H2D/D2H copy": ["copy_", "_to_copy", "to_copy", "memcpy", "::to"],
            "dispatch (CANN aclnn)": ["aclnn"],
            "dispatch (PyTorch aten)": ["aten::"],
            "framework/comm": ["c10d::", "profiler", "broadcast_"],
            "compile": ["compile", "opcompile"],
        },
    },

    "communication": {
        # --- 摘要与类型分解 ---
        "wait_dominant_ratio": 0.8,         # wait/total above this = DEFINITE sync-bound
        "per_type_wait_ratio": 0.9,         # per-type wait ratio above this = SIGNAL
        "per_type_min_count": 10,           # min count for per-type signal
        "small_packet_ratio": 0.3,          # 延迟主导消息占比超此 → [SIGNAL]（M2 小包判定）
        # --- M1 溯源 ---
        "trace_top_k": 3,                   # 默认报告自动溯源的算子数
        "comm_host_op_map": {               # hcom → Hccl host 名映射（首选名在前；计数匹配者用于对齐）
            "hcom_allGather": ["HcclAllGather", "HcclAllgatherBase"],
            "hcom_allReduce": ["HcclAllReduce"],
            "hcom_alltoall": ["HcclAllToAllV", "HcclAllToAll"],
            "hcom_reduceScatter": ["HcclReduceScatterV", "HcclReduceScatter"],
            "hcom_broadcast": ["HcclBroadcast"],
            "hcom_send": ["HcomSend", "HcclSend"],
            "hcom_recv": ["HcomRecv", "HcclRecv"],
        },
        "overlap_risk_ratio": 0.02,         # 同域时间区间重叠率超此 → 序号对齐标记不可靠
        "host_ts_lag_tolerance_us": 1000,   # trace ts 校验容忍的时钟偏差
        "suspect_min_elapse_ms": 100,       # Top op elapse 低于此不算疑点（溯源成本控制触发）
        # --- M1 瞬态防护 ---
        "head_wait_ratio": 0.6,             # 头部区段 wait 占比超此 → 疑瞬态污染
        "head_window_frac": 0.1,            # 头部区段 = 通信总时窗前 10%
        # --- M2 Matrix ---
        "skew_ratio": 3.0,                  # rank-pair size 倾斜比
        "skew_min_size_mb": 10,             # 倾斜检查排除的小消息
        "slow_link_size_buckets": [1, 10, 100],  # size 对数分档边界（MB）
        "bw_p50_ratio": 0.5,                # 大 size 档内低于 P50×此值 = 慢链路
        "unaligned_link_ratio": 0.3,        # 非 512B 对齐 link 占比超此 → [SIGNAL]
        "alignment_bytes": 512,             # HCCS 对齐要求
        "rdma_retransmission_ms": 4000,     # Transit 超此 = 疑似重传
        "latency_dominated_multiple": 3.0,  # transit < 此×L̂ = 延迟主导小包
        "latency_est_min_samples": 10,      # 最小档样本低于此 → L̂ 不可信，降级分档下界缺省
        "soc_bw_table": {},                 # 芯片带宽参考表（默认空 = 绝对基准不启用）
        # --- M3 跨 Rank ---
        "straggler_min_wait_gap": 0.5,      # 高 wait 算子上 max/min wait 超出 1+此值才投票
        "r_wait_sort_top_k": 5,             # R_wait 画像排序展示条数（无阈值判定）
        "overlap_severe_pct": 5,            # 假性重叠判定的重叠下限（触发以 Comm(NotOvl) 占比为准）
        "false_overlap_comm_pct": 20,       # 假性重叠判定
        "alltoall_cv_signal": 0.20,         # alltoall 量/kernel 耗时跨 rank CV（负载不均交叉验证）
    },

    "api_statistic": {
        "host_precompute_ratio": 0.2,       # ratio — tiling+workspace / total above this = signal
        "dominant_category_pct": 20,        # % — dominant API category above this = signal
    },
}
