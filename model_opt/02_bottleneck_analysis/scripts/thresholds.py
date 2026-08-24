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
        "wait_dominant_ratio": 0.8,         # wait/total above this = DEFINITE sync-bound
        "per_type_wait_ratio": 0.9,         # per-type wait ratio above this = SIGNAL
        "per_type_min_count": 10,           # min count for per-type signal
        "low_bw_ratio": 0.3,                # bandwidth below avg*this = low bandwidth link
        "low_bw_min_size_mb": 1,            # MB — min size for low bandwidth link signal
        "small_packet_mb": 1.0,             # MB — packets below this = small
        "small_packet_ratio": 0.3,          # small packet ratio above this = SIGNAL
    },

    "api_statistic": {
        "host_precompute_ratio": 0.2,       # ratio — tiling+workspace / total above this = signal
        "dominant_category_pct": 20,        # % — dominant API category above this = signal
    },

    "multi_rank": {
        # --- Phase 1: straggler detection ---
        # (T_max - T_avg) / T_avg above this = straggler (Tail Card)
        "tail_card_ratio": 0.10,           # 10% — above this = DEFINITE straggler
        "tail_card_ratio_strong": 0.15,    # 15% — above this = severe straggler
        # straggler detection — how much slower than median makes a rank a straggler
        "straggler_ratio": 1.10,            # x — total time > median * this = DEFINITE straggler
        "straggler_margin_ms": 2000,        # ms — minimum absolute margin to flag straggler (avoid noise)
        # load imbalance — coefficient of variation (std/mean) across ranks
        "comm_cv_signal": 0.30,            # comm time CV above this = SIGNAL imbalance
        "comm_cv_definite": 0.50,          # comm time CV above this = DEFINITE imbalance
        "compute_cv_signal": 0.05,         # computing time CV above this = SIGNAL (compute should be uniform)

        # --- Phase 2: overlap & parallel efficiency ---
        # Overlapped / Total below this = parallel bottleneck (comm not overlapped with compute)
        "overlap_low_pct": 10,             # % — below this = SIGNAL parallel bottleneck
        "overlap_severe_pct": 5,           # % — below this = DEFINITE severe parallel bottleneck
        # false overlap: if overlap > 0 but comm_not_ovl still high, overlap may be ineffective
        "false_overlap_comm_pct": 20,      # % — comm_not_ovl / total above this with overlap = false overlap hint

        # --- Phase 3.4: communication deep analysis ---
        # R_wait = 1 - (T_avg / T_max) per comm type — sync straggler metric
        "r_wait_definite": 0.30,           # R_wait above this = DEFINITE sync straggler
        # comm wait imbalance — which rank is the victim (waits the most)
        "wait_imbalance_ratio": 2.0,       # max_rank_wait / min_rank_wait above this = DEFINITE
        # small packet: single comm transfer below this = small packet overhead
        "small_packet_mb": 32,            # MB — transfers below this = small packet (Hermes guide)
        "small_packet_ratio": 0.30,        # % — small packet ratio above this = SIGNAL
        # byte alignment: HCCS requires 512-byte aligned transfer sizes
        "alignment_bytes": 512,           # bytes — HCCS alignment requirement
        # RDMA retransmission: transit time above this = potential retransmission
        "rdma_retransmission_ms": 4000,  # ms — transit time above this = SIGNAL (network issue)
        # HBM contention: if overlapped comm co-occurs with compute, check for slowdown

        # --- Phase 3.3: compute analysis ---
        # per-op cross-rank variance
        "op_cv_signal": 0.20,             # op total time CV above this = SIGNAL (imbalanced op)
        "op_cv_min_share": 0.01,           # min share of total op time to consider for variance signal
        # TransData / format conversion ops — these indicate private format conversion overhead
        "transdata_keywords": ["TransData", "TransForm", "FormatTransfer"],

        # --- Phase 5: MoE / AlltoAll load imbalance ---
        # AlltoAll token distribution CV above this = load imbalance
        "alltoall_cv_signal": 0.20,       # CV above this = SIGNAL load imbalance

        # --- Phase 1: step spike detection ---
        # step time > previous step * this ratio = sudden spike
        "step_spike_ratio": 2.0,          # x — step time > prev * this = SIGNAL spike

        # --- bandwidth across ranks ---
        "link_bw_cv_signal": 0.30,        # per-link bandwidth CV across ranks above this = SIGNAL
    },
}
