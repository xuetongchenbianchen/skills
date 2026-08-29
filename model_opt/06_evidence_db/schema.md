# 案例库 Schema 说明

## 目录

- [定位](#定位) / [当前阶段目标](#当前阶段目标)
- [Schema 字段说明](#schema-字段说明)——记录案例时按字段查：phenomenon / analysis_path / root_cause / parallel_splitting（切分案例专用）/ optimization / final_state / platform_findings / context
- [填写原则](#填写原则) / [目录结构](#目录结构)

## 定位

本文件是案例库的**构造语法说明**,agent 据此在项目工作目录的 `evidence_db/` 下记录优化案例。

> **路径约定**: 案例数据存在**项目工作目录** `<workspace>/evidence_db/` 下(与 `profiling/` 同级),**不在 skill 目录中**。skill 目录只存本 schema 定义。

## 当前阶段目标

现阶段案例库的唯一目的是**把优化过程中的信息尽可能完整地记录下来**。消费端(检索、匹配、统计)是后续的事——先确保"存对了",再谈"怎么用"。

## Schema 字段说明

```yaml
- id: <string>
  # 唯一标识符。格式: <模型架构缩写>-<核心现象关键词>-<日期YYYYMMDD>
  # 例: "llm-transpose-layout-20260706", "moe-scatter-backward-20260710"
  # 规则: 用现象/根因描述而非模型全名,确保跨项目可检索

  depends_on: <list[string], optional>
    # 本案例依赖的前序案例 id。如 FastContraction(bmm) 依赖 FastContraction(index_select)
    # 无依赖则省略

  phenomenon:
    # 记录 agent 观察到的所有相关信号,尽可能完整
    # (优化案例来自 profiling;切分案例来自冒烟测试/OOM 分诊)
    signals:
      - source: <string>  # 产出此信号的脚本名+参数,如 "parse_op_statistic" 或 "parse_kernel_details --filter Transpose"
        content: <string>  # 脚本输出的原文摘录(关键数值+判断),不做解读,只记事实
    raw_context: <string, optional>
      # 补充任何脚本没覆盖但 agent 观察到的信息
      # 如: trace_view 中肉眼看到的 pattern、源码中发现的结构特征

  analysis_path:
    # 记录从现象到根因的完整推理过程,按实际执行顺序
    profiling_ref: <string, optional>
      # 本案例分析所基于的 profiling 数据路径或时间戳
      # 如 "profiling/L1_20260729_134016" 或 "profiling/latest_L1"
    steps:
      - action: <string>   # agent 做了什么(跑了什么脚本/读了什么代码/做了什么推理)
        observation: <string>  # 得到了什么结果/看到了什么
        reasoning: <string, optional>  # 为什么做这一步/这个结果说明什么

  root_cause:
    description: <string>  # 最终确认的根因
    bottleneck_type: <enum, optional>
      # Host-Bound / Compute-Bound / Memory-Bound / Allocator-Bound / Execution-Mode
      # 不明确或属于多种: 写 "mixed" 并在 description 中说明
    evidence: <string>  # 支撑此根因判断的关键证据

  parallel_splitting: <object, optional>
    # 多卡切分案例专用字段(07_parallel_splitting 轨道)。切分是独立于优化循环的前置结构变更
    # (发生在 Phase 2 之前),不记在 optimization.attempts 里——切分完成时写独立的切分案例,
    # 后续优化案例用 depends_on 指向它。填写本节的案例,optimization 节留空;反之亦然。
    split_position: <string>  # input（输入边界）/ module（模型内部），见 07_parallel_splitting「切分怎么接入模型」
    split_dimension: <string>  # horizontal / vertical
    implementation_mode: <string>  # 输入边界编排 / monkey-patch / 子类覆写 / 源码内嵌（替换机制）
    tensor_distribution_table:
      # 逐张量声明分布方式。须覆盖所有大张量及其配套张量（mask/index/
      # position_ids 等）——配套漏切是实测高频错误；未声明 = 潜在 bug
      - tensor: <string>
        global_shape: <list[int]>
        distribution: <string>
        per_card_shape: <list[int]>
        consumer_op: <string>
        recovery_comm: <string>
    quantitative:
      single_card_peak_bytes: <int>
      per_card_peak_after_split_bytes: <int>
      hbm_bytes: <int>
      communication_volume_bytes: <int>
      communication_time_ms: <float>
      break_even: <string>  # 通信耗时 vs 计算节省的对比结论
    proofs:
      - operation: <string>
        equivalence_proof: <string>  # 切分等价性论证
        lower_bound_comm_bytes: <int>  # 理论通信下界
        actual_comm_bytes: <int>  # 实际通信量
        ratio: <float>  # actual / lower_bound，< 1.0 说明估算有误
    verification:
      method: <string>  # verify_split.py + split-type；缩配 baseline 注明缩配参数
      result: <string>  # verify_report 摘要（overall + 各 tier 结论）
    known_pitfalls:
      - pitfall: <string>
        mitigation: <string>
    numeric_sensitivity: <string, optional>
      # 数值敏感性评估结论（评测/RL 等敏感场景的验收策略与容差档选择，
      # 见 07 implementation_guide「验收标准」）
    rollback: <string>  # 恢复机制，如 disable_parallel() / git revert

  optimization:
    attempts:
      - description: <string>  # 做了什么改动
        dimension: <string, optional>
          # eliminate_redundancy / reuse_and_precompute / hide_latency / equivalent_substitution
        implementation_detail: <string>  # 具体代码层面怎么改的(文件、函数、改法)
        equivalence_verification:
          method: <string>  # 怎么验证等价性的
          metrics: <list, optional>
            # 结构化的度量信息,便于检索
            - metric: <string>    # cosine_similarity / max_abs_diff / relative_error / kl_divergence / match_rate
              threshold: <string> # 如 ">= 0.9999" 或 "< 1e-4"
              value: <string>     # 实际值,如 "0.99999" 或 "3.2e-7"
          result: <string>  # 验证结论(通过/失败,可用文字补充细节)
        performance_result:
          metric: <string>  # 用什么指标衡量
          baseline_ref: <string, optional>
            # before 的参照对象,如 "raw model (无优化)" / "上一轮优化后" / "L0 基线"
          before: <string>
          after: <string>
          verdict: <string> # accepted / rejected / partial
        failure_reason: <string, optional>  # 如果 rejected,为什么失败

  final_state:
    adopted: <string>  # 最终采纳了哪个方案(或"无方案被采纳")
    files_modified: <list[string], optional>
      # 本案例修改/新增的文件列表
      # 如: ["mace/modules/fast_contraction.py", "scripts/benchmark_full.py"]
    end_to_end_before: <string>
    end_to_end_after: <string>
    remaining_bottleneck: <string, optional>  # 优化后暴露的新瓶颈
    is_terminal: <bool, optional>  # 是否判定为终局

  platform_findings: <list[string], optional>
    # 跨越单个优化案例的 NPU 平台级行为洞察,对后续项目有指导价值
    # 如: "NPU async pipeline (TASK_QUEUE_ENABLE=2) makes host-side optimizations counterproductive"
    # 如: "data_ptr() cache is unsafe on NPU due to memory address reuse"
    # 如: "torch.einsum internal decomposition is worse than opt_einsum_fx on NPU"
    # 如: "NPU AllToAll requires .contiguous() after chunk() (GPU usually doesn't)"
    # 如: "HCCL backend requires importing torch_npu before torch.npu.is_available() check"
    # 如: "NPU operators may be non-deterministic; per-sample seeding (torch/random/np.random) needed"
    # 如: "Multi-node NPU needs HCCL_CONNECT_TIMEOUT=600 to avoid connection timeout"

  context:
    hardware: <string>   # 如 "Ascend 910B"
    cann_version: <string>  # 如 "CANN 8.0.0 / torch_npu 2.3.1"
    model_arch: <string>  # 架构类型而非具体模型名
    input_spec: <string>  # 测试输入规格
    profiling_level: <string>  # "L0" / "L1"
    date: <string>  # YYYY-MM-DD
    notes: <string, optional>
```

## 填写原则

1. **完整优先**: 不确定某信息是否有用时,记下来。optional 字段能填就填。
2. **原文摘录**: phenomenon.signals.content 和 analysis_path.steps.observation 尽量贴脚本原始输出。
3. **失败必记**: optimization.attempts 中 rejected 的方案和 failure_reason 是最有价值的信息。
4. **切分与优化分开记录**: 多卡切分完成时写一个**切分案例**(填 `parallel_splitting` 节,`optimization` 留空);每经过一轮完整的 Phase 2->4 写一个**优化案例**(填 `optimization.attempts`,切分模型上的优化用 `depends_on` 指向所基于的切分案例)。
5. **不强求归类**: bottleneck_type 和 dimension 能判断就写,判断不了写"mixed"并在 description 中说明。
6. **平台发现独立记录**: NPU 特有的行为洞察写在 platform_findings 中,不要埋在 notes 或 failure_reason 里——这些发现跨越单个案例,对后续项目有指导价值。

## 目录结构

```
<workspace>/
├── profiling/           # profiling 数据
├── evidence_db/         # 案例库(项目工作目录下)
│   ├── <id>.yaml        # 每个案例一个文件,扁平存放
│   └── ...
└── ...
```

文件名 = id 字段值 + `.yaml`
