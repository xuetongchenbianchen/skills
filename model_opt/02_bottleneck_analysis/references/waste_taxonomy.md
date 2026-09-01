# 十类性能浪费（统一归因分类）

> 跨线共用的候选归因分类：Line B 用它做现象归因（识别信号见 [profiling_to_action.md](profiling_to_action.md) 归因层）；Line A 用它填疑点的 `waste_class`（见 [proactive_source_analysis.md](proactive_source_analysis.md)）；合并阶段按它给候选归类并度量收益上限（[merge_analysis.md](merge_analysis.md)）。

| # | 类别 | 定义（消除后省的是哪类时间） | 收益上限的度量 |
|---|------|---------------------------|---------------|
| ① | 显式同步开销 | host 主动等 device（.item/.numpy/显式 sync/empty_cache） | 同步类 host 时间占比 |
| ② | dispatch/调度开销 | host 在框架调度（Module.__call__、hook、分发）而非计算 | dispatch 类 host 时间占比 |
| ③ | 内存管理阻塞 | host 卡在内存管理 API（分配/释放/映射） | 内存管理类 host 时间占比 |
| ④ | 在线编译/重编译 | 每步重新编译算子 | 编译时间占比 |
| ⑤ | 内存带宽受限 | kernel 在搬数据而非算 | 带宽受限段的 device 时间 |
| ⑥ | compute 饱和 | 计算密集且硬件已充分利用 | 该 kernel 群 device 时间 |
| ⑦ | 布局/格式转换 | 运行时 transpose/cast/format 转换 | 数据搬运类耗时占比 |
| ⑧ | 通信开销 | 通信时间花在等（同步等待）或传得慢（带宽/协议） | 可消除的通信时间 |
| ⑨ | 小算子碎片 | 大量短 kernel 串行 | 碎片段累计耗时 |
| ⑩ | 延迟未掩盖 | 存在可并行的独立工作但未重叠 | 可掩盖的 idle/延迟 |
