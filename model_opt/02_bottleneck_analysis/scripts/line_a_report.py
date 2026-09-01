#!/usr/bin/env python3
"""Line A 源码分析报告渲染器（校验 + 渲染，不做分析）。

职责（确定性工作）：
  1. findings schema 校验：疑点必须引用存在的事实 id；三层事实非空；枚举合法
  2. 热路径覆盖检查：标记 hot 的组件须在实现逻辑层有审视记录
  3. 渲染报告落盘（findings 同目录 line_a_report.md）+ stdout 摘要

不做（判断力工作，留给 agent）：
  - 三层事实记录与疑点推导：见 ../references/proactive_source_analysis.md
  - 疑点量化与候选合并（合并阶段）：见 model_opt/references/execution_protocol.md

用法：
  python line_a_report.py --findings analysis/round_0/line_a_findings.yaml [--output path.md]

输入支持 .yaml（需 PyYAML）与 .json。非法输入直接报错退出（fail loudly），不做静默修复。
"""
import argparse
import datetime
import json
import os
import sys

LAYERS = ("structure", "implementation", "algorithm")
KINDS = ("component", "dataflow", "lifecycle", "controlflow", "access", "algo")
DIMENSIONS = ("eliminate", "reuse", "hide", "substitute")
CONFIDENCES = ("high", "medium", "low")
LAYER_ZH = {"structure": "结构层", "implementation": "实现逻辑层", "algorithm": "算法层"}
KIND_ZH = {"component": "组件", "dataflow": "数据流", "lifecycle": "生命周期",
           "controlflow": "控制流", "access": "访问模式", "algo": "算法"}
DIM_ZH = {"eliminate": "去重", "reuse": "复用", "hide": "掩盖", "substitute": "替换"}


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    try:
        import yaml
    except ImportError:
        sys.exit("[ERROR] findings 为 YAML 时需要 PyYAML（pip install pyyaml），或改用 .json")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate(data):
    errors, warnings = [], []
    meta = data.get("meta") or {}
    for key in ("model", "source_commit", "round"):
        if meta.get(key) in (None, ""):
            errors.append(f"meta.{key} 缺失")

    facts = data.get("facts") or []
    suspects = data.get("suspects") or []
    if not facts:
        errors.append("facts 为空——三层事实记录缺失")

    fact_ids = set()
    for i, fact in enumerate(facts):
        fid = fact.get("id")
        if not fid:
            errors.append(f"facts[{i}].id 缺失")
            continue
        if fid in fact_ids:
            errors.append(f"事实 id 重复: {fid}")
        fact_ids.add(fid)
        if fact.get("layer") not in LAYERS:
            errors.append(f"{fid}: layer 非法 {fact.get('layer')!r}（应为 {LAYERS} 之一）")
        if fact.get("kind") not in KINDS:
            errors.append(f"{fid}: kind 非法 {fact.get('kind')!r}（应为 {KINDS} 之一）")
        if not str(fact.get("content") or "").strip():
            errors.append(f"{fid}: content 为空")

    for layer in LAYERS:
        if not any(f.get("layer") == layer for f in facts):
            errors.append(f"{LAYER_ZH[layer]}（{layer}）无任何事实记录")

    hot_components = [f for f in facts if f.get("kind") == "component" and f.get("hot")]
    impl_components = {f.get("component") for f in facts
                       if f.get("layer") == "implementation" and f.get("component")}
    for h in hot_components:
        name = h.get("component")
        if not name:
            warnings.append(f"热路径组件事实 {h.get('id')} 缺 component 字段，跳过覆盖检查")
        elif name not in impl_components:
            errors.append(f"热路径组件「{name}」在实现逻辑层无审视记录"
                          f"（需要一条 component='{name}' 的 implementation 事实）")
    if facts and not hot_components:
        warnings.append("无任何组件标记 hot=true——热路径应由生成模式推导并显式标记")

    suspect_ids = set()
    for i, s in enumerate(suspects):
        sid = s.get("id")
        if not sid:
            errors.append(f"suspects[{i}].id 缺失")
            continue
        if sid in suspect_ids:
            errors.append(f"疑点 id 重复: {sid}")
        suspect_ids.add(sid)
        refs = s.get("fact_refs") or []
        if not refs:
            errors.append(f"{sid}: fact_refs 为空——疑点必须引用事实 id")
        for ref in refs:
            if ref not in fact_ids:
                errors.append(f"{sid}: 引用了不存在的事实 id {ref!r}")
        if s.get("layer") not in LAYERS:
            errors.append(f"{sid}: layer 非法 {s.get('layer')!r}")
        if not str(s.get("location") or "").strip():
            errors.append(f"{sid}: location 为空（join key，格式须统一为 文件:行 + 函数）")
        if not str(s.get("suspicion") or "").strip():
            errors.append(f"{sid}: suspicion 为空")
        if s.get("dimension") not in DIMENSIONS:
            errors.append(f"{sid}: dimension 非法 {s.get('dimension')!r}")
        if s.get("confidence") not in CONFIDENCES:
            errors.append(f"{sid}: confidence 非法 {s.get('confidence')!r}")
        if not str(s.get("impact_qualitative") or "").strip():
            warnings.append(f"{sid}: impact_qualitative 为空（建议填写定性影响：重复度×频率）")
        if not s.get("waste_class"):
            warnings.append(f"{sid}: waste_class 为空（判据外机会可空，合并阶段补）")

    if facts and not suspects:
        warnings.append("suspects 为空——完整分析产出零疑点属于异常，确认已逐条对照判据表")
    return errors, warnings


def render(data):
    meta = data.get("meta") or {}
    facts = data.get("facts") or []
    suspects = data.get("suspects") or []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"# Line A 源码分析报告（round {meta.get('round', '?')}）\n")
    lines.append("| 模型 | 源码版本 | 轮次 | 分析范围 | 生成时间 |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| {meta.get('model', '')} | {meta.get('source_commit', '')} | "
                 f"{meta.get('round', '')} | {meta.get('scope', '全面审视')} | {now} |\n")

    lines.append("## 1. 架构理解（结构层）\n")
    rows = [f for f in facts if f.get("layer") == "structure"]
    lines.append("| id | 组件 | 事实 | 热路径 |")
    lines.append("|---|---|---|---|")
    for f in rows:
        hot = "✓" if f.get("hot") else ""
        lines.append(f"| {f.get('id')} | {f.get('component') or ''} | {f.get('content', '')} | {hot} |")
    lines.append("")

    lines.append("## 2. 实现逻辑理解\n")
    impl = [f for f in facts if f.get("layer") == "implementation"]
    for kind in ("dataflow", "lifecycle", "controlflow", "access"):
        lines.append(f"### {KIND_ZH[kind]}\n")
        kind_rows = [f for f in impl if f.get("kind") == kind]
        if not kind_rows:
            lines.append("（无记录）\n")
            continue
        lines.append("| id | 组件 | 事实 |")
        lines.append("|---|---|---|")
        for f in kind_rows:
            lines.append(f"| {f.get('id')} | {f.get('component') or ''} | {f.get('content', '')} |")
        lines.append("")
    other = [f for f in impl if f.get("kind") not in
             ("dataflow", "lifecycle", "controlflow", "access")]
    if other:
        lines.append("### 其他\n")
        lines.append("| id | 组件 | 事实 |")
        lines.append("|---|---|---|")
        for f in other:
            lines.append(f"| {f.get('id')} | {f.get('component') or ''} | {f.get('content', '')} |")
        lines.append("")

    lines.append("## 3. 算法理解\n")
    rows = [f for f in facts if f.get("layer") == "algorithm"]
    lines.append("| id | 计算块 | 事实 |")
    lines.append("|---|---|---|")
    for f in rows:
        lines.append(f"| {f.get('id')} | {f.get('component') or ''} | {f.get('content', '')} |")
    lines.append("")

    lines.append("## 4. 疑点汇总\n")
    lines.append("| id | 层 | 位置 | 依据事实 | 疑点 | 维度 | 浪费类别 | 影响范围（定性） | 置信度 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in suspects:
        lines.append(
            f"| {s.get('id')} | {LAYER_ZH.get(s.get('layer'), s.get('layer', ''))} "
            f"| {s.get('location', '')} | {', '.join(s.get('fact_refs') or [])} "
            f"| {s.get('suspicion', '')} | {DIM_ZH.get(s.get('dimension'), s.get('dimension', ''))} "
            f"| {s.get('waste_class') or '—'} | {s.get('impact_qualitative') or '—'} "
            f"| {s.get('confidence', '')} |")
    lines.append("")
    lines.append("> 报告于 §4 封卷。疑点量化与候选合并在合并阶段进行"
                 "（结构根因 + 量化影响合成完整候选），见 execution_protocol。\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Line A 报告渲染器（校验 + 渲染）")
    parser.add_argument("--findings", required=True, help="findings.yaml / findings.json 路径")
    parser.add_argument("--output", default=None, help="报告输出路径（默认 findings 同目录 line_a_report.md）")
    args = parser.parse_args()

    data = load(args.findings)
    if not isinstance(data, dict):
        sys.exit("[ERROR] findings 根节点必须是 mapping/dict")

    errors, warnings = validate(data)
    for w in warnings:
        print(f"[WARN] {w}")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(f"\n校验失败（{len(errors)} 项错误）——修复 findings 后重跑，报告不渲染")

    out = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.findings)), "line_a_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(data))

    facts = data.get("facts") or []
    suspects = data.get("suspects") or []
    by_layer = {layer: sum(1 for s in suspects if s.get("layer") == layer) for layer in LAYERS}
    by_dim = {d: sum(1 for s in suspects if s.get("dimension") == d) for d in DIMENSIONS}
    print(f"报告已写入: {out}")
    print(f"事实: {len(facts)} 条（{'/'.join(str(sum(1 for f in facts if f.get('layer') == l)) for l in LAYERS)} 按三层）")
    print(f"疑点: {len(suspects)} 条（结构 {by_layer['structure']}/实现 {by_layer['implementation']}"
          f"/算法 {by_layer['algorithm']}；维度 去重{by_dim['eliminate']}/复用{by_dim['reuse']}"
          f"/掩盖{by_dim['hide']}/替换{by_dim['substitute']}）")


if __name__ == "__main__":
    main()
