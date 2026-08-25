# .gitignore 模板

目录结构与提交范围的完整定义见 [standardized_operations.md](../../references/standardized_operations.md)「标准项目目录结构」，本模板与其保持一致。

```gitignore
# 原始大体积数据（目录级忽略）
weights/
profiling/
golden/
baseline/

# 运行时临时文件
__pycache__/
*.pyc
kernel_meta/
*.log
.venv/
```

**应当纳入 git** 的：源代码与脚本（`scripts/`、主目录入口）、结论性记录（`evidence_db/`、`comparison_records/` 摘要）。
