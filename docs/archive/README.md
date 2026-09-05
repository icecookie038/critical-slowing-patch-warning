# v3.5–v3.7 历史归档与来源

本归档衔接云端任务《重新分析调整方案》与当前 QIR 研究分支。
原始研究结论、小型数值结果与新方向的先导结果分别保留，不能相互替代。

## 恢复范围

- 29 个源码、测试、脚本、文档及 JSON 文件已按原始字节恢复。独立归档提交
  `2b6071c` 位于 `archive/v3.7-source-recovered-2026-09-05`。
- 25 个 CSV 已从云端压缩包恢复，逐文件核对原始清单。
- 3 张 PNG 从阶段总结 DOCX 的内嵌图片中原样提取，并核对原始清单。
- 另 3 张 PNG 由恢复的绘图代码、公开原始数据和历史结果 CSV 重新生成。
  它们具有新的文件校验值，**不是云端原 PNG 的逐字节副本**。

原清单的 60 个文件均有可用对应物，其中 57 个原文件已校验恢复，
3 张图为明确标记的重绘版本。完整原始云端 Git 提交并未导入，不能声称
本地提交与 `9473acb724ff67c9780f7a80c4cdc01c135c157f` 相同。

原始清单见 [original_manifest.json](original_manifest.json)，逐文件恢复状态
及本次工作分支文件校验值见 [recovery_manifest.json](recovery_manifest.json)。
[source_manifest.json](source_manifest.json) 另行保留 29 个源文件的核对记录。
当前 QIR 分支包含 Pandas 3 兼容修复、相应测试及导航更新；原始源文件请
按上述独立归档提交取用，不应将后续修改后的工作文件冒充原件。
清单中的工作分支校验值按 Git 内容记录，避免本地换行符差异。

## 图片来源

以下原图来自 `stage_summary_2026-09-04.docx` 的 `word/media/`：

- `image2.png` → `results/v3_5_local_recovery_spatial_explanation/summary_figure.png`
- `image3.png` → `results/v3_6_published_csd_benchmark/benchmark_summary.png`
- `image4.png` → `results/v3_7_time_safe_event_survival/time_safe_survival_summary.png`

以下是重绘版本，保留原路径以便旧报告链接可用：

- `results/v3_5_local_recovery_spatial_explanation/patch_spatial_explanation_64m.png`
- `results/v3_6_published_csd_benchmark/macroalgal_patch_explanation.png`
- `results/v3_6_published_csd_benchmark/salt_marsh_precision_audit.png`

运行 `python -m scripts.analysis.rebuild_archive_figures` 可重新生成这三张图，
默认输出到 `results/rebuilt_archive_figures/`，不会修改历史数值 CSV。
[figure_rebuild_manifest.json](figure_rebuild_manifest.json) 记录公开输入、
绘图代码、依赖版本和输出校验值。盐沼局部场需要从公开数据重新计算；
这次图形重建不是对所有历史大型实验的重新验证。

## 传输与检验

源云端记录的归档提交为 `9473acb724ff67c9780f7a80c4cdc01c135c157f`，
父提交为 `945804cf54c34d54bb017ec0e132ee2acbd7f462`。
本地已核对的传输包：

- 源码包：76,835 字节，SHA256
  `1425c09410b874ee469c0751f03711020b196258c7f00abf6c24801fc8db4f9e`。
- CSV 与原始清单包：18,853 字节，SHA256
  `e79ecf4b463c280778181f5af3f2c8f62bee5dd3b6122267b8dce2891776898d`。
- 阶段总结 DOCX：SHA256
  `fc7fdcfc75d21feb01698c64b05fe939253781b0e1fb07ebac19b213488238ba`。

本地历史与 QIR 测试合计 31 项通过。测试验证小型逻辑、事件定义与兼容性，
不代表已重跑历史全量分析或证明新方法有效。
原始数据、缓存、权重及大规模中间文件不进入本归档。

临时 GitHub 部署密钥已撤销，云端临时私钥已删除；本归档不包含凭据。
旧的 [云端恢复记录](../RECOVERY_v3_7_2026-09-05.md) 保持原文，
其中“本地”“当前运行环境”指原云端环境，实际转移状态以本文及校验清单为准。

## 继续研究

从 [科研入口](../research_start_here_zh.md) 进入当前方案。
QIR 盐沼先导试验尚未显示稳定双向优势；历史样条基线复核暴露了源域训练与
校准压力范围不重叠的问题。下一轮先处理这个验证设计问题，保留失败结果，
再进行多随机种子比较及独立系统验证。论文内部草稿不能作为投稿完成稿。
