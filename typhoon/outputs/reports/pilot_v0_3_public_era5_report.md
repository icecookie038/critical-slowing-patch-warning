# v0.3 匿名开放 ERA5 稀疏环境基线

## 数据与范围

- 轨迹：10 条（5 个暂定匹配对）。
- 时间锚点：-72, -48, -24, -3 h。
- 环境量：850 hPa 相对涡度、600/700/850 hPa 平均相对湿度、200--850 hPa 风切变。
- 空间尺度：300, 500 km，主分析为 500 km。
- 来源：NSF NCAR Curated ERA5 on AWS，匿名远程分块读取。

本实验是四时次稀疏 Pilot，不等同于原计划的完整 3 小时 ERA5 序列。它的作用是先检查此前冷云信号是否仅由大尺度环境解释。

## 留一匹配对结果

| model                                    |   n_cases |   n_pairs |   n_features |   roc_auc |   pair_bootstrap_ci_low |   pair_bootstrap_ci_high |   pr_auc |   brier_score |   exact_within_pair_permutation_p_one_sided |   n_exact_permutations |
|:-----------------------------------------|----------:|----------:|-------------:|----------:|------------------------:|-------------------------:|---------:|--------------:|--------------------------------------------:|-----------------------:|
| metadata_only                            |        10 |         5 |            4 |      0.88 |                    0.76 |                        1 | 0.925    |      0.204498 |                                      0.0625 |                     32 |
| environment_only                         |        10 |         5 |            3 |      0.72 |                    0.68 |                        1 | 0.794444 |      0.220179 |                                      0.125  |                     32 |
| cloud_area_only                          |        10 |         5 |            2 |      1    |                    1    |                        1 | 1        |      0.1693   |                                      0.0625 |                     32 |
| environment_plus_cloud                   |        10 |         5 |            5 |      0.84 |                    0.76 |                        1 | 0.885    |      0.174143 |                                      0.125  |                     32 |
| environment_plus_cloud_plus_organization |        10 |         5 |            9 |      0.88 |                    0.76 |                        1 | 0.902857 |      0.151968 |                                      0.0625 |                     32 |

## 增量诊断

- 元数据模型 AUC：0.880。
- 仅环境模型 AUC：0.720。
- 仅冷云面积模型 AUC：1.000。
- 环境 + 冷云 AUC：0.840，相对仅环境变化 +0.120。
- 加入组织指标后 AUC：0.880，相对环境 + 冷云变化 +0.040。

组织指标没有显示至少 0.05 的稳定增量，本阶段不支持直接进入 DynCL、SINDy 或临界慢化结论。

## 敏感性分析

| sample      |   radius_km | model                                    |   n_cases |   n_pairs |   roc_auc |
|:------------|------------:|:-----------------------------------------|----------:|----------:|----------:|
| all_5_pairs |         300 | metadata_only                            |        10 |         5 |    0.88   |
| all_5_pairs |         300 | environment_only                         |        10 |         5 |    0.72   |
| all_5_pairs |         300 | cloud_area_only                          |        10 |         5 |    1      |
| all_5_pairs |         300 | environment_plus_cloud                   |        10 |         5 |    0.88   |
| all_5_pairs |         300 | environment_plus_cloud_plus_organization |        10 |         5 |    0.92   |
| drop_pair_5 |         300 | metadata_only                            |         8 |         4 |    0.6875 |
| drop_pair_5 |         300 | environment_only                         |         8 |         4 |    0.8125 |
| drop_pair_5 |         300 | cloud_area_only                          |         8 |         4 |    1      |
| drop_pair_5 |         300 | environment_plus_cloud                   |         8 |         4 |    0.9375 |
| drop_pair_5 |         300 | environment_plus_cloud_plus_organization |         8 |         4 |    1      |
| all_5_pairs |         500 | metadata_only                            |        10 |         5 |    0.88   |
| all_5_pairs |         500 | environment_only                         |        10 |         5 |    0.72   |
| all_5_pairs |         500 | cloud_area_only                          |        10 |         5 |    1      |
| all_5_pairs |         500 | environment_plus_cloud                   |        10 |         5 |    0.84   |
| all_5_pairs |         500 | environment_plus_cloud_plus_organization |        10 |         5 |    0.88   |
| drop_pair_5 |         500 | metadata_only                            |         8 |         4 |    0.6875 |
| drop_pair_5 |         500 | environment_only                         |         8 |         4 |    0.875  |
| drop_pair_5 |         500 | cloud_area_only                          |         8 |         4 |    1      |
| drop_pair_5 |         500 | environment_plus_cloud                   |         8 |         4 |    0.9375 |
| drop_pair_5 |         500 | environment_plus_cloud_plus_organization |         8 |         4 |    1      |

300 km 与 500 km 的总体结论相近。删除匹配最差的第 5 对后指标普遍升高，这不能视为性能证明，反而说明当前结果对样本组成仍然敏感。

## 涡度计算交叉校验

以 ERA5 直接归档的 VO 变量对 3 个锚点交叉校验，球面风场导数法的最大绝对相对差为 0.83%。

## 统计边界

只有 5 对案例，成对标签的精确置换检验最小单侧 p 值为 0.0625。AUC 的高低只能用于方法筛选，不能作为可发表的泛化性能。当前未发展样本仍存在地域、季节和“已被最佳路径追踪”的选择偏差。

## 下一决策

1. 扩大并客观追踪开放洋面的未发展云团，改善纬度、月份和环境匹配；
2. 在扩大样本上重复“仅环境 / 环境 + 冷云 / 环境 + 组织”消融；
3. 只有组织变量在分组跨年份验证中重复提供增量，才构造潜在组织状态并做动力学识别；
4. 临界慢化检验必须晚于状态变量有效性、空模型和假阳性控制。
