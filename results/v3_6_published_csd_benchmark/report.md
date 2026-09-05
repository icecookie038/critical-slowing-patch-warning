# v3.6 已发表生态临界慢化数据集基准与精度审计

## 结论

本轮在四篇已发表研究的数据上建立了统一基准。结果支持：**采样精度会明显影响间接预警指标，但它不是此前斑块路线失败的唯一原因。** 高精度数据中仍存在驱动依赖、指标依赖和对照假阳性；因此不能通过单纯提高分辨率把斑块重新定义为通用 CSD 检测器。

- 主检测层：直接恢复率或论文已验证的动力学 EWS。
- 空间解释层：斑块描述慢恢复/转变区域如何形成、连通和扩展。
- 当前决策：`NO-GO`（斑块主检测器）；`GO, descriptive and sensitivity-audited`（斑块空间解释）。

## 数据集筛选

| study | system | sampling_design | spatial_patch_data | benchmark_status |
| --- | --- | --- | --- | --- |
| van Belzen et al. 2017 | Tidal-marsh vegetation | 0.25 m binary maps; 8 and 11 irregular dates | 2-D | RUN: v3.5 direct recovery + patch explanation |
| Clements & Ozgul 2016 | Predator-prey microcosms | Daily observations; 37 populations; up to 45 d | None | RUN: replication + temporal downsampling |
| Dai et al. 2015 | Cooperative yeast populations | 48 parallel populations; 21 daily states per driver | None | RUN: driver comparison + replicate subsampling |
| Rindi et al. 2018 | Intertidal macroalgal canopy | 2 y; 16 transects; 5x30 cells per map | 2-D | RUN: spatial EWS + patch explanation + subsampling |
| Drake & Griffen 2010 | Daphnia extinction experiment | High-frequency population trajectories | None | SCREENED: repository download currently denied |
| Veraart et al. 2012 | Cyanobacterial chemostats | Parallel chemostats with pulse perturbations | None | SCREENED: public endpoint returned an anti-bot HTML file |
| Butitta et al. 2017 | Whole-lake trophic manipulation | 11 weekly maps; thousands of georeferenced samples/map | 2-D points | SCREENED: package metadata open, data entity restricted |

前三个新增基准与既有盐沼基准均使用已发表论文公开的数据。另三组数据完成了来源和结构核对，但公开端点当前不能无认证取得，因此没有把失败下载的网页当作数据参与统计。

## 1. 捕食—猎物微宇宙：时间精度

按作者 LOESS 实现推断出 17 个跨越增长率阈值的种群，其中 16 个来自恶化处理；常量处理 Tube 14 按原代码排除。作者推荐的 `CV + mean size + size SD` 复合指标在完整每日数据上得到更高真阳性率和更低假阳性率。

| sampling_interval_days | class | eligible_fraction | signal_rate_all | normalized_metric_score |
| --- | --- | --- | --- | --- |
| 1 | constant | 1.0000 | 0.1000 | 0.5250 |
| 1 | deteriorating | 1.0000 | 0.6250 | 0.5250 |
| 2 | constant | 1.0000 | 0.0000 | 0.0625 |
| 2 | deteriorating | 1.0000 | 0.0625 | 0.0625 |
| 3 | constant | 1.0000 | 0.0000 | 0.0000 |
| 3 | deteriorating | 0.6458 | 0.0000 | 0.0000 |
| 4 | constant | 0.0000 | 0.0000 | 0.0000 |
| 4 | deteriorating | 0.0000 | 0.0000 | 0.0000 |

降采样结果用于回答观测间隔问题。`eligible_fraction` 同时报告，因为快速崩溃种群在每 3–4 天采样时甚至无法保留足够的转折前观测；把这些样本静默删除会高估性能。

## 2. 酵母实验：同等精度下的驱动依赖

死亡率上升和蔗糖下降均有 48 个平行种群。用相邻日平均密度比首次低于 0.5 确定论文所述崩溃起点，并比较崩溃前 10 日中末 3 日与前 5 日的指标。

| driver | environment | indicator | tipping_index | late_minus_early | kendall_tau | kendall_p |
| --- | --- | --- | --- | --- | --- | --- |
| dilution_factor | dilution_factor | cv | 16 | 0.2844 | 0.9111 | 2.976e-05 |
| dilution_factor | dilution_factor | ar1_slope | 16 | 0.1637 | 0.2889 | 0.2912 |
| dilution_factor | control | cv | 16 | 0.0137 | 0.2889 | 0.2912 |
| dilution_factor | control | ar1_slope | 16 | -0.1005 | -0.2000 | 0.4843 |
| sucrose | sucrose | cv | 20 | 0.1202 | 0.8222 | 3.577e-04 |
| sucrose | sucrose | ar1_slope | 20 | 0.5525 | 0.3778 | 0.1557 |
| sucrose | control | cv | 20 | 0.0129 | 0.3778 | 0.1557 |
| sucrose | control | ar1_slope | 20 | 0.3350 | 0.6444 | 0.0091 |

两种驱动的 CV 均明显上升，但自回归/自相关信号强度不同，且对应控制窗口本身也可能出现正趋势。这是在观测数量相同的情况下发生的，直接说明“低精度”不能解释全部成败。

平行重复数降采样结果：

| driver | indicator | sample_size | positive_direction_rate | greater_than_control_rate | median_rank_similarity |
| --- | --- | --- | --- | --- | --- |
| dilution_factor | ar1_slope | 48 | 1.0000 | 1.0000 | 1.0000 |
| dilution_factor | ar1_slope | 24 | 0.8800 | 0.9580 | 0.8788 |
| dilution_factor | ar1_slope | 12 | 0.6800 | 0.8140 | 0.6970 |
| dilution_factor | ar1_slope | 6 | 0.6360 | 0.6800 | 0.4545 |
| dilution_factor | cv | 48 | 1.0000 | 1.0000 | 1.0000 |
| dilution_factor | cv | 24 | 1.0000 | 1.0000 | 1.0000 |
| dilution_factor | cv | 12 | 1.0000 | 1.0000 | 0.9879 |
| dilution_factor | cv | 6 | 1.0000 | 1.0000 | 0.9515 |
| sucrose | ar1_slope | 48 | 1.0000 | 1.0000 | 1.0000 |
| sucrose | ar1_slope | 24 | 1.0000 | 0.7900 | 0.8424 |
| sucrose | ar1_slope | 12 | 0.9900 | 0.6060 | 0.6545 |
| sucrose | ar1_slope | 6 | 0.8480 | 0.5280 | 0.4909 |
| sucrose | cv | 48 | 1.0000 | 1.0000 | 1.0000 |
| sucrose | cv | 24 | 1.0000 | 1.0000 | 0.9879 |
| sucrose | cv | 12 | 1.0000 | 1.0000 | 0.9394 |
| sucrose | cv | 6 | 1.0000 | 0.9980 | 0.8667 |

## 3. 大型藻类野外实验：空间精度与斑块解释

按论文的 5×30 网格、二维平面去趋势和 0–75% 冠层移除梯度复现空间统计。斑块以单元中藻坪覆盖至少 50% 为主定义；它们只标记空间组织，不用于定义临界慢化标签。

| metric | metric_role | year_adjusted_slope | one_sided_block_permutation_p | spearman_rho |
| --- | --- | --- | --- | --- |
| spatial_sd | published_spatial_ews | 0.2991 | 0.0010 | 0.7447 |
| spatial_cv | published_spatial_ews | 0.0084 | 0.0010 | 0.8234 |
| spatial_skewness | published_spatial_ews | 0.0574 | 0.0010 | 0.8567 |
| moran_lag1 | published_spatial_ews | 0.0023 | 0.0200 | 0.4329 |
| low_frequency_power | published_spatial_ews | 28.4380 | 0.2930 | 0.6327 |
| turf_fraction | patch_explanation | 0.0076 | 0.0010 | 0.8828 |
| component_density | patch_explanation | 2.000e-04 | 0.1810 | 0.2272 |
| largest_turf_patch_fraction | patch_explanation | 0.0057 | 0.0020 | 0.8244 |
| edge_density | patch_explanation | 0.0034 | 0.0010 | 0.8195 |

空间 SD、CV 和偏度的方向与分块置换结果均支持已发表趋势。低频功率方向为正，但本项目采用的预阈值线性分块检验与论文 GAMM 不同，置换检验未显著；因此只记为方向性复现。Moran 的简单线性斜率虽为正，原论文将其判为不稳定指标，本项目不把它计入预期成功数。

空间单元随机保留后的稳健性：

| sampling_fraction | metric | direction_retention_rate | median_rank_similarity |
| --- | --- | --- | --- |
| 1.0000 | component_density | 1.0000 | 1.0000 |
| 0.5000 | component_density | 1.0000 | 0.2259 |
| 0.2500 | component_density | 1.0000 | 0.1280 |
| 1.0000 | edge_density | 1.0000 | 1.0000 |
| 0.5000 | edge_density | 1.0000 | 0.9009 |
| 0.2500 | edge_density | 1.0000 | 0.7480 |
| 1.0000 | largest_turf_patch_fraction | 1.0000 | 1.0000 |
| 0.5000 | largest_turf_patch_fraction | 1.0000 | 0.8945 |
| 0.2500 | largest_turf_patch_fraction | 1.0000 | 0.8199 |
| 1.0000 | low_frequency_power | 1.0000 | 1.0000 |
| 0.5000 | low_frequency_power | 1.0000 | 0.8732 |
| 0.2500 | low_frequency_power | 1.0000 | 0.7392 |
| 1.0000 | moran_lag1 | 1.0000 | 1.0000 |
| 0.5000 | moran_lag1 | 0.9900 | 0.6272 |
| 0.2500 | moran_lag1 | 0.8000 | 0.3836 |
| 1.0000 | spatial_cv | 1.0000 | 1.0000 |
| 0.5000 | spatial_cv | 1.0000 | 0.9885 |
| 0.2500 | spatial_cv | 1.0000 | 0.9725 |
| 1.0000 | spatial_sd | 1.0000 | 1.0000 |
| 0.5000 | spatial_sd | 1.0000 | 0.9685 |
| 0.2500 | spatial_sd | 1.0000 | 0.9179 |
| 1.0000 | spatial_skewness | 1.0000 | 1.0000 |
| 0.5000 | spatial_skewness | 1.0000 | 0.9809 |
| 0.2500 | spatial_skewness | 1.0000 | 0.8754 |
| 1.0000 | turf_fraction | 1.0000 | 1.0000 |
| 0.5000 | turf_fraction | 1.0000 | 0.9878 |
| 0.2500 | turf_fraction | 1.0000 | 0.9718 |

斑块阈值敏感性（25%、50%、75% 藻坪覆盖）：

| minimum_turf_cover_percent | metric | year_adjusted_slope | one_sided_block_permutation_p |
| --- | --- | --- | --- |
| 25 | turf_fraction | 0.0089 | 0.0010 |
| 25 | component_density | -3.533e-04 | 0.9760 |
| 25 | largest_turf_patch_fraction | 0.0086 | 0.0010 |
| 25 | edge_density | 0.0013 | 0.0560 |
| 50 | turf_fraction | 0.0076 | 0.0010 |
| 50 | component_density | 2.000e-04 | 0.1660 |
| 50 | largest_turf_patch_fraction | 0.0057 | 0.0030 |
| 50 | edge_density | 0.0034 | 0.0010 |
| 75 | turf_fraction | 0.0053 | 0.0020 |
| 75 | component_density | 7.100e-04 | 0.0020 |
| 75 | largest_turf_patch_fraction | 0.0030 | 0.0050 |
| 75 | edge_density | 0.0040 | 0.0010 |

该实验清楚显示了斑块的合适用途：随着冠层移除，较大的藻坪连通域与更多边界出现，解释了空间方差、偏度和低频结构为何改变。但这些斑块是受控压力的空间响应，不等同于独立的恢复率测量。

## 4. 对“是不是原数据精度太低”的判断

### 精度确实贡献了失败

1. 高频捕食—猎物数据被稀释后，快速崩溃种群很快失去足够的转折前观测。
2. 酵母平行重复数减少后，指标曲线对完整 48 重复结果的排序一致性下降。
3. 大型藻类空间单元减少后，尤其是连通域数量和边界等斑块指标，更容易被缺测人为打碎。

### 但精度不是唯一或主要可修复原因

1. 酵母在相同 48 重复、相同日频率下，不同环境驱动仍产生不同的指标强度。
2. 大型藻类原论文及本复现中，不同空间指标表现不同；Moran 并不是稳定单调信号。
3. 盐沼的直接恢复率在两个地点均成功，而斑块的跨地点增量不稳定。这是“直接动力学量成功、形态代理失败”，不是没有 CSD。
4. 盐沼只有 8/11 个不规则日期，这严重限制动态斑块谱系；但每幅图的 0.25 m 空间分辨率本身并不低。

所以更准确的诊断是：**时间采样稀疏降低了动态斑块检验力，二值分类和环境异质性又削弱了特异性；即使增加空间像元，形态代理仍未必含有恢复率之外的信息。**

## 5. 推荐修改方向

1. 不恢复 PWSI/斑块作为主检测器，也不重新调权重。
2. 用四个已发表基准形成“检测层”：直接恢复率、CV/AC1、论文预先给定的复合指标。
3. 斑块形成独立“解释层”：覆盖、最大连通域、边界和连通域密度；明确报告阈值与缺测敏感性。
4. 在新数据选择上，优先获取“高频时间 + 重复扰动/对照 + 2-D 空间图”的组合；单纯更高像元分辨率不是充分条件。
5. 论文主线可改为：**直接动力学证据如何受到观测设计影响，以及空间斑块何时只解释、何时能提供额外信息。**

## 可复现判定

```json
{
  "published_datasets_run": 4,
  "clements_transition_set_exact": true,
  "clements_trait_composite_true_positive_rate": 0.625,
  "clements_trait_composite_false_positive_rate": 0.1,
  "dai_cv_positive_for_both_drivers": true,
  "rindi_expected_spatial_metrics_positive": 4,
  "rindi_expected_spatial_metrics_total": 4,
  "rindi_expected_spatial_metrics_block_significant": 3,
  "rindi_low_frequency_replication": "positive direction; blocked p > 0.05",
  "rindi_positive_patch_descriptors": 4,
  "rindi_patch_descriptors_total": 4,
  "sampling_precision_contributes": true,
  "sampling_precision_is_sole_explanation": false,
  "patch_primary_detector": "NO-GO",
  "patch_spatial_explanation": "GO, descriptive and sensitivity-audited"
}
```
