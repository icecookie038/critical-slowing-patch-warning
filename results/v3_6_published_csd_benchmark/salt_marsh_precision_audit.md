# 盐沼斑块精度与尺度审计

## 结论

原始 0.25 m 数据没有挽救斑块增量，因此此前失败不能归因于 1 m 聚合造成的信息损失。5 m 在预设 `alpha=0.1` 下出现双向 deviance 改善，但该信号没有同时改善两向 log-RMSE，并且对正则化强度和分析单元宽度敏感。

这更符合“存在可能的生态尺度/去噪窗口”，而不是“空间精度越高越能检测临界慢化”。斑块继续作为空间解释和尺度敏感性结果，不升级为主要检测器。

## 像元分辨率

| patch_resolution_m | train_site | test_site | deviance_change_vs_stress_percent | log_rmse_change_vs_stress_percent |
| --- | --- | --- | --- | --- |
| 0.250 | Paulina | Hellegat | -12.641 | -4.900 |
| 0.250 | Hellegat | Paulina | 3.827 | 6.285 |
| 1.000 | Paulina | Hellegat | -10.751 | -4.279 |
| 1.000 | Hellegat | Paulina | 11.659 | 12.943 |
| 5.000 | Paulina | Hellegat | -10.020 | -4.050 |
| 5.000 | Hellegat | Paulina | -8.172 | 2.333 |
| 10.000 | Paulina | Hellegat | -11.313 | -4.618 |
| 10.000 | Hellegat | Paulina | 103.129 | 57.340 |

## 5 m 正则化敏感性

| patch_resolution_m | ridge_alpha | train_site | test_site | deviance_change_vs_stress_percent | log_rmse_change_vs_stress_percent |
| --- | --- | --- | --- | --- | --- |
| 5.000 | 0.010 | Paulina | Hellegat | -50.752 | -8.974 |
| 5.000 | 0.010 | Hellegat | Paulina | 12.150 | 15.645 |
| 5.000 | 0.100 | Paulina | Hellegat | -10.020 | -4.050 |
| 5.000 | 0.100 | Hellegat | Paulina | -8.172 | 2.333 |
| 5.000 | 1.000 | Paulina | Hellegat | -2.199 | -0.954 |
| 5.000 | 1.000 | Hellegat | Paulina | 5.995 | 4.501 |
| 5.000 | 10.000 | Paulina | Hellegat | -0.253 | -0.115 |
| 5.000 | 10.000 | Hellegat | Paulina | 0.910 | 0.502 |

## 5 m 分析单元敏感性

| block_width_m | train_site | test_site | deviance_change_vs_stress_percent | log_rmse_change_vs_stress_percent |
| --- | --- | --- | --- | --- |
| 32.000 | Paulina | Hellegat | -9.998 | -3.962 |
| 32.000 | Hellegat | Paulina | 25.118 | 18.916 |
| 64.000 | Paulina | Hellegat | -10.020 | -4.050 |
| 64.000 | Hellegat | Paulina | -8.172 | 2.333 |
| 128.000 | Paulina | Hellegat | -5.098 | -1.608 |
| 128.000 | Hellegat | Paulina | -32.867 | -17.439 |

## 事后候选：5 m 像元 × 128 m 单元

该组合在 `alpha=0.01` 和 `0.1` 时同时改善两向 deviance 与 log-RMSE，但在更强正则化下消失。更重要的是，它只含 Hellegat 10 个、Paulina 15 个空间块，未达到预设的每站至少 20 个空间块要求。因此它是下一数据集应预注册检验的尺度候选，不是当前的确证结果。

| patch_resolution_m | block_width_m | ridge_alpha | train_site | test_site | deviance_change_vs_stress_percent | log_rmse_change_vs_stress_percent |
| --- | --- | --- | --- | --- | --- | --- |
| 5.000 | 128.000 | 0.010 | Paulina | Hellegat | -33.767 | -8.620 |
| 5.000 | 128.000 | 0.010 | Hellegat | Paulina | -55.360 | -34.583 |
| 5.000 | 128.000 | 0.100 | Paulina | Hellegat | -5.098 | -1.608 |
| 5.000 | 128.000 | 0.100 | Hellegat | Paulina | -32.867 | -17.439 |
| 5.000 | 128.000 | 1.000 | Paulina | Hellegat | -1.293 | -0.073 |
| 5.000 | 128.000 | 1.000 | Hellegat | Paulina | 0.797 | 0.926 |
| 5.000 | 128.000 | 10.000 | Paulina | Hellegat | -0.168 | 0.005 |
| 5.000 | 128.000 | 10.000 | Hellegat | Paulina | 0.261 | 0.146 |

| site | block_width_m | supported_units | represented_blocks | recoveries | exposure_years |
| --- | --- | --- | --- | --- | --- |
| Hellegat | 128.000 | 137 | 10 | 440789 | 4541974.000 |
| Paulina | 128.000 | 73 | 15 | 451694 | 3456375.000 |

## 可复现判定

```json
{
  "resolutions_tested_m": [
    0.25,
    1.0,
    5.0,
    10.0
  ],
  "resolutions_passing_two_way_deviance": [
    5.0
  ],
  "resolutions_passing_two_way_deviance_and_log_rmse": [],
  "five_metre_alphas_tested": [
    0.01,
    0.1,
    1.0,
    10.0
  ],
  "five_metre_alphas_passing_two_way_deviance": [
    0.1
  ],
  "five_metre_block_widths_passing_two_way_deviance": [
    64.0,
    128.0
  ],
  "candidate_5m_128m_alphas_passing_both_losses": [
    0.01,
    0.1
  ],
  "candidate_5m_128m_supported_blocks": {
    "Hellegat": 10,
    "Paulina": 15
  },
  "candidate_5m_128m_data_requirement_pass": false,
  "candidate_5m_128m_status": "exploratory; pre-register for new data",
  "higher_native_resolution_rescues_patch_model": false,
  "scale_sensitivity_present": true,
  "precision_is_sole_explanation": false,
  "patch_primary_detector": "NO-GO",
  "patch_spatial_explanation": "GO"
}
```
