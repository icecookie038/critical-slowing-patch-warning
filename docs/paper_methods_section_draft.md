# Draft Methods Section

## 2. Methods

### 2.1 Overview of the proposed framework

This study develops and evaluates a patch-based early-warning framework for spatial critical-transition risk prediction. The overall workflow consists of five main steps:

```text
1. Generate spatial SEIR simulation data.
2. Extract image sequences and patch-structure features from spatial infection fields.
3. Construct traditional early-warning signal features for comparison.
4. Train deep learning models with different feature groups.
5. Evaluate prediction performance, lead-time proxy, and feature interpretability.
```

The main objective is to test whether patch-structure features provide a stronger and more interpretable representation for spatial critical-transition early warning than traditional early-warning signals.

The complete experiment chain includes:

```text
classic patch baseline
dynamic patch indicators
feature group comparison
traditional EWS baseline
lead-time proxy analysis
feature-value diagnostic importance
```

---

## 2.2 Spatial SEIR simulation model

A spatial SEIR simulation model was used as the baseline data-generating system. The model represents the spatial domain as a two-dimensional grid. Each grid cell can be interpreted as a local spatial unit whose state is related to disease or disturbance spreading.

The SEIR process contains four main epidemiological states:

```text
S: susceptible
E: exposed
I: infected
R: recovered
```

At each simulation step, local spreading occurs through spatial interactions among neighboring cells. The infected region evolves over time and forms spatial patches with different sizes, shapes, aggregation levels, and boundary structures.

The simulation was designed to generate spatial spreading trajectories with different outbreak patterns. These trajectories were then used to construct supervised learning samples for risk prediction.

The main simulation setting used in the formal experiments was:

```text
num_sims = 300
grid = 64 × 64
sim_steps = 200
prediction horizon = 30
random seeds = 42, 123, 2026, 3407, 7777
```

For each seed, 300 independent simulations were generated. The same data-generating setting was used across the main feature comparison experiments to ensure fair comparison among methods.

---

## 2.3 Risk prediction task

The prediction task was formulated as a binary early-warning classification problem.

For each simulation trajectory, a sequence of spatial states was used as input. The model was trained to predict whether the system would enter a high-risk state within a future prediction horizon.

The risk label can be described as:

```text
y_risk = 1 if a critical or high-risk transition occurs within the prediction horizon
y_risk = 0 otherwise
```

The prediction horizon was set to:

```text
horizon = 30 time steps
```

Therefore, the model learns whether the current spatial structure contains warning information about an upcoming critical transition within the next 30 time steps.

In addition to risk classification, the pipeline also stores the remaining time to transition for the validation samples. This value is later used to compute a lead-time proxy.

---

## 2.4 Image sequence representation

For image-based inputs, each sample is represented as a sequence of spatial infection fields.

The image input has the general form:

```text
X_img: (N, T, C, H, W)
```

where:

```text
N = number of samples
T = input sequence length
C = number of channels
H = grid height
W = grid width
```

In the current implementation, the spatial infection field is represented as a single-channel grid. This image sequence is used by the convolutional branch of the deep learning model.

The image-only feature group uses only this spatial image sequence as model input.

---

## 2.5 Patch extraction

Patch extraction was performed on the spatial infection fields. The infection field was first converted into a binary target region using a threshold. Connected infected cells were then identified as spatial patches.

The basic workflow is:

```text
spatial infection field
→ thresholded infected region
→ connected component labeling
→ infected patches
→ patch-structure feature extraction
```

A patch is defined as a connected component of infected cells. Patch-level information was then summarized into numerical features describing the spatial structure of the infection pattern.

This patch-based representation is designed to capture spatial warning information such as patch expansion, aggregation, dominance, fragmentation, and displacement.

---

## 2.6 Classic patch-structure features

Classic patch features describe the spatial structure of infected patches at each time step.

The main classic patch features include:

```text
patch_count
total_patch_area
largest_patch_area
mean_patch_area
largest_patch_ratio
patch_area_std
patch_area_cv
gini
edge_density
aggregation_index
centroid_x
centroid_y
spread_x
spread_y
perimeter
perimeter_area_ratio
compactness
fragmentation
nearest_patch_distance
patch_density
```

These indicators describe several aspects of spatial structure:

```text
patch abundance
infected area
dominant patch formation
patch-size heterogeneity
spatial aggregation
centroid displacement
spatial spread
boundary and perimeter structure
fragmentation
patch density
```

The classic patch feature matrix has the form:

```text
X_patch: (N, T, F)
```

where `F` is the number of patch features.

The classic patch feature group uses these features as the main input for risk prediction.

---

## 2.7 Dynamic patch indicators

In addition to static patch-structure features, dynamic patch indicators were developed to describe short-term temporal changes in patch structure.

The first-generation dynamic patch indicators include:

```text
PDSI: Patch-size Distribution Shift Index
DPCI: Dominant Patch Change / Growth Index
FAI: Fragmentation Acceleration Index
PSII: Patch-Structure Instability Index
```

These indicators aim to capture changes in patch-size distribution, dominant patch growth, fragmentation acceleration, and structural instability over time.

The dynamic patch feature group uses these dynamic indicators only.

The classic + dynamic patch feature group combines classic patch features and dynamic patch indicators.

---

## 2.8 Second-generation dynamic patch indicators

A second-generation dynamic patch indicator set was also tested as a diagnostic extension.

The v2 indicators include:

```text
BCI: Boundary Complexity Index
FPV: Front Propagation Velocity
PMR: Patch Merging Rate
GCGR: Giant Component Growth Rate
PPI: Percolation Proximity Index
```

These indicators were designed to better describe spatial spreading mechanisms, including boundary complexity, front movement, patch merging, giant component formation, and percolation-like expansion.

However, the v2 indicators were only evaluated as a diagnostic experiment under seed 42. Because they did not provide sufficient predictive improvement under the current SEIR setting, they were not included in the main five-seed method comparison table.

Thus, dynamic patch v2 is treated as an exploratory supplementary analysis rather than a formal main method.

---

## 2.9 Traditional early-warning signal baseline

Traditional early-warning signals were implemented as an important baseline.

The traditional EWS feature set includes:

```text
total_infected
temporal_variance
temporal_ac1
temporal_skewness
return_rate
kendall_trend
spatial_variance
moran_i
spatial_skewness
patch_size_slope
```

These features cover common temporal and spatial early-warning signals.

Temporal indicators include:

```text
variance
lag-1 autocorrelation
skewness
return rate
Kendall trend
```

Spatial indicators include:

```text
spatial variance
Moran's I
spatial skewness
patch-size distribution slope
```

The traditional EWS baseline allows the proposed patch-based framework to be compared with standard early-warning approaches.

The traditional EWS feature group is denoted as:

```text
traditional_ews_only
```

---

## 2.10 Feature groups

To evaluate the contribution of different information sources, six main feature groups were compared:

```text
traditional_ews_only
image_only
classic_patch_only
dynamic_patch_only
classic_dynamic_patch
full
```

Their meanings are:

| Feature group | Description |
|---|---|
| traditional_ews_only | Uses only traditional early-warning signal features |
| image_only | Uses only spatial image sequences |
| classic_patch_only | Uses only classic patch-structure features |
| dynamic_patch_only | Uses only first-generation dynamic patch indicators |
| classic_dynamic_patch | Uses classic patch features and dynamic patch indicators |
| full | Uses image sequence and patch-based features together |

This design allows the study to test whether patch-structure features outperform traditional EWS and image-only baselines, and whether dynamic patch indicators provide additional predictive information.

---

## 2.11 Deep learning prediction model

A deep learning model was used to predict future critical-transition risk.

The model contains two major input pathways:

```text
1. A convolutional neural network branch for spatial image sequences.
2. A temporal sequence branch for patch or EWS feature sequences.
```

The image branch extracts spatial representations from infection fields. The patch feature branch processes time-dependent numerical features using a recurrent neural network structure.

The model outputs:

```text
risk probability
```

For some settings, the model also contains a remaining-time prediction output. However, when the remaining-time regression weight is set to zero, the main optimization target is risk classification.

The model is trained using the same training configuration across different feature groups to ensure fair comparison.

---

## 2.12 Training and validation setting

For each random seed, spatial SEIR simulations were generated and converted into supervised learning samples.

The data were split by simulation ID to reduce leakage between training and validation samples. This ensures that validation samples come from different simulation trajectories rather than merely different time windows from the same trajectory.

The main formal experiment setting was:

```text
num_sims = 300
grid = 64
sim_steps = 200
horizon = 30
seeds = 42, 123, 2026, 3407, 7777
```

Each method was evaluated over five random seeds. Reported results are shown as:

```text
mean ± standard deviation
```

across the five seeds.

---

## 2.13 Evaluation metrics

The main classification metrics include:

```text
AUC
AUPRC
accuracy
F1 score
precision
recall
```

AUC measures the overall ranking ability of the model. AUPRC is especially useful when the positive risk class is imbalanced. F1 score summarizes the balance between precision and recall under a selected classification threshold.

Precision and recall are interpreted as:

```text
precision = proportion of predicted alarms that are true risk samples
recall = proportion of true risk samples that are successfully detected
```

The best threshold was selected based on validation performance.

---

## 2.14 Lead-time proxy analysis

To evaluate whether the model provides early warnings before critical transition, a lead-time proxy analysis was conducted.

The current prediction files contain:

```text
y_remaining_true
y_remaining_pred
y_risk_true
risk_prob
```

However, they do not contain:

```text
simulation ID
time index
critical time
```

Therefore, the current analysis reports a sample-level lead-time proxy rather than a strict trajectory-level first-alarm lead time.

The lead-time proxy is defined as:

```text
the average remaining time to critical transition among correctly predicted risk samples
```

This indicates how early the model tends to issue correct risk alarms at the sample level.

Because simulation-level first alarm timing cannot yet be reconstructed, the lead-time proxy should be interpreted cautiously.

A future version should save simulation ID, time index, critical time, and remaining time for each validation sample to enable strict first-alarm lead-time analysis.

---

## 2.15 Feature-value diagnostic importance

A feature-value diagnostic importance analysis was conducted for the classic patch features.

This analysis does not estimate neural network attribution directly. Instead, it measures how strongly each classic patch feature separates risk and non-risk samples.

Three statistics were used:

```text
absolute Cohen's d
single-feature AUC
absolute mean difference
```

For each classic patch feature, sample-level feature values were summarized across the input sequence. Risk and non-risk samples were then compared.

Features with larger absolute Cohen's d and higher single-feature AUC were interpreted as having stronger risk/non-risk separation.

This analysis was used to identify which patch-structure indicators provide the most interpretable warning information.

---

## 2.16 Result aggregation and table generation

After training, validation results were collected from all feature groups and random seeds.

The main result summaries were saved as CSV files and paper-ready tables.

The main output tables include:

```text
docs/paper_tables/method_comparison_table.md
docs/paper_tables/lead_time_proxy_table.md
docs/paper_tables/feature_importance_table.md
```

The method comparison table reports classification performance. The lead-time proxy table reports alarm timing and alarm quality. The feature importance table reports the most discriminative patch features.

---

## 2.17 Reproducibility

All experiments were organized through versioned scripts and Git branches.

The main versions used in the current study are:

```text
v0.7   classic patch baseline
v0.8   dynamic patch indicators v1
v0.9   feature group comparison
v0.9.1 dynamic patch indicators v2 diagnostic
v1.0   traditional EWS baseline
v1.1   lead-time proxy and feature importance
```

This versioned workflow ensures that each experimental component can be traced and reproduced.

---

## 2.18 Methodological limitation

The current SEIR model is a baseline spatial spreading model and produces relatively smooth spatial expansion.

This may limit the predictive contribution of dynamic patch indicators, because many dynamic changes are already captured by classic patch-structure features.

In addition, the current lead-time result is only a proxy. A strict first-alarm lead-time analysis requires simulation-level temporal metadata.

Future work should extend the current framework by introducing:

```text
strict lead-time metadata
extended SEIR with spatial heterogeneity and stochasticity
spatial stochastic cellular automaton
external ecological or remote-sensing validation
```