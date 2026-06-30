# v2.6C Supervisor Report: Patch Formation Process and Precursor Warning

## 1. Research focus adjustment

Following the supervisor's suggestion on patch dynamic analysis and precursor warning, the focus of the project has been adjusted from pure prediction performance comparison to:

**Patch formation process analysis + precursor warning indicator framework**

The revised research question is:

**How do spatial patches form before critical transitions, and can this formation process provide precursor warning signals?**

Therefore, CNN/RNN models and conventional prediction models are no longer treated as the central contribution. Instead, they are retained as auxiliary validation tools. The main focus is now placed on the explanatory role of prepatch indicators, PWSI, dynamic patch indicators, and visible patch indicators in describing the patch formation process before critical transitions.

---

## 2. Indicator framework

The current indicator framework contains four major layers.

### 2.1 Prepatch spatial organization

This layer describes spatial organization before visible patches become apparent.

Representative indicators include:

- Moran's I
- Local neighbor correlation
- Sync edge ratio
- Boundary sharpness
- Spectral or modal organization indicators

These indicators are used to detect early spatial organization before obvious patch fragmentation or spreading.

### 2.2 PWSI application index

PWSI is used as a compact application-oriented warning index. It summarizes multiple prepatch signals into a more practical warning score.

In the current interpretation:

- raw prepatch indicators are mainly used for mechanism-level explanation;
- PWSI is mainly used as an applied precursor warning index.

### 2.3 Dynamic patch restructuring or spreading

This layer describes the dynamic evolution of patches after early organization has appeared.

In Vegetation CA, this layer mainly reflects:

- patch restructuring;
- boundary hardening;
- fragmentation tendency;
- patch connectivity changes.

In SEIR, this layer mainly reflects:

- infection patch spreading;
- growth of connected infected regions;
- rapid expansion of spatial outbreak patterns.

### 2.4 Visible patch manifestation

This layer describes already visible patch structures.

Representative indicators include:

- patch count;
- edge density;
- largest patch ratio;
- total patch ratio;
- visible patch metric.

This layer is usually later than the prepatch layer and is closer to visible ecological degradation or visible spreading.

---

## 3. Vegetation CA results

Vegetation CA shows a relatively clear gradual patch formation process.

The main pattern is:

**prepatch spatial organization → dynamic restructuring / boundary complexity → visible fragmentation → critical transition**

The stage lead-time result shows that prepatch indicators appear earlier than visible patch indicators. This suggests that vegetation degradation is not preceded only by visible fragmentation. Instead, before obvious visible patch fragmentation, the system has already developed spatial organization and dynamic restructuring signals.

The order probability result further supports this interpretation. In Vegetation CA, weak order, strict order, and margin-based order probabilities are relatively high, indicating a stable temporal ordering among patch formation stages.

Therefore, Vegetation CA can serve as the main system for demonstrating the gradual formation process of ecological patches before critical transitions.

---

## 4. SEIR results

SEIR shows a different but compatible process.

The main pattern is:

**prepatch / PWSI early spatial organization → rapid dynamic spreading and visible patch manifestation → critical transition**

Unlike Vegetation CA, SEIR does not show a clearly separated dynamic-to-visible stage. Instead, dynamic spreading and visible patch manifestation often occur nearly synchronously.

This is reasonable because SEIR is a spatial spreading system. Once local infection organization appears, spreading and visible infected patches may emerge rapidly.

Therefore, the SEIR result should not be interpreted as a failure of the patch formation framework. Instead, it shows that:

1. prepatch and PWSI form a robust early precursor layer;
2. later-stage patch dynamics depend on the specific spatial system;
3. in fast spreading systems, dynamic and visible patch signals may appear almost simultaneously.

The cross-system conclusion is therefore not that all systems share exactly the same stage order. The stronger and more defensible conclusion is:

**Prepatch spatial organization is a robust early precursor layer, while later patch formation dynamics are system-dependent.**

---

## 5. Control experiments

### 5.1 Spatial-shuffle control

Spatial shuffling preserves the marginal value distribution within each frame but destroys spatial arrangement.

The result shows that core spatial organization indicators, especially Moran's I and local neighbor correlation, are strongly attenuated after spatial shuffling.

This confirms that prepatch indicators capture genuine spatial organization rather than only global mean, variance, or marginal value distribution.

This is an important validation for the patch formation process because it shows that the early warning signals are spatially meaningful.

---

### 5.2 Stable-window false-alarm control

Stable-window control compares far-stable windows with late precursor windows.

The results show that dynamic patch and visible patch indicators have low alarm rates in far-stable windows but high alarm rates in late precursor windows.

This supports the interpretation that dynamic and visible patch indicators are more stage-specific near-transition signals.

Prepatch and PWSI may show earlier responses, especially in Vegetation CA. This should not be interpreted simply as false alarms. Instead, it suggests that they capture earlier spatial organization before visible patch manifestation.

Thus, the stable-window result supports a layered interpretation:

- prepatch and PWSI: early spatial organization;
- dynamic patch and visible patch: later-stage patch manifestation.

---

### 5.3 Event-time permutation control

Event-time permutation randomly permutes transition times across simulations. It tests whether precursor signals are aligned with the true transition time rather than arbitrary fake event times.

The results show that alarm rates around true event times are higher than those around permuted event times.

For Vegetation CA, the strongest evidence appears in the early precursor window.

For SEIR, the early precursor window shows strong alignment for prepatch and PWSI, while dynamic and visible patch indicators become strongly aligned in the late precursor window.

This supports the interpretation that:

1. early spatial organization appears before the true transition;
2. dynamic and visible patch indicators are closer to late-stage transition manifestation;
3. the temporal structure of patch formation differs between gradual degradation systems and fast spreading systems.

---

## 6. Main conclusions

The current results support the following conclusions.

### Conclusion 1

Critical transitions are preceded by measurable patch formation processes.

### Conclusion 2

Prepatch indicators can detect early spatial organization before visible patch manifestation.

### Conclusion 3

PWSI can be used as an applied composite precursor warning index.

### Conclusion 4

Vegetation CA shows a clearer gradual patch formation chain:

**prepatch organization → dynamic restructuring / boundary complexity → visible fragmentation**

### Conclusion 5

SEIR shows an early organization layer followed by rapid synchronized spreading and visible patch emergence:

**prepatch / PWSI → dynamic spreading + visible patch manifestation**

### Conclusion 6

Control experiments support the spatial and temporal specificity of the proposed indicators.

Spatial-shuffle control confirms that key prepatch indicators depend on spatial organization.

Event-time permutation control confirms that precursor signals are aligned with true transition times.

---

## 7. Recommended figures for supervisor report

The following four figures are recommended as the main figures for the supervisor report.

### Figure 1. Stage lead time

File:

`fig1_stage_lead_time_presentation.png`

Purpose:

This figure shows when different patch-formation stages appear before the critical transition.

Main message:

Prepatch and PWSI signals generally appear earlier than visible patch indicators. Vegetation CA shows a clearer gradual formation process, while SEIR shows early organization followed by rapid dynamic and visible patch emergence.

---

### Figure 2. Key order probability

File:

`fig2_key_order_probability_presentation.png`

Purpose:

This figure shows whether patch-formation stages follow the expected temporal order.

Main message:

Vegetation CA has strong weak, strict, and margin-based order probabilities, indicating a clearer staged process. SEIR has high weak order but low strict order, suggesting that dynamic spreading and visible patch manifestation often occur nearly simultaneously.

---

### Figure 3. Spatial-shuffle attenuation

File:

`fig3_spatial_shuffle_presentation.png`

Purpose:

This figure validates whether prepatch indicators depend on spatial organization.

Main message:

Moran's I and local correlation are strongly attenuated after spatial shuffling, confirming that prepatch indicators capture spatial structure rather than only global distributional changes.

---

### Figure 4. Event-time permutation control

File:

`fig5_event_time_permutation_presentation.png`

Purpose:

This figure tests whether precursor signals align with true transition times.

Main message:

Real event-aligned alarm rates are higher than permuted-event alarm rates, indicating that the signals are temporally associated with the true critical transition.

---

## 8. Figure kept as supplementary material

The stable-window false-alarm figure can be used as supplementary evidence.

File:

`fig4_stable_window_presentation.png`

Reason:

This figure is useful, but prepatch and PWSI may show high alarm rates in far-stable windows, especially in Vegetation CA. This requires careful explanation. Therefore, it is better placed in supplementary material or used only when discussing the layered interpretation of early and late indicators.

---

## 9. Current paper storyline

The current paper storyline can be written as follows:

Critical transitions are often preceded by spatial reorganization. Traditional early warning signals mainly focus on temporal statistics and may not explain how spatial patches form before transitions. This study proposes a multi-level patch formation indicator framework, including prepatch spatial organization, PWSI, dynamic patch restructuring or spreading, and visible patch manifestation.

Across Vegetation CA and SEIR systems, the results show that early spatial organization appears before visible patch formation. Vegetation CA exhibits a relatively clear gradual patch formation process, whereas SEIR shows an early organization layer followed by rapid dynamic spreading and visible patch emergence. Control experiments further confirm that these signals depend on spatial structure and are aligned with true transition times.

Therefore, patch formation analysis can strengthen precursor warning of ecological critical transitions.

---

## 10. Next work

The next stage should not focus on adding more models immediately.

The recommended next steps are:

1. Refine the Results section around the patch formation process.
2. Use the four main figures for the supervisor report.
3. Keep CNN/RNN prediction results as auxiliary validation.
4. Prepare a formal Methods section for prepatch, PWSI, dynamic patch, and control experiments.
5. Later, if time allows, add real remote-sensing or forest disturbance validation as external evidence.

The current manuscript direction can be summarized as:

**Patch formation process as precursor warning signals for ecological critical transitions**

