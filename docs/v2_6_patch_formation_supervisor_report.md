# v2.6 Patch Formation Process Supervisor Report

## 1. Research focus adjustment

The current project focus has shifted from pure prediction performance to patch-formation process analysis for precursor warning.

The revised research question is:

How do spatial patches form before critical transitions, and can this formation process provide precursor warning signals?

This change does not invalidate previous work. Previous SEIR, Vegetation CA, prepatch, PWSI, dynamic patch, threshold sensitivity, and prediction experiments are now reorganized into a stronger mechanistic evidence chain.

---

## 2. Current evidence chain

The current evidence chain contains five parts:

1. Event-based transition labeling  
2. Cross-system spatial simulations  
3. Multi-level patch-formation indicators  
4. Patch-formation stage order analysis  
5. Control experiments validating spatial and temporal specificity  

The core indicator layers are:

- Prepatch spatial organization
- PWSI application index
- Dynamic patch restructuring / spreading
- Boundary complexity
- Visible patch manifestation

---

## 3. Vegetation CA result

Vegetation CA shows a relatively clear gradual patch-formation process.

The main pattern is:

prepatch spatial organization  
→ dynamic restructuring / boundary hardening  
→ visible fragmentation  
→ critical transition

This indicates that vegetation degradation is not preceded only by visible patch fragmentation. Before visible fragmentation becomes obvious, the system already shows spatial organization and dynamic restructuring signals.

This result supports the supervisor's suggestion that patch indicators can be used not only for prediction, but also for exploring the patch formation process.

---

## 4. SEIR result

SEIR shows a different but compatible process.

The main pattern is:

prepatch / PWSI early spatial organization  
→ rapid dynamic spreading and visible patch manifestation  
→ critical transition

Unlike Vegetation CA, SEIR does not show a clearly separated dynamic-to-visible stage. Dynamic spreading and visible patch manifestation often occur nearly synchronously.

This is reasonable because SEIR is a spatial spreading system. Once local infection organization appears, spreading and visible infection patches may emerge rapidly.

Therefore, the cross-system conclusion is not that all systems share exactly the same stage order. The stronger conclusion is:

Prepatch spatial organization is a robust early precursor layer, while later patch dynamics depend on the specific system.

---

## 5. Control experiments

### 5.1 Spatial-shuffle control

Spatial shuffling preserves the value distribution within each frame but destroys spatial arrangement.

The raw spatial-shuffle analysis shows that core spatial organization indicators, especially Moran's I and local neighbor correlation, are strongly attenuated after shuffling.

This means that the prepatch signals are genuinely spatial and cannot be explained only by global mean, variance, or marginal value distribution.

### 5.2 Stable-window false-alarm control

Dynamic patch and visible patch indicators have low alarm rates in far-stable windows and high alarm rates in precursor windows.

This supports their stage specificity.

Prepatch and PWSI sometimes activate earlier, especially in Vegetation CA. This should not be interpreted simply as false alarms. Instead, it suggests that they capture earlier spatial organization before visible patch manifestation.

### 5.3 Event-time permutation control

Event-time permutation tests whether precursor signals are aligned with true transition times.

Vegetation CA shows strongest event alignment in the early precursor window.

SEIR shows early event alignment for prepatch and PWSI, while dynamic and visible patch indicators become strongly aligned in full and late precursor windows.

This supports the interpretation that different systems have different patch-formation timing, but share an early spatial-organization precursor layer.

---

## 6. Main conclusion

The current results support the revised thesis:

Patch-formation indicators are not merely machine-learning features. They describe a measurable precursor process before critical transitions.

The strongest current conclusion is:

Spatial systems show early prepatch organization before visible patch manifestation. This early organization can be detected by prepatch indicators and summarized by PWSI. Later dynamic and visible patch indicators describe system-specific patch restructuring, fragmentation, or spreading processes.

---

## 7. Recommended figures for supervisor report

Use the following figures:

1. fig_v2_6_stage_lead_time.png  
   Purpose: show when different patch-formation stages appear before transition.

2. fig_v2_6_order_probability.png  
   Purpose: show whether the stage order is stable.

3. fig_v2_6_spatial_shuffle_raw_effect.png  
   Purpose: prove that prepatch indicators depend on spatial organization.

4. fig_v2_6_stable_window_effect.png  
   Purpose: show that dynamic/visible patch indicators are stage-specific.

5. fig_v2_6_event_time_permutation.png  
   Purpose: show that precursor signals align with true event times.

---

## 8. Next work

The next stage should not immediately add new models.

The next work should be:

1. Improve figure readability.
2. Write the Results section around patch-formation process.
3. Prepare a concise supervisor report.
4. Decide whether to add true no-transition simulations later.
5. Keep CNN/RNN prediction as auxiliary validation, not the main story.

---

## 9. Current paper storyline

The final paper storyline should be:

Critical transitions are often preceded by spatial reorganization. Traditional early warning signals focus mainly on temporal statistics and may not explain how spatial patches form. This study proposes a multi-level patch-formation indicator framework, including prepatch spatial organization, dynamic patch restructuring, visible patch manifestation, and PWSI. Across Vegetation CA and SEIR systems, the results show that early spatial organization appears before visible patch formation. Control experiments further confirm that these signals depend on spatial structure and are aligned with true transition times. Therefore, patch-formation analysis can strengthen precursor warning of ecological critical transitions.

