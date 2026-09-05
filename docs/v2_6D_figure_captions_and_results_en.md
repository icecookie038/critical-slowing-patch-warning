# v2.6D Figure Captions and Results Draft

## Main Figures

### Figure 1. Patch-formation stage lead time across systems

This figure summarizes the median lead time of different patch-formation stages before the critical transition.

In Vegetation CA, prepatch indicators appear earlier than visible patch indicators, suggesting that vegetation degradation is preceded by a gradual spatial organization process before visible fragmentation becomes dominant.

In SEIR, prepatch, PWSI, and boundary-related signals also appear earlier than dynamic and visible patch indicators. This suggests that even in a fast spreading system, early spatial organization can be detected before visible outbreak patches become dominant.

Overall, this figure supports the idea that patch formation is not an instantaneous event. Instead, it can be decomposed into multiple precursor stages.

---

### Figure 2. Key patch-formation order probability

This figure compares the temporal ordering of patch-formation stages in Vegetation CA and SEIR.

Vegetation CA shows relatively high weak, strict, and margin-based order probabilities. This indicates a clearer staged process, where prepatch organization is followed by dynamic restructuring and then visible fragmentation.

SEIR shows high weak order probability but low strict and margin-based probabilities. This means that the expected stage order is usually not violated, but dynamic spreading and visible patch manifestation often occur nearly simultaneously.

Therefore, SEIR should not be interpreted as a failure case. Instead, it reflects the fast spreading nature of epidemic-like spatial systems.

---

### Figure 3. Spatial-shuffle attenuation of core spatial indicators

This figure shows the effect of spatial shuffling on core spatial indicators.

Spatial shuffling preserves the marginal value distribution within each frame but destroys spatial arrangement. After shuffling, Moran's I and local correlation are strongly reduced in both Vegetation CA and SEIR.

This confirms that the prepatch indicators capture genuine spatial organization rather than only global mean, variance, or marginal value distribution.

This result provides important structural validation for the proposed patch-formation indicator framework.

---

### Figure 4. Event-time permutation control

This figure compares the alarm rate around true transition times with the alarm rate around randomly permuted transition times.

Positive values indicate that the real event-aligned alarm rate is higher than the permuted-event alarm rate.

In Vegetation CA, early precursor signals are more strongly aligned with the true transition time than with permuted event times.

In SEIR, early prepatch/PWSI signals and late dynamic/visible patch signals are strongly aligned with the true transition time.

This confirms that the detected patch-formation signals are temporally associated with the actual critical transition process.

---

## Results Draft

### 1. Cross-system patch-formation stages

The results show that spatial patch formation before critical transition can be decomposed into multiple stages. In Vegetation CA, the stage lead-time pattern suggests a gradual formation chain from prepatch spatial organization to dynamic restructuring and finally visible fragmentation. Prepatch indicators appear earlier than visible patch indicators, indicating that spatial degradation is already organized before obvious fragmentation becomes visible.

In SEIR, the stage pattern is different but compatible with the same conceptual framework. Prepatch and PWSI signals appear before dynamic and visible patch indicators, while dynamic spreading and visible patch manifestation tend to occur close together. This suggests that fast spreading systems may have a compressed late-stage patch formation process.

Therefore, the cross-system result supports a layered interpretation. Prepatch spatial organization is a robust early precursor layer, while later patch-formation dynamics are system-dependent.

---

### 2. Temporal ordering of patch-formation stages

The order probability analysis further supports the staged interpretation of patch formation. Vegetation CA shows relatively high weak, strict, and margin-based order probabilities, indicating that the patch-formation process is temporally ordered and gradual.

In contrast, SEIR shows high weak order probability but low strict and margin-based probabilities. This indicates that the expected order is generally preserved, but the temporal separation between dynamic spreading and visible patch manifestation is small. This is consistent with the fast spatial spreading mechanism of SEIR.

Thus, the two systems should not be forced into an identical stage order. A more defensible conclusion is that both systems share an early prepatch organization layer, while the later patch dynamics differ across systems.

---

### 3. Spatial-shuffle validation

The spatial-shuffle control confirms that the proposed prepatch indicators depend on spatial organization. Since spatial shuffling preserves the marginal value distribution but destroys spatial arrangement, the attenuation of Moran's I and local correlation after shuffling indicates that these indicators are not merely responding to global value changes.

This result strengthens the mechanistic interpretation of prepatch indicators. They capture spatial organization before visible patches emerge, rather than only reflecting changes in overall system mean or variance.

---

### 4. Event-time permutation validation

The event-time permutation control tests whether the detected precursor signals are aligned with true transition times. The results show that real event-aligned alarm rates are generally higher than permuted-event alarm rates.

In Vegetation CA, the clearest event alignment appears in the early precursor window. In SEIR, early prepatch/PWSI signals and late dynamic/visible patch signals are aligned with the true event time.

These results suggest that the proposed indicators are not arbitrary alarms. Instead, they are temporally associated with the real transition process.

---

## Summary Statement

The current results support the revised research focus: patch-formation indicators are not merely machine-learning features, but can describe a measurable precursor process before critical transitions.

Vegetation CA demonstrates a gradual patch-formation process, while SEIR demonstrates an early spatial organization layer followed by rapid spreading and visible patch manifestation. Spatial-shuffle and event-time permutation controls further support the spatial and temporal specificity of the proposed indicators.

The current manuscript direction can therefore be summarized as:

Patch formation process as precursor warning signals for ecological critical transitions.
