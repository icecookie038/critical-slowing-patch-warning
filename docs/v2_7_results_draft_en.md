# v2.7 Results Draft

## 3. Results

### 3.1 Multi-level patch-formation stages emerge before critical transitions

We first examined whether patch formation before critical transitions could be decomposed into multiple precursor stages. Across the two spatial systems, the results show that patch-related signals generally appeared before the critical transition, but the temporal pattern differed between Vegetation CA and SEIR.

In Vegetation CA, prepatch spatial organization appeared earliest among the main indicator groups. PWSI, dynamic patch indicators, and boundary-related indicators also appeared before visible patch manifestation. Visible patch indicators showed shorter lead times, suggesting that visible fragmentation was a later-stage manifestation rather than the initial signal of degradation.

This pattern supports a gradual patch-formation process in Vegetation CA:

**prepatch spatial organization → dynamic restructuring / boundary complexity → visible fragmentation → critical transition**

In SEIR, prepatch and PWSI signals also appeared before dynamic and visible patch indicators. However, dynamic spreading and visible patch manifestation had shorter and more similar lead times. This indicates that SEIR has a more compressed late-stage process, where dynamic spreading and visible infected patches emerge rapidly once early spatial organization has formed.

Therefore, the two systems share an early spatial organization layer, but differ in their later patch-formation dynamics. Vegetation CA shows a clearer gradual degradation process, whereas SEIR shows early organization followed by rapid spreading and visible patch emergence.

This result supports the revised focus of the study: patch indicators are not only prediction features, but can also describe a measurable patch-formation process before critical transitions.

---

### 3.2 Vegetation CA shows a clearer temporal ordering of patch-formation stages

We next evaluated whether patch-formation stages followed a consistent temporal order. Three order metrics were used: weak order probability, strict order probability, and margin-based order probability.

Vegetation CA showed relatively high weak, strict, and margin-based order probabilities. This indicates that the expected stage order was not only generally preserved, but also separated by a meaningful temporal margin in many simulations.

This result suggests that Vegetation CA exhibits a relatively gradual and ordered patch-formation process. In this system, prepatch spatial organization tends to appear before dynamic restructuring and visible fragmentation. Therefore, Vegetation CA provides the clearest evidence for progressive patch formation before critical transition.

The order probability result is consistent with the stage lead-time analysis. Both results indicate that visible degradation patches are preceded by earlier spatial organization and patch restructuring signals.

---

### 3.3 SEIR shows weak temporal ordering but compressed late-stage dynamics

SEIR showed a different pattern. The weak order probability was high, indicating that early prepatch organization generally did not occur later than dynamic or visible patch signals. However, strict and margin-based order probabilities were low.

This does not imply that the patch-formation framework fails in SEIR. Instead, it suggests that SEIR has compressed late-stage dynamics. In a spatial spreading system, once local organization appears, dynamic spreading and visible patch manifestation may occur almost simultaneously.

Therefore, SEIR should be interpreted as a fast-spreading validation system rather than a gradual degradation system. The important result is that prepatch and PWSI signals still appeared as early organization layers, while later dynamic and visible patch signals were temporally close.

This supports a system-dependent interpretation:

**prepatch spatial organization is a shared early precursor layer, whereas later patch dynamics depend on the type of spatial process.**

---

### 3.4 Spatial-shuffle control confirms that prepatch indicators capture spatial organization

To test whether the prepatch signals truly depended on spatial structure, we performed a spatial-shuffle control. Spatial shuffling preserves the marginal value distribution within each frame, but destroys the spatial arrangement of values.

After spatial shuffling, Moran's I and local neighbor correlation were strongly attenuated in both Vegetation CA and SEIR. This result indicates that these indicators are sensitive to spatial organization rather than only to global mean, variance, or marginal value distribution.

This validation is important because it supports the mechanistic interpretation of prepatch indicators. The early warning signals are not simply caused by changes in the overall system state. Instead, they reflect the emergence of spatial organization before visible patch manifestation.

PWSI also showed attenuation after spatial shuffling, although the effect was weaker than raw spatial indicators such as Moran's I and local neighbor correlation. This is reasonable because PWSI is a composite index and may combine several types of precursor information.

Overall, the spatial-shuffle control supports the claim that prepatch indicators capture genuine spatial organization before critical transitions.

---

### 3.5 Stable-window control shows stage specificity of dynamic and visible patch indicators

We further compared alarm rates in far-stable windows and late precursor windows. This control was used to test whether indicators alarmed continuously throughout the trajectory or became more active near the transition.

Dynamic patch and visible patch indicators showed low alarm rates in far-stable windows and high alarm rates in late precursor windows. This indicates that these indicators are more stage-specific and are mainly associated with near-transition patch manifestation.

Prepatch and PWSI showed higher alarm rates in far-stable windows, especially in Vegetation CA. This should not be interpreted simply as false alarms. Instead, it suggests that these indicators may capture earlier spatial organization before visible patch manifestation. Therefore, the stable-window result supports a layered interpretation:

- prepatch and PWSI indicate early spatial organization;
- dynamic and visible patch indicators indicate later-stage patch restructuring and manifestation.

For this reason, the stable-window control is best treated as supplementary evidence rather than the main validation figure.

---

### 3.6 Event-time permutation confirms temporal alignment with true transitions

Finally, we used event-time permutation to test whether patch-formation signals were specifically aligned with the true transition time. In this control, transition times were randomly permuted across simulations, and alarm rates were recomputed around the permuted event times.

In Vegetation CA, alarm rates around the true event time were higher than those around permuted event times, especially in the early precursor window. This indicates that the early patch-formation signals were temporally associated with the actual transition process.

In SEIR, early prepatch and PWSI signals were more strongly aligned with true event times than with permuted event times. Dynamic and visible patch indicators became strongly event-aligned in the late precursor window. This supports the interpretation that SEIR has an early organization layer followed by rapid late-stage spreading and visible patch emergence.

The event-time permutation control therefore confirms that the proposed indicators do not simply produce arbitrary alarms. Instead, their activation is temporally related to the true critical transition process.

---

### 3.7 Summary of results

Together, the results support a multi-level patch-formation interpretation of precursor warning.

Vegetation CA demonstrates a relatively gradual patch-formation chain, in which prepatch organization appears before dynamic restructuring, boundary complexity, and visible fragmentation.

SEIR demonstrates an early spatial organization layer followed by rapid dynamic spreading and visible patch manifestation.

Spatial-shuffle control confirms that core prepatch indicators depend on spatial organization. Event-time permutation confirms that the signals are aligned with true transition times.

These results support the revised research direction:

**patch formation process analysis can strengthen precursor warning of ecological critical transitions.**
