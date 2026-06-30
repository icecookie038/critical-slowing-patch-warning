# v2.5 Control Experiments Summary

## Purpose

The v2.5 experiments were designed to test whether the patch-formation signals are robust precursor signals rather than artifacts of spatial distribution, random alarm timing, or arbitrary event alignment.

The control experiments include:

1. Spatial-shuffle control
2. Stable-window false-alarm control
3. Event-time permutation control

---

## v2.5A Spatial-shuffle control

Spatial shuffling preserves the marginal value distribution within each frame but destroys spatial arrangement.

The first-rise based shuffle comparison was not used as the main evidence because random shuffling may artificially create fragmentation, sharp boundaries, and local gradients.

The final interpretation is based on raw spatial-organization attenuation.

Main result:

- Moran's I and local neighbor correlation were strongly reduced after spatial shuffling.
- This was consistent in both Vegetation CA and SEIR.
- Full and late precursor windows showed consistent spatial attenuation.

Interpretation:

The key prepatch indicators capture genuine spatial organization rather than only global mean, variance, or marginal value distribution.

---

## v2.5B Stable-window false-alarm control

The stable-window control compared far-stable windows with precursor windows.

Main result:

- Dynamic patch and visible patch indicators showed very low alarm rates in far-stable windows.
- Their alarm rates became nearly universal in full and late precursor windows.
- Prepatch and PWSI showed earlier responses, especially in Vegetation CA.

Interpretation:

Dynamic and visible patch indicators are more stage-specific near-transition signals. Prepatch and PWSI should be interpreted as earlier spatial-organization signals rather than late-stage patch manifestation indicators.

---

## v2.5C Event-time permutation control

Event-time permutation randomly permutes transition times across simulations and recomputes precursor-window alarm rates.

Main result:

Vegetation CA:
- The early precursor window provides the strongest evidence.
- Real event-aligned alarm rates were higher than permuted event-aligned alarm rates.
- Full and late precursor windows were close to saturation and therefore less informative.

SEIR:
- In the early precursor window, prepatch and PWSI were strongly aligned with true event times.
- Dynamic and visible patch indicators were not yet event-aligned in the early window.
- In full and late precursor windows, dynamic and visible patch indicators became strongly aligned with true event times.

Interpretation:

The two systems share a common early spatial-organization precursor layer, but their later patch-formation dynamics differ.

Vegetation CA shows a clearer gradual patch-formation process:

prepatch spatial organization -> dynamic restructuring -> visible fragmentation

SEIR shows a faster spreading process:

prepatch / PWSI early organization -> rapid dynamic spreading and visible patch manifestation

---

## Final interpretation

The v2.5 controls support the revised thesis focus:

Patch-formation indicators are not merely prediction features. They describe a measurable precursor process before critical transitions.

The most robust mechanistic evidence comes from:

- raw spatial attenuation of Moran's I and local neighbor correlation after spatial shuffling
- low far-stable alarm rates for dynamic and visible patch indicators
- event-time alignment of prepatch/PWSI and later dynamic/visible patch signals

## Caution

Do not overstate that all indicators have low far-stable false-alarm rates.

Prepatch and PWSI may activate earlier than dynamic and visible patch indicators. They should be described as early spatial-organization signals rather than false positives.

Do not use first-rise spatial-shuffle results as the main evidence, because random shuffling can create artificial fragmentation.
