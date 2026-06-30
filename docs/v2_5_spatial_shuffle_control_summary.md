# v2.5A Spatial-shuffle control summary

## Purpose

This experiment tests whether the proposed prepatch and patch-formation indicators depend on spatial organization rather than only marginal value distributions.

Spatial shuffling preserves the value distribution within each frame, but destroys spatial arrangement.

## Main conclusion

The first-rise based shuffle comparison is not suitable as the main evidence, because random pixel shuffling may create artificial fragmentation, sharp boundaries, and local gradients. Therefore, the main evidence is based on raw spatial-organization attenuation.

## Key result

In both Vegetation CA and SEIR systems, core spatial-organization indicators are strongly attenuated after spatial shuffling.

The strongest evidence comes from:

- prepatch_moran_i
- prepatch_local_neighbor_corr_mean

In the late and full precursor windows, both systems show consistent original > shuffled values.

This indicates that the prepatch signal is genuinely spatial and cannot be explained only by global mean, variance, or marginal value distribution.

## Interpretation

Vegetation CA:
Spatial shuffling strongly reduces Moran's I and local neighbor correlation, especially in full and late precursor windows. This supports the claim that vegetation degradation is preceded by spatial organization before visible fragmentation.

SEIR:
Spatial shuffling also strongly reduces Moran's I and local neighbor correlation. This supports the claim that infection spreading is preceded by local spatial synchronization before visible outbreak patches.

## How to use this in the paper

The spatial-shuffle control should be reported as a structural validation experiment.

Recommended wording:

Spatial-shuffle controls showed that the key prepatch indicators, especially Moran's I and local neighbor correlation, were strongly attenuated when the spatial arrangement was destroyed while preserving marginal value distributions. This confirms that the prepatch signals reflect genuine spatial organization rather than global changes in mean or variance.

## Caution

PWSI is a composite index and may behave differently across baseline and early precursor windows. It should be used as an applied warning index, while raw prepatch indicators should be used as the primary mechanism-level evidence.

First-rise based shuffle results should not be used as the main spatial-shuffle evidence.
