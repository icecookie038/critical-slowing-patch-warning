# v2.9 Discussion Draft: Patch Formation Dynamics

## 4.X Patch formation as a precursor dynamics process

The present results suggest that patch formation before critical transitions should be interpreted as a staged precursor dynamics process rather than a single static warning signal.

Previous early warning studies often focus on temporal indicators, spatial autocorrelation, variance, or visible patch statistics. In contrast, the present framework separates patch formation into multiple state layers: prepatch organization, dynamic restructuring, boundary complexity, visible patch manifestation, and overall formation activation.

This layered representation provides a more detailed interpretation of how spatial warning signals emerge before transitions.

---

## 4.X.1 From indicators to state dynamics

A key contribution of this study is that patch indicators were not only used as predictive features. Instead, they were reorganized into low-dimensional state variables that describe the formation process itself.

P_state captures early spatial organization before visible patches become obvious. D_state and B_state describe intermediate restructuring and boundary development. V_state captures visible patch manifestation. F_state summarizes the overall formation process.

This transformation from individual indicators to state variables allows the warning process to be analyzed dynamically.

---

## 4.X.2 System-dependent patch formation pathways

The two systems showed different patch formation pathways.

Vegetation CA showed a gradual degradation pathway:

P_state -> D_state / B_state -> V_state

This suggests that vegetation degradation patches develop through early organization, intermediate restructuring, and later visible manifestation.

SEIR showed a more compressed spreading pathway:

P_state -> rapid D_state / B_state / V_state activation

This reflects the nature of epidemic-like spatial spreading, where visible infected patches can emerge rapidly after early spatial organization is established.

Therefore, the proposed framework does not assume one universal patch formation pathway. Instead, it provides a general state-based framework that can represent different types of spatial transition dynamics.

---

## 4.X.3 Why coupled dynamics matter

The comparison between diagonal and full coupled state-space models further clarifies the role of state interactions.

In SEIR, the full coupled model clearly improved recursive reconstruction, indicating that state interactions are important for capturing the rapid spreading process.

In Vegetation CA, the diagonal baseline performed well for overall smooth trajectories, but the full coupled model better recovered visible patch manifestation and the ordering of D_state / B_state before V_state. This suggests that even when early states have strong self-persistence, visible patch emergence still depends on coupled state dynamics.

Thus, coupled dynamics are particularly important for explaining the transition from intermediate restructuring to visible patch manifestation.

---

## 4.X.4 Implications for early warning

The proposed patch formation dynamics framework has two implications for early warning.

First, early warning should not rely only on visible patch statistics. In both systems, prepatch organization emerged before visible patch manifestation. This means that relying only on visible patches may miss earlier warning opportunities.

Second, warning signals should be interpreted as a process. The sequence from P_state to D_state/B_state and then to V_state provides a mechanistic interpretation of how spatial warning signals develop.

This supports the idea of precursor warning rather than simple threshold-based alarm detection.

---

## 4.X.5 Limitations and future extensions

The current state-space model is intentionally minimal and data-driven. It is designed to test whether low-dimensional patch formation states can reconstruct trajectories and temporal ordering. It should not be interpreted as a complete mechanistic ecological equation.

Future work can extend this framework in several directions. SINDy may be used to discover sparse governing equations among P_state, D_state, B_state, and V_state. Reaction-diffusion or PDE models may provide a stronger mechanistic basis for spatial degradation and spreading processes. PINN-based approaches may be useful when explicit physical constraints are available. Generative diffusion models may be explored for spatial scenario generation or data augmentation, but they should not replace the interpretable patch formation dynamics framework.

In the current manuscript, these advanced models should be treated as future extensions rather than core results.

---

## 4.X.6 Main conclusion

Overall, the results indicate that patch formation before ecological critical transitions can be described as a low-dimensional staged and coupled dynamics process.

This strengthens the manuscript's main contribution:

patch-based early warning should move from static patch indicators toward dynamic patch formation analysis.
