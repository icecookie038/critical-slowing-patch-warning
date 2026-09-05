# v2.9 Results Draft: Patch Formation Dynamics

## 3.X Patch formation can be represented as a staged precursor dynamics process

To further investigate the formation process of spatial patches before critical transitions, we constructed a low-dimensional patch formation dynamics framework based on the existing patch indicators. Five state variables were defined: P_state for prepatch spatial organization, D_state for dynamic patch restructuring, B_state for boundary complexity, V_state for visible patch manifestation, and F_state for overall patch formation activation.

This analysis was designed to move beyond static indicator comparison and to test whether patch formation could be interpreted as a staged and dynamically coupled process.

---

### 3.X.1 Patch formation states show ordered activation before transitions

The normalized state trajectories revealed different patch formation patterns in Vegetation CA and SEIR.

In Vegetation CA, P_state increased earliest, followed by B_state and D_state, while V_state increased later and closer to the transition. This indicates that visible patch manifestation was not the initial signal of degradation. Instead, visible patches were preceded by prepatch spatial organization and intermediate restructuring processes.

The dominant formation pathway in Vegetation CA can therefore be summarized as:

P_state -> B_state / D_state -> V_state -> critical transition

In SEIR, P_state also increased earlier than the other states. However, D_state, B_state, and V_state rose more rapidly and more synchronously near the transition. This suggests a compressed late-stage spreading process, where early spatial organization is followed by rapid dynamic and visible patch activation.

The dominant formation pathway in SEIR can be summarized as:

P_state -> rapid D_state / B_state / V_state activation -> critical transition

These results indicate that both systems share an early prepatch organization layer, but differ in their later patch formation dynamics.

---

### 3.X.2 Lag-correlation analysis supports temporal leading relationships among states

Lag-correlation analysis was used to quantify whether earlier patch formation states tended to lead later states.

In Vegetation CA, P_state showed positive leading relationships with D_state, B_state, and V_state. D_state and B_state also showed positive leading relationships with V_state. This supports a gradual formation process from prepatch organization to dynamic restructuring and finally to visible patch manifestation.

In SEIR, P_state also led D_state, B_state, and V_state, but the best leading lags were shorter. This suggests that SEIR retains an early spatial organization stage, but its late-stage dynamic, boundary, and visible patch states are more temporally compressed.

Thus, lag-correlation results support the interpretation that patch formation is not a simultaneous process. Instead, it contains an ordered temporal structure, with system-specific differences in the duration of late-stage patch development.

---

### 3.X.3 State-increment regression identifies growth pathways among patch formation states

To test whether current patch formation states could explain future state growth, we performed state-increment regression. Unlike direct next-state prediction, this analysis focused on future growth increments, such as Delta_D, Delta_B, and Delta_V.

In SEIR, P_state positively explained future growth of D_state, B_state, V_state, and F_state. D_state and B_state also positively explained future growth of V_state. This indicates that prepatch spatial organization is associated with subsequent spreading, boundary development, and visible patch manifestation in the SEIR system.

In Vegetation CA, P_state positively explained future growth of D_state and B_state, while D_state positively explained future growth of V_state. P_state had weak direct contribution to V_state growth, suggesting that the effect of prepatch organization on visible patch manifestation is mediated through dynamic restructuring and boundary complexity.

Therefore, the state-increment results support two system-specific dynamic pathways:

Vegetation CA:
P_state -> D_state / B_state -> V_state

SEIR:
P_state -> D_state / B_state / V_state

These results strengthen the interpretation of patch formation as a coupled state-growth process rather than a set of independent static indicators.

---

### 3.X.4 Minimal state-space simulation supports coupled patch formation dynamics

Finally, we constructed a minimal state-space model to test whether the patch formation process could be reconstructed from low-dimensional state dynamics. The state vector was defined as:

X(t) = [P_state, D_state, B_state, V_state, F_state]

Two models were compared: a diagonal baseline model, in which each state only predicted itself, and a full coupled model, in which all states could interact.

In SEIR, the full coupled model substantially improved recursive trajectory reconstruction compared with the diagonal baseline. The recursive ALL R2 increased from approximately -0.039 to 0.369, while RMSE decreased from approximately 0.290 to 0.215. The full coupled model also better recovered the expected temporal order, especially P_state before D_state, B_state, and V_state.

In Vegetation CA, the diagonal baseline achieved higher overall recursive R2 than the full coupled model, indicating that some state trajectories had strong self-persistent smooth dynamics. However, the full coupled model better reconstructed V_state and substantially improved order consistency. In particular, the consistency of B_state before V_state and D_state before V_state increased from near zero in the diagonal baseline to 1.0 in the full coupled model.

This indicates that although Vegetation CA has strong smooth self-evolution in early and intermediate states, visible patch manifestation is better explained by coupled state dynamics.

---

### 3.X.5 Summary

Together, v2.8 results demonstrate that patch formation before critical transitions can be represented as a low-dimensional staged dynamics process.

The evidence includes:

1. Ordered activation of P_state, D_state, B_state, V_state, and F_state.
2. Positive leading relationships among patch formation states.
3. State-increment regression showing that current states explain future state growth.
4. State-space simulation showing that coupled dynamics improve reconstruction of temporal ordering and visible patch formation.

These findings support the central claim that patch formation is not merely a static spatial pattern. Instead, it is a staged and coupled precursor dynamics process that can provide mechanistic support for early warning of ecological critical transitions.
