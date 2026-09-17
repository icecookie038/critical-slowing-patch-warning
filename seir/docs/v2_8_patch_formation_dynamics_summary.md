# v2.8 Patch Formation Dynamics Model Summary

## 1. Purpose

The purpose of v2.8 is to move beyond descriptive patch indicators and construct a low-dimensional patch formation dynamics framework.

The central question is:

Can patch formation before ecological critical transitions be represented as a staged and dynamically coupled process?

To answer this question, we constructed five state variables:

- P_state: prepatch spatial organization
- D_state: dynamic patch restructuring
- B_state: boundary complexity
- V_state: visible patch manifestation
- F_state: overall patch formation activation

These states were used to analyze temporal ordering, lagged relationships, state growth, and minimal state-space dynamics.

---

## 2. v2.8A State-variable construction

The first step constructed normalized 0-1 patch formation state variables.

In Vegetation CA, P_state increased earliest, followed by B_state and D_state, while V_state appeared later and closer to the transition. This supports a gradual patch formation process:

P_state -> B_state / D_state -> V_state -> transition

In SEIR, P_state also increased earlier, but D_state, B_state, and V_state rose rapidly near the transition. This indicates a compressed late-stage spreading process:

P_state -> rapid D/B/V activation -> transition

---

## 3. v2.8B Lag-correlation analysis

Lag-correlation analysis quantified whether earlier states lead later states.

In Vegetation CA, P_state showed positive leading lags with D_state, B_state, and V_state. D_state and B_state also showed positive leading relationships with V_state. This supports a gradual state progression from prepatch organization to visible patch manifestation.

In SEIR, P_state also led D_state, B_state, and V_state, but the best lags were shorter. This suggests that SEIR has early spatial organization followed by rapid late-stage coupling among dynamic, boundary, and visible patch states.

---

## 4. v2.8C-fixed State-increment regression

State-increment regression tested whether current states explain future state growth.

In SEIR, P_state positively explained future growth of D_state, B_state, V_state, and F_state. D_state and B_state also positively explained future V_state growth. This supports a coupled spreading process in which prepatch organization precedes later visible patch formation.

In Vegetation CA, P_state positively explained future growth of D_state and B_state, while D_state positively explained future V_state growth. P_state had weak direct contribution to V_state growth, suggesting that visible patch formation is mediated through dynamic restructuring and boundary complexity.

Therefore, the main dynamic pathway in Vegetation CA is:

P_state -> D_state / B_state -> V_state

The main dynamic pathway in SEIR is:

P_state -> D_state / B_state / V_state

---

## 5. v2.8D Minimal state-space simulation

A minimal state-space model was constructed:

X(t+1) = A X(t) + c

where:

X(t) = [P_state, D_state, B_state, V_state, F_state]

Two models were compared:

1. Diagonal baseline: each state only predicts itself.
2. Full coupled model: all states can interact.

In SEIR, the full coupled model clearly improved recursive reconstruction relative to the diagonal baseline. The recursive ALL R2 increased from approximately -0.039 to 0.369, and RMSE decreased from approximately 0.290 to 0.215. The full coupled model also better recovered the expected temporal ordering, especially P_state before D_state, B_state, and V_state.

In Vegetation CA, the diagonal baseline had higher overall recursive R2 than the full coupled model. However, the full coupled model better reconstructed V_state and dramatically improved order consistency. In particular, the consistency of B_state before V_state and D_state before V_state increased from near zero in the diagonal baseline to 1.0 in the full coupled model.

This suggests that Vegetation CA has strong self-persistent smooth state trajectories, but the emergence of visible patch manifestation is better explained by coupled state dynamics.

---

## 6. Main conclusion

v2.8 demonstrates that patch formation before critical transitions can be represented as a low-dimensional staged dynamics process.

The evidence includes:

1. State trajectories showing ordered activation of P/D/B/V/F states.
2. Lag correlations showing that earlier states lead later states.
3. State-increment regression showing that current states explain future state growth.
4. State-space simulation showing that coupled state dynamics improve reconstruction of temporal ordering and visible patch formation.

Overall, patch formation should not be treated as a single static indicator. Instead, it can be interpreted as a staged and coupled process:

prepatch organization -> dynamic restructuring / boundary complexity -> visible patch manifestation -> critical transition

This provides a mechanistic explanation for precursor warning signals before ecological critical transitions.

---

## 7. Recommended use in manuscript

v2.8 should be written as a new Results subsection:

Patch formation can be represented as a low-dimensional staged dynamics process.

Recommended main figures:

1. P/D/B/V/F state trajectories.
2. State onset timing.
3. Lag-correlation or best leading lag summary.
4. State-increment regression coefficients.
5. State-space model order-consistency or recursive simulation summary.

The full performance figures can be placed in the Supplementary Materials.
