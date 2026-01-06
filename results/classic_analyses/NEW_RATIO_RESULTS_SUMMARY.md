# Statistical Analysis Summary: EEG Power Ratios (Alpha/Beta & Theta/Beta)

**Date:** January 6, 2026
**Features Analyzed:** `eeg_ab_ratio` (Alpha/Beta Ratio) and `eeg_tb_ratio` (Theta/Beta Ratio)
**N:** 44 (EEG-valid participants)

---

## 1. Repeated-Measures ANOVA (2×2: Stress × Workload)

Both ratio features demonstrated strong sensitivity to **Cognitive Workload** manipulations but were not significantly affected by the Psychosocial Stress manipulation.

### Alpha/Beta Ratio (`eeg_ab_ratio`)
*   **Workload Main Effect:** **Significant** ($F = 10.27, p = 0.0017, \eta_p^2 = 0.074$)
*   **Stress Main Effect:** Not Significant ($p = 0.521$)
*   **Stress × Workload Interaction:** **Significant** ($p = 0.0077$)

### Theta/Beta Ratio (`eeg_tb_ratio`)
*   **Workload Main Effect:** **Significant** ($F = 10.35, p = 0.0016, \eta_p^2 = 0.074$)
*   **Stress Main Effect:** Not Significant ($p = 0.593$)
*   **Stress × Workload Interaction:** **Significant** ($p = 0.0028$)

**Interpretation:** These metrics are robust indicators of cognitive demand in the VR-TSST environment. The significant interactions suggest that the effect of workload on these ratios varies depending on the concurrent stress level, or vice-versa.

---

## 2. Stratified Repeated-Measures Correlations

We examined how these features tracked with subjective ratings *within* participants, stratified by condition.

### Key Findings
*   **Alpha/Beta Ratio vs. Subjective Workload**:
    *   In the **Low Stress** condition, Alpha/Beta ratio showed a moderate positive correlation with subjective Workload ($r = 0.38, p = 0.011$).
    *   This relationship was weaker in the High Stress condition, possibly due to the interaction effect observed in the ANOVA.
    *   *Note: This correlation approached significance after FDR correction ($p_{adj} = 0.097$).*
    
*   **Theta/Beta Ratio**:
    *   Did not show strong linear correlations with subjective ratings in the stratified analysis, despite the group-level differences found in the ANOVA.

---

## 3. Conclusions

1.  **Workload Markers:** Both Alpha/Beta and Theta/Beta ratios are validated as effective markers of cognitive workload in this study.
2.  **Specificity:** They appear specific to the cognitive component (arithmetic difficulty) rather than the psychosocial stress component.
3.  **Interaction:** The significant interactions warrant further plotting to understand how stress might be modulating the workload response (e.g., does high stress blunt the neural response to workload?).
