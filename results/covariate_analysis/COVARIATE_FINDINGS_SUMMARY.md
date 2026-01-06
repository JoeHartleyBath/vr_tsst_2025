# Covariate Analysis Summary: Impact of Response Rate on Physiological Features

**Date:** 2025-02-18
**Script:** `scripts/analysis/covariates/rerun_covariate_analysis.R`
**Covariate:** `response_rate` (z-scored per participant/session)
**Objective:** To determine if observed experimental effects (Stress/Workload) are driven simply by behavioral performance differences (Response Rate).

## Executive Summary

Controlling for **Response Rate** does **not** diminish the experimental effects. In fact, for key EEG metrics, it **strengthens** the statistical significance of the Workload effect. This suggests that the neural signatures of Workload (Alpha/Beta Ratio, Theta/Beta Ratio, Frontal Theta) are robust indicators of the task state, distinct from behavioral success/failure.

## Detailed Findings by Feature

### 1. Alpha/Beta Ratio (`eeg_ab_ratio`)
*   **Baseline Model:** Significant Workload effect ($p = 0.0019$).
*   **Covariate Model:**
    *   **Response Rate:** Significant predictor ($p = 0.025$). Better performance predicts higher A/B ratio.
    *   **Workload Effect:** REMAINS Significant ($p = 0.0002$).
    *   **Effect Size:** The coefficient for Workload **increased** from 0.11 to 0.22.
*   **Interpretation:** The Workload effect is robust. Controlling for performance "de-noises" the data, making the workload impact clearer.

### 2. Theta/Beta Ratio (`eeg_tb_ratio`)
*   **Baseline Model:** Significant Workload effect ($p = 0.011$).
*   **Covariate Model:**
    *   **Response Rate:** Marginally significant ($p = 0.053$).
    *   **Workload Effect:** REMAINS Significant ($p = 0.002$).
    *   **Effect Size:** The coefficient for Workload **increased** from 0.12 to 0.24.
*   **Interpretation:** Similar to A/B ratio, the Workload effect is robust and strengthened by controlling for performance.

### 3. Frontal Midline Theta (`eeg_fm_theta_power`)
*   **Baseline Model:** Workload effect was **NOT** significant ($p = 0.20$).
*   **Covariate Model:**
    *   **Response Rate:** Marginally significant ($p = 0.052$).
    *   **Workload Effect:** **EMERGED as Significant** ($p = 0.021$).
*   **Interpretation:** This is a crucial finding. The suppressive effect of performance variability was masking the increase in Theta due to Workload. When performance is held constant, High Workload significantly increases Frontal Theta.

### 4. EDA Tonic Mean (`eda_tonic_mean`)
*   **Baseline Model:** Significant Stress effect ($p = 0.011$).
*   **Covariate Model:**
    *   **Response Rate:** Marginally significant ($p = 0.08$).
    *   **Stress Effect:** REMAINS Significant ($p = 0.005$).
*   **Interpretation:** The physiological stress response is robust to behavioral performance.

### 5. Other Features
*   **Heart Rate (`hr_med`):** Stress effect is marginal ($p \approx 0.07$ in both models).
*   **Feature Beta/Alpha Power:** No significant effects in either model.

## Conclusion

The addition of the new canonical features (`eeg_ab_ratio`, `eeg_tb_ratio`) has proven highly valuable. They show strong sensitivity to Workload that is independent of behavioral performance. Furthermore, the covariate analysis revealed a hidden Workload effect in `eeg_fm_theta_power`.

The pipeline is now validated to produce robust, statistically significant results that survive rigorous covariate testing.
