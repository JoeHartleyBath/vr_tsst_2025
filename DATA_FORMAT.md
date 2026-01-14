# Data Interface Specification: Adaptive Workload Model

## Overview
This document defines the contract between the Data Preparation Pipeline (`pipelines/12_...`) and the Deep Learning Model (`models/deep_learning/`).

## 1. Input Data Format (From Data Prep)
**Format:** MNE Epochs (`.fif`)
**Location:** `output/adaptive_workload/mne_epochs/`

### Features (X)
*   **Source:** Cleaned EEG from `.set` files (128 channels).
*   **Time Resolution:** 10s windows, 5s overlap (50% overlap).
*   **Preprocessing:** 
    *   Subject-wise Z-scoring is recommended (StandardScaler per subject).
    *   **Baseline Note:** SVM testing showed that absolute power (subject-scaled) performed significantly better ($60.0\%$ mean accuracy, $87.1\%$ on P10) than pre-condition subtracted power ($35.7\%$). 
    *   Resampling to 250Hz is applied to reduce file size.

### Labels (y)
*   `0`: `LowStress_LowCog_Task` (Baseline Workload)
*   `1`: `LowStress_HighCog1022_Task` (High Workload)

## 2. Output Data Format (From Model)
**Format:** `.csv`
**Location:** `output/adaptive_workload/predictions/`

### Schema
| timestamp | subject_id | predicted_workload | confidence | latency_ms |
|-----------|------------|--------------------|------------|------------|
| ...       | ...        | ...                | ...        | ...        |

## 3. Real-time Simulation
*   How we simulate the stream for testing inference speed.
