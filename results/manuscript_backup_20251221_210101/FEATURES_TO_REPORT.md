# MANUSCRIPT FEATURE NOTES
**Date:** December 19, 2025

## Features to INCLUDE in Manuscript:

### ✅ EEG Spectral Features:
- **Power ratios:** Theta/Beta, Alpha/Beta (cognitive load markers)
- **Band power:** Theta, alpha, beta across regions
- **Spectral entropy:** Frontal-midline, frontal left/right, temporal, parietal
  - Present in data: 22 spectral entropy features
  - Selected in 84% of stress folds (all domain)
  - Selected in 77% of stress folds (eeg domain)
  - Moderate selection for workload (32%)
- **Frontal asymmetry:** If present

### ✅ Peripheral Physiology:
- HRV (RMSSD, SDNN)
- Heart Rate (mean, variability)
- EDA/GSR (mean, variability)
- Pupil dilation

## Features to EXCLUDE from Manuscript:

### ❌ NOT in our data:
- **Sample Entropy (SampEn)** - NOT computed
- **Approximate Entropy (ApEn)** - NOT computed
- **EEG Connectivity:** wPLI, coherence, phase synchrony - NOT computed
- **Network measures:** Any functional connectivity - NOT computed

## Manuscript Language:

**Correct:** "We extracted physiological markers including EEG spectral features (power ratios, band power, spectral entropy), heart rate variability, and electrodermal activity."

**Avoid:** Mentioning "sample entropy," "approximate entropy," "connectivity," or "network" analyses.

## Data Confirmation:
- Total features in `final_data_eeg_valid.rds`: 242
- Spectral entropy features: 22 (verified present)
- Sample/Approximate entropy: 0 (confirmed absent)
