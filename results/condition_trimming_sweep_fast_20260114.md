# Condition-Level Trimming Sweep Results (Fast Mode)

**Date:** 2026-01-14  
**Mode:** Fast (15 participants, 2100 base windows)  
**Window Size:** 10s @ 125Hz (1249 samples)  
**CV:** 3-fold GroupKFold by participant  

## Summary

| Config | Window Range | Windows | Accuracy | Std | F1 |
|--------|-------------|---------|----------|-----|-----|
| **trim_both_20s** | 5-31 | 1620 | **62.3%** | 5.8% | 0.616 |
| trim_both_30s | 7-29 | 1380 | 58.5% | 7.4% | 0.524 |
| all_windows (baseline) | 1-35 | 2100 | 58.0% | 2.3% | 0.579 |
| trim_last_10s | 1-33 | 1980 | 57.0% | 2.5% | 0.555 |
| trim_first_10s | 3-35 | 1980 | 56.3% | 3.6% | 0.548 |
| middle_60s | 13-23 | 660 | 55.9% | 2.1% | 0.529 |
| late_only_60s | 25-35 | 660 | 52.3% | 1.0% | 0.486 |
| trim_both_10s | 3-33 | 1860 | 51.9% | 1.2% | 0.431 |

## Full Dataset Validation (43 participants)

| Config | Window Range | Windows | Accuracy | Std | F1 |
|--------|-------------|---------|----------|-----|-----|
| **trim_both_20s** | 5-31 | 4644 | **60.9%** | 3.3% | 0.606 |

## Key Findings

1. **Best performer:** `trim_both_20s` (windows 5-31)
   - Fast subset: **62.3%** (±5.8%)
   - Full dataset: **60.9%** (±3.3%), F1=0.606
   - Trims first 20s and last 20s of each 180s condition
   - Remains superior to all other trims tested

2. **More trimming hurts:** `trim_both_30s` (58.5%) is worse than 20s trimming
   - Extremely high variance (7.4%) - Fold 2 dropped to chance (50%)
   - Removing too many windows destabilizes training

3. **Middle-only fails:** `middle_60s` at 55.9%
   - Uses only the middle 60s of each condition (windows 13-23)
   - Misses the important late-condition workload differentiation
   - Limited samples (660) hurt generalization

4. **Baseline:** `all_windows` at 58.0% - uses all windows from each condition

5. **Late-only fails:** Using only the last 60s of each condition (52.3%)
   - Too few samples (660 vs 2100)
   - Late cognitive state alone insufficient

## Interpretation

The 20s edge trimming showing the best result suggests:
- Transition effects at condition boundaries may add noise
- But more aggressive trimming (20s) is needed vs. conservative (10s)
- The specific window indices matter - non-linear relationship

## Next Steps

1. Apply `trim_both_20s` in the late-fusion pipeline to see if it lifts the 65.8% fusion baseline
2. Run proper window length sweep (5s, 10s, 15s, 20s actual window sizes) with actual windowing
3. If needed, try mild LR decay or +10 epochs for the full-dataset run to test if training stabilizes further

## Fold-by-Fold Details

### trim_both_20s (BEST)
- Fold 1: 64.4%
- Fold 2: 54.4%  
- Fold 3: 68.1%

### all_windows (baseline)
- Fold 1: 54.9%
- Fold 2: 58.9%
- Fold 3: 60.3%

### trim_first_10s
- Fold 1: 61.2%
- Fold 2: 55.0%
- Fold 3: 52.6%

### trim_last_10s
- Fold 1: 59.4%
- Fold 2: 58.0%
- Fold 3: 53.6%

### trim_both_10s
- Fold 1: 52.4%
- Fold 2: 50.3%
- Fold 3: 53.1%

### late_only_60s
- Fold 1: 53.2%
- Fold 2: 52.7%
- Fold 3: 50.9%

### trim_both_30s
- Fold 1: 68.0%
- Fold 2: 50.0% (chance)
- Fold 3: 57.4%

### middle_60s
- Fold 1: 58.6%
- Fold 2: 53.6%
- Fold 3: 55.5%

### Full dataset: trim_both_20s
- Fold 1: 56.6%
- Fold 2: 64.7%
- Fold 3: 61.5%

---

*Note: 5s_first_half, 5s_second_half, and 5s_trimmed_both configs were conceptually flawed (slicing existing windows rather than creating proper shorter windows) and results are not included.*
