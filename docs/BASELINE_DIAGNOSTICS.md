# Baseline diagnostics (workload TCNet)

This repo’s workload deep-learning training uses baseline segments from the cleaned continuous EEG (`output/cleaned_eeg/*.set`) aligned to forest-onset annotations, and applies an optional per-epoch baseline adjustment in `pipelines/13_workload_deep_learning_training/train_tcnet.py`.

## Key numbers (smoke probe)

For P01 (cleaned `.set`, 180s baseline segment starting at forest onset):

- Baseline segment median channel std: ~`5.7e-9 V` (~`0.0057 µV`)
- Baseline mean magnitude (median over channels): ~`6e-12 V` (~`0.000006 µV`)
- Interpretation: the **baseline mean is expected to be near zero** after filtering/detrending, while the **baseline std is non-trivial**.

When applying std-based normalization (e.g., `divstd`), the epoch scale becomes **unitless** and the typical epoch std becomes ~`O(1)` (because data is divided by a per-channel baseline std of ~`6e-9 V`).

## Conclusion

Mean subtraction is effectively a no-op under the current cleaning; std-based normalization (`divstd`/`zscore`) is the lever that materially changes scale.
