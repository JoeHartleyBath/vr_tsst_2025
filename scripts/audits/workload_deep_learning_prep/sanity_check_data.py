"""EEG sanity check for TCNet-style training (sliding windows).

Key facts (validated in this repo):
- MNE reads EEGLAB amplitudes in Volts (V). Report microvolts (uV) via ×1e6.
- Task blocks are extracted as sliding windows: 10s windows, 5s step across 180s.

This script produces quick, unit-aware diagnostics for a small PID subset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

import warnings


warnings.filterwarnings("ignore")

# Windows consoles often default to cp1252 when output is redirected/piped.
# Force UTF-8, but also keep output ASCII-safe (uV, ->, [OK]/[WARN]/[ERR]).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).parent.parent.parent
CLEANED_DIR = PROJECT_ROOT / "output" / "cleaned_eeg"

# Default: 5 participants, deterministic.
DEFAULT_TEST_PIDS = [3, 10, 20, 30, 40]

WINDOW_DUR_S = 10.0
STEP_DUR_S = 5.0
BLOCK_DUR_S = 180.0

TASK_EVENT_DESCS = ("101", "102", "103", "104")
FOREST_EVENT_DESCS = ("20", "21", "23", "24")


# Event code mapping from config/conditions.yaml (same mapping used by export_mne_epochs.py)
EVENT_CODE_TO_CONDITION = {
    "101": "HighStress_HighCog_Task",
    "102": "HighStress_LowCog_Task",
    "103": "LowStress_HighCog_Task",
    "104": "LowStress_LowCog_Task",
}

TARGET_CONDITIONS = {
    "LowWorkload": ["HighStress_LowCog_Task", "LowStress_LowCog_Task"],
    "HighWorkload": ["HighStress_HighCog_Task", "LowStress_HighCog_Task"],
}

CONDITION_TO_WORKLOAD_CLASS = {
    cond: cls for cls, conds in TARGET_CONDITIONS.items() for cond in conds
}


def _as_uv(x_v: np.ndarray | float) -> np.ndarray | float:
    return x_v * 1e6


def _safe_std(x: np.ndarray, axis=None) -> np.ndarray:
    out = np.std(x, axis=axis, ddof=0)
    return out


@dataclass(frozen=True)
class BlockEvent:
    desc: str
    onset_samp: int
    onset_sec: float


def _load_raw_cleaned(pid: int) -> mne.io.BaseRaw | None:
    set_path = CLEANED_DIR / f"P{pid:02d}_cleaned.set"
    if not set_path.exists():
        set_path = CLEANED_DIR / f"P{pid}_cleaned.set"
        if not set_path.exists():
            print(f"[ERR] File not found: {set_path}")
            return None

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    return raw


def _summarize_annotations(raw: mne.io.BaseRaw) -> pd.Series:
    desc = list(raw.annotations.description)
    if len(desc) == 0:
        return pd.Series(dtype=int)
    return pd.Series(desc).value_counts()


def _extract_block_events(
    raw: mne.io.BaseRaw, valid_descs: Iterable[str]
) -> list[BlockEvent]:
    valid_descs = set(valid_descs)

    if raw.annotations is None or len(raw.annotations) == 0:
        return []

    blocks: list[BlockEvent] = []
    for onset_sec, desc in zip(raw.annotations.onset, raw.annotations.description, strict=False):
        if desc in valid_descs:
            onset_samp = int(round(onset_sec * raw.info["sfreq"]))
            blocks.append(BlockEvent(desc=desc, onset_samp=onset_samp, onset_sec=float(onset_sec)))

    blocks.sort(key=lambda b: b.onset_samp)
    return blocks


def _sliding_windows_for_block(
    data: np.ndarray,
    onset_samp: int,
    sfreq: float,
    window_dur_s: float,
    step_dur_s: float,
    block_dur_s: float,
) -> tuple[np.ndarray, list[int], int]:
    """Return (windows, start_samples, dropped_windows_count)."""
    window_samps = int(round(window_dur_s * sfreq))
    step_samps = int(round(step_dur_s * sfreq))
    block_samps = int(round(block_dur_s * sfreq))
    n_expected = int((block_dur_s - window_dur_s) / step_dur_s) + 1

    windows: list[np.ndarray] = []
    start_samps: list[int] = []
    dropped = 0

    block_end = onset_samp + block_samps
    for win_idx in range(n_expected):
        start = onset_samp + win_idx * step_samps
        stop = start + window_samps

        if stop > block_end:
            dropped += (n_expected - win_idx)
            break
        if stop > data.shape[1]:
            dropped += (n_expected - win_idx)
            break

        windows.append(data[:, start:stop].astype(np.float32, copy=False))
        start_samps.append(start)

    if len(windows) == 0:
        return np.zeros((0, data.shape[0], window_samps), dtype=np.float32), [], dropped

    return np.stack(windows, axis=0), start_samps, dropped


def _pick_psd_channels(raw: mne.io.BaseRaw, psd_all_channels: bool, n_front: int, n_central: int) -> list[int]:
    if psd_all_channels:
        return list(range(len(raw.ch_names)))

    locs = np.array([raw.info["chs"][i]["loc"][:3] for i in range(len(raw.ch_names))], dtype=float)
    y = locs[:, 1]
    x = locs[:, 0]
    z = locs[:, 2]

    # "Frontal": large +y; prefer near-midline (|x| small) and higher z.
    y_thr = float(np.quantile(y, 0.80))
    frontal_idx = np.where(y >= y_thr)[0]
    frontal_sorted = sorted(frontal_idx.tolist(), key=lambda i: (abs(x[i]), -z[i], -y[i]))
    frontal = frontal_sorted[:n_front]

    # "Central": near y≈0; prefer near-midline and higher z.
    central_sorted = sorted(range(len(raw.ch_names)), key=lambda i: (abs(y[i]), abs(x[i]), -z[i]))
    central: list[int] = []
    for i in central_sorted:
        if i in frontal:
            continue
        central.append(i)
        if len(central) >= n_central:
            break

    picks = frontal + central
    picks = list(dict.fromkeys(picks))  # preserve order, de-dupe
    return picks


def _bandpower_log10(
    epochs: np.ndarray,
    sfreq: float,
    bands_hz: dict[str, tuple[float, float]],
    nperseg_s: float = 2.0,
) -> tuple[np.ndarray, list[str]]:
    """Compute log10 bandpower via Welch.

    Parameters
    ----------
    epochs : array, shape (n_epochs, n_channels, n_times)
    """
    nperseg = int(round(nperseg_s * sfreq))
    nperseg = max(8, min(nperseg, epochs.shape[-1]))
    noverlap = nperseg // 2

    freqs, psd = welch(
        epochs,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    # psd: (n_epochs, n_channels, n_freqs)

    features = []
    band_names = []
    for band_name, (fmin, fmax) in bands_hz.items():
        idx = (freqs >= fmin) & (freqs < fmax)
        if not np.any(idx):
            raise RuntimeError(f"No PSD bins in band {band_name} [{fmin},{fmax})")
        bp = np.trapz(psd[..., idx], freqs[idx], axis=-1)  # (n_epochs, n_channels)
        features.append(np.log10(bp + 1e-20))
        band_names.append(band_name)

    # (n_epochs, n_channels, n_bands)
    return np.stack(features, axis=-1), band_names


def _cohens_d(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Vectorized Cohen's d along axis=0 (samples)."""
    high = np.asarray(high)
    low = np.asarray(low)
    mu_h = np.mean(high, axis=0)
    mu_l = np.mean(low, axis=0)
    s_h = np.std(high, axis=0, ddof=0)
    s_l = np.std(low, axis=0, ddof=0)
    pooled = np.sqrt((s_h**2 + s_l**2) / 2.0)
    return (mu_h - mu_l) / (pooled + 1e-20)


def _participant_report(
    pid: int,
    seed: int,
    psd_all_channels: bool,
    psd_n_front: int,
    psd_n_central: int,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run per-participant checks. Returns (baseline_corrected_epochs, workload_labels)."""
    print(f"\n{'='*70}")
    print(f"PARTICIPANT {pid:02d}: TCNet Sanity Check")
    print(f"{'='*70}")

    raw = _load_raw_cleaned(pid)
    if raw is None:
        return None

    sfreq = float(raw.info["sfreq"])
    data = raw.get_data()  # (n_channels, n_times), in Volts

    print(f"[OK] Loaded cleaned .set via MNE (units: Volts)")
    print(f"  sfreq: {sfreq:.3f} Hz | samples: {raw.n_times} | channels: {len(raw.ch_names)}")
    print(f"  duration: {raw.n_times / sfreq:.1f} s")

    ann_counts = _summarize_annotations(raw)
    if len(ann_counts) == 0:
        print("[ERR] No annotations found; cannot extract blocks")
        return None

    print("\nEvent annotations (counts; showing task+forest):")
    for k in list(TASK_EVENT_DESCS) + list(FOREST_EVENT_DESCS):
        if k in ann_counts.index:
            print(f"  {k}: {int(ann_counts.loc[k])}")
        else:
            print(f"  {k}: 0")

    task_blocks = _extract_block_events(raw, TASK_EVENT_DESCS)
    forest_blocks = _extract_block_events(raw, FOREST_EVENT_DESCS)

    if len(task_blocks) == 0:
        print("[ERR] No task block onsets (101-104) found")
        return None
    if len(forest_blocks) == 0:
        print("[ERR] No forest baseline onsets (20/21/23/24) found")
        return None

    print("\nSliding-window extraction (10s windows, 5s step, 180s block):")
    expected_per_block = int((BLOCK_DUR_S - WINDOW_DUR_S) / STEP_DUR_S) + 1
    print(f"  expected windows per block: {expected_per_block}")

    # Baseline: compute per-forest mean/std per channel.
    forest_stats: dict[str, dict[str, np.ndarray]] = {}
    block_samps = int(round(BLOCK_DUR_S * sfreq))
    for fb in forest_blocks:
        start = fb.onset_samp
        stop = start + block_samps
        if stop > data.shape[1]:
            if debug:
                print(
                    f"  DEBUG forest {fb.desc}: onset_samp={start}, stop={stop} > n_times={data.shape[1]} (skipping)"
                )
            continue
        seg = data[:, start:stop]
        forest_stats[fb.desc] = {
            "mean_v": np.mean(seg, axis=1),
            "std_v": _safe_std(seg, axis=1),
            "onset_samp": np.array([start], dtype=int),
        }

    if len(forest_stats) == 0:
        print("[ERR] No valid forest baseline segments within recording bounds")
        return None

    # Amplitude diagnostics on a deterministic random 10s segment (continuous).
    rng = np.random.default_rng(seed + pid)
    win_samps = int(round(WINDOW_DUR_S * sfreq))
    if data.shape[1] <= win_samps:
        rand_start = 0
    else:
        rand_start = int(rng.integers(0, data.shape[1] - win_samps))
    rand_seg = data[:, rand_start : rand_start + win_samps]
    seg_std_v = _safe_std(rand_seg, axis=1)
    seg_ptp_v = np.ptp(rand_seg, axis=1)

    print("\nAmplitude (random 10s segment; report in V and uV):")
    print(
        f"  median per-channel std:  {np.median(seg_std_v):.3e} V  |  {_as_uv(np.median(seg_std_v)):.2f} uV"
    )
    print(
        f"  median per-channel ptp:  {np.median(seg_ptp_v):.3e} V  |  {_as_uv(np.median(seg_ptp_v)):.2f} uV"
    )

    # Baseline variability summary across forest blocks.
    forest_descs_present = [d for d in FOREST_EVENT_DESCS if d in forest_stats]
    forest_median_std_uv = []
    forest_median_mean_uv = []
    forest_std_by_block = []  # (n_blocks, n_channels)
    for d in forest_descs_present:
        mu_uv = _as_uv(forest_stats[d]["mean_v"])
        sd_uv = _as_uv(forest_stats[d]["std_v"])
        forest_median_mean_uv.append(float(np.median(mu_uv)))
        forest_median_std_uv.append(float(np.median(sd_uv)))
        forest_std_by_block.append(sd_uv)

    forest_std_by_block_arr = np.stack(forest_std_by_block, axis=0) if forest_std_by_block else None
    if forest_std_by_block_arr is not None and forest_std_by_block_arr.shape[0] >= 2:
        per_chan_cv = np.std(forest_std_by_block_arr, axis=0, ddof=0) / (
            np.mean(forest_std_by_block_arr, axis=0) + 1e-12
        )
        median_per_chan_cv = float(np.median(per_chan_cv))
        block_level_cv = float(np.std(forest_median_std_uv, ddof=0) / (np.mean(forest_median_std_uv) + 1e-12))
        block_level_range = float(np.max(forest_median_std_uv) - np.min(forest_median_std_uv))
    else:
        median_per_chan_cv = float("nan")
        block_level_cv = float("nan")
        block_level_range = float("nan")

    print("\nForest baseline (180s) per-block summary (uV):")
    for d, med_mu, med_sd in zip(forest_descs_present, forest_median_mean_uv, forest_median_std_uv, strict=False):
        print(f"  Forest {d}: median(mean)={med_mu:.3f} uV | median(std)={med_sd:.2f} uV")
    print("Baseline variability across forest blocks:")
    print(f"  block-level CV of median(std): {block_level_cv:.3f} | range: {block_level_range:.2f} uV")
    print(f"  median per-channel CV (std across blocks): {median_per_chan_cv:.3f}")

    # Map each task block to the nearest preceding forest block (purely from annotations).
    forest_sorted = sorted(
        [(d, int(forest_stats[d]["onset_samp"][0])) for d in forest_descs_present], key=lambda x: x[1]
    )

    def assign_forest(task_onset_samp: int) -> str | None:
        prior = [d for (d, s) in forest_sorted if s < task_onset_samp]
        return prior[-1] if prior else None

    # Extract sliding windows for tasks and create baseline-corrected versions.
    all_epochs_raw: list[np.ndarray] = []
    all_epochs_baseline_adj: list[np.ndarray] = []
    workload_labels: list[str] = []

    debug_bad_counts = False
    for block_idx, tb in enumerate(task_blocks, start=1):
        condition = EVENT_CODE_TO_CONDITION.get(tb.desc, tb.desc)
        workload_class = CONDITION_TO_WORKLOAD_CLASS.get(condition, "Unknown")

        block_epochs, start_samps, dropped = _sliding_windows_for_block(
            data,
            onset_samp=tb.onset_samp,
            sfreq=sfreq,
            window_dur_s=WINDOW_DUR_S,
            step_dur_s=STEP_DUR_S,
            block_dur_s=BLOCK_DUR_S,
        )
        n_windows = int(block_epochs.shape[0])

        baseline_desc = assign_forest(tb.onset_samp)
        baseline_mu = None
        baseline_sd = None
        if baseline_desc is not None and baseline_desc in forest_stats:
            baseline_mu = forest_stats[baseline_desc]["mean_v"].astype(np.float32, copy=False)
            baseline_sd = forest_stats[baseline_desc]["std_v"].astype(np.float32, copy=False)

        if n_windows != expected_per_block:
            debug_bad_counts = True

        block_end_samp = tb.onset_samp + int(round(BLOCK_DUR_S * sfreq))
        print(
            f"  Block {block_idx} Task {tb.desc} ({condition} -> {workload_class}): onset={tb.onset_sec:.3f}s | "
            f"onset_samp={tb.onset_samp} | end_samp={block_end_samp} | windows={n_windows} | dropped={dropped} | "
            f"baseline_forest={baseline_desc}"
        )

        if debug and (n_windows != expected_per_block or dropped > 0):
            print("    DEBUG windowing:")
            if len(start_samps) > 0:
                print(f"      first_start={start_samps[0]} last_start={start_samps[-1]} win_samps={win_samps}")
            print(f"      raw_n_times={data.shape[1]} block_end_samp={block_end_samp}")

        # Append epochs.
        if n_windows > 0:
            all_epochs_raw.append(block_epochs)
            if baseline_mu is None:
                # No baseline available; keep as raw for baseline-adj too.
                all_epochs_baseline_adj.append(block_epochs)
            else:
                adj = block_epochs - baseline_mu[None, :, None]
                all_epochs_baseline_adj.append(adj)

            workload_labels.extend([workload_class] * n_windows)

        # Baseline-relative z-score diagnostics (optional but always printed compactly).
        if baseline_mu is not None and baseline_sd is not None and n_windows > 0:
            z = (block_epochs - baseline_mu[None, :, None]) / (baseline_sd[None, :, None] + 1e-12)
            z_std = float(np.std(z, ddof=0))
            z_mean = float(np.mean(z))
            print(f"    baseline-relative z (diagnostic): mean={z_mean:+.3f}, std={z_std:.3f}")

    if len(all_epochs_baseline_adj) == 0:
        print("[ERR] No task windows extracted")
        return None

    epochs_baseline_adj = np.concatenate(all_epochs_baseline_adj, axis=0)
    labels = np.array(workload_labels)

    # Count summary.
    total_windows = int(epochs_baseline_adj.shape[0])
    expected_total = expected_per_block * 4
    print("\nWindow counts:")
    print(f"  total windows: {total_windows} (expected {expected_total} for 4 blocks)")
    if total_windows != expected_total:
        print("  [WARN] Count mismatch: enabling debug details recommended (--debug)")

    # PSD discriminability on a small, position-based channel subset by default.
    picks = _pick_psd_channels(raw, psd_all_channels=psd_all_channels, n_front=psd_n_front, n_central=psd_n_central)
    pick_names = [raw.ch_names[i] for i in picks]
    print("\nPSD discriminability (Welch bandpower; log10 power):")
    print(f"  channels used ({len(picks)}): {', '.join(pick_names[:16])}{'...' if len(picks) > 16 else ''}")

    bands = {"theta(4-8)": (4.0, 8.0), "alpha(8-13)": (8.0, 13.0), "beta(13-30)": (13.0, 30.0)}
    feat, band_names = _bandpower_log10(epochs_baseline_adj[:, picks, :], sfreq=sfreq, bands_hz=bands)
    # feat: (n_epochs, n_picks, n_bands)

    high_mask = labels == "HighWorkload"
    low_mask = labels == "LowWorkload"
    n_high = int(np.sum(high_mask))
    n_low = int(np.sum(low_mask))
    print(f"  class counts: HighWorkload={n_high}, LowWorkload={n_low}")

    if n_high > 5 and n_low > 5:
        for b_idx, b_name in enumerate(band_names):
            x = feat[:, :, b_idx]  # (n_epochs, n_ch)
            d = _cohens_d(x[high_mask], x[low_mask])  # (n_ch,)
            delta = np.mean(x[high_mask], axis=0) - np.mean(x[low_mask], axis=0)
            order = np.argsort(-np.abs(d))
            print(f"\n  Top 10 |d| for {b_name}:")
            for rank, j in enumerate(order[:10], start=1):
                print(
                    f"    {rank:2d}. {pick_names[j]:>4s}: d={float(d[j]):+.3f} | dMean(logP)={float(delta[j]):+.3f}"
                )
    else:
        print("  [WARN] Not enough labeled epochs to compute d (need both classes)")

    if debug_bad_counts and debug:
        print("\nDEBUG NOTE: Some blocks had unexpected window counts.")
        print("  Suggestion: inspect task/forest onsets above and recording bounds.")

    return epochs_baseline_adj, labels


def _normalization_demo(all_epochs: np.ndarray, seed: int) -> None:
    print(f"\n{'='*70}")
    print("NORMALIZATION CHECKS")
    print(f"{'='*70}")

    # Sample deterministically for speed (avoid huge memory).
    rng = np.random.default_rng(seed)
    n = all_epochs.shape[0]
    sample_n = min(n, 200)
    idx = rng.choice(n, size=sample_n, replace=False) if n > sample_n else np.arange(n)
    sample = all_epochs[idx].astype(np.float64, copy=False)

    print(f"Using {sample.shape[0]} epochs for normalization demo (deterministic sample)")

    # Global z-score over all values (epochs × channels × time).
    mu = float(np.mean(sample))
    sd = float(np.std(sample, ddof=0))
    if sd < 1e-20:
        print("[ERR] Near-zero variance; cannot z-score")
        return

    z = (sample - mu) / sd
    print("\nGlobal z-score (computed on same data):")
    print(f"  mean={float(np.mean(z)):+.4f} | std={float(np.std(z, ddof=0)):.4f} (expect ~0 and ~1)")

    # Fold-safe demo: fit mean/std on train only.
    perm = rng.permutation(sample.shape[0])
    split = int(round(0.8 * sample.shape[0]))
    train_idx, test_idx = perm[:split], perm[split:]

    train = sample[train_idx]
    test = sample[test_idx]
    mu_tr = float(np.mean(train))
    sd_tr = float(np.std(train, ddof=0))
    z_tr = (train - mu_tr) / (sd_tr + 1e-20)
    z_te = (test - mu_tr) / (sd_tr + 1e-20)

    print("\nFold-safe normalisation demo (train-fitted mean/std):")
    print(f"  train mean={float(np.mean(z_tr)):+.4f} | train std={float(np.std(z_tr, ddof=0)):.4f} (expect ~1)")
    print(f"  test  mean={float(np.mean(z_te)):+.4f} | test  std={float(np.std(z_te, ddof=0)):.4f} (should be close to 1)")


def _cross_subject_summary(per_subject: list[dict]) -> None:
    if len(per_subject) == 0:
        return

    print(f"\n{'='*70}")
    print("CROSS-SUBJECT SUMMARY (baseline-adjusted task windows)")
    print(f"{'='*70}")

    df = pd.DataFrame(per_subject)
    cols = ["pid", "n_epochs", "median_chan_std_uv", "median_chan_ptp_uv"]
    print(df[cols].to_string(index=False))

    print("\nAcross-subject variability (uV):")
    print(f"  median(std) mean={df['median_chan_std_uv'].mean():.2f} | sd={df['median_chan_std_uv'].std(ddof=0):.2f}")
    print(f"  median(ptp) mean={df['median_chan_ptp_uv'].mean():.2f} | sd={df['median_chan_ptp_uv'].std(ddof=0):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG sanity check for TCNet sliding-window training")
    parser.add_argument(
        "--pids",
        type=str,
        default=",".join(str(x) for x in DEFAULT_TEST_PIDS),
        help="Comma-separated participant IDs (default: 3,10,20,30,40)",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Deterministic RNG seed")
    parser.add_argument(
        "--psd-all-channels",
        action="store_true",
        help="Compute PSD discriminability on all channels (slower)",
    )
    parser.add_argument("--psd-n-front", type=int, default=8, help="Number of frontal channels (pos-based) for PSD")
    parser.add_argument("--psd-n-central", type=int, default=8, help="Number of central channels (pos-based) for PSD")
    parser.add_argument("--debug", action="store_true", help="Extra debug prints when counts differ")
    args = parser.parse_args()

    pids = [int(x.strip()) for x in args.pids.split(",") if x.strip()]
    print("=" * 70)
    print("EEG SANITY CHECK (TCNet sliding windows)")
    print("=" * 70)
    print(f"Participants: {pids}")
    print(f"Windowing: {WINDOW_DUR_S:.0f}s windows, {STEP_DUR_S:.0f}s step, {BLOCK_DUR_S:.0f}s blocks")
    print("Units: MNE reads EEGLAB in Volts; uV shown for reporting only")

    per_subject_rows: list[dict] = []
    all_epochs_for_norm: list[np.ndarray] = []

    for pid in pids:
        try:
            result = _participant_report(
                pid,
                seed=args.seed,
                psd_all_channels=bool(args.psd_all_channels),
                psd_n_front=int(args.psd_n_front),
                psd_n_central=int(args.psd_n_central),
                debug=bool(args.debug),
            )
            if result is None:
                continue
            epochs_baseline_adj, labels = result

            # Quick per-subject amplitude summaries on baseline-adjusted windows (uV).
            chan_std_uv = _as_uv(_safe_std(epochs_baseline_adj, axis=(0, 2)))
            chan_ptp_uv = _as_uv(np.ptp(epochs_baseline_adj, axis=2).mean(axis=0))
            per_subject_rows.append(
                {
                    "pid": pid,
                    "n_epochs": int(epochs_baseline_adj.shape[0]),
                    "median_chan_std_uv": float(np.median(chan_std_uv)),
                    "median_chan_ptp_uv": float(np.median(chan_ptp_uv)),
                }
            )

            all_epochs_for_norm.append(epochs_baseline_adj)
        except Exception as e:
            print(f"\n[ERR] Error processing P{pid:02d}: {e}")
            import traceback

            traceback.print_exc()

    _cross_subject_summary(per_subject_rows)

    if len(all_epochs_for_norm) > 0:
        combined = np.concatenate(all_epochs_for_norm, axis=0)
        _normalization_demo(combined, seed=args.seed)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
