"""
EEG-TCNet Training Script for Workload Classification (PyTorch Port)
Combines EEGNet with Temporal Convolutional Network + Attention Mechanisms.

Reference: Ingolfsson et al., 2020 - EEG-TCNet: An Accurate Temporal Convolutional
Network for Embedded Motor-Imagery Brain-Machine Interfaces

Enhancements:
- Squeeze-and-Excitation channel attention
- Temporal attention before classification
- Class-weighted loss for imbalanced data
- Data augmentation support
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import re
from pathlib import Path
import hashlib
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json
from datetime import datetime

import mne

from mne_dataloader import load_workload_data, create_group_splits, MNEEpochsDataset

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = r'C:\vr_tsst_2025\output\adaptive_workload\mne_epochs'

# Cleaned continuous EEG (EEGLAB .set) used for baseline segments.
CLEANED_DIR = r'C:\vr_tsst_2025\output\cleaned_eeg'

# Baseline/task block spec
EXPECTED_SFREQ_HZ = 125.0
SFREQ_ABS_TOL_HZ = 0.2
BLOCK_DUR_S = 180.0
WINDOW_DUR_S = 10.0
STEP_DUR_S = 5.0

# Event codes (as strings after normalization)
FOREST_CODES = {"20", "21", "23", "24"}
TASK_CODES = {"101", "102", "103", "104"}

# Attention toggles (architecture stays the same; just toggling optional modules)
USE_SE_ATTENTION = False
USE_TEMPORAL_ATTENTION = False

# Safety / reproducibility
SEED = 1337
SMOKE_TEST = False

# Baseline adjustment (configured via CLI)
BASELINE_ADJUST = "zscore"  # 'none'|'mean'|'divstd'|'zscore'
BASELINE_CACHE_PATH = r'C:\vr_tsst_2025\results\baseline_stats_cache.pkl'


def _md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest()


def _hash_pid_list(pids: np.ndarray | list[int]) -> str:
    arr = np.asarray(pids, dtype=int).reshape(-1)
    joined = ",".join(str(int(x)) for x in sorted(arr.tolist()))
    return _md5_text(joined)


def _count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _state_dict_key_md5(model: nn.Module) -> str:
    keys = sorted(list(model.state_dict().keys()))
    return _md5_text("\n".join(keys))


def _print_parity_block(title: str, lines: list[str]) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n[PARITY] {title}\n{bar}")
    for ln in lines:
        print(f"[PARITY] {ln}")


def _dataset_signature(dataset: MNEEpochsDataset) -> dict:
    data_shape = tuple(getattr(dataset, "data").shape) if hasattr(dataset, "data") else None
    labels = np.asarray(getattr(dataset, "labels")) if hasattr(dataset, "labels") else np.asarray([])
    pids = np.asarray(getattr(dataset, "pids")) if hasattr(dataset, "pids") else np.asarray([])
    return {
        "n_samples": int(len(dataset)),
        "data_shape": data_shape,
        "n_unique_pids": int(len(np.unique(pids))) if pids.size else 0,
        "class_counts": class_counts(labels) if labels.size else {},
    }


def build_dataset(*, augment: bool, normalize: bool | None, subset_pids: list[int] | None):
    """Build the dataset in a way that can match tune_tcnet.py defaults.

    Note: tune_tcnet.py does not pass normalize=... explicitly, relying on the
    loader default (normalize=False). We support both behaviors via normalize=None.
    """
    kwargs: dict = {
        "subset_pids": subset_pids,
        "augment": bool(augment),
        "target_chans": CHANS,
        "target_time_points": TIME_POINTS,
    }
    if normalize is not None:
        kwargs["normalize"] = bool(normalize)
    return load_workload_data(DATA_DIR, **kwargs)


def fmt(x: float, unit: str) -> str:
    return f"{float(x):.3e} {unit}"


def _attention_tag() -> str:
    if USE_SE_ATTENTION and USE_TEMPORAL_ATTENTION:
        return "attn_se_temp"
    if USE_SE_ATTENTION:
        return "attn_se"
    if USE_TEMPORAL_ATTENTION:
        return "attn_temp"
    return "no_attn"


def make_results_path(*, baseline_adjust: str) -> str:
    tag = _attention_tag()
    return rf"C:\vr_tsst_2025\results\workload_tcnet_baseline_{baseline_adjust}_{tag}_safe_cv.txt"


def make_fold_log_path(*, baseline_adjust: str) -> str:
    tag = _attention_tag()
    return rf"C:\vr_tsst_2025\results\workload_tcnet_baseline_{baseline_adjust}_{tag}_safe_cv_folds.jsonl"


def make_window_pred_path(*, baseline_adjust: str) -> str:
    tag = _attention_tag()
    return rf"C:\vr_tsst_2025\results\workload_tcnet_baseline_{baseline_adjust}_{tag}_safe_cv_window_preds.jsonl"


def normalize_desc(desc: object) -> str:
    """Normalize EEGLAB->MNE annotation descriptions for robust event matching.

    Rules:
    - Strip whitespace.
    - If numeric-like (e.g., "20", "20.0"), coerce to int then back to string.
    """
    s = str(desc).strip()
    if s == "":
        return s
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        try:
            f = float(s)
            if abs(f - round(f)) < 1e-6:
                return str(int(round(f)))
        except Exception:
            return s
    return s


def extract_onsets(raw: mne.io.BaseRaw, codes: set[str]) -> list[int]:
    """Extract onset samples for annotation descriptions matching any of `codes`."""
    if raw.annotations is None or len(raw.annotations) == 0:
        return []

    sfreq = float(raw.info["sfreq"])
    onsets: list[int] = []
    for onset_sec, desc in zip(raw.annotations.onset, raw.annotations.description, strict=False):
        if normalize_desc(desc) in codes:
            onsets.append(int(round(float(onset_sec) * sfreq)))
    onsets.sort()
    return onsets


def _print_annotation_diagnostics(raw: mne.io.BaseRaw, max_unique: int = 40) -> None:
    if raw.annotations is None or len(raw.annotations) == 0:
        print("  [ERR] No annotations found")
        return
    raw_vals = [str(d) for d in raw.annotations.description]
    raw_unique = sorted(set(raw_vals))
    print("  Annotation descriptions (unique; raw -> normalized):")
    for rv in raw_unique[:max_unique]:
        print(f"    {rv!r} -> {normalize_desc(rv)!r}")
    if len(raw_unique) > max_unique:
        print(f"    ... ({len(raw_unique) - max_unique} more raw unique values)")


def debug_baseline_probe(dataset: MNEEpochsDataset, pid: int, n_examples: int = 5) -> None:
    """One-PID baseline extraction probe to diagnose near-zero baseline means.

    Prints channel alignment checks, forest onsets, and baseline/epoch amplitude stats in µV.
    """
    pid = int(pid)
    n_examples = int(n_examples)
    if n_examples <= 0:
        print("[DEBUG_BASELINE] n_examples <= 0; skipping")
        return

    if not hasattr(dataset, "ch_names") or getattr(dataset, "ch_names") is None:
        raise AssertionError("Dataset missing ch_names; update mne_dataloader.MNEEpochsDataset to expose it")
    ds_ch_names = list(getattr(dataset, "ch_names"))

    set_path = _find_cleaned_set_path(pid)
    if set_path is None:
        raise FileNotFoundError(f"[DEBUG_BASELINE] cleaned .set not found for pid={pid}: searched under {CLEANED_DIR}")

    print(f"\n[DEBUG_BASELINE] pid={pid} set_path={set_path}")
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    print(f"[DEBUG_BASELINE] raw sfreq={sfreq:.6f} Hz, n_times={int(raw.n_times)}")

    # Annotation diagnostics
    _print_annotation_diagnostics(raw)

    forest_onsets = extract_onsets(raw, FOREST_CODES)
    print(f"[DEBUG_BASELINE] forest codes={sorted(FOREST_CODES)}")
    print(f"[DEBUG_BASELINE] forest_onsets count={len(forest_onsets)}")
    if len(forest_onsets) > 0:
        first10 = forest_onsets[:10]
        first10_s = [s / sfreq for s in first10]
        print(f"[DEBUG_BASELINE] forest_onsets samp (first 10): {first10}")
        print(f"[DEBUG_BASELINE] forest_onsets sec  (first 10): {[round(x, 3) for x in first10_s]}")

    # Channel alignment: require exact set equality so we don't silently probe the wrong channels.
    raw_ch_names = list(raw.ch_names)
    raw_set = set(raw_ch_names)
    ds_set = set(ds_ch_names)
    missing = sorted(ds_set - raw_set)
    extras = sorted(raw_set - ds_set)
    if missing or extras:
        raise RuntimeError(
            "[DEBUG_BASELINE] Channel mismatch between dataset epochs and cleaned raw. "
            f"missing_in_raw={missing}; extras_in_raw={extras}"
        )
    raw.pick_channels(ds_ch_names, ordered=True)
    print(f"[DEBUG_BASELINE] channel alignment OK: n_ch={len(raw.ch_names)}")

    if not hasattr(dataset, "pids") or not hasattr(dataset, "start_samp"):
        raise AssertionError("[DEBUG_BASELINE] dataset must expose pids and start_samp")

    idx = np.where(np.asarray(dataset.pids) == pid)[0][:n_examples]
    print(f"[DEBUG_BASELINE] dataset epochs for pid={pid}: found={int(np.sum(np.asarray(dataset.pids)==pid))}, probing={len(idx)}")
    if idx.size == 0:
        print("[DEBUG_BASELINE] no dataset epochs for this pid; skipping")
        return

    forest_onsets_arr = np.asarray(forest_onsets, dtype=np.int64)
    block_samps = int(round(float(BLOCK_DUR_S) * sfreq))

    for k, i in enumerate(idx.tolist()):
        epoch_start_samp = int(np.asarray(dataset.start_samp, dtype=np.int64)[i])
        if forest_onsets_arr.size == 0:
            print(f"[DEBUG_BASELINE] ex{k}: epoch_i={i} epoch_start_samp={epoch_start_samp} -> no forest onsets")
            continue

        # nearest preceding forest onset (< epoch_start_samp)
        pos = int(np.searchsorted(forest_onsets_arr, epoch_start_samp, side="left"))
        if pos <= 0:
            print(f"[DEBUG_BASELINE] ex{k}: epoch_i={i} epoch_start_samp={epoch_start_samp} -> no prior forest onset")
            continue
        baseline_onset_samp = int(forest_onsets_arr[pos - 1])
        delta_samp = int(epoch_start_samp - baseline_onset_samp)
        delta_sec = float(delta_samp / sfreq)
        print(
            f"[DEBUG_BASELINE] ex{k}: epoch_i={i} epoch_start_samp={epoch_start_samp} "
            f"baseline_onset_samp={baseline_onset_samp} delta_samp={delta_samp} delta_sec={delta_sec:.3f}"
        )

        # Baseline segment stats (180s from forest onset)
        base_start = int(baseline_onset_samp)
        base_stop = int(baseline_onset_samp + block_samps)
        if base_start >= int(raw.n_times):
            print(f"[DEBUG_BASELINE] ex{k}: baseline_start beyond raw length; skipping")
            continue
        if base_stop > int(raw.n_times):
            base_stop = int(raw.n_times)
        seg = raw.get_data(start=base_start, stop=base_stop)  # (C, T) in Volts

        chan_std_v = np.std(seg, axis=1, ddof=0)
        med_chan_std_v = float(np.median(chan_std_v))
        med_chan_std_uv = med_chan_std_v * 1e6
        abs_vals_v = np.abs(seg).ravel()
        p50_v, p95_v, p99_v = [float(x) for x in np.percentile(abs_vals_v, [50, 95, 99])]
        p50, p95, p99 = [p50_v * 1e6, p95_v * 1e6, p99_v * 1e6]
        med_abs_mean_v = float(np.median(np.abs(seg.mean(axis=1))))
        med_abs_mean_uv = med_abs_mean_v * 1e6
        print(
            f"[DEBUG_BASELINE] ex{k}: baseline seg len_samp={int(seg.shape[1])} "
            f"median_chan_std={med_chan_std_v:.3e} V ({med_chan_std_uv:.6f} µV) "
            f"abs_amp_p50/p95/p99={p50_v:.3e}/{p95_v:.3e}/{p99_v:.3e} V ({p50:.6f}/{p95:.6f}/{p99:.6f} µV) "
            f"median_abs_chan_mean={med_abs_mean_v:.3e} V ({med_abs_mean_uv:.6f} µV)"
        )

        # Epoch window stats from dataset (Volts -> µV)
        x_epoch_v = np.asarray(dataset.data[i, 0, :, :], dtype=np.float64)
        x_abs_v = np.abs(x_epoch_v).ravel()
        ep_p50_v, ep_p95_v, ep_p99_v = [float(x) for x in np.percentile(x_abs_v, [50, 95, 99])]
        print(
            f"[DEBUG_BASELINE] ex{k}: epoch abs_amp_p50/p95/p99="
            f"{ep_p50_v:.3e}/{ep_p95_v:.3e}/{ep_p99_v:.3e} V "
            f"({ep_p50_v * 1e6:.6f}/{ep_p95_v * 1e6:.6f}/{ep_p99_v * 1e6:.6f} µV)"
        )


def _find_cleaned_set_path(pid: int) -> str | None:
    p1 = os.path.join(CLEANED_DIR, f"P{pid:02d}_cleaned.set")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(CLEANED_DIR, f"P{pid}_cleaned.set")
    if os.path.exists(p2):
        return p2
    return None


def _compute_baseline_stats_for_onset(
    raw: mne.io.BaseRaw,
    *,
    pid: int,
    baseline_onset_samp: int,
    block_dur_s: float = BLOCK_DUR_S,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (baseline_mean[C], baseline_std[C], baseline_len_samp) in Volts."""
    sfreq = float(raw.info["sfreq"])
    block_samps = int(round(float(block_dur_s) * sfreq))
    n_times = int(raw.n_times)
    start = int(baseline_onset_samp)
    stop = int(baseline_onset_samp + block_samps)
    if start >= n_times:
        print(
            f"  [WARN] baseline: P{pid:02d} baseline_start={start} beyond data_len={n_times}; using zeros"
        )
        return np.zeros((CHANS,), dtype=np.float32), np.ones((CHANS,), dtype=np.float32), 0
    if stop > n_times:
        print(
            f"  [WARN] baseline: P{pid:02d} baseline segment truncated (stop={stop} > len={n_times}); clipping"
        )
        stop = n_times
    baseline_len = int(stop - start)
    if baseline_len <= 0:
        print(f"  [WARN] baseline: P{pid:02d} baseline segment empty; using zeros")
        return np.zeros((CHANS,), dtype=np.float32), np.ones((CHANS,), dtype=np.float32), 0
    baseline_data = raw.get_data(start=start, stop=stop)  # (C, T)
    if baseline_data.shape[0] != CHANS:
        print(
            f"  [WARN] baseline: P{pid:02d} channel count {baseline_data.shape[0]} != {CHANS}; using zeros"
        )
        return np.zeros((CHANS,), dtype=np.float32), np.ones((CHANS,), dtype=np.float32), baseline_len
    baseline_mean = baseline_data.mean(axis=1).astype(np.float32, copy=False)
    baseline_std = baseline_data.std(axis=1, ddof=0).astype(np.float32, copy=False)
    return baseline_mean, baseline_std, baseline_len


def _load_baseline_cache(path: str) -> dict[tuple[int, int], object]:
    try:
        p = Path(path)
        if not p.exists():
            return {}
        with p.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and all(isinstance(k, tuple) and len(k) == 2 for k in obj.keys()):
            return obj
        # Support wrapped cache format.
        if isinstance(obj, dict) and "stats" in obj and isinstance(obj["stats"], dict):
            return obj["stats"]
        return {}
    except Exception as e:
        print(f"  [WARN] baseline cache: failed to load {path}: {e}")
        return {}


def _save_baseline_cache(path: str, cache: dict[tuple[int, int], object]) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"  [WARN] baseline cache: failed to save {path}: {e}")


def apply_baseline_adjustment_inplace(
    dataset: MNEEpochsDataset,
    *,
    baseline_adjust: str,
    baseline_cache_path: str | None,
    debug_baseline: bool = False,
) -> dict:
    """Apply baseline adjustment to dataset.data in-place (Volts)."""
    if baseline_adjust == "none":
        # Expose a consistent attribute for fold-level sanity logging.
        dataset._baseline_mean_per_epoch = np.zeros((len(dataset), CHANS), dtype=np.float32)
        dataset._baseline_std_per_epoch = np.ones((len(dataset), CHANS), dtype=np.float32)
        return {"enabled": False}

    if baseline_adjust not in {"mean", "divstd", "zscore"}:
        raise ValueError(f"Unsupported baseline_adjust: {baseline_adjust}")

    # Hard correctness checks about start samples.
    if not hasattr(dataset, "start_samp"):
        raise AssertionError("Dataset missing start_samp; update mne_dataloader to expose it")

    start_samp = np.asarray(dataset.start_samp, dtype=np.int64)
    assert len(start_samp) == len(dataset) == len(dataset.pids), (
        f"start_samp alignment mismatch: start_samp={len(start_samp)}, dataset={len(dataset)}, pids={len(dataset.pids)}"
    )

    # Warn (don’t crash) if non-decreasing violated within pid.
    for pid in sorted({int(p) for p in np.unique(dataset.pids)}):
        idx = np.where(dataset.pids == pid)[0]
        if idx.size < 2:
            continue
        ss = start_samp[idx]
        if np.any(np.diff(ss) < 0):
            print(f"  [WARN] baseline: P{pid:02d} start_samp not non-decreasing within pid")

    cache = _load_baseline_cache(baseline_cache_path) if baseline_cache_path else {}

    n = int(len(dataset))
    baseline = np.zeros((n, 1, CHANS, 1), dtype=np.float32)
    baseline_std = np.ones((n, 1, CHANS, 1), dtype=np.float32)
    baseline_onset_samp_arr = np.full((n,), -1, dtype=np.int64)
    baseline_len_samp_arr = np.zeros((n,), dtype=np.int64)

    missing_forests = 0
    first_pid = None
    first_pid_examples = 0
    first_pid_printed_onsets = False

    # Process per pid (load each cleaned .set once).
    for pid in sorted({int(p) for p in np.unique(dataset.pids)}):
        pid_idx = np.where(dataset.pids == pid)[0]
        if pid_idx.size == 0:
            continue
        if first_pid is None:
            first_pid = pid

        set_path = _find_cleaned_set_path(pid)
        if set_path is None:
            print(f"  [WARN] baseline: P{pid:02d} cleaned .set not found; baselines will be zeros")
            missing_forests += int(pid_idx.size)
            continue

        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

        # Channel alignment: require raw channels match exported epochs channels.
        if not hasattr(dataset, "ch_names") or getattr(dataset, "ch_names") is None:
            raise AssertionError("Dataset missing ch_names; update mne_dataloader.MNEEpochsDataset to expose it")
        ds_ch_names = list(getattr(dataset, "ch_names"))
        raw_ch_names = list(raw.ch_names)
        missing = sorted(set(ds_ch_names) - set(raw_ch_names))
        extras = sorted(set(raw_ch_names) - set(ds_ch_names))
        if missing or extras:
            raise ValueError(
                "baseline channel mismatch between epochs and cleaned raw; "
                f"pid={pid}; missing_in_raw={missing}; extras_in_raw={extras}"
            )
        raw.pick_channels(ds_ch_names, ordered=True)

        sfreq = float(raw.info["sfreq"])
        if not np.isclose(sfreq, EXPECTED_SFREQ_HZ, atol=SFREQ_ABS_TOL_HZ):
            print(
                f"  [WARN] baseline: P{pid:02d} expected sfreq {EXPECTED_SFREQ_HZ}Hz, got {sfreq:.3f}Hz; continuing"
            )

        forest_onsets = extract_onsets(raw, FOREST_CODES)
        if len(forest_onsets) == 0:
            print(f"  [WARN] baseline: P{pid:02d} no forest onsets found for codes {sorted(FOREST_CODES)}")
            missing_forests += int(pid_idx.size)

            # If this is the first pid, print annotation descriptions to diagnose code mismatch.
            if pid == first_pid:
                _print_annotation_diagnostics(raw)
            continue

        if debug_baseline and pid == first_pid and not first_pid_printed_onsets:
            print(f"  [DEBUG] baseline: P{pid:02d} forest onsets samp: {forest_onsets}")
            print(f"  [DEBUG] baseline: P{pid:02d} forest onsets sec:  {[round(s / sfreq, 3) for s in forest_onsets]}")
            first_pid_printed_onsets = True

        for i in pid_idx:
            epoch_start = int(start_samp[i])
            prior = [s for s in forest_onsets if s < epoch_start]
            if not prior:
                missing_forests += 1
                continue

            baseline_onset = int(prior[-1])
            assert baseline_onset < epoch_start, (
                f"baseline_onset must be < epoch_start (pid={pid}, onset={baseline_onset}, start={epoch_start})"
            )
            baseline_onset_samp_arr[i] = baseline_onset

            key = (int(pid), int(baseline_onset))
            cached = cache.get(key)
            mean_vec: np.ndarray | None = None
            std_vec: np.ndarray | None = None
            if isinstance(cached, dict):
                mean_vec = cached.get("mean")
                std_vec = cached.get("std")
            elif isinstance(cached, np.ndarray):
                # Old cache: array is baseline mean only.
                mean_vec = cached
                std_vec = None
            else:
                mean_vec = None
                std_vec = None

            if mean_vec is None or std_vec is None:
                mean_vec_new, std_vec_new, base_len = _compute_baseline_stats_for_onset(
                    raw,
                    pid=pid,
                    baseline_onset_samp=baseline_onset,
                    block_dur_s=BLOCK_DUR_S,
                )
                if mean_vec is None:
                    mean_vec = mean_vec_new
                if std_vec is None:
                    std_vec = std_vec_new
                cache[key] = {
                    "mean": np.asarray(mean_vec, dtype=np.float32),
                    "std": np.asarray(std_vec, dtype=np.float32),
                }
            else:
                # Length can be recomputed cheaply for logging.
                base_len = int(min(raw.n_times, baseline_onset + int(round(BLOCK_DUR_S * sfreq))) - baseline_onset)

            mean_vec = np.asarray(mean_vec, dtype=np.float32)
            std_vec = np.asarray(std_vec, dtype=np.float32)

            baseline[i, 0, :, 0] = mean_vec
            baseline_std[i, 0, :, 0] = std_vec
            baseline_len_samp_arr[i] = int(base_len)

            if pid == first_pid and (debug_baseline or first_pid_examples < 10):
                if debug_baseline and first_pid_examples < 5:
                    med_std_v = float(np.median(std_vec))
                    med_abs_mean_v = float(np.median(np.abs(mean_vec)))
                    # Epoch window stats (pre-adjust) from dataset.data.
                    x_epoch_v = np.asarray(dataset.data[int(i), 0, :, :], dtype=np.float64)
                    abs_epoch_v = np.abs(x_epoch_v).ravel()
                    ep_p50_v, ep_p95_v, ep_p99_v = [float(x) for x in np.percentile(abs_epoch_v, [50, 95, 99])]
                    print(
                        f"  [DEBUG] baseline ex: pid={pid:02d} epoch_i={int(i)} start_samp={epoch_start} "
                        f"baseline_onset_samp={baseline_onset} baseline_len_samp={int(base_len)}"
                    )
                    print(
                        "  [DEBUG] baseline seg stats: "
                        f"median_chan_std={med_std_v:.3e} V ({med_std_v * 1e6:.6f} µV); "
                        f"median_abs_chan_mean={med_abs_mean_v:.3e} V ({med_abs_mean_v * 1e6:.6f} µV); "
                        f"median_baseline_std={med_std_v:.3e} V ({med_std_v * 1e6:.6f} µV)"
                    )
                    print(
                        "  [DEBUG] epoch abs amp p50/p95/p99: "
                        f"{ep_p50_v:.3e}/{ep_p95_v:.3e}/{ep_p99_v:.3e} V "
                        f"({ep_p50_v * 1e6:.6f}/{ep_p95_v * 1e6:.6f}/{ep_p99_v * 1e6:.6f} µV)"
                    )
                else:
                    abs_med_uv = float(np.median(np.abs(mean_vec))) * 1e6
                    print(
                        f"  baseline ex: pid={pid:02d} epoch_i={int(i)} start_samp={epoch_start} "
                        f"baseline_onset_samp={baseline_onset} baseline_len_samp={int(base_len)} "
                        f"median|baseline_mean|={abs_med_uv:.6f} µV"
                    )
                first_pid_examples += 1

    if baseline_cache_path:
        _save_baseline_cache(baseline_cache_path, cache)

    dataset.data = dataset.data.astype(np.float32, copy=False)
    # Clamp std before any division.
    baseline_std = np.maximum(baseline_std, 1e-12).astype(np.float32, copy=False)

    if baseline_adjust == "mean":
        dataset.data -= baseline
    elif baseline_adjust == "divstd":
        dataset.data /= baseline_std
    elif baseline_adjust == "zscore":
        dataset.data = (dataset.data - baseline) / baseline_std
    else:
        raise AssertionError(f"Unhandled baseline_adjust: {baseline_adjust}")

    dataset._baseline_mean_per_epoch = baseline[:, 0, :, 0]
    dataset._baseline_std_per_epoch = baseline_std[:, 0, :, 0]

    abs_med_v = float(np.median(np.abs(baseline)))
    abs_med_uv = abs_med_v * 1e6
    med_std_v = float(np.median(dataset._baseline_std_per_epoch))
    med_std_uv = med_std_v * 1e6

    # Baseline mean near 0 is expected after filtering/detrending.
    if baseline_adjust == "mean":
        print(
            "  [NOTE] baseline: baseline mean near 0 is expected after filtering/detrending; "
            "use zscore/divstd to change scale"
        )

    # Only warn about scale/flatness if baseline std is tiny.
    tiny_std_thresh_v = 1e-9
    if med_std_v < tiny_std_thresh_v:
        print(
            f"  [WARN] baseline: median baseline std is tiny ({med_std_v:.3e} V / {med_std_uv:.6f} µV). "
            "Signal may be overly scaled or near-flat."
        )

    return {
        "enabled": True,
        "mode": baseline_adjust,
        "cache_path": baseline_cache_path,
        "baseline_abs_median_v": abs_med_v,
        "baseline_abs_median_uv": abs_med_uv,
        "baseline_std_median_v": med_std_v,
        "baseline_std_median_uv": med_std_uv,
        "missing_forests": int(missing_forests),
        "total_epochs": int(n),
    }

# All 44 EEG-valid participants (P44 excluded - no valid windows)
ALL_PIDS = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 43, 45, 47, 48]
SUBSET_PIDS = ALL_PIDS

# Model hyperparameters
CHANS = 128
TIME_POINTS = 1250
CLASSES = 2

# Model hyperparameters (can be overridden by tuned params)
F1 = 8
D = 2
KERN_LENGTH = 64
DROPOUT_EEG = 0.2
TCN_FILTERS = 12
TCN_KERNEL = 4
TCN_DEPTH = 2
DROPOUT_TCN = 0.3

# Training hyperparameters (can be overridden by tuned params)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 10
WEIGHT_DECAY = 0.0
LABEL_SMOOTHING = 0.0

# Debug limits (used in SMOKE_TEST)
MAX_BATCHES_PER_EPOCH = None  # e.g., 5

# Augmentation settings
USE_AUGMENTATION = False  # Disabled for baseline test
NOISE_SIGMA = 0.02  # Reduced from 0.05
TIME_SHIFT = 3       # Reduced from 5
CHANNEL_DROPOUT = 0.05  # Reduced from 0.1

# ==============================================================================
# ATTENTION MODULES
# ==============================================================================

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, channels, height, width) or (batch, channels, time)
        # Global average pooling
        if x.dim() == 4:
            squeeze = x.mean(dim=(2, 3))  # (batch, channels)
        else:
            squeeze = x.mean(dim=2)  # (batch, channels)
        
        # Excitation
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)
        
        # Scale
        if x.dim() == 4:
            return x * excitation.unsqueeze(2).unsqueeze(3)
        else:
            return x * excitation.unsqueeze(2)


class TemporalAttention(nn.Module):
    """Lightweight 1D convolutional attention for temporal features."""
    
    def __init__(self, channels, kernel_size=7):
        super(TemporalAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(channels, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, channels, time)
        attn = self.conv(x)  # (batch, 1, time)
        attn = self.sigmoid(attn)
        return x * attn


# ==============================================================================
# EEG-TCNet Model with Attention (PyTorch Port)
# ==============================================================================

class TemporalBlock(nn.Module):
    """Single TCN block with dilated causal convolutions and residual connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation  # Causal padding
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               dilation=dilation, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, channels, time)
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Causal: trim to original length
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # Causal: trim to original length
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCN(nn.Module):
    """Temporal Convolutional Network with exponentially increasing dilation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, depth=2, dropout=0.3):
        super(TCN, self).__init__()
        
        layers = []
        for i in range(depth):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else out_channels
            layers.append(TemporalBlock(in_ch, out_channels, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class EEGTCNet(nn.Module):
    """
    EEG-TCNet with Attention: EEGNet + SE Attention + TCN + Temporal Attention.
    
    Architecture:
    1. EEGNet feature extractor (temporal + spatial convolutions)
    2. Squeeze-and-Excitation channel attention after spatial conv
    3. TCN for temporal pattern learning
    4. Temporal attention before classification
    5. Dense classification head
    """
    
    def __init__(self, chans=128, classes=2, time_points=1249,
                 F1=8, D=2, kernLength=64, dropout_eeg=0.2,
                 tcn_filters=12, tcn_kernel=4, tcn_depth=2, dropout_tcn=0.3,
                 use_se_attention=True, use_temporal_attention=True, se_reduction=4):
        super(EEGTCNet, self).__init__()
        
        self.use_se_attention = use_se_attention
        self.use_temporal_attention = use_temporal_attention
        
        F2 = F1 * D
        
        # ===== EEGNet Block =====
        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Block 2: Depthwise spatial convolution
        self.depthwise = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 8))
        self.drop1 = nn.Dropout(dropout_eeg)
        
        # Squeeze-and-Excitation after depthwise conv
        if use_se_attention:
            self.se_block = SqueezeExcitation(F2, reduction=se_reduction)
        
        # Block 3: Separable convolution
        self.separable_depth = nn.Conv2d(F2, F2, (1, 16), padding='same', groups=F2, bias=False)
        self.separable_point = nn.Conv2d(F2, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_eeg)
        
        # ===== TCN Block =====
        self.tcn = TCN(F2, tcn_filters, kernel_size=tcn_kernel, depth=tcn_depth, dropout=dropout_tcn)
        
        # Temporal attention after TCN
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(tcn_filters, kernel_size=7)
        
        # ===== Classification Head =====
        self.fc = nn.Linear(tcn_filters, classes)
        
    def forward(self, x):
        # x: (batch, 1, chans, time)
        
        # EEGNet Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        
        # EEGNet Block 2 with SE attention
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        if self.use_se_attention:
            x = self.se_block(x)
        
        # EEGNet Block 3
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Reshape for TCN: (batch, F2, 1, time') -> (batch, F2, time')
        x = x.squeeze(2)
        
        # TCN
        x = self.tcn(x)
        
        # Temporal attention
        if self.use_temporal_attention:
            x = self.temporal_attention(x)
        
        # Take last time step for classification
        x = x[:, :, -1]
        
        # Classification
        x = self.fc(x)
        return x


def set_reproducibility(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"WARNING: torch.use_deterministic_algorithms(True) failed: {e}")


def build_criterion(class_weights: torch.Tensor, label_smoothing: float):
    try:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(label_smoothing))
    except TypeError:
        if float(label_smoothing) > 0.0:
            print("[WARN] label_smoothing not supported by this torch; falling back to 0.0")
        return nn.CrossEntropyLoss(weight=class_weights)


def evaluate_normalized(model, loader, criterion, device, mean_c, std_c, max_batches=None):
    """Evaluate model with fold-safe normalization applied.

    This is the ONLY evaluation implementation used in this script.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for bi, (batch_x, batch_y) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            assert batch_y.dtype == torch.long, f"batch_y dtype must be torch.long, got {batch_y.dtype}"
            batch_x = batch_x.to(device)
            batch_x = normalize_batch(batch_x, mean_c, std_c)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += float(loss.item())
            n_batches += 1
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.detach().cpu().numpy())

    mean_loss = total_loss / max(1, n_batches)
    acc = float(accuracy_score(all_labels, all_preds)) if len(all_labels) else 0.0
    f1_macro = float(f1_score(all_labels, all_preds, average='macro', zero_division=0)) if len(all_labels) else 0.0
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1]) if len(all_labels) else np.zeros((2, 2), dtype=int)
    return mean_loss, acc, f1_macro, np.asarray(all_labels), np.asarray(all_preds), cm


def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def compute_train_channel_stats(dataset, train_indices, batch_size=64):
    """Compute per-channel mean/std on TRAIN only across epochs and timepoints.

    Returns tensors shaped (1, 128, 1) on CPU.
    """
    loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=False)

    total_count = 0
    sum_c = None
    sumsq_c = None
    nan_found = False

    for batch_x, _ in loader:
        # batch_x: (B, 1, C, T)
        if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
            nan_found = True
        x = batch_x.squeeze(1)  # (B, C, T)
        b, c, t = x.shape
        x = x.reshape(b, c, t)
        if sum_c is None:
            sum_c = x.sum(dim=(0, 2), dtype=torch.float64)  # (C,)
            sumsq_c = (x.double() ** 2).sum(dim=(0, 2))
        else:
            sum_c += x.sum(dim=(0, 2), dtype=torch.float64)
            sumsq_c += (x.double() ** 2).sum(dim=(0, 2))
        total_count += b * t

    mean = (sum_c / float(total_count)).float().view(1, -1, 1)
    var = (sumsq_c / float(total_count) - (mean.view(-1) ** 2)).clamp_min(0.0)
    std = torch.sqrt(var).float().view(1, -1, 1)
    std = std.clamp_min(1e-12)
    return mean, std, nan_found


def normalize_batch(batch_x, mean_c, std_c):
    # batch_x: (B, 1, C, T); mean_c/std_c: (1, C, 1)
    assert batch_x.ndim == 4, f"batch_x must be 4D (B,1,C,T), got {tuple(batch_x.shape)}"
    assert batch_x.shape[1] == 1, f"batch_x must have singleton dim=1, got {tuple(batch_x.shape)}"
    assert batch_x.shape[2] == CHANS, f"batch_x channels mismatch: got {batch_x.shape[2]}, expected {CHANS}"
    assert tuple(mean_c.shape) == (1, CHANS, 1), f"mean_c shape must be (1,{CHANS},1), got {tuple(mean_c.shape)}"
    assert tuple(std_c.shape) == (1, CHANS, 1), f"std_c shape must be (1,{CHANS},1), got {tuple(std_c.shape)}"
    mean = mean_c.unsqueeze(1)  # (1, 1, C, 1)
    std = std_c.unsqueeze(1)    # (1, 1, C, 1)
    return (batch_x - mean) / std


def summarize_std(std_c: torch.Tensor):
    # std_c shape: (1, C, 1) in Volts
    std_flat = std_c.detach().cpu().view(-1)
    median_v = float(std_flat.median().item())
    median_uv = median_v * 1e6
    return median_v, median_uv


def class_counts(labels: np.ndarray):
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def run_cv_evaluation(
    *,
    baseline_adjust: str,
    baseline_cache_path: str | None,
    debug_baseline: bool = False,
    export_window_preds: bool = False,
    parity_mode: bool = False,
    trial_number: int = 0,
    train_seed: int | None = None,
    split_seed: int | None = None,
    save_checkpoints: bool = False,
    tuned_reference_macro_f1: float | None = None,
):
    """Run 3-fold GroupKFold cross-validation with augmentation and class weighting."""
    effective_split_seed = int(SEED if split_seed is None else split_seed)

    # Parity with tune_tcnet.py Stage2
    # - Stage2 uses trial_seed = SEED + 100000 + trial.number
    # - That seed drives model init + DataLoader shuffling order.
    if parity_mode:
        trial_seed = int(SEED + 100000 + int(trial_number))
        effective_train_seed = int(trial_seed)
    else:
        trial_seed = None
        effective_train_seed = int(SEED if train_seed is None else train_seed)

    # Determinism: must match the tuner when parity_mode is enabled.
    set_reproducibility(effective_train_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Augmentation: {USE_AUGMENTATION} (parity_mode forces False)")
    print(f"Seed (train RNG): {effective_train_seed}")
    print(f"Seed (splits): {effective_split_seed}")
    print(f"Baseline adjust: {baseline_adjust}")
    print(f"Label smoothing: {LABEL_SMOOTHING}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    if parity_mode:
        print("Parity mode: ENABLED (tuner-aligned)")
    if baseline_cache_path:
        print(f"Baseline cache:  {baseline_cache_path}")
    else:
        print("Baseline cache:  <DISABLED>")

    n_splits = 2 if SMOKE_TEST else 3
    max_epochs = 3 if SMOKE_TEST else (60 if parity_mode else EPOCHS)
    max_batches = 5 if SMOKE_TEST else MAX_BATCHES_PER_EPOCH
    if SMOKE_TEST:
        print("SMOKE_TEST enabled: n_splits=2, EPOCHS=3, max_batches_per_epoch=5")
    results_path = make_results_path(baseline_adjust=baseline_adjust)
    fold_log_path = make_fold_log_path(baseline_adjust=baseline_adjust)
    window_pred_path = make_window_pred_path(baseline_adjust=baseline_adjust)
    ensure_parent_dir(results_path)
    ensure_parent_dir(fold_log_path)
    if export_window_preds:
        ensure_parent_dir(window_pred_path)
    
    # Load data WITHOUT augmentation first (for class weights and splits)
    # Use normalize=None in parity_mode to exactly match tune_tcnet.py call signature.
    dataset = build_dataset(augment=False, normalize=None if parity_mode else False, subset_pids=SUBSET_PIDS)
    ds_sig = _dataset_signature(dataset)
    _print_parity_block(
        "RUN START",
        [
            f"dataset: n_samples={ds_sig['n_samples']} shape={ds_sig['data_shape']} unique_pids={ds_sig['n_unique_pids']} class_counts={ds_sig['class_counts']}",
            # Parity with tune_tcnet.py Stage2
            f"trial_number={int(trial_number)} trial_seed={effective_train_seed}" if parity_mode else "trial_number=<n/a> trial_seed=<n/a>",
            f"baseline_adjust={baseline_adjust} baseline_cache={'enabled' if baseline_cache_path else 'disabled'}",
            f"splits: n_splits={n_splits} seed={effective_split_seed}",
            f"train: max_epochs={max_epochs} early_stop_patience={int(EARLY_STOP_PATIENCE)} batch_size={int(BATCH_SIZE)} lr={float(LEARNING_RATE):.6g} wd={float(WEIGHT_DECAY):.6g} label_smoothing={float(LABEL_SMOOTHING):.6g}",
            f"tuned_reference_macro_f1={tuned_reference_macro_f1 if tuned_reference_macro_f1 is not None else 'n/a'}",
        ],
    )
    print(f"Total samples: {len(dataset)}, Shape: {dataset.data.shape}")
    if hasattr(dataset, 'shape_fixes'):
        print(f"Shape fixes: {dataset.shape_fixes}")

    unique_pids = np.unique(dataset.pids)
    expected_pids = set(map(int, SUBSET_PIDS))
    present_pids = set(map(int, unique_pids.tolist()))
    missing_pids = sorted(expected_pids - present_pids)
    extra_pids = sorted(present_pids - expected_pids)
    if missing_pids:
        print(f"[WARN] Missing PIDs in loaded dataset (expected {len(expected_pids)}): {missing_pids}")
    if extra_pids:
        print(f"[WARN] Unexpected extra PIDs in loaded dataset: {extra_pids}")
    print(f"Loaded unique PIDs: {len(present_pids)}")

    # If requested, run the debug probe even when baseline_adjust=none.
    if bool(debug_baseline):
        debug_pid = int(unique_pids[0])
        debug_baseline_probe(dataset, pid=debug_pid, n_examples=5)

    assert len(dataset.pids) == len(dataset), f"pid length mismatch: {len(dataset.pids)} vs {len(dataset)}"
    if hasattr(dataset, "start_samp"):
        assert len(dataset.start_samp) == len(dataset), f"start_samp length mismatch: {len(dataset.start_samp)} vs {len(dataset)}"

    # Global label safety
    assert set(np.unique(dataset.labels)).issubset({0, 1}), f"Unexpected labels: {np.unique(dataset.labels)}"
    
    # Compute class weights for imbalanced data
    class_weights = dataset.get_class_weights().to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Load augmented version for training.
    # Tuner is always augmentation-free; parity_mode must force augmentation off.
    use_aug = bool(USE_AUGMENTATION) and (not bool(parity_mode))
    if use_aug:
        dataset_aug = load_workload_data(
            DATA_DIR,
            subset_pids=SUBSET_PIDS,
            augment=True,
            noise_sigma=NOISE_SIGMA,
            time_shift=TIME_SHIFT,
            channel_dropout=CHANNEL_DROPOUT,
            normalize=False,
            target_chans=CHANS,
            target_time_points=TIME_POINTS,
        )
    else:
        dataset_aug = dataset

    if USE_AUGMENTATION:
        assert len(dataset_aug) == len(dataset), f"dataset_aug length mismatch: {len(dataset_aug)} vs {len(dataset)}"

    # Optional baseline adjustment (in-place on BOTH datasets).
    baseline_info = apply_baseline_adjustment_inplace(
        dataset,
        baseline_adjust=baseline_adjust,
        baseline_cache_path=baseline_cache_path,
        debug_baseline=bool(debug_baseline),
    )
    if dataset_aug is not dataset:
        _ = apply_baseline_adjustment_inplace(
            dataset_aug,
            baseline_adjust=baseline_adjust,
            baseline_cache_path=baseline_cache_path,
            debug_baseline=False,
        )
    if baseline_info.get("enabled"):
        _print_parity_block(
            "BASELINE",
            [
                f"mode={baseline_adjust} cache_path={baseline_info.get('cache_path')}",
                f"median_abs_baseline_mean_uv={baseline_info.get('baseline_abs_median_uv'):.6g} median_baseline_std_uv={baseline_info.get('baseline_std_median_uv'):.6g}",
                f"missing_forests={baseline_info.get('missing_forests')} total_epochs={baseline_info.get('total_epochs')}",
            ],
        )

    # Post-adjust quick scale check (dataset-wide sample).
    sample_n = min(256, len(dataset))
    if sample_n > 0:
        x = dataset.data[:sample_n, 0, :, :]  # (S,C,T)
        med_epoch_std = float(np.median(np.std(x, axis=2)))
        if baseline_adjust in {"divstd", "zscore"}:
            print(f"Post-adjust median epoch std (sample): {fmt(med_epoch_std, 'unitless')}")
        else:
            print(f"Post-adjust median epoch std (sample): {fmt(med_epoch_std * 1e6, 'µV')}")
    
    fold_results = []
    
    # Reset fold log file each run.
    with open(fold_log_path, 'w', encoding='utf-8') as _:
        pass

    if export_window_preds:
        with open(window_pred_path, 'w', encoding='utf-8') as _:
            pass

    # Precompute deterministic splits ONCE and reuse across folds (tuner parity).
    splits = list(create_group_splits(dataset, n_splits=n_splits, seed=effective_split_seed))
    if len(splits) != int(n_splits):
        raise RuntimeError(f"Expected {n_splits} splits, got {len(splits)}")
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}")
        print(f"{'='*50}")
        
        train_pids = np.unique(dataset.pids[train_idx])
        test_pids = np.unique(dataset.pids[test_idx])
        print(f"Train PIDs: {len(train_pids)} subjects, Test PIDs: {len(test_pids)} subjects")

        test_pid_hash = _hash_pid_list(test_pids)
        _print_parity_block(
            f"FOLD {fold + 1} SPLIT",
            [
                f"test_pids_sorted={sorted([int(x) for x in test_pids.tolist()])}",
                f"test_pid_md5={test_pid_hash}",
            ],
        )

        # Hard subject-level separation assert
        overlap = np.intersect1d(train_pids, test_pids)
        assert overlap.size == 0, f"Subject leakage: overlap={overlap.tolist()}"

        # Index/pid consistency asserts
        assert np.all(np.isin(dataset.pids[train_idx], train_pids)), "Train indices contain pids outside train_pids"
        assert np.all(np.isin(dataset.pids[test_idx], test_pids)), "Test indices contain pids outside test_pids"
        assert set(train_idx).isdisjoint(set(test_idx)), "Train/test indices overlap"

        y_train = dataset.labels[train_idx]
        y_test = dataset.labels[test_idx]
        assert set(np.unique(y_train)).issubset({0, 1}), f"Unexpected train labels: {np.unique(y_train)}"
        assert set(np.unique(y_test)).issubset({0, 1}), f"Unexpected test labels: {np.unique(y_test)}"
        print(f"Train class counts: {class_counts(y_train)}")
        print(f"Test  class counts: {class_counts(y_test)}")

        # Baseline-adjustment sanity stats (fold-level)
        if baseline_info.get("enabled"):
            # median abs baseline mean (train fold only) in µV
            base = getattr(dataset, "_baseline_mean_per_epoch", None)
            if base is None:
                base_abs_med_v = float("nan")
                base_abs_med_uv = float("nan")
            else:
                train_baseline_abs = np.abs(base[np.asarray(train_idx, dtype=int), :])  # (N,C)
                base_abs_med_v = float(np.median(train_baseline_abs))
            base_abs_med_uv = base_abs_med_v * 1e6

            # median std of adjusted epochs (sampled)
            sample_n = min(256, len(train_idx))
            sample_idx = np.asarray(train_idx[:sample_n], dtype=int)
            x = dataset.data[sample_idx, 0, :, :]  # (S,C,T)
            med_epoch_std = float(np.median(np.std(x, axis=2)))
            if baseline_adjust in {"divstd", "zscore"}:
                print(
                    "Baseline sanity (train fold): "
                    f"median |baseline_mean|={fmt(base_abs_med_uv, 'µV')}; "
                    f"median epoch std(after baseline)={fmt(med_epoch_std, 'unitless')}"
                )
            else:
                med_epoch_std_uv = med_epoch_std * 1e6
                print(
                    "Baseline sanity (train fold): "
                    f"median |baseline_mean|={fmt(base_abs_med_uv, 'µV')}; "
                    f"median epoch std(after baseline)={fmt(med_epoch_std_uv, 'µV')}"
                )

        # Fold-safe normalization (train only)
        mean_c, std_c, nan_in_train = compute_train_channel_stats(dataset, train_idx)
        mean_c_d = mean_c.to(device)
        std_c_d = std_c.to(device)
        med_std_v, med_std_uv = summarize_std(std_c)
        mean_has_bad = bool(torch.isnan(mean_c).any() or torch.isinf(mean_c).any())
        std_has_bad = bool(torch.isnan(std_c).any() or torch.isinf(std_c).any())
        if baseline_adjust in {"divstd", "zscore"}:
            print(f"Train-only per-channel std (median): {fmt(med_std_v, 'unitless')}")
        else:
            print(f"Train-only per-channel std (median): {fmt(med_std_uv, 'µV')}")

        _print_parity_block(
            f"FOLD {fold + 1} NORMALIZATION",
            [
                f"nan_or_inf_found_in_train_subset={bool(nan_in_train)}",
                f"mean_has_nan_or_inf={mean_has_bad} std_has_nan_or_inf={std_has_bad}",
                f"median_per_channel_std={'unitless' if baseline_adjust in {'divstd','zscore'} else 'uV'}={med_std_v if baseline_adjust in {'divstd','zscore'} else med_std_uv:.6g}",
            ],
        )
        
        # Create data loaders - use augmented for train, non-augmented for test
        train_subset = Subset(dataset_aug, train_idx)  # Augmented
        test_subset = Subset(dataset, test_idx)        # Not augmented
        
        generator = None
        if parity_mode:
            # Parity with tune_tcnet.py Stage2: seed DataLoader shuffling with the Stage2 trial seed.
            generator = torch.Generator()
            generator.manual_seed(int(effective_train_seed))

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            generator=generator,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Preflight asserts (PARITY ONLY): must not consume train_loader iterator.
        # Calling next(iter(train_loader)) would advance the shuffle generator and change
        # the training order vs tune_tcnet.py.
        if parity_mode:
            _check_train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            _bx, _by = next(iter(_check_train_loader))
            assert _bx.shape[1:] == (1, CHANS, TIME_POINTS), f"train batch_x shape={tuple(_bx.shape)}"
            assert _by.ndim == 1, f"train batch_y shape={tuple(_by.shape)}"
            assert _by.dtype == torch.long, f"train batch_y dtype={_by.dtype}"
            _bx2, _by2 = next(iter(test_loader))
            assert _bx2.shape[1:] == (1, CHANS, TIME_POINTS), f"test batch_x shape={tuple(_bx2.shape)}"
            assert _by2.ndim == 1, f"test batch_y shape={tuple(_by2.shape)}"
            assert _by2.dtype == torch.long, f"test batch_y dtype={_by2.dtype}"
        
        # Initialize model (attention configurable via Optuna tuning)
        model = EEGTCNet(
            chans=CHANS, classes=CLASSES, time_points=TIME_POINTS,
            F1=F1,
            D=D,
            kernLength=KERN_LENGTH,
            dropout_eeg=DROPOUT_EEG,
            tcn_filters=TCN_FILTERS,
            tcn_kernel=TCN_KERNEL,
            tcn_depth=TCN_DEPTH,
            dropout_tcn=DROPOUT_TCN,
            use_se_attention=False if parity_mode else USE_SE_ATTENTION,
            use_temporal_attention=False if parity_mode else USE_TEMPORAL_ATTENTION,
        ).to(device)

        _print_parity_block(
            f"FOLD {fold + 1} MODEL",
            [
                f"param_count={_count_params(model)}",
                f"state_dict_key_md5={_state_dict_key_md5(model)}",
                f"arch: F1={int(F1)} D={int(D)} kernLength={int(KERN_LENGTH)} dropout_eeg={float(DROPOUT_EEG):.6g} tcn_filters={int(TCN_FILTERS)} tcn_kernel={int(TCN_KERNEL)} tcn_depth={int(TCN_DEPTH)} dropout_tcn={float(DROPOUT_TCN):.6g}",
            ],
        )
        
        # Use weighted loss for class imbalance
        criterion = build_criterion(class_weights, LABEL_SMOOTHING)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Training loop with early stopping
        best_acc = -1.0
        best_f1_macro = -1.0
        best_test_loss = float("inf")
        best_epoch = -1
        best_state_dict = None
        patience_counter = 0
        use_early_stop = EARLY_STOP_PATIENCE is not None and int(EARLY_STOP_PATIENCE) > 0
        
        for epoch in range(max_epochs):
            # --- Train ---
            model.train()
            total_loss = 0.0
            n_batches_train = 0
            train_all_preds = []
            train_all_labels = []
            for bi, (batch_x, batch_y) in enumerate(train_loader):
                if max_batches is not None and bi >= max_batches:
                    break
                assert batch_y.dtype == torch.long, f"batch_y dtype must be torch.long, got {batch_y.dtype}"
                batch_x = batch_x.to(device)
                batch_x = normalize_batch(batch_x, mean_c_d, std_c_d)
                batch_y = batch_y.to(device)

                if parity_mode:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                n_batches_train += 1
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                train_all_preds.extend(preds)
                train_all_labels.extend(batch_y.detach().cpu().numpy())

            train_loss = total_loss / max(1, n_batches_train)
            train_acc = accuracy_score(train_all_labels, train_all_preds) if len(train_all_labels) else 0.0

            # --- Eval (normalized, single implementation) ---
            test_loss, test_acc, test_f1_macro, _, _, _ = evaluate_normalized(
                model,
                test_loader,
                criterion,
                device,
                mean_c_d,
                std_c_d,
                max_batches=None,
            )

            if parity_mode or (epoch + 1) % 10 == 0 or SMOKE_TEST:
                _print_parity_block(
                    f"FOLD {fold + 1} EPOCH {epoch + 1}",
                    [
                        f"train_loss={train_loss:.6g} train_acc={float(train_acc):.6g}",
                        f"test_loss={float(test_loss):.6g} test_acc={float(test_acc):.6g} test_macro_f1={float(test_f1_macro):.6g}",
                    ],
                )
            
            # Early stopping check
            # Parity with tune_tcnet.py Stage2
            # Best checkpoint tie-break:
            #   - prefer higher macro-F1
            #   - if macro-F1 ties within 1e-10, prefer higher accuracy
            # Non-parity keeps legacy behavior (uses loss as tie-break).
            if parity_mode:
                is_better = (float(test_f1_macro) > float(best_f1_macro)) or (
                    abs(float(test_f1_macro) - float(best_f1_macro)) < 1e-10
                    and float(test_acc) > float(best_acc)
                )
            else:
                f1_diff = float(test_f1_macro) - float(best_f1_macro)
                is_better = False
                if f1_diff > 1e-10:
                    is_better = True
                elif abs(f1_diff) <= 1e-10:
                    if float(test_loss) < float(best_test_loss) - 1e-12:
                        is_better = True
                    elif abs(float(test_loss) - float(best_test_loss)) <= 1e-12 and float(test_acc) > float(best_acc):
                        is_better = True
            if is_better:
                best_acc = float(test_acc)
                best_f1_macro = float(test_f1_macro)
                best_test_loss = float(test_loss)
                # Parity with tune_tcnet.py Stage2: store 0-based epoch index.
                best_epoch = int(epoch) if parity_mode else int(epoch + 1)
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                if use_early_stop:
                    patience_counter += 1
                    if patience_counter >= int(EARLY_STOP_PATIENCE):
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        # Restore best checkpoint before final fold reporting
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        if parity_mode:
            _print_parity_block(
                f"FOLD {fold + 1} BEST",
                [
                    f"best_epoch={int(best_epoch)}",
                    f"best_macro_f1={float(best_f1_macro):.6g}",
                    f"best_acc={float(best_acc):.6g}",
                    f"test_pid_md5={str(test_pid_hash)}",
                ],
            )

        if save_checkpoints and best_state_dict is not None:
            ckpt_dir = r"C:\\vr_tsst_2025\\results\\tcnet_checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(
                ckpt_dir,
                f"tcnet_fold{fold + 1}_baseline_{baseline_adjust}_seed{effective_train_seed}.pt",
            )
            torch.save(
                {
                    "fold": int(fold + 1),
                    "baseline_adjust": str(baseline_adjust),
                    "train_seed": int(effective_train_seed),
                    "split_seed": int(effective_split_seed),
                    "best_epoch": int(best_epoch),
                    "best_macro_f1": float(best_f1_macro),
                    "best_accuracy": float(best_acc),
                    "best_test_loss": float(best_test_loss),
                    "state_dict": best_state_dict,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint: {ckpt_path}")

        # Final metrics + confusion matrix (normalized)
        final_loss, final_acc, final_f1_macro, y_true, y_pred, cm = evaluate_normalized(
            model,
            DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False),
            criterion,
            device,
            mean_c_d,
            std_c_d,
            max_batches=None,
        )
        if export_window_preds:
            test_idx_arr = np.asarray(test_idx, dtype=int)
            conds = np.asarray(dataset.conditions)[test_idx_arr]
            win_idx = np.asarray(dataset.window_indices)[test_idx_arr]
            pids = np.asarray(dataset.pids)[test_idx_arr]
            if len(y_true) != len(test_idx_arr):
                raise AssertionError(
                    f"Prediction length mismatch: y_true={len(y_true)} vs test_idx={len(test_idx_arr)}"
                )
            with open(window_pred_path, 'a', encoding='utf-8') as f_pred:
                for i in range(len(test_idx_arr)):
                    rec = {
                        'timestamp': datetime.utcnow().isoformat() + 'Z',
                        'fold': int(fold + 1),
                        'pid': int(pids[i]),
                        'condition': str(conds[i]),
                        'window_idx': int(win_idx[i]),
                        'y_true': int(y_true[i]),
                        'y_pred': int(y_pred[i]),
                        'baseline_adjust': str(baseline_adjust),
                    }
                    f_pred.write(json.dumps(rec) + "\n")
        print(f"Fold {fold+1} Best Epoch={best_epoch}: Macro-F1={best_f1_macro:.3f}, Acc={best_acc:.3f}, Loss={best_test_loss:.4f}")
        print(f"Fold {fold+1} Final (restored best): Macro-F1={final_f1_macro:.3f}, Acc={final_acc:.3f}")
        print(f"Fold {fold+1} Confusion Matrix [rows=true 0/1, cols=pred 0/1]:\n{cm}")

        warnings = []
        if hasattr(dataset, 'shape_fixes'):
            if dataset.shape_fixes.get('cropped_time', 0) > 0 or dataset.shape_fixes.get('padded_time', 0) > 0:
                warnings.append(
                    f"shape_fixes: cropped_time={dataset.shape_fixes.get('cropped_time')}, padded_time={dataset.shape_fixes.get('padded_time')}"
                )
            if dataset.shape_fixes.get('nan_found', False):
                warnings.append('nan_or_inf_found_in_loaded_data')
        if nan_in_train:
            warnings.append('nan_or_inf_found_in_train_subset')

        fold_record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'fold': int(fold + 1),
            'seed_train_rng': int(effective_train_seed),
            'seed_splits': int(effective_split_seed),
            'smoke_test': bool(SMOKE_TEST),
            'n_splits': int(n_splits),
            'baseline_adjust': str(baseline_adjust),
            'train_subjects': [int(x) for x in train_pids.tolist()],
            'test_subjects': [int(x) for x in test_pids.tolist()],
            'test_pid_md5': str(test_pid_hash),
            'train_epochs': int(len(train_idx)),
            'test_epochs': int(len(test_idx)),
            'train_class_counts': class_counts(y_train),
            'test_class_counts': class_counts(y_test),
            'train_std_median_v': float(med_std_v),
            'train_std_median_uv': float(med_std_uv) if baseline_adjust not in {'divstd', 'zscore'} else float('nan'),
            'normalization': 'train-only per-channel zscore',
            'data_units': 'unitless' if baseline_adjust in {'divstd', 'zscore'} else 'Volts (MNE)',
            'best_epoch': int(best_epoch),
            'best_macro_f1': float(best_f1_macro),
            'best_accuracy': float(best_acc),
            'best_test_loss': float(best_test_loss),
            'final_macro_f1': float(final_f1_macro),
            'final_accuracy': float(final_acc),
            'confusion_matrix': cm.tolist(),
            'warnings': warnings,
        }
        with open(fold_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(fold_record) + "\n")

        # Parity with tune_tcnet.py Stage2:
        # The fold score is best-over-epoch macro-F1/accuracy, not recomputed final metrics.
        if parity_mode:
            fold_results.append({'accuracy': float(best_acc), 'f1': float(best_f1_macro)})
        else:
            fold_results.append({'accuracy': float(final_acc), 'f1': float(final_f1_macro)})
    
    # Aggregate results
    mean_acc = float(np.mean([r['accuracy'] for r in fold_results])) if fold_results else 0.0
    std_acc = float(np.std([r['accuracy'] for r in fold_results])) if fold_results else 0.0
    mean_f1 = float(np.mean([r['f1'] for r in fold_results])) if fold_results else 0.0
    std_f1 = float(np.std([r['f1'] for r in fold_results])) if fold_results else 0.0
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS - EEG-TCNet safe-CV" + (" (PARITY: best-over-epoch)" if parity_mode else ""))
    print(f"{'='*50}")
    print(f"Per-Fold Accuracy: {[r['accuracy'] for r in fold_results]}")
    print(f"Mean Accuracy: {mean_acc:.3f} (+/- {std_acc:.3f})")
    print(f"Per-Fold Macro-F1: {[r['f1'] for r in fold_results]}")
    if parity_mode:
        print(f"Mean best Macro-F1: {mean_f1:.3f} (+/- {std_f1:.3f})")
    else:
        print(f"Mean Macro-F1: {mean_f1:.3f} (+/- {std_f1:.3f})")
    
    # Save results
    with open(results_path, 'w') as f:
        f.write(f"EEG-TCNet safe-CV - Workload Classification (N={len(SUBSET_PIDS)} subjects)\n")
        f.write(
            "Architecture: EEGNet + "
            + (
                "SE+Temporal Attention"
                if (USE_SE_ATTENTION and USE_TEMPORAL_ATTENTION)
                else ("SE Attention" if USE_SE_ATTENTION else ("Temporal Attention" if USE_TEMPORAL_ATTENTION else "no attention"))
            )
            + " + TCN (architecture unchanged; safety patch only)\n"
        )
        f.write(f"Baseline adjust: {baseline_adjust}\n")
        f.write(f"Data: {len(dataset)} epochs, {TIME_POINTS} samples @ 125Hz\n")
        f.write(f"Augmentation: noise={NOISE_SIGMA}, shift={TIME_SHIFT}, ch_drop={CHANNEL_DROPOUT}\n")
        f.write(f"Training: {max_epochs} max epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}\n")
        f.write(f"Weight decay: {WEIGHT_DECAY}\n")
        f.write(f"Label smoothing: {LABEL_SMOOTHING}\n")
        f.write(f"CV: n_splits={n_splits}, seed_splits={effective_split_seed}, seed_train_rng={effective_train_seed}, smoke_test={SMOKE_TEST}\n")
        f.write(f"Class weights: {class_weights.cpu().numpy()}\n")
        f.write(f"Fold log: {fold_log_path}\n")
        if export_window_preds:
            f.write(f"Window preds: {window_pred_path}\n")
        f.write("-" * 50 + "\n")
        for i, r in enumerate(fold_results):
            f.write(f"Fold {i+1}: Accuracy={r['accuracy']:.4f}, F1={r['f1']:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
        if parity_mode:
            f.write(f"Mean best Macro-F1: {mean_f1:.4f} (+/- {std_f1:.4f})\n")
        else:
            f.write(f"Mean Macro-F1: {mean_f1:.4f} (+/- {std_f1:.4f})\n")
    
    print(f"\nResults saved to {results_path}")
    
    # Baseline reference (keep only the SVM baseline; other numbers are stale/confusing)
    print("\n--- BASELINE REFERENCE ---")
    print("SVM Baseline:      0.564 Accuracy")
    
    return mean_acc, mean_f1


def _read_fold_f1_from_log(fold_log_path: str) -> list[float]:
    vals: list[float] = []
    try:
        with open(fold_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and 'final_macro_f1' in obj:
                    vals.append(float(obj['final_macro_f1']))
    except Exception as e:
        print(f"[WARN] baseline sweep: failed reading fold log {fold_log_path}: {e}")
    return vals


def run_baseline_sweep(*, baseline_cache_path: str | None) -> None:
    modes = ["none", "divstd", "zscore"]
    results: list[dict] = []

    out_path = r"C:\vr_tsst_2025\results\workload_tcnet_baseline_sweep.json"
    ensure_parent_dir(out_path)

    for mode in modes:
        print("\n" + "=" * 60)
        print(f"BASELINE SWEEP: running mode={mode}")
        print("=" * 60)

        mean_acc, mean_f1 = run_cv_evaluation(baseline_adjust=mode, baseline_cache_path=baseline_cache_path, debug_baseline=False)
        fold_log_path = make_fold_log_path(baseline_adjust=mode)
        per_fold_f1 = _read_fold_f1_from_log(fold_log_path)
        std_f1 = float(np.std(per_fold_f1)) if len(per_fold_f1) > 0 else float('nan')

        results.append({
            "baseline": mode,
            "mean_acc": float(mean_acc),
            "mean_f1": float(mean_f1),
            "std_f1": std_f1,
            "per_fold_f1": [float(x) for x in per_fold_f1],
        })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Winner: highest mean macro-F1; if tie, lowest std of per-fold F1.
    def key_fn(r: dict) -> tuple[float, float]:
        return (float(r.get('mean_f1', float('-inf'))), -float(r.get('std_f1', float('inf'))))

    winner = sorted(results, key=key_fn, reverse=True)[0] if results else None
    print("\nBASELINE SWEEP COMPLETE")
    if winner is None:
        print("Winner: <none>")
        return
    print(f"Winner: {winner['baseline']}")
    print(f"Mean Macro-F1: {winner['mean_f1']:.3f} ± {winner['std_f1']:.3f}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG-TCNet on workload epochs with optional baseline adjustment")
    parser.add_argument(
        "--baseline_adjust",
        choices=["none", "mean", "divstd", "zscore"],
        default="zscore",
        help="Baseline adjustment mode applied to each epoch using per-block forest baseline (default: zscore)",
    )
    parser.add_argument(
        "--baseline_cache_path",
        type=str,
        default=BASELINE_CACHE_PATH,
        help="Optional pickle cache for baseline means (default: results/baseline_stats_cache.pkl)",
    )
    parser.add_argument(
        "--disable_baseline_cache",
        action="store_true",
        help="Disable baseline cache reads/writes (debugging; default uses cache)",
    )
    parser.add_argument(
        "--baseline_sweep",
        action="store_true",
        help="Run baseline sweep over modes [none, divstd, zscore] and write results JSON",
    )
    parser.add_argument(
        "--debug_baseline",
        action="store_true",
        help="If set, print baseline stats for first pid (max 5 epochs) during baseline adjustment",
    )
    parser.add_argument(
        "--export_window_preds",
        action="store_true",
        help="Write per-window test predictions to JSONL for time-indexed accuracy plots",
    )
    parser.add_argument(
        "--use_tuned_params",
        action="store_true",
        help="Load tuned hyperparameters from a JSON file and override defaults",
    )
    parser.add_argument(
        "--tuned_params_path",
        type=str,
        default=r"C:\vr_tsst_2025\results\best_tcnet_focused_params.json",
        help="Path to tuned params JSON (default: results/best_tcnet_focused_params.json)",
    )
    parser.add_argument(
        "--parity_mode",
        action="store_true",
        help="Enable tuner-parity mode (deterministic seeding, no aug/attn, epoch-by-epoch parity report)",
    )
    parser.add_argument(
        "--trial_number",
        type=int,
        default=0,
        help="Optuna trial number for Stage2 parity seeding (parity_mode only). Stage2 uses trial_seed = 1337 + 100000 + trial_number.",
    )
    parser.add_argument(
        "--no_parity_mode",
        action="store_true",
        help="Force-disable parity mode even when loading a Stage2 summary JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override training RNG seed (default: derived from Stage2 best trial when available, else 1337)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Override split seed for create_group_splits (default: 1337)",
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        help="Save best checkpoint per fold to results/tcnet_checkpoints (enabled by parity mode)",
    )
    args = parser.parse_args()

    if bool(args.baseline_sweep):
        run_baseline_sweep(baseline_cache_path=str(args.baseline_cache_path))
        raise SystemExit(0)

    tuned_reference_macro_f1: float | None = None
    inferred_trial_number: int | None = None

    if bool(args.use_tuned_params):
        with open(str(args.tuned_params_path), "r", encoding="utf-8") as f:
            tuned = json.load(f)
        best = tuned.get("best_params", {})
        fixed = tuned.get("fixed_params", {})
        arch = tuned.get("best_arch", {})

        # Heuristic: Stage2 overall summary produced by pipelines/13_workload_deep_learning_training/tune_tcnet.py
        is_stage2_summary = (
            isinstance(tuned, dict)
            and "best_value_macro_f1" in tuned
            and "best_arch" in tuned
            and "best_params" in tuned
        )

        if is_stage2_summary:
            tuned_reference_macro_f1 = float(tuned.get("best_value_macro_f1"))
            # Prefer the top-1 trial number if present (tune_tcnet uses trial_seed = SEED + trial.number)
            top = tuned.get("top_k_trials")
            if isinstance(top, list) and len(top) > 0 and isinstance(top[0], dict) and "trial_number" in top[0]:
                try:
                    inferred_trial_number = int(top[0]["trial_number"])
                except Exception:
                    inferred_trial_number = None

        if "F1" in arch:
            globals()["F1"] = int(arch["F1"])
        if "D" in arch:
            globals()["D"] = int(arch["D"])
        if "kernLength" in arch:
            globals()["KERN_LENGTH"] = int(arch["kernLength"])
        if "tcn_filters" in arch:
            globals()["TCN_FILTERS"] = int(arch["tcn_filters"])
        if "tcn_kernel" in arch:
            globals()["TCN_KERNEL"] = int(arch["tcn_kernel"])
        if "tcn_depth" in arch:
            globals()["TCN_DEPTH"] = int(arch["tcn_depth"])

        if "F1" in best:
            globals()["F1"] = int(best["F1"])
        if "D" in best:
            globals()["D"] = int(best["D"])
        if "kernLength" in best:
            globals()["KERN_LENGTH"] = int(best["kernLength"])
        if "dropout_eeg" in best:
            globals()["DROPOUT_EEG"] = float(best["dropout_eeg"])
        if "tcn_filters" in best:
            globals()["TCN_FILTERS"] = int(best["tcn_filters"])
        if "tcn_kernel" in best:
            globals()["TCN_KERNEL"] = int(best["tcn_kernel"])
        if "tcn_depth" in best:
            globals()["TCN_DEPTH"] = int(best["tcn_depth"])
        if "dropout_tcn" in best:
            globals()["DROPOUT_TCN"] = float(best["dropout_tcn"])
        if "learning_rate" in best:
            globals()["LEARNING_RATE"] = float(best["learning_rate"])
        if "batch_size" in best:
            globals()["BATCH_SIZE"] = int(best["batch_size"])
        if "weight_decay" in best:
            globals()["WEIGHT_DECAY"] = float(best["weight_decay"])
        if "label_smoothing" in best:
            globals()["LABEL_SMOOTHING"] = float(best["label_smoothing"])

        if "use_se_attention" in fixed:
            globals()["USE_SE_ATTENTION"] = bool(fixed["use_se_attention"])
        if "use_temporal_attention" in fixed:
            globals()["USE_TEMPORAL_ATTENTION"] = bool(fixed["use_temporal_attention"])
        if "use_augmentation" in fixed:
            globals()["USE_AUGMENTATION"] = bool(fixed["use_augmentation"])
        if "early_stop_patience" in fixed:
            globals()["EARLY_STOP_PATIENCE"] = int(fixed["early_stop_patience"])
        elif "best_arch" in tuned:
            globals()["EARLY_STOP_PATIENCE"] = 0

        # Match the tuner default (MAX_EPOCHS=60) when we are clearly loading Stage2 tuned params.
        # This matters because the Optuna objective reports best-over-epoch macro-F1.
        if is_stage2_summary and "max_epochs" not in tuned:
            globals()["EPOCHS"] = 60

        # Stage2 tuning assumes baseline zscore is applied (separate from fold-safe CV normalization).
        if is_stage2_summary and str(args.baseline_adjust) != "zscore":
            print(
                "[WARN] Stage2 tuned params were optimized with baseline_adjust='zscore', "
                f"but this run is using baseline_adjust='{args.baseline_adjust}'. "
                "This mismatch can materially reduce performance."
            )

        print(f"Loaded tuned params from {args.tuned_params_path}")
        print(
            "Tuned model params: "
            f"F1={F1}, D={D}, kernLength={KERN_LENGTH}, dropout_eeg={DROPOUT_EEG}, "
            f"tcn_filters={TCN_FILTERS}, tcn_kernel={TCN_KERNEL}, tcn_depth={TCN_DEPTH}, dropout_tcn={DROPOUT_TCN}"
        )
        print(
            "Tuned training params: "
            f"batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, "
            f"label_smoothing={LABEL_SMOOTHING}, early_stop_patience={EARLY_STOP_PATIENCE}, "
            f"use_se_attention={USE_SE_ATTENTION}, use_temporal_attention={USE_TEMPORAL_ATTENTION}, "
            f"use_augmentation={USE_AUGMENTATION}"
        )
        if is_stage2_summary:
            print(f"Stage2 parity hints: expected_baseline_adjust=zscore, epochs={EPOCHS}")

    # Baseline cache safety: default matches tuner (enabled). Allow disabling for debugging.
    baseline_cache_path = None if bool(args.disable_baseline_cache) else str(args.baseline_cache_path)

    # Parity mode: enabled explicitly or auto-enabled for Stage2 summary unless user forces it off.
    parity_mode = bool(args.parity_mode) or (inferred_trial_number is not None)
    if bool(args.no_parity_mode):
        parity_mode = False

    # Training seed selection
    # - Non-parity mode preserves legacy behavior (train_seed can be overridden).
    # - Parity mode uses --trial_number (or inferred from a Stage2 summary) to derive the Stage2 trial_seed.
    if args.seed is not None:
        train_seed = int(args.seed)
    else:
        train_seed = int(SEED)

    trial_number = int(args.trial_number)
    if trial_number == 0 and inferred_trial_number is not None:
        trial_number = int(inferred_trial_number)

    if parity_mode:
        print(f"[PARITY] trial_number={trial_number} -> trial_seed={int(SEED + 100000 + trial_number)}")
    elif inferred_trial_number is not None:
        print(f"Inferred Stage2 best trial_number={inferred_trial_number} (parity_mode disabled)")

    run_cv_evaluation(
        baseline_adjust=str(args.baseline_adjust),
        baseline_cache_path=baseline_cache_path,
        debug_baseline=bool(args.debug_baseline),
        export_window_preds=bool(args.export_window_preds),
        parity_mode=bool(parity_mode),
        trial_number=int(trial_number),
        train_seed=int(train_seed),
        split_seed=int(SEED if args.split_seed is None else int(args.split_seed)),
        save_checkpoints=bool(args.save_checkpoints) or bool(parity_mode),
        tuned_reference_macro_f1=tuned_reference_macro_f1,
    )
