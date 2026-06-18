"""Export VR-TSST EEG as per-participant continuous HDF5 files.

Pipeline
--------
  1. Load raw XDF (two 64-channel eego streams at 500 Hz) via the existing
     VR-TSST xdf_to_set utilities.
  2. Resample 500 Hz → 128 Hz (scipy.signal.resample_poly, ratio 32:125).
  3. Apply the MATB preprocessing contract using adaptive_matb_2026's
     EegPreprocessor: causal Butterworth 0.5–40 Hz order 4, causal IIR
     notch 50 Hz Q=30, Common Average Reference.
  4. Reconstruct per-block condition labels from the Unity metadata CSV and
     config/conditions.yaml using the existing VR-TSST label pipeline.
  5. Compute segment boundaries (sample indices) for task, forest, and
     fixation blocks.
  6. Write one HDF5 file per participant with continuous preprocessed EEG
     and segment metadata.  Downstream code handles windowing/epoching.

Output (per participant)
-----------------------
  output/matb_pretrain/continuous/P01.h5:
    /eeg                    (n_samples, 128) float32  — full session, gzip
    /task_onsets             (n_tasks,)       int64    — block start (sample)
    /task_offsets            (n_tasks,)       int64    — block end   (sample)
    /task_labels             (n_tasks,)       int8     — 0=LOW, 2=HIGH
    /task_block_order        (n_tasks,)       int8     — temporal order 0–3
    /forest_onsets           (n_forests,)     int64
    /forest_offsets          (n_forests,)     int64
    /forest_block_order      (n_forests,)     int8
    /fixation_onsets         (n_fix,)         int64
    /fixation_offsets        (n_fix,)         int64
    attrs: srate, pid, n_channels, channels (JSON),
           task_conditions (JSON), forest_conditions (JSON),
           fixation_conditions (JSON), preprocessing_config_hash

  output/matb_pretrain/continuous/manifest.json:
    Build metadata and per-participant summaries.

Usage
-----
  
  python ... --participants 1 3 5
  python ... --workers 4

Dependencies
------------
  pip install pyxdf scipy numpy h5py pyyaml pandas
  MATB repo must be accessible (default: C:/phd_projects/adaptive_matb_2026).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyxdf
import yaml
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Path wiring
# ---------------------------------------------------------------------------
_PIPELINE_DIR = Path(__file__).resolve().parent
_VR_TSST_ROOT = _PIPELINE_DIR.parent.parent          # c:/vr_tsst_2025
_XDF_TO_SET_DIR = _VR_TSST_ROOT / "pipelines" / "01_xdf_to_set"

# Add VR-TSST xdf_to_set utilities to path
sys.path.insert(0, str(_XDF_TO_SET_DIR))
from xdf_to_set import (  # noqa: E402
    load_and_merge,
    add_exposure_type_from_config,
    align_timestamps,
    extract_event_timestamps,
)

# ---------------------------------------------------------------------------
# Resampling constants  500 Hz → 128 Hz  (GCD=4, up=32, down=125)
# ---------------------------------------------------------------------------
_HW_SRATE = 500.0
_TARGET_SRATE = 128.0
_RESAMPLE_UP = 32
_RESAMPLE_DOWN = 125

# ---------------------------------------------------------------------------
# Condition → binary workload label mapping
# ---------------------------------------------------------------------------
# VR-TSST conditions contain numeric task variants e.g.
#   HighStress_HighCog1022_Task, LowStress_HighCog2043_Task
# Matching uses substring checks (same logic as pipeline 12 export_mne_epochs.py):
#   "HighCog" in label → HIGH workload
#   "LowCog"  in label → LOW  workload
# Conditions not ending in "_Task" (preambles, finishes, etc.) are ignored.

def _workload_class(condition_label: str) -> str | None:
    """Return 'HIGH', 'LOW', or None if not a task condition."""
    if not condition_label.endswith("_Task"):
        return None
    if "HighCog" in condition_label:
        return "HIGH"
    if "LowCog" in condition_label:
        return "LOW"
    return None

# Numeric labels matching adaptive_matb_2026 LABEL_MAP
_LABEL_HIGH: int = 2
_LABEL_LOW: int = 0

# ---------------------------------------------------------------------------
# Baseline condition identifiers
# ---------------------------------------------------------------------------
_FOREST_LABELS = {"Forest1", "Forest2", "Forest3", "Forest4"}
_FIXATION_LABELS = {
    "Pre_Exposure_Blank_Fixation_Cross",
    "Pre_Exposure_Room_Fixation_Cross",
}

# ---------------------------------------------------------------------------
# Study design constants
# ---------------------------------------------------------------------------
_BLOCK_DUR_S: float = 180.0   # nominal task block duration (seconds)
_FIXATION_DUR_S: float = 60.0  # nominal fixation cross duration (seconds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_hash(params: dict) -> str:
    """Stable 8-char hex hash of preprocessing parameters."""
    s = json.dumps(params, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _load_matb_modules(matb_repo: Path) -> tuple[Any, Any]:
    """Import EegPreprocessor and EegPreprocessingConfig from the MATB repo.

    The MATB eeg package's __init__.py imports pylsl (a live-recording
    dependency not available in the VR-TSST environment).  We inject a
    minimal stub into sys.modules before importing so the rest of the
    package — which does not use pylsl — loads cleanly.
    """
    import types

    # Stub out pylsl so eeg/__init__.py (which imports EegInlet → pylsl)
    # does not raise ModuleNotFoundError.
    if "pylsl" not in sys.modules:
        sys.modules["pylsl"] = types.ModuleType("pylsl")

    matb_src = matb_repo / "src"
    if str(matb_src) not in sys.path:
        sys.path.insert(0, str(matb_src))

    from eeg import (  # type: ignore[import]  # noqa: E402
        EegPreprocessingConfig,
        EegPreprocessor,
    )
    return EegPreprocessingConfig, EegPreprocessor


def _load_channel_names(matb_repo: Path) -> list[str]:
    """Load the canonical 128-channel name list from the MATB config."""
    meta_path = matb_repo / "config" / "eeg_metadata.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    return meta["channel_names"]


def _load_metadata_csv(meta_path: Path) -> pd.DataFrame:
    """Load a Unity metadata CSV and parse the LSL_Timestamp index."""
    df = pd.read_csv(meta_path, low_memory=False)
    df["LSL_Timestamp"] = pd.to_datetime(df["LSL_Timestamp"], unit="s", origin="unix")
    df = df.set_index("LSL_Timestamp")
    return df


def _onset_to_sample(onset_ts: np.datetime64,
                     eeg_start_ts: np.datetime64,
                     srate: float) -> int:
    """Convert an absolute onset datetime to a sample index in the resampled EEG."""
    delta_s = (onset_ts - eeg_start_ts) / np.timedelta64(1, "s")
    return max(0, int(round(delta_s * srate)))


# ---------------------------------------------------------------------------
# Per-participant processing
# ---------------------------------------------------------------------------

def process_participant(
    pid: int,
    xdf_dir: Path,
    meta_dir: Path,
    conditions_yaml: Path,
    matb_repo: Path,
    out_dir: Path,
    channel_names: list[str],
    cfg_hash: str,
) -> dict | None:
    """Process one VR-TSST participant: preprocess EEG and write .h5 file.

    Writes a per-participant HDF5 with continuous preprocessed EEG and
    segment boundary metadata.  Returns a summary dict on success,
    None on failure.
    """
    pid_str = f"P{pid:02d}"

    # Skip if already processed
    fpath = out_dir / f"{pid_str}.h5"
    if fpath.exists():
        print(f"  [SKIP] {pid_str}: {fpath.name} already exists")
        return None

    print(f"\n{'='*60}")
    print(f"  {pid_str}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Locate XDF file(s) for this participant
    # XDF directories use plain numeric names (e.g. data/RAW/1/, data/RAW/10/)
    # ------------------------------------------------------------------
    xdf_files = (
        sorted(xdf_dir.glob(f"{pid}/*.xdf")) or
        sorted(xdf_dir.glob(f"{pid_str}/*.xdf")) or
        sorted(xdf_dir.glob(f"P{pid}/*.xdf"))
    )
    if not xdf_files:
        print(f"  [SKIP] No XDF files found under {xdf_dir}/{pid_str}/")
        return None
    if len(xdf_files) > 1:
        print(f"  [WARN] Multiple XDF files found; using first: {xdf_files[0].name}")
    xdf_path = xdf_files[0]
    print(f"  XDF: {xdf_path.name}")

    # ------------------------------------------------------------------
    # 2. Locate metadata CSV
    # ------------------------------------------------------------------
    meta_candidates = [
        meta_dir / f"{pid_str}.csv",
        meta_dir / f"P{pid}.csv",
    ]
    meta_path = next((p for p in meta_candidates if p.exists()), None)
    if meta_path is None:
        print(f"  [SKIP] Metadata CSV not found (tried {[str(c) for c in meta_candidates]})")
        return None
    print(f"  Metadata: {meta_path.name}")

    # ------------------------------------------------------------------
    # 3. Load and merge two EEG streams (500 Hz, 128 ch)
    # ------------------------------------------------------------------
    try:
        merged = load_and_merge(xdf_path)
    except Exception as exc:
        print(f"  [SKIP] XDF load/merge failed: {exc}")
        return None

    eeg_hw = merged["data"].T.astype(np.float32)   # (128, n_samples) at 500 Hz
    eeg_ts_hw = merged["timestamps"]                # LSL timestamps at 500 Hz
    n_ch, n_samples_hw = eeg_hw.shape
    print(f"  Loaded: {n_ch} ch x {n_samples_hw} samples at {_HW_SRATE:.0f} Hz")

    if n_ch != 128:
        print(f"  [SKIP] Expected 128 channels, got {n_ch}")
        return None

    # ------------------------------------------------------------------
    # 4. Resample 500 Hz → 128 Hz
    # ------------------------------------------------------------------
    eeg_128 = resample_poly(eeg_hw, up=_RESAMPLE_UP, down=_RESAMPLE_DOWN, axis=1)
    eeg_128 = eeg_128.astype(np.float32)
    n_samples_128 = eeg_128.shape[1]
    print(f"  Resampled: {n_samples_128} samples at {_TARGET_SRATE:.0f} Hz")

    # ------------------------------------------------------------------
    # 5. Apply MATB preprocessing contract at 128 Hz
    # ------------------------------------------------------------------
    EegPreprocessingConfig, EegPreprocessor = _load_matb_modules(matb_repo)

    prep_config = EegPreprocessingConfig(
        bp_low_hz=0.5,
        bp_high_hz=40.0,
        bp_order=4,
        notch_freq=50.0,
        notch_quality=30.0,
        apply_car=True,
        srate=_TARGET_SRATE,
    )
    preprocessor = EegPreprocessor(prep_config)
    preprocessor.initialize_filters(n_ch)
    eeg_proc = preprocessor.process(eeg_128)   # (128, n_samples) preprocessed
    print("  Preprocessing applied (0.5-40 Hz causal BP, 50 Hz notch, CAR)")

    # ------------------------------------------------------------------
    # 6. Reconstruct condition labels from metadata CSV
    # ------------------------------------------------------------------
    try:
        df_meta = _load_metadata_csv(meta_path)
    except Exception as exc:
        print(f"  [SKIP] Failed to load metadata CSV: {exc}")
        return None

    try:
        df_meta = add_exposure_type_from_config(df_meta, config_path=conditions_yaml)
    except Exception as exc:
        print(f"  [SKIP] Condition assignment failed: {exc}")
        return None

    # align_timestamps expects (n_samples, n_ch) for shape check — pass transposed data
    # align_timestamps shifts EEG timestamps into the physio CSV reference frame,
    # correcting any wall-clock offset between the two streams' start times.
    # We only need [0] as the reference origin for converting onsets → sample indices.
    aligned_eeg_dt = align_timestamps(
        df_physio=df_meta,
        eeg_data=eeg_hw.T,        # (n_samples_500hz, 128) — only .shape[0] used
        eeg_ts=eeg_ts_hw,
        srate=_HW_SRATE,          # alignment uses original 500 Hz timestamps
    )
    # aligned_eeg_dt[0] == physio df_meta.index[0] (shared reference frame).
    # extract_event_timestamps returns timestamps also in that reference frame,
    # so delta = onset_ts - aligned_eeg_dt[0] gives the correct offset in seconds.
    eeg_aligned_start = aligned_eeg_dt[0]   # datetime64[ns] in physio ref frame

    event_ts = extract_event_timestamps(df_meta)

    task_events = {
        label: ts for label, ts in event_ts.items()
        if _workload_class(label) is not None
    }
    forest_events = {
        label: ts for label, ts in event_ts.items()
        if label in _FOREST_LABELS
    }
    fixation_events = {
        label: ts for label, ts in event_ts.items()
        if label in _FIXATION_LABELS
    }
    if not task_events:
        print("  [SKIP] No task condition onsets found in metadata")
        return None

    print(f"  Task conditions: {list(task_events.keys())}")
    print(f"  Forest baselines: {list(forest_events.keys())}")
    print(f"  Fixation baselines: {list(fixation_events.keys())}")

    # ------------------------------------------------------------------
    # 7. Compute segment boundaries (sample indices)
    # ------------------------------------------------------------------
    task_block_samples = int(_BLOCK_DUR_S * _TARGET_SRATE)
    forest_block_samples = int(_BLOCK_DUR_S * _TARGET_SRATE)
    fixation_block_samples = int(_FIXATION_DUR_S * _TARGET_SRATE)

    task_onsets: list[int] = []
    task_offsets: list[int] = []
    task_labels_list: list[int] = []
    task_conditions: list[str] = []

    for block_i, (condition, onset_ts) in enumerate(
            sorted(task_events.items(), key=lambda x: x[1])):
        wl_class = _workload_class(condition)
        if wl_class is None:
            print(f"  [WARN] Skipping unmapped condition: {condition}")
            continue
        label_int = _LABEL_HIGH if wl_class == "HIGH" else _LABEL_LOW

        onset = _onset_to_sample(onset_ts, eeg_aligned_start, _TARGET_SRATE)
        offset = min(onset + task_block_samples, n_samples_128)
        task_onsets.append(onset)
        task_offsets.append(offset)
        task_labels_list.append(label_int)
        task_conditions.append(condition)

        dur_s = (offset - onset) / _TARGET_SRATE
        label_name = "HIGH" if label_int == _LABEL_HIGH else "LOW"
        print(f"  {condition}: {dur_s:.1f}s -> {label_name} (block {block_i})")

    if not task_onsets:
        print("  [SKIP] No task conditions found")
        return None

    # Forest baseline segments
    forest_onsets: list[int] = []
    forest_offsets: list[int] = []
    forest_conditions: list[str] = []
    for block_i, (condition, onset_ts) in enumerate(
            sorted(forest_events.items(), key=lambda x: x[1])):
        onset = _onset_to_sample(onset_ts, eeg_aligned_start, _TARGET_SRATE)
        offset = min(onset + forest_block_samples, n_samples_128)
        forest_onsets.append(onset)
        forest_offsets.append(offset)
        forest_conditions.append(condition)
        dur_s = (offset - onset) / _TARGET_SRATE
        print(f"  {condition}: {dur_s:.1f}s (forest block {block_i})")

    # Fixation baseline segments (Pre_Exposure, 60 s each)
    fixation_onsets: list[int] = []
    fixation_offsets: list[int] = []
    fixation_conditions: list[str] = []
    for condition, onset_ts in sorted(fixation_events.items(), key=lambda x: x[1]):
        onset = _onset_to_sample(onset_ts, eeg_aligned_start, _TARGET_SRATE)
        offset = min(onset + fixation_block_samples, n_samples_128)
        fixation_onsets.append(onset)
        fixation_offsets.append(offset)
        fixation_conditions.append(condition)
        dur_s = (offset - onset) / _TARGET_SRATE
        print(f"  {condition}: {dur_s:.1f}s fixation")

    # ------------------------------------------------------------------
    # 8. Write per-participant HDF5
    # ------------------------------------------------------------------
    fpath = out_dir / f"{pid_str}.h5"
    eeg_out = eeg_proc.T.astype(np.float32)  # (n_samples, n_channels)

    with h5py.File(fpath, "w") as f:
        # Continuous preprocessed EEG — chunked for fast time-slicing
        f.create_dataset("eeg", data=eeg_out,
                         chunks=(1024, eeg_out.shape[1]))

        # Task segment boundaries
        f.create_dataset("task_onsets",
                         data=np.array(task_onsets, dtype=np.int64))
        f.create_dataset("task_offsets",
                         data=np.array(task_offsets, dtype=np.int64))
        f.create_dataset("task_labels",
                         data=np.array(task_labels_list, dtype=np.int8))
        f.create_dataset("task_block_order",
                         data=np.arange(len(task_onsets), dtype=np.int8))

        # Forest segment boundaries
        f.create_dataset("forest_onsets",
                         data=np.array(forest_onsets, dtype=np.int64))
        f.create_dataset("forest_offsets",
                         data=np.array(forest_offsets, dtype=np.int64))
        f.create_dataset("forest_block_order",
                         data=np.arange(len(forest_onsets), dtype=np.int8))

        # Fixation segment boundaries
        f.create_dataset("fixation_onsets",
                         data=np.array(fixation_onsets, dtype=np.int64))
        f.create_dataset("fixation_offsets",
                         data=np.array(fixation_offsets, dtype=np.int64))

        # Metadata attributes
        f.attrs["srate"] = _TARGET_SRATE
        f.attrs["pid"] = pid_str
        f.attrs["n_channels"] = eeg_out.shape[1]
        f.attrs["channels"] = json.dumps(channel_names)
        f.attrs["preprocessing_config_hash"] = cfg_hash
        f.attrs["task_conditions"] = json.dumps(task_conditions)
        f.attrs["forest_conditions"] = json.dumps(forest_conditions)
        f.attrs["fixation_conditions"] = json.dumps(fixation_conditions)

    file_mb = fpath.stat().st_size / 1e6
    eeg_dur_s = round(n_samples_128 / _TARGET_SRATE, 1)
    print(f"  Written: {fpath.name} ({file_mb:.1f} MB, {eeg_dur_s}s, "
          f"{len(task_onsets)} task, {len(forest_onsets)} forest, "
          f"{len(fixation_onsets)} fix)")

    return {
        "n_samples": n_samples_128,
        "duration_s": eeg_dur_s,
        "task_blocks": len(task_onsets),
        "forest_blocks": len(forest_onsets),
        "fixation_blocks": len(fixation_onsets),
        "task_labels": task_labels_list,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--participants", type=int, nargs="+", default=None,
        help="Participant IDs to process (default: 1–48)",
    )
    parser.add_argument(
        "--matb-repo", type=Path,
        default=Path("C:/phd_projects/adaptive_matb_2026"),
        help="Path to adaptive_matb_2026 repository root",
    )
    parser.add_argument(
        "--raw-dir", type=Path,
        default=_VR_TSST_ROOT / "data" / "RAW",
        help="Directory containing per-participant XDF subdirectories",
    )
    parser.add_argument(
        "--meta-dir", type=Path,
        default=_VR_TSST_ROOT / "data" / "raw" / "metadata",
        help="Directory containing Unity metadata CSVs (P01.csv, P02.csv, ...)",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=_VR_TSST_ROOT / "output" / "matb_pretrain",
        help="Output directory (files written to <out-dir>/continuous/)",
    )
    parser.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1),
        help="Number of parallel workers (default: min(4, cpu_count))",
    )
    parser.add_argument(
        "--no-qc-filter", action="store_true",
        help="Include QC-excluded participants (process all)",
    )
    args = parser.parse_args()

    # Validate MATB repo
    matb_src = args.matb_repo / "src"
    if not matb_src.exists():
        sys.exit(f"[ERROR] MATB src/ not found at {matb_src}. Check --matb-repo.")

    conditions_yaml = _VR_TSST_ROOT / "config" / "conditions.yaml"
    if not conditions_yaml.exists():
        sys.exit(f"[ERROR] conditions.yaml not found at {conditions_yaml}")

    # Load QC exclusions from MATB config
    qc_yaml = args.matb_repo / "config" / "tsst_qc.yaml"
    excluded_pids: set[str] = set()
    if qc_yaml.exists():
        with open(qc_yaml) as f:
            qc = yaml.safe_load(f)
        excluded_pids = set(qc.get("excluded_participants", {}).keys())
        print(f"QC exclusions: {len(excluded_pids)} from {qc_yaml.name}")
    else:
        print(f"[WARN] {qc_yaml} not found — no QC exclusions applied")

    cont_dir = args.out_dir / "continuous"
    cont_dir.mkdir(parents=True, exist_ok=True)

    # Build participant list: all 1–48 or --participants, minus QC exclusions
    all_pids = args.participants or list(range(1, 49))
    if args.no_qc_filter:
        participants = all_pids
    else:
        participants = [p for p in all_pids if f"P{p:02d}" not in excluded_pids]
    n_workers = max(1, args.workers)

    # Load channel names from MATB for provenance
    channel_names = _load_channel_names(args.matb_repo)

    # Build preprocessing config hash (matches MATB _config_hash convention)
    prep_params = {
        "bp_low_hz": 0.5, "bp_high_hz": 40.0, "bp_order": 4,
        "notch_freq": 50.0, "notch_quality": 30.0,
        "apply_car": True, "srate": _TARGET_SRATE,
    }
    cfg_hash = _config_hash(prep_params)

    print(f"VR-TSST -> MATB continuous EEG export")
    print(f"Target srate : {_TARGET_SRATE} Hz")
    print(f"Participants : {participants}")
    print(f"Workers      : {n_workers}")
    print(f"Output       : {cont_dir}")

    # Process participants — each worker writes its own .h5 file
    summaries: dict[str, dict] = {}
    skipped: list[int] = []

    def _submit_args(pid: int) -> dict:
        return dict(
            pid=pid,
            xdf_dir=args.raw_dir,
            meta_dir=args.meta_dir,
            conditions_yaml=conditions_yaml,
            matb_repo=args.matb_repo,
            out_dir=cont_dir,
            channel_names=channel_names,
            cfg_hash=cfg_hash,
        )

    if n_workers <= 1:
        for pid in participants:
            result = process_participant(**_submit_args(pid))
            if result is None:
                skipped.append(pid)
            else:
                summaries[f"P{pid:02d}"] = result
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers) as pool:
            futures = {
                pool.submit(process_participant, **_submit_args(pid)): pid
                for pid in participants
            }
            for future in concurrent.futures.as_completed(futures):
                pid = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"  [ERROR] P{pid:02d} raised {exc!r}")
                    skipped.append(pid)
                    continue
                if result is None:
                    skipped.append(pid)
                else:
                    summaries[f"P{pid:02d}"] = result
                    print(f"  [OK] P{pid:02d}")

    if not summaries:
        sys.exit("[ERROR] No participants produced valid output.")

    # Write manifest
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": "vr_tsst_2025",
        "format": "continuous",
        "preprocessing_config_hash": cfg_hash,
        "preprocessing_params": prep_params,
        "srate": _TARGET_SRATE,
        "n_channels": 128,
        "label_map": {"LOW": _LABEL_LOW, "HIGH": _LABEL_HIGH},
        "participants": {
            k: {kk: vv for kk, vv in v.items() if kk != "task_labels"}
            for k, v in summaries.items()
        },
        "skipped": skipped,
    }
    manifest_path = cont_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_written = len(summaries)
    total_samples = sum(v["n_samples"] for v in summaries.values())
    total_dur = sum(v["duration_s"] for v in summaries.values())
    print(f"\nDone.")
    print(f"  Participants : {n_written} written, {len(skipped)} skipped {skipped}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total duration: {total_dur / 60:.1f} min")
    print(f"  Output       : {cont_dir}")
    print(f"  Manifest     : {manifest_path}")


if __name__ == "__main__":
    main()
