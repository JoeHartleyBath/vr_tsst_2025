from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
import yaml

# ============================================================================== 
# CONFIGURATION
# ============================================================================== 
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_CONDITIONS = PROJECT_ROOT / "config" / "conditions.yaml"

CLEANED_DIR = PROJECT_ROOT / "output" / "cleaned_eeg"
OUTPUT_DIR = PROJECT_ROOT / "output" / "adaptive_workload" / "mne_epochs"

# Sliding windowing across each task block.
WINDOW_DUR_S = 10.0
STEP_DUR_S = 5.0
BLOCK_DUR_S = 180.0

# Cleaned data is already resampled to 125 Hz in the pipeline.
EXPECTED_SFREQ_HZ = 125.0
SFREQ_ABS_TOL_HZ = 0.2

# All 44 EEG-valid participants
ALL_PIDS = [1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
            42, 43, 44, 45, 47, 48]
SUBSET_PIDS = ALL_PIDS  # Use all 44 participants

# All 4 conditions (LowStress + HighStress)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass(frozen=True)
class BlockEvent:
    raw_desc: str
    norm_desc: str
    onset_samp: int
    onset_sec: float


def _load_conditions_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _canonicalize_task_label(label: str) -> str:
    # Collapse variants like HighCog1022_Task / HighCog2043_Task -> HighCog_Task.
    return re.sub(r"\d+_Task$", "_Task", label)


def _normalize_annot_desc(desc: object) -> str:
    """Normalize EEGLAB->MNE annotation descriptions for robust event matching.

    Rules:
    - Strip whitespace.
    - If numeric-like (e.g., "101", "101.0"), coerce to int then back to string ("101").
    - Otherwise, keep as-is (after stripping).
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


def _build_event_mappings(cfg: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Return (task_code->canonical_label, forest_code->forest_label)."""
    export = cfg.get("export_event_labels", {})
    if not export:
        raise RuntimeError("config/conditions.yaml missing export_event_labels")

    # Invert label->code into code->labels.
    code_to_labels: dict[str, list[str]] = {}
    for label, code in export.items():
        code_str = str(int(code))
        code_to_labels.setdefault(code_str, []).append(str(label))

    # Task codes are the 4 core onsets (merged across 1022/2043 via shared codes).
    task_codes = ("101", "102", "103", "104")
    forest_codes = ("20", "21", "23", "24")

    task_code_to_label: dict[str, str] = {}
    for c in task_codes:
        labels = [lbl for lbl in code_to_labels.get(c, []) if lbl.endswith("_Task")]
        if len(labels) == 0:
            raise RuntimeError(f"No _Task labels found for event code {c} in export_event_labels")
        canon = sorted({_canonicalize_task_label(lbl) for lbl in labels})
        if len(canon) != 1:
            raise RuntimeError(f"Ambiguous task label mapping for code {c}: {labels} -> {canon}")
        task_code_to_label[c] = canon[0]

    forest_code_to_label: dict[str, str] = {}
    for c in forest_codes:
        labels = [lbl for lbl in code_to_labels.get(c, []) if lbl.startswith("Forest")]
        if len(labels) != 1:
            raise RuntimeError(f"Expected exactly one Forest label for code {c}, got: {labels}")
        forest_code_to_label[c] = labels[0]

    return task_code_to_label, forest_code_to_label


def _workload_class_for_condition(condition_label: str) -> str:
    # Workload is based on cognitive load (HighCog vs LowCog).
    if "HighCog" in condition_label:
        return "HighWorkload"
    if "LowCog" in condition_label:
        return "LowWorkload"
    return "Unknown"


def _extract_block_events(raw: mne.io.BaseRaw, valid_descs: Iterable[str]) -> list[BlockEvent]:
    valid_descs = set(valid_descs)
    if raw.annotations is None or len(raw.annotations) == 0:
        return []

    blocks: list[BlockEvent] = []
    sfreq = float(raw.info["sfreq"])
    for onset_sec, desc in zip(raw.annotations.onset, raw.annotations.description, strict=False):
        raw_desc = str(desc)
        norm_desc = _normalize_annot_desc(raw_desc)
        if norm_desc in valid_descs:
            onset_samp = int(round(float(onset_sec) * sfreq))
            blocks.append(
                BlockEvent(raw_desc=raw_desc, norm_desc=norm_desc, onset_samp=onset_samp, onset_sec=float(onset_sec))
            )

    blocks.sort(key=lambda b: b.onset_samp)
    return blocks


def _sliding_windows_for_block(
    data: np.ndarray,
    onset_samp: int,
    sfreq: float,
    window_dur_s: float,
    step_dur_s: float,
    block_dur_s: float,
) -> tuple[list[np.ndarray], list[int], str | None]:
    window_samps = int(round(window_dur_s * sfreq))
    step_samps = int(round(step_dur_s * sfreq))
    block_samps = int(round(block_dur_s * sfreq))
    n_expected = int((block_dur_s - window_dur_s) / step_dur_s) + 1

    windows: list[np.ndarray] = []
    starts: list[int] = []
    stop_reason: str | None = None
    block_end = onset_samp + block_samps
    for win_idx in range(n_expected):
        start = onset_samp + win_idx * step_samps
        stop = start + window_samps
        if stop > block_end:
            stop_reason = "block_end"
            break
        if stop > data.shape[1]:
            stop_reason = "data_end"
            break
        windows.append(data[:, start:stop].astype(np.float32, copy=False))
        starts.append(start)

    return windows, starts, stop_reason


def _print_annotation_diagnostics(raw: mne.io.BaseRaw, max_unique: int = 40) -> None:
    if raw.annotations is None or len(raw.annotations) == 0:
        print("  [ERR] No annotations found")
        return

    raw_vals = [str(d) for d in raw.annotations.description]
    norm_vals = [_normalize_annot_desc(d) for d in raw_vals]

    raw_unique = sorted(set(raw_vals))
    norm_unique = sorted(set(norm_vals))
    print("  Annotation descriptions (unique; raw -> normalized):")
    for rv in raw_unique[:max_unique]:
        print(f"    {rv!r} -> {_normalize_annot_desc(rv)!r}")
    if len(raw_unique) > max_unique:
        print(f"    ... ({len(raw_unique) - max_unique} more raw unique values)")

    print(f"  Unique normalized descriptions (sample): {norm_unique[:min(len(norm_unique), max_unique)]}")


def export_subject_epochs(
    pid: int,
    task_code_to_label: dict[str, str],
    forest_code_to_label: dict[str, str],
    *,
    debug: bool = False,
    print_ann_sample: bool = False,
) -> dict:
    set_path = os.path.join(CLEANED_DIR, f'P{pid:02d}_cleaned.set')
    if not os.path.exists(set_path):
        # Try without leading zero if P01 fails
        set_path = os.path.join(CLEANED_DIR, f'P{pid}_cleaned.set')
        if not os.path.exists(set_path):
            print(f"Skipping P{pid}: File not found ({set_path})")
            return {"pid": pid, "skipped": True}

    print(f"\nProcessing P{pid}...")

    # Load cleaned EEG. EEGLAB files often have .fdt in same dir.
    raw = mne.io.read_raw_eeglab(set_path, preload=True)
    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, EXPECTED_SFREQ_HZ, atol=SFREQ_ABS_TOL_HZ):
        raise AssertionError(
            f"P{pid}: expected cleaned data at {EXPECTED_SFREQ_HZ} Hz, got {sfreq:.3f} Hz (no resampling here)."
        )

    # Extract event onsets from MNE annotations (derived from EEGLAB events).
    if print_ann_sample or debug:
        _print_annotation_diagnostics(raw)

    task_codes = tuple(task_code_to_label.keys())
    forest_codes = tuple(forest_code_to_label.keys())
    task_blocks = _extract_block_events(raw, task_codes)
    forest_blocks = _extract_block_events(raw, forest_codes)

    assert len(task_blocks) > 0, f"P{pid}: no task block onsets found (codes {task_codes})"
    assert len(forest_blocks) > 0, f"P{pid}: no forest onsets found (codes {forest_codes})"

    # Sort forest onsets and assign each task to the nearest preceding forest.
    forest_sorted = sorted(
        [(fb.norm_desc, fb.onset_samp) for fb in forest_blocks],
        key=lambda x: x[1],
    )

    def assign_forest(task_onset_samp: int) -> str | None:
        prior = [code for (code, s) in forest_sorted if s < task_onset_samp]
        return prior[-1] if prior else None

    # Sliding windows for each task block.
    data = raw.get_data()  # (n_channels, n_times)
    expected_per_block = int((BLOCK_DUR_S - WINDOW_DUR_S) / STEP_DUR_S) + 1
    win_samps = int(round(WINDOW_DUR_S * sfreq))
    assert win_samps == 1250, f"P{pid}: expected 10s window to be 1250 samples at 125Hz; got {win_samps}"

    all_epoch_data: list[np.ndarray] = []
    all_metadata: list[dict] = []

    # Assertions: ensure the config-driven mapping matches the expected 4-condition collapse.
    expected_mapping = {
        "101": "HighStress_HighCog_Task",
        "102": "HighStress_LowCog_Task",
        "103": "LowStress_HighCog_Task",
        "104": "LowStress_LowCog_Task",
    }
    assert task_code_to_label == expected_mapping, (
        "Task code->condition mapping mismatch. "
        f"Got {task_code_to_label}; expected {expected_mapping}. Check config/conditions.yaml export_event_labels."
    )

    global_window_idx = 0
    windows_per_block: list[int] = []
    bad_blocks: list[dict] = []
    for block_idx, tb in enumerate(task_blocks, start=1):
        condition = task_code_to_label.get(tb.norm_desc)
        assert condition is not None, f"P{pid}: unmapped task event code: {tb.norm_desc}"

        workload_class = _workload_class_for_condition(condition)
        assert workload_class in ("LowWorkload", "HighWorkload"), (
            f"P{pid}: condition did not map to workload class: {condition} -> {workload_class}"
        )

        baseline_code = assign_forest(tb.onset_samp)
        baseline_label = forest_code_to_label.get(baseline_code) if baseline_code else None

        windows, starts, stop_reason = _sliding_windows_for_block(
            data,
            onset_samp=tb.onset_samp,
            sfreq=sfreq,
            window_dur_s=WINDOW_DUR_S,
            step_dur_s=STEP_DUR_S,
            block_dur_s=BLOCK_DUR_S,
        )

        # Defensive truncation check: warn if the nominal 180s block extends past recording end.
        block_samps = int(round(BLOCK_DUR_S * sfreq))
        block_end_samp = tb.onset_samp + block_samps
        tol_samps = 2
        if block_end_samp > data.shape[1] + tol_samps:
            print(
                f"  [WARN] P{pid} block {block_idx}: nominal block_end_samp={block_end_samp} exceeds data_len={data.shape[1]} "
                f"by {block_end_samp - data.shape[1]} samples; block appears truncated."
            )

        # Per-block diagnostics.
        n_windows = len(windows)
        windows_per_block.append(n_windows)

        data_len_samp = int(data.shape[1])
        print(
            f"  Block {block_idx}: pid={pid:02d} matched_event_desc={tb.raw_desc!r} norm_desc={tb.norm_desc!r} "
            f"onset_sec={tb.onset_sec:.3f} onset_samp={tb.onset_samp} data_len_samp={data_len_samp} "
            f"block_end_samp={block_end_samp} n_windows={n_windows}"
        )

        if n_windows < expected_per_block:
            if stop_reason == "data_end":
                print("    [WARN] Windowing stopped due to recording end (data truncation).")
            elif stop_reason == "block_end":
                print("    [WARN] Windowing stopped due to block_end bound (rounding/logic issue).")
            else:
                print("    [WARN] Window count below expected; unknown stop reason.")

        # Assertion: each task block should yield exactly expected_per_block windows.
        if n_windows != expected_per_block:
            bad_blocks.append(
                {
                    "pid": pid,
                    "block_idx": block_idx,
                    "event": tb.norm_desc,
                    "condition": condition,
                    "n_windows": n_windows,
                }
            )

        for local_idx, (epoch, start_samp) in enumerate(zip(windows, starts, strict=False), start=1):
            tmin = float(start_samp / sfreq)
            tmax = float((start_samp + epoch.shape[1]) / sfreq)
            start_sec_in_block = float((start_samp - tb.onset_samp) / sfreq)

            meta = {
                "pid": pid,
                "window_idx": global_window_idx,
                "window_in_block": local_idx,
                "block_idx": block_idx,
                "event_code": tb.norm_desc,
                "event_label": condition,
                "workload_class": workload_class,
                "baseline_event_code": baseline_code if baseline_code is not None else "None",
                "baseline_id": baseline_label if baseline_label is not None else "None",
                "start_samp": int(start_samp),
                "start_sec_in_block": start_sec_in_block,
                "tmin": tmin,
                "tmax": tmax,
            }
            all_epoch_data.append(epoch)
            all_metadata.append(meta)
            global_window_idx += 1

    if not all_epoch_data:
        print(f"  No valid windows found for P{pid}")
        return

    # Enforce consistent sample counts (should already be fixed-length windows).
    n_samples = sorted({int(d.shape[1]) for d in all_epoch_data})
    if len(n_samples) != 1:
        raise AssertionError(f"P{pid}: inconsistent window sample counts: {n_samples}")
    epochs_data = np.stack(all_epoch_data, axis=0)
    
    # Create EpochsArray
    info = raw.info.copy()
    epochs = mne.EpochsArray(epochs_data, info)
    epochs.metadata = pd.DataFrame(all_metadata)
    
    # Event IDs: 1 for Low, 2 for High
    epochs.event_id = {"LowWorkload": 1, "HighWorkload": 2}
    events = np.zeros((len(all_metadata), 3), dtype=int)
    events[:, 0] = epochs.metadata["start_samp"].astype(int).to_numpy()
    events[:, 1] = 0
    events[:, 2] = [1 if m["workload_class"] == "LowWorkload" else 2 for m in all_metadata]
    epochs.events = events

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"P{pid:02d}_workload-epo.fif")
    epochs.save(out_path, overwrite=True)
    print(f"  Saved to {out_path}")

    return {
        "pid": pid,
        "skipped": False,
        "n_task_blocks": int(len(task_blocks)),
        "windows_per_block": windows_per_block,
        "total_windows": int(len(all_metadata)),
        "bad_blocks": bad_blocks,
    }

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MNE Epochs from cleaned EEGLAB .set using event markers")
    parser.add_argument(
        "--pids",
        type=str,
        default="all",
        help="Comma-separated participant IDs (e.g. 1,3) or 'all'",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable extra diagnostics",
    )
    args = parser.parse_args()

    cfg = _load_conditions_config(CONFIG_CONDITIONS)
    task_code_to_label, forest_code_to_label = _build_event_mappings(cfg)

    if args.pids.strip().lower() == "all":
        pids = list(SUBSET_PIDS)
    else:
        pids = [int(x.strip()) for x in args.pids.split(",") if x.strip() != ""]

    results: list[dict] = []
    bad_blocks_all: list[dict] = []
    failed_pids: list[int] = []
    for i, pid in enumerate(pids):
        try:
            res = export_subject_epochs(
                pid,
                task_code_to_label,
                forest_code_to_label,
                debug=bool(args.debug),
                print_ann_sample=(i == 0),
            )
            results.append(res)
            if res.get("bad_blocks"):
                bad_blocks_all.extend(res["bad_blocks"])
        except Exception as e:
            print(f"Error processing P{pid}: {e}")
            import traceback

            traceback.print_exc()
            failed_pids.append(int(pid))

    print("\n=== SUMMARY ===")
    for res in results:
        if res.get("skipped"):
            print(f"P{res.get('pid'):02d}: skipped")
            continue
        pid = int(res["pid"])
        w = res.get("windows_per_block", [])
        print(
            f"P{pid:02d}: task_blocks={int(res.get('n_task_blocks', 0))} windows_per_block={w} "
            f"total_windows={int(res.get('total_windows', 0))}"
        )

    if len(failed_pids) > 0:
        print(f"\nFailed PIDs: {failed_pids}")

    if len(bad_blocks_all) > 0:
        print("\nBlocks not yielding ~35 windows:")
        for b in bad_blocks_all:
            print(
                f"  P{int(b['pid']):02d} block={int(b['block_idx'])} event={b['event']} condition={b['condition']} "
                f"n_windows={int(b['n_windows'])}"
            )
    elif len(failed_pids) == 0:
        print("\nAll blocks yielded ~35 windows per condition.")
