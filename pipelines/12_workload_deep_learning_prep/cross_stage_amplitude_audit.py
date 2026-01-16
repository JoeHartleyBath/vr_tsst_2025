"""Compatibility wrapper.

The canonical amplitude audit script lives at:
  scripts/audits/cross_stage_amplitude_audit.py

This file remains so older references keep working.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve()
    root = here
    for p in [here] + list(here.parents):
        if (p / "scripts").exists() and (p / "results").exists() and (p / "config").exists():
            root = p
            break
    target = root / "scripts" / "audits" / "cross_stage_amplitude_audit.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
"""cross_stage_amplitude_audit.py

Court-proof cross-stage amplitude audit across:
  - Raw XDF EEG (native numeric units)
  - Raw EEGLAB .set (EEGLAB stored numeric values; typically µV)
  - Raw EEGLAB .set via MNE (Volts)
  - Cleaned EEGLAB .set (stored numeric; typically µV)
  - Cleaned EEGLAB .set via MNE (Volts)

Key properties of this audit:
  - Selects the segment by time (t0..t1 seconds) within each file.
  - Uses consistent channel mapping by name where possible.
  - Falls back to deterministic, declared assumptions when metadata is missing.
  - Reads EEGLAB .fdt directly whenever present (no MATLAB required).
  - Emits a per-participant JSON report with all key numbers + verdicts.

Ratios (as requested):
  R1 = raw_set_matlab_uV / xdf_native
       Interpret against hypotheses:
         - if XDF is µV numerically, expect R1 ≈ 1
         - if XDF is mV numerically, expect R1 ≈ 1e3  (since 1 mV = 1e3 µV)
         - if XDF is V  numerically, expect R1 ≈ 1e6  (since 1 V  = 1e6 µV)
  R2 = cleaned_set_matlab_uV / raw_set_matlab_uV
  R3 = raw_set_mne_V / raw_set_matlab_uV
       If raw_set_matlab_uV is truly µV, expect R3 ≈ 1e-6
  R4 = cleaned_set_mne_V / cleaned_set_matlab_uV
       If cleaned_set_matlab_uV is truly µV, expect R4 ≈ 1e-6
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _normalize_channel_name(name: str) -> str:
    if name is None:
        return ""
    n = str(name).strip().upper()
    for prefix in ("EEG ", "EEG_", "EEG-"):
        if n.startswith(prefix):
            n = n[len(prefix):]
    for ch in (" ", "\t", "-", "_", "."):
        n = n.replace(ch, "")
    return n


def _median_channel_std(data_ch_by_time: np.ndarray) -> float:
    if data_ch_by_time.ndim != 2:
        raise ValueError(f"Expected 2D (channels x samples), got {data_ch_by_time.shape}")
    if data_ch_by_time.shape[1] < 2:
        return float("nan")
    return float(np.median(np.std(data_ch_by_time, axis=1, ddof=0)))


def _median_channel_mad(data_ch_by_time: np.ndarray) -> float:
    if data_ch_by_time.ndim != 2:
        raise ValueError(f"Expected 2D (channels x samples), got {data_ch_by_time.shape}")
    med = np.median(data_ch_by_time, axis=1, keepdims=True)
    mad = np.median(np.abs(data_ch_by_time - med), axis=1)
    return float(np.median(mad))


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size < 3:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _pick_eeg_indices_by_name(ch_names: Sequence[str], wanted: Sequence[str]) -> Tuple[List[int], List[str]]:
    index_by_norm = {_normalize_channel_name(n): i for i, n in enumerate(ch_names)}
    picks: List[int] = []
    picked_names: List[str] = []
    for w in wanted:
        key = _normalize_channel_name(w)
        if key in index_by_norm:
            i = index_by_norm[key]
            picks.append(i)
            picked_names.append(ch_names[i])
    return picks, picked_names


def _find_workspace_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "data").exists() and (p / "output").exists() and (p / "config").exists():
            return p
    return Path.cwd().resolve()


@dataclass(frozen=True)
class SegmentWindow:
    t0: float
    t1: float

    @property
    def duration(self) -> float:
        return float(self.t1 - self.t0)


@dataclass
class StageStats:
    stage: str
    unit: str
    n_channels: int
    n_samples: int
    srate_hz: Optional[float]
    median_std: float
    median_mad: float


def _format_float(x: Optional[float]) -> str:
    if x is None:
        return "(missing)"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.10e}"


def _load_xdf_streams(xdf_path: Path) -> List[Dict[str, Any]]:
    try:
        import pyxdf
    except Exception as e:
        raise ImportError("pyxdf is required to load .xdf files") from e

    streams, _header = pyxdf.load_xdf(str(xdf_path))
    eeg_streams = [s for s in streams if s.get("info", {}).get("type", [""])[0] == "EEG"]
    if not eeg_streams:
        raise ValueError(f"No EEG streams found in XDF: {xdf_path}")
    return eeg_streams


def _extract_xdf_channel_labels(stream: Dict[str, Any]) -> Optional[List[str]]:
    try:
        channels = stream["info"]["desc"][0]["channels"][0]["channel"]
        labels = []
        for ch in channels:
            lab = None
            if "label" in ch.dtype.fields:
                lab = ch["label"][0]
            elif "labels" in ch.dtype.fields:
                lab = ch["labels"][0]
            if lab is None:
                labels.append("")
            else:
                labels.append(str(lab))
        return labels
    except Exception:
        return None


def _xdf_stream_to_eeg(stream: Dict[str, Any], *, strip_aux: bool = True) -> Tuple[np.ndarray, np.ndarray, float, Optional[List[str]], Optional[str]]:
    data = np.asarray(stream["time_series"], dtype=float)  # (samples, channels)
    ts = np.asarray(stream["time_stamps"], dtype=float)
    srate = float(stream["info"]["nominal_srate"][0])

    unit = None
    try:
        unit = stream["info"]["desc"][0]["channels"][0]["channel"][0]["unit"][0]
        unit = str(unit)
    except Exception:
        unit = None

    labels = _extract_xdf_channel_labels(stream)

    if strip_aux and data.shape[1] >= 66:
        data = data[:, :64]
        if labels is not None and len(labels) >= 64:
            labels = labels[:64]

    return data, ts, srate, labels, unit


def _align_xdf_streams(a: Tuple[np.ndarray, np.ndarray, float, Optional[List[str]], Optional[str]],
                       b: Tuple[np.ndarray, np.ndarray, float, Optional[List[str]], Optional[str]]) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    data_a, ts_a, srate_a, _labels_a, _unit_a = a
    data_b, ts_b, srate_b, _labels_b, _unit_b = b
    if abs(srate_a - srate_b) > 1e-6:
        raise ValueError(f"XDF EEG streams have different sampling rates: {srate_a} vs {srate_b}")

    start = max(ts_a[0], ts_b[0])
    end = min(ts_a[-1], ts_b[-1])
    if start >= end:
        raise ValueError("XDF EEG streams do not overlap in time")

    ia0 = int(np.searchsorted(ts_a, start))
    ia1 = int(np.searchsorted(ts_a, end))
    ib0 = int(np.searchsorted(ts_b, start))
    ib1 = int(np.searchsorted(ts_b, end))

    da = data_a[ia0:ia1, :]
    db = data_b[ib0:ib1, :]
    ta = ts_a[ia0:ia1]
    tb = ts_b[ib0:ib1]

    n = min(da.shape[0], db.shape[0])
    return (da[:n, :], ta[:n]), (db[:n, :], tb[:n])


def _build_xdf_matrix(xdf_path: Path, *, channel_mode: str, eeg_metadata_labels: Optional[List[str]]) -> Dict[str, Any]:
    eeg_streams = _load_xdf_streams(xdf_path)

    # For robustness, pick the first two EEG streams if present.
    stream_a = _xdf_stream_to_eeg(eeg_streams[0], strip_aux=True)
    stream_b = _xdf_stream_to_eeg(eeg_streams[1], strip_aux=True) if len(eeg_streams) > 1 else None

    data_a, ts_a, srate_a, labels_a, unit_a = stream_a
    xdf_out: Dict[str, Any] = {
        "srate_hz": srate_a,
        "unit_metadata": unit_a,
        "has_two_streams": stream_b is not None,
    }

    if channel_mode == "common64" or stream_b is None:
        # Use stream A only (64 channels).
        rel_t = ts_a - ts_a[0]
        ch_names = labels_a
        mapping_source = "xdf_metadata" if (ch_names and any(ch_names)) else "(missing)"
        if (not ch_names or not any(ch_names)) and eeg_metadata_labels and len(eeg_metadata_labels) >= 64:
            ch_names = eeg_metadata_labels[:64]
            mapping_source = "config_assumed_order"
        xdf_out.update(
            {
                "data_samples_by_channels": data_a,  # (samples, channels)
                "rel_time_s": rel_t,
                "ch_names": ch_names,
                "channel_mapping_source": mapping_source,
                "mode": "streamA_64",
            }
        )
        return xdf_out

    if channel_mode == "all128":
        assert stream_b is not None
        (da, ta), (db, tb) = _align_xdf_streams(stream_a, stream_b)
        merged = np.concatenate([da, db], axis=1)  # (samples, 128)
        rel_t = ta - ta[0]

        # Channel names: prefer XDF metadata if it looks usable; otherwise assume config order.
        ch_names: Optional[List[str]] = None
        mapping_source = "(missing)"
        if labels_a and any(labels_a) and stream_b[3] and any(stream_b[3]):
            ch_names = list(labels_a[:64]) + list(stream_b[3][:64])
            mapping_source = "xdf_metadata"
        elif eeg_metadata_labels and len(eeg_metadata_labels) >= 128:
            ch_names = eeg_metadata_labels[:128]
            mapping_source = "config_assumed_order"

        xdf_out.update(
            {
                "data_samples_by_channels": merged,
                "rel_time_s": rel_t,
                "ch_names": ch_names,
                "channel_mapping_source": mapping_source,
                "mode": "merged_128",
            }
        )
        return xdf_out

    raise ValueError(f"Unknown channel_mode: {channel_mode}")


def _read_eeg_metadata_channel_names(root: Path) -> Optional[List[str]]:
    meta_path = root / "config" / "eeg_metadata.yaml"
    if not meta_path.exists():
        return None
    try:
        import yaml
    except Exception:
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        ch = meta.get("channel_names")
        if isinstance(ch, list) and all(isinstance(x, str) for x in ch):
            return ch
        return None
    except Exception:
        return None


def _read_eeglab_stored_data_uV(set_path: Path) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[float], str, Any]:
    """Return EEGLAB stored numeric data (channels x samples) and assume units are µV.

    Strategy:
      1) Use MNE to parse metadata (nchan, n_times, srate, ch_names).
      2) If .fdt exists, read it directly (float32) and reshape.
      3) Else try scipy.io.loadmat for inline data (.set as MATLAB v5 MAT).
      4) If v7.3/HDF5 or loadmat fails, return None data but still return MNE raw.
    """
    raw = None
    ch_names: Optional[List[str]] = None
    n_ch: Optional[int] = None
    n_samp: Optional[int] = None
    srate: Optional[float] = None

    # 1) Try MNE first (preferred for standard EEGLAB files).
    try:
        import mne

        raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose=False)
        n_ch = int(raw.info["nchan"])
        n_samp = int(raw.n_times)
        srate = float(raw.info["sfreq"])
        ch_names = list(raw.ch_names)
    except Exception as e:
        # Non-standard .set files (e.g., scipy.savemat outputs) can break MNE.
        raw = None
        mne_error = type(e).__name__
    else:
        mne_error = None

    # 2) Try .fdt direct read whenever present.
    fdt_path = set_path.with_suffix(".fdt")
    if fdt_path.exists():
        # Need dimensions; if MNE didn't load, try loadmat to recover nbchan/pnts.
        if n_ch is None or n_samp is None or srate is None or ch_names is None:
            try:
                import scipy.io

                mat = scipy.io.loadmat(set_path, squeeze_me=True, struct_as_record=False)
                EEG = mat.get("EEG")
                if EEG is not None and not hasattr(EEG, "data"):
                    try:
                        EEG = EEG[0, 0]
                    except Exception:
                        pass
                if EEG is not None:
                    srate = float(getattr(EEG, "srate", srate if srate is not None else np.nan))
                    n_ch = int(getattr(EEG, "nbchan", n_ch if n_ch is not None else 0))
                    n_samp = int(getattr(EEG, "pnts", n_samp if n_samp is not None else 0))
                    # Chanlocs labels
                    cl = getattr(EEG, "chanlocs", None)
                    if cl is not None:
                        names: List[str] = []
                        for item in np.ravel(cl):
                            lab = None
                            if hasattr(item, "labels"):
                                lab = getattr(item, "labels")
                            elif isinstance(item, np.void) and item.dtype.names and "labels" in item.dtype.names:
                                lab = item["labels"]
                            if lab is not None:
                                names.append(str(lab))
                        if names:
                            ch_names = names
            except Exception:
                pass

        if n_ch is None or n_samp is None:
            return None, ch_names, srate, f"fdt_present_but_missing_dims(mne_error={mne_error})", raw

        data_flat = np.fromfile(fdt_path, dtype=np.float32)
        expected = int(n_ch) * int(n_samp)
        if data_flat.size != expected:
            return None, ch_names, srate, f"fdt_size_mismatch(expected={expected}, got={data_flat.size}, mne_error={mne_error})", raw
        # EEGLAB writes MATLAB arrays (column-major) to .fdt. Use Fortran order.
        data_uV = data_flat.reshape((int(n_ch), int(n_samp)), order="F")
        return data_uV, ch_names, srate, "fdt_float32", raw

    # 3) Inline .set (MATLAB v5 .mat) fallback.
    try:
        import scipy.io

        mat = scipy.io.loadmat(set_path, squeeze_me=True, struct_as_record=False)
        EEG = mat.get("EEG")
        if EEG is None:
            if "data" in mat:
                data_uV = np.asarray(mat["data"], dtype=float)
                if data_uV.ndim == 2:
                    if n_ch is None:
                        n_ch = int(data_uV.shape[0])
                    if n_samp is None:
                        n_samp = int(data_uV.shape[1])
                    return data_uV, ch_names, srate, f"loadmat_flat_data(mne_error={mne_error})", raw
            return None, ch_names, srate, f"loadmat_no_EEG(mne_error={mne_error})", raw

        if not hasattr(EEG, "data"):
            try:
                EEG = EEG[0, 0]
            except Exception:
                pass

        # Metadata
        if srate is None and hasattr(EEG, "srate"):
            srate = float(getattr(EEG, "srate"))

        # Chanlocs
        if ch_names is None:
            cl = getattr(EEG, "chanlocs", None)
            if cl is not None:
                names = []
                for item in np.ravel(cl):
                    lab = None
                    if hasattr(item, "labels"):
                        lab = getattr(item, "labels")
                    elif isinstance(item, np.void) and item.dtype.names and "labels" in item.dtype.names:
                        lab = item["labels"]
                    if lab is not None:
                        names.append(str(lab))
                if names:
                    ch_names = names

        data_field = getattr(EEG, "data", None)
        if data_field is None:
            return None, ch_names, srate, f"loadmat_EEG_no_data(mne_error={mne_error})", raw
        data_uV = np.asarray(data_field, dtype=float)
        if data_uV.ndim == 2:
            n_ch = int(data_uV.shape[0])
            n_samp = int(data_uV.shape[1])
            return data_uV, ch_names, srate, f"loadmat_EEG_data(mne_error={mne_error})", raw

        return None, ch_names, srate, f"loadmat_unexpected_shape({data_uV.shape}, mne_error={mne_error})", raw
    except Exception as e:
        return None, ch_names, srate, f"loadmat_failed({type(e).__name__}, mne_error={mne_error})", raw


def _slice_by_time_srate(n_times: int, srate_hz: float, window: SegmentWindow) -> Tuple[int, int]:
    start = int(np.round(window.t0 * srate_hz))
    stop = int(np.round(window.t1 * srate_hz))
    start = max(0, min(int(n_times), start))
    stop = max(0, min(int(n_times), stop))
    if stop <= start:
        raise ValueError(f"Invalid time window from srate: start={start}, stop={stop}, n_times={n_times}")
    return start, stop


def _slice_by_time_mne(raw: Any, window: SegmentWindow) -> Tuple[int, int]:
    start, stop = raw.time_as_index([window.t0, window.t1], use_rounding=True)
    start = int(max(0, start))
    stop = int(min(raw.n_times, stop))
    if stop <= start:
        raise ValueError(f"Invalid time window after indexing: start={start}, stop={stop}")
    return start, stop


def _slice_by_time_xdf(rel_time_s: np.ndarray, window: SegmentWindow) -> Tuple[int, int]:
    i0 = int(np.searchsorted(rel_time_s, window.t0, side="left"))
    i1 = int(np.searchsorted(rel_time_s, window.t1, side="right"))
    i0 = max(0, i0)
    i1 = min(rel_time_s.size, i1)
    if i1 <= i0:
        raise ValueError(f"Invalid time window in XDF: i0={i0}, i1={i1}")
    return i0, i1


def _interpret_r1(raw_set_uV_over_xdf_native: float) -> Dict[str, Any]:
    r1 = float(raw_set_uV_over_xdf_native)
    if not np.isfinite(r1) or r1 <= 0:
        return {"r1": r1, "best_hypothesis": None, "distance_log10": None}

    targets = {
        "xdf_is_uV": 1.0,
        "xdf_is_mV": 1e3,
        "xdf_is_V": 1e6,
    }
    log_r1 = np.log10(r1)
    best = None
    best_dist = float("inf")
    for name, target in targets.items():
        dist = abs(log_r1 - np.log10(target))
        if dist < best_dist:
            best = name
            best_dist = dist

    # Also report whether it is "close" to mV->uV (×1000) as commonly expected here.
    close_to_1e3 = abs(log_r1 - 3.0) <= np.log10(1.2)  # within ±20%
    close_to_1e6 = abs(log_r1 - 6.0) <= np.log10(1.2)
    close_to_1 = abs(log_r1 - 0.0) <= np.log10(1.2)
    return {
        "r1": r1,
        "best_hypothesis": best,
        "distance_log10": float(best_dist),
        "close_to_1": bool(close_to_1),
        "close_to_1e3": bool(close_to_1e3),
        "close_to_1e6": bool(close_to_1e6),
    }


def _compute_stage_stats(stage: str, unit: str, data_ch_by_time: np.ndarray, srate_hz: Optional[float]) -> StageStats:
    return StageStats(
        stage=stage,
        unit=unit,
        n_channels=int(data_ch_by_time.shape[0]),
        n_samples=int(data_ch_by_time.shape[1]),
        srate_hz=None if srate_hz is None else float(srate_hz),
        median_std=_median_channel_std(data_ch_by_time),
        median_mad=_median_channel_mad(data_ch_by_time),
    )


def audit_participant(
    *,
    root: Path,
    participant: str,
    window: SegmentWindow,
    channel_mode: str,
    outdir: Path,
) -> Dict[str, Any]:
    xdf_path = root / "data" / "RAW" / "eeg" / f"{participant}.xdf"
    raw_set_path = root / "output" / "sets" / f"{participant}.set"
    cleaned_set_path = root / "output" / "cleaned_eeg" / f"{participant}_cleaned.set"

    report: Dict[str, Any] = {
        "participant": participant,
        "t0_s": window.t0,
        "t1_s": window.t1,
        "duration_s": window.duration,
        "channel_mode": channel_mode,
        "paths": {
            "xdf": str(xdf_path),
            "raw_set": str(raw_set_path),
            "cleaned_set": str(cleaned_set_path),
        },
        "stages": {},
        "ratios": {},
        "sanity": {},
        "verdicts": {},
    }

    eeg_meta = _read_eeg_metadata_channel_names(root)

    # --- Stage: XDF ---
    if not xdf_path.exists():
        raise FileNotFoundError(f"Missing XDF: {xdf_path}")

    xdf = _build_xdf_matrix(xdf_path, channel_mode=channel_mode, eeg_metadata_labels=eeg_meta)
    xdf_data = np.asarray(xdf["data_samples_by_channels"], dtype=float)  # (samples, ch)
    xdf_rel_t = np.asarray(xdf["rel_time_s"], dtype=float)
    xdf_srate = float(xdf["srate_hz"])
    xdf_ch_names = xdf.get("ch_names")

    xi0, xi1 = _slice_by_time_xdf(xdf_rel_t, window)
    xdf_seg = xdf_data[xi0:xi1, :].T  # -> (ch, samples)
    xdf_stats = _compute_stage_stats("xdf", "native", xdf_seg, xdf_srate)
    report["stages"]["xdf"] = {
        **xdf_stats.__dict__,
        "xdf_mode": xdf.get("mode"),
        "xdf_unit_metadata": xdf.get("unit_metadata"),
        "channel_mapping_source": xdf.get("channel_mapping_source"),
        "time_index": {"start": xi0, "stop": xi1},
    }

    # --- Stage: raw .set ---
    raw_set: Dict[str, Any] = {"exists": raw_set_path.exists()}
    raw_uV = None
    raw_mne = None
    raw_ch_names = None
    if raw_set_path.exists():
        raw_uV, raw_ch_names, raw_srate, raw_method, raw_mne = _read_eeglab_stored_data_uV(raw_set_path)
        raw_set.update(
            {
                "stored_data_method": raw_method,
                "srate_hz": raw_srate,
                "n_channels": len(raw_ch_names) if raw_ch_names else None,
            }
        )
        try:
            if raw_mne is not None:
                rs0, rs1 = _slice_by_time_mne(raw_mne, window)
                raw_set["time_index"] = {"start": rs0, "stop": rs1}
            elif raw_uV is not None and raw_srate is not None:
                rs0, rs1 = _slice_by_time_srate(raw_uV.shape[1], float(raw_srate), window)
                raw_set["time_index"] = {"start": rs0, "stop": rs1}
        except Exception as e:
            raw_set["time_index_error"] = f"{type(e).__name__}: {e}"
    report["stages"]["raw_set"] = raw_set

    # --- Stage: cleaned .set ---
    if not cleaned_set_path.exists():
        raise FileNotFoundError(f"Missing cleaned .set: {cleaned_set_path}")
    clean_uV, clean_ch_names, clean_srate, clean_method, clean_mne = _read_eeglab_stored_data_uV(cleaned_set_path)
    if clean_mne is not None:
        cs0, cs1 = _slice_by_time_mne(clean_mne, window)
    elif clean_uV is not None and clean_srate is not None:
        cs0, cs1 = _slice_by_time_srate(clean_uV.shape[1], float(clean_srate), window)
    else:
        raise RuntimeError("Cannot slice cleaned .set by time: no MNE raw and no stored data")
    report["stages"]["cleaned_set"] = {
        "exists": True,
        "stored_data_method": clean_method,
        "srate_hz": clean_srate,
        "n_channels": len(clean_ch_names) if clean_ch_names else None,
        "time_index": {"start": cs0, "stop": cs1},
    }

    # --- Channel selection / mapping ---
    # Prefer to compare the SAME channel set across stages.
    # If we have names for both XDF and EEGLAB, match by normalized name.
    # Otherwise, fall back to index-based selection for the first N channels.
    wanted_names = None
    if xdf_ch_names and any(xdf_ch_names):
        wanted_names = xdf_ch_names
    elif eeg_meta:
        # If XDF names are missing, use the canonical label order used by the pipeline.
        wanted_names = eeg_meta[: (64 if channel_mode == "common64" else 128)]

    # Build picks for each stage
    xdf_picks = None
    if wanted_names and xdf_ch_names and any(xdf_ch_names):
        xdf_picks, _ = _pick_eeg_indices_by_name(xdf_ch_names, wanted_names)
    else:
        xdf_picks = list(range(min(xdf_seg.shape[0], 64 if channel_mode == "common64" else xdf_seg.shape[0])))

    report["sanity"]["channel_mapping"] = {
        "wanted_names_source": "xdf_metadata" if (xdf_ch_names and any(xdf_ch_names)) else ("config" if eeg_meta else "index"),
        "n_wanted": None if wanted_names is None else len(wanted_names),
    }

    # Apply XDF picks
    xdf_seg_sel = xdf_seg[np.asarray(xdf_picks, dtype=int), :]
    report["stages"]["xdf"]["selected_channels"] = {
        "count": int(xdf_seg_sel.shape[0]),
        "mode": "by_name" if (wanted_names and xdf_ch_names and any(xdf_ch_names)) else "by_index",
    }
    report["stages"]["xdf"]["median_std_selected"] = float(_median_channel_std(xdf_seg_sel))

    # Raw-set stats (stored + MNE) if available
    raw_stats_uV = None
    raw_stats_mne_V = None
    if raw_set_path.exists() and (raw_uV is not None or raw_mne is not None):
        rs0 = report["stages"]["raw_set"].get("time_index", {}).get("start")
        rs1 = report["stages"]["raw_set"].get("time_index", {}).get("stop")
        if rs0 is not None and rs1 is not None and raw_uV is not None:
            rs0 = int(rs0)
            rs1 = int(rs1)
            raw_uV_seg = raw_uV[:, rs0:rs1]

            raw_mne_source = "mne" if raw_mne is not None else "uV_to_V_assumed"
            if raw_mne is not None:
                raw_mne_data_V = raw_mne.get_data(start=rs0, stop=rs1)
                raw_ch_names_for_pick = list(raw_mne.ch_names)
            else:
                raw_mne_data_V = raw_uV_seg * 1e-6
                raw_ch_names_for_pick = list(raw_ch_names) if raw_ch_names else None

            # Channel selection: match wanted names against EEGLAB.
            if wanted_names and raw_ch_names_for_pick:
                raw_picks, raw_picked_names = _pick_eeg_indices_by_name(raw_ch_names_for_pick, wanted_names)
            else:
                raw_picks = list(range(min(raw_uV_seg.shape[0], xdf_seg_sel.shape[0])))
                raw_picked_names = [raw_ch_names_for_pick[i] for i in raw_picks] if raw_ch_names_for_pick else []

            raw_uV_sel = raw_uV_seg[np.asarray(raw_picks, dtype=int), :]
            raw_mne_sel_V = raw_mne_data_V[np.asarray(raw_picks, dtype=int), :]

            raw_stats_uV = _compute_stage_stats("raw_set_matlab", "uV", raw_uV_sel, report["stages"]["raw_set"].get("srate_hz"))
            raw_stats_mne_V = _compute_stage_stats("raw_set_mne", "V", raw_mne_sel_V, report["stages"]["raw_set"].get("srate_hz"))
            report["stages"]["raw_set"]["stats"] = {
                "matlab_uV": raw_stats_uV.__dict__,
                "mne_V": raw_stats_mne_V.__dict__,
                "mne_source": raw_mne_source,
                "mne_as_uV_median_std": float(raw_stats_mne_V.median_std * 1e6),
                "selected_channels": {"count": int(raw_uV_sel.shape[0]), "example_names": raw_picked_names[:10]},
            }

            if raw_mne is not None:
                # Sanity: true MNE(V) should match stored_uV * 1e-6 (same waveforms).
                corr = []
                for ch in range(raw_uV_sel.shape[0]):
                    corr.append(_safe_corrcoef(raw_mne_sel_V[ch], raw_uV_sel[ch] * 1e-6))
                corr = np.asarray(corr, dtype=float)
                report["sanity"]["raw_set_mne_vs_stored_corr_median"] = float(np.nanmedian(corr))

    # Cleaned stats (stored + MNE)
    clean_uV_seg = None
    if clean_uV is not None:
        clean_uV_seg = clean_uV[:, cs0:cs1]
    clean_mne_source = "mne" if clean_mne is not None else "uV_to_V_assumed"
    if clean_mne is not None:
        clean_mne_data_V = clean_mne.get_data(start=cs0, stop=cs1)
        clean_ch_names_for_pick = list(clean_mne.ch_names)
    else:
        # Best-effort proxy for MNE volts, assuming stored data are µV.
        clean_uV_seg_proxy = None if clean_uV is None else clean_uV[:, cs0:cs1]
        if clean_uV_seg_proxy is None:
            raise RuntimeError("Cannot construct cleaned volts view without MNE or stored data")
        clean_mne_data_V = clean_uV_seg_proxy * 1e-6
        clean_ch_names_for_pick = list(clean_ch_names) if clean_ch_names else None

    if wanted_names and clean_ch_names_for_pick:
        clean_picks, clean_picked_names = _pick_eeg_indices_by_name(clean_ch_names_for_pick, wanted_names)
    else:
        # Fallback to index selection; keep it symmetric with XDF selection.
        n = min(clean_mne_data_V.shape[0], xdf_seg_sel.shape[0])
        clean_picks = list(range(n))
        clean_picked_names = [clean_ch_names_for_pick[i] for i in clean_picks] if clean_ch_names_for_pick else []

    clean_mne_sel_V = clean_mne_data_V[np.asarray(clean_picks, dtype=int), :]
    clean_stats_mne_V = _compute_stage_stats("cleaned_set_mne", "V", clean_mne_sel_V, clean_srate)
    report["stages"]["cleaned_set"]["stats"] = {
        "mne_V": clean_stats_mne_V.__dict__,
        "mne_source": clean_mne_source,
        "mne_as_uV_median_std": float(clean_stats_mne_V.median_std * 1e6),
        "selected_channels": {"count": int(clean_mne_sel_V.shape[0]), "example_names": clean_picked_names[:10]},
    }

    clean_stats_uV = None
    if clean_uV_seg is not None:
        clean_uV_sel = clean_uV_seg[np.asarray(clean_picks, dtype=int), :]
        clean_stats_uV = _compute_stage_stats("cleaned_set_matlab", "uV", clean_uV_sel, clean_srate)
        report["stages"]["cleaned_set"]["stats"]["matlab_uV"] = clean_stats_uV.__dict__

        if clean_mne is not None:
            corr = []
            for ch in range(clean_uV_sel.shape[0]):
                corr.append(_safe_corrcoef(clean_mne_sel_V[ch], clean_uV_sel[ch] * 1e-6))
            corr = np.asarray(corr, dtype=float)
            report["sanity"]["cleaned_set_mne_vs_stored_corr_median"] = float(np.nanmedian(corr))

    # --- Ratios ---
    ratios: Dict[str, Any] = {}
    if raw_stats_uV is not None:
        r1 = raw_stats_uV.median_std / float(_median_channel_std(xdf_seg_sel))
        ratios["R1_rawset_uV_over_xdf_native"] = float(r1)
        ratios["R1_interpretation"] = _interpret_r1(r1)

    if raw_stats_uV is not None and clean_stats_uV is not None:
        ratios["R2_clean_uV_over_raw_uV"] = float(clean_stats_uV.median_std / raw_stats_uV.median_std)

    if raw_stats_uV is not None and raw_stats_mne_V is not None:
        ratios["R3_raw_mne_V_over_raw_uV"] = float(raw_stats_mne_V.median_std / raw_stats_uV.median_std)
        ratios["R3_source"] = report.get("stages", {}).get("raw_set", {}).get("stats", {}).get("mne_source")
        ratios["R3_expected"] = {
            "value": 1e-6,
            "reason": "If EEGLAB stored numeric values are µV, then converting to V multiplies by 1e-6. MNE returns Volts.",
        }

    if clean_stats_uV is not None:
        ratios["R4_clean_mne_V_over_clean_uV"] = float(clean_stats_mne_V.median_std / clean_stats_uV.median_std)
        ratios["R4_source"] = report.get("stages", {}).get("cleaned_set", {}).get("stats", {}).get("mne_source")
        ratios["R4_expected"] = {
            "value": 1e-6,
            "reason": "If EEGLAB stored numeric values are µV, then converting to V multiplies by 1e-6. MNE returns Volts.",
        }

    report["ratios"] = ratios

    # --- Verdicts ---
    verdicts: Dict[str, Any] = {}
    # MNE-vs-stored should be very close to 1.0 correlation if our reads + channel selection are correct.
    verdicts["cleaned_mne_vs_stored_ok"] = bool(
        report["sanity"].get("cleaned_set_mne_vs_stored_corr_median") is not None
        and report["sanity"]["cleaned_set_mne_vs_stored_corr_median"] > 0.995
    )
    if "raw_set_mne_vs_stored_corr_median" in report["sanity"]:
        verdicts["raw_mne_vs_stored_ok"] = bool(report["sanity"]["raw_set_mne_vs_stored_corr_median"] > 0.995)

    if "R3_raw_mne_V_over_raw_uV" in ratios:
        r3 = ratios["R3_raw_mne_V_over_raw_uV"]
        verdicts["R3_unit_consistency_ok"] = bool(0.9e-6 <= r3 <= 1.1e-6)
    if "R4_clean_mne_V_over_clean_uV" in ratios:
        r4 = ratios["R4_clean_mne_V_over_clean_uV"]
        verdicts["R4_unit_consistency_ok"] = bool(0.9e-6 <= r4 <= 1.1e-6)

    if "R1_interpretation" in ratios:
        verdicts["R1_best_hypothesis"] = ratios["R1_interpretation"].get("best_hypothesis")
        # In this project, the conversion intends XDF(mV) -> EEGLAB(µV), so "close_to_1e3" is the target.
        verdicts["xdf_to_set_scaling_x1000_likely"] = bool(ratios["R1_interpretation"].get("close_to_1e3"))

    report["verdicts"] = verdicts

    # --- Write JSON ---
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"amplitude_audit_{participant}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["json_path"] = str(out_path)

    return report


def _print_human_summary(report: Dict[str, Any]) -> None:
    p = report["participant"]
    print("=" * 100)
    print(f"CROSS-STAGE AMPLITUDE AUDIT: {p}  (t={report['t0_s']}-{report['t1_s']} s, mode={report['channel_mode']})")
    print("=" * 100)

    def show_stage(key: str, label: str) -> None:
        st = report["stages"].get(key, {})
        if not st.get("exists", True) and key != "xdf":
            print(f"- {label:<24} (missing)")
            return
        stats = st.get("stats")
        if key == "xdf":
            med = st.get("median_std_selected")
            print(f"- {label:<24} median_std(native)={_format_float(med)}")
            return
        if stats is None:
            print(f"- {label:<24} (no stats; stored read method={st.get('stored_data_method')})")
            return
        if "matlab_uV" in stats:
            print(f"- {label:<24} stored(median_std_uV)={_format_float(stats['matlab_uV']['median_std'])}")
        print(f"  {'':<24} mne(median_std_V)={_format_float(stats['mne_V']['median_std'])}")
        print(f"  {'':<24} mne(median_std_uV)={_format_float(stats['mne_as_uV_median_std'])}")

    show_stage("xdf", "XDF")
    show_stage("raw_set", "Raw .set")
    show_stage("cleaned_set", "Cleaned .set")

    print("\nRatios (requested):")
    ratios = report.get("ratios", {})
    if "R1_rawset_uV_over_xdf_native" in ratios:
        r1 = ratios["R1_rawset_uV_over_xdf_native"]
        interp = ratios.get("R1_interpretation", {})
        print(f"- R1 raw_set_uV / xdf_native = {_format_float(r1)}")
        print("  Expectations: if XDF is µV -> ~1; if mV -> ~1e3; if V -> ~1e6")
        print(f"  Best hypothesis: {interp.get('best_hypothesis')} (log10 distance={interp.get('distance_log10')})")
    else:
        print("- R1: (skipped; raw .set missing or unreadable)")

    if "R2_clean_uV_over_raw_uV" in ratios:
        print(f"- R2 cleaned_uV / raw_uV = {_format_float(ratios['R2_clean_uV_over_raw_uV'])}")
    else:
        print("- R2: (skipped; need stored raw+clean data)")

    if "R3_raw_mne_V_over_raw_uV" in ratios:
        print(f"- R3 raw_mne_V / raw_uV = {_format_float(ratios['R3_raw_mne_V_over_raw_uV'])} (expect ~1e-6)")
    else:
        print("- R3: (skipped; need stored raw data)")

    if "R4_clean_mne_V_over_clean_uV" in ratios:
        print(f"- R4 clean_mne_V / clean_uV = {_format_float(ratios['R4_clean_mne_V_over_clean_uV'])} (expect ~1e-6)")
    else:
        print("- R4: (skipped; need stored clean data)")

    print("\nSanity checks:")
    sanity = report.get("sanity", {})
    if "raw_set_mne_vs_stored_corr_median" in sanity:
        print(f"- Raw MNE vs stored corr (median) = {sanity['raw_set_mne_vs_stored_corr_median']:.6f}")
    print(f"- Cleaned MNE vs stored corr (median) = {sanity.get('cleaned_set_mne_vs_stored_corr_median', float('nan')):.6f}")

    print("\nVerdicts:")
    for k, v in report.get("verdicts", {}).items():
        print(f"- {k}: {v}")
    print(f"\nJSON: {report.get('json_path')}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-stage amplitude audit (XDF → raw .set → cleaned .set)")
    parser.add_argument("--participants", nargs="+", required=True, help="Participant IDs like P01 P03 P10")
    parser.add_argument("--t0", type=float, required=True, help="Window start time in seconds from recording start")
    parser.add_argument("--t1", type=float, required=True, help="Window end time in seconds from recording start")
    parser.add_argument("--channels", choices=["common64", "all128"], default="all128", help="Channel mode")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for JSON reports")
    args = parser.parse_args(argv)

    if args.t1 <= args.t0:
        raise SystemExit("--t1 must be > --t0")

    here = Path(__file__).resolve()
    root = _find_workspace_root(here)
    outdir = Path(args.outdir) if args.outdir else (root / "tmp" / "amplitude_audit")

    window = SegmentWindow(t0=float(args.t0), t1=float(args.t1))

    all_reports = []
    for p in args.participants:
        rep = audit_participant(root=root, participant=p, window=window, channel_mode=args.channels, outdir=outdir)
        _print_human_summary(rep)
        all_reports.append(rep)

    # Also write an index file for convenience.
    index_path = outdir / "amplitude_audit_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"participants": [r["participant"] for r in all_reports], "reports": [r.get("json_path") for r in all_reports]}, f, indent=2)

    print("\nDone.")
    print(f"Index JSON: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
