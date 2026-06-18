from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = PROJECT_ROOT / "share" / "P26_package_2026-02-03"
REPORTS_DIR = PACKAGE_DIR / "reports"


@dataclass
class FileInfo:
    path: str
    exists: bool
    size_bytes: int | None = None


def file_info(path: Path) -> FileInfo:
    if not path.exists():
        return FileInfo(path=str(path), exists=False)
    st = path.stat()
    return FileInfo(path=str(path), exists=True, size_bytes=st.st_size)


def load_export_event_mapping(conditions_yaml_path: Path) -> dict[str, int]:
    with conditions_yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mapping = cfg.get("export_event_labels", {})
    if not isinstance(mapping, dict):
        raise ValueError("export_event_labels missing or not a mapping")
    # mapping: label -> code
    return {str(k): int(v) for k, v in mapping.items()}


def invert_mapping(mapping: dict[str, int]) -> dict[str, str]:
    # code -> one label (note: some labels intentionally share a code)
    inv: dict[str, str] = {}
    for label, code in mapping.items():
        k = str(code)
        inv.setdefault(k, label)
    return inv


def read_eeglab_raw(set_path: Path):
    import mne

    return mne.io.read_raw_eeglab(str(set_path), preload=False, verbose="ERROR")


def _extract_event_onsets_by_code(raw) -> dict[str, list[float]]:
    onsets: dict[str, list[float]] = {}
    if raw.annotations is None:
        return onsets

    for onset, _, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        code = str(desc)
        onsets.setdefault(code, []).append(float(onset))
    return onsets


def _get_segment(raw, onset_s: float, duration_s: float) -> np.ndarray | None:
    sfreq = float(raw.info["sfreq"])
    start = int(round(onset_s * sfreq))
    stop = int(round((onset_s + duration_s) * sfreq))
    if start < 0 or stop > raw.n_times or stop <= start:
        return None
    data = raw.get_data(start=start, stop=stop)
    return data


def compute_condition_psd(
    raw,
    event_onsets: dict[str, list[float]],
    codes: list[str],
    window_s: float = 10.0,
    max_events_per_code: int = 3,
    fmin: float = 1.0,
    fmax: float = 40.0,
):
    import mne

    sfreq = float(raw.info["sfreq"])

    freqs_out = None
    psd_by_code: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for code in codes:
        onsets = event_onsets.get(code, [])
        if not onsets:
            continue

        segments = []
        for onset_s in onsets[:max_events_per_code]:
            seg = _get_segment(raw, onset_s, window_s)
            if seg is None:
                continue
            segments.append(seg)

        if not segments:
            continue

        # Average PSD across events, then across channels
        psds = []
        for seg in segments:
            psd, freqs = mne.time_frequency.psd_array_welch(
                seg,
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                n_fft=min(seg.shape[1], 1024),
                n_overlap=0,
                verbose="ERROR",
            )
            psds.append(psd)  # (n_ch, n_freq)
        psd_mean = np.mean(np.stack(psds, axis=0), axis=0)  # (n_ch, n_freq)
        psd_mean_ch = np.mean(psd_mean, axis=0)  # (n_freq,)

        freqs_out = freqs
        psd_by_code[code] = (freqs, psd_mean_ch)

    return psd_by_code


def band_power(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return float("nan")
    # NumPy 1.24+: trapz is deprecated in favor of trapezoid.
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(psd[mask], freqs[mask]))
    return float(np.trapz(psd[mask], freqs[mask]))


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_xdf = PROJECT_ROOT / "data" / "RAW" / "eeg" / "P26.xdf"
    raw_meta = PROJECT_ROOT / "data" / "RAW" / "metadata" / "P26.csv"
    raw_set_with_chanlocs = PACKAGE_DIR / "raw" / "P26_raw_chanlocs.set"
    raw_set_original = PROJECT_ROOT / "output" / "sets" / "P26.set"

    clean_set = PROJECT_ROOT / "output" / "cleaned_eeg" / "P26_cleaned.set"
    conditions_yaml = PROJECT_ROOT / "config" / "conditions.yaml"

    manifest = {
        "files": {
            "raw_xdf": asdict(file_info(raw_xdf)),
            "raw_meta": asdict(file_info(raw_meta)),
            "raw_set_original": asdict(file_info(raw_set_original)),
            "raw_set_with_chanlocs": asdict(file_info(raw_set_with_chanlocs)),
            "clean_set": asdict(file_info(clean_set)),
        }
    }

    export_map = load_export_event_mapping(conditions_yaml)
    code_to_label = invert_mapping(export_map)

    # Prefer PSD on cleaned set (always available)
    cleaned = read_eeglab_raw(clean_set)
    manifest["cleaned"] = {
        "nchan": int(cleaned.info["nchan"]),
        "sfreq": float(cleaned.info["sfreq"]),
        "ntimes": int(cleaned.n_times),
        "duration_s": float(cleaned.n_times) / float(cleaned.info["sfreq"]),
        "n_annotations": int(len(cleaned.annotations)) if cleaned.annotations is not None else 0,
    }

    event_onsets_clean = _extract_event_onsets_by_code(cleaned)
    codes_of_interest = sorted({str(v) for v in export_map.values()})

    # Write a small event mapping + counts table for convenience
    import csv

    event_mapping_csv = REPORTS_DIR / "event_mapping.csv"
    with event_mapping_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label", "code"])
        w.writeheader()
        for label, code in sorted(export_map.items(), key=lambda kv: (int(kv[1]), str(kv[0]))):
            w.writerow({"label": label, "code": code})

    event_counts_csv = REPORTS_DIR / "event_counts_cleaned.csv"
    with event_counts_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "label", "n_events"])
        w.writeheader()
        for code in sorted(event_onsets_clean.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            w.writerow(
                {
                    "code": code,
                    "label": code_to_label.get(str(code), ""),
                    "n_events": len(event_onsets_clean.get(code, [])),
                }
            )

    psd_clean = compute_condition_psd(cleaned, event_onsets_clean, codes_of_interest)

    # Optional: try PSD on raw-with-chanlocs (if present)
    psd_raw = {}
    raw_summary = {"available": False}
    if raw_set_with_chanlocs.exists():
        try:
            raw = read_eeglab_raw(raw_set_with_chanlocs)
            raw_summary = {
                "available": True,
                "nchan": int(raw.info["nchan"]),
                "sfreq": float(raw.info["sfreq"]),
                "ntimes": int(raw.n_times),
                "duration_s": float(raw.n_times) / float(raw.info["sfreq"]),
                "n_annotations": int(len(raw.annotations)) if raw.annotations is not None else 0,
            }
            event_onsets_raw = _extract_event_onsets_by_code(raw)
            psd_raw = compute_condition_psd(raw, event_onsets_raw, codes_of_interest)
        except Exception as e:
            raw_summary = {"available": False, "error": f"{type(e).__name__}: {e}"}

    manifest["raw_with_chanlocs"] = raw_summary

    # Build a simple PSD plot for cleaned (condition-level)
    import matplotlib.pyplot as plt

    def label_for_code(code: str) -> str:
        return code_to_label.get(code, f"code_{code}")

    fig = plt.figure(figsize=(12, 7), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    for code, (freqs, psd) in sorted(psd_clean.items(), key=lambda kv: int(kv[0])):
        ax.plot(freqs, 10 * np.log10(psd + 1e-20), label=f"{code}: {label_for_code(code)}")

    ax.set_title("P26 cleaned EEG: condition-level PSD (mean across channels; first 1–3 events per code)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_xlim(1, 40)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    psd_plot_path = REPORTS_DIR / "P26_cleaned_condition_psd.png"
    fig.tight_layout()
    fig.savefig(psd_plot_path)
    plt.close(fig)

    # Tabulate a tiny “sanity” summary: theta/alpha/beta band power per code
    bands = {
        "theta_4_7": (4.0, 7.0),
        "alpha_8_12": (8.0, 12.0),
        "beta_13_30": (13.0, 30.0),
    }

    rows = []
    for code, (freqs, psd) in sorted(psd_clean.items(), key=lambda kv: int(kv[0])):
        row = {"code": code, "label": label_for_code(code)}
        for name, (lo, hi) in bands.items():
            row[name] = band_power(freqs, psd, lo, hi)
        rows.append(row)

    report = {
        "psd_plot": str(psd_plot_path),
        "band_power_summary": rows,
    }

    report_path = REPORTS_DIR / "P26_psd_sanity.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    manifest["reports"] = {
        "psd_plot": str(psd_plot_path),
        "psd_sanity_json": str(report_path),
        "event_mapping_csv": str(event_mapping_csv),
        "event_counts_cleaned_csv": str(event_counts_csv),
    }

    # Focused comparison: HighStress_HighCog task (101) vs LowStress_LowCog task (104)
    codes_compare = ["101", "104"]
    compare = {"codes": codes_compare, "labels": {c: label_for_code(c) for c in codes_compare}}

    def _band_table(psd_by_code: dict[str, tuple[np.ndarray, np.ndarray]]):
        out = {}
        for code in codes_compare:
            if code not in psd_by_code:
                continue
            freqs, psd = psd_by_code[code]
            out[code] = {
                "theta_4_7": band_power(freqs, psd, 4.0, 7.0),
                "alpha_8_12": band_power(freqs, psd, 8.0, 12.0),
                "beta_13_30": band_power(freqs, psd, 13.0, 30.0),
            }
        return out

    compare["cleaned_band_power"] = _band_table(psd_clean)
    compare["raw_band_power"] = _band_table(psd_raw) if raw_summary.get("available") else {}

    compare_plot_path = REPORTS_DIR / "P26_compare_psd_raw_vs_cleaned_101_vs_104.png"
    fig = plt.figure(figsize=(12, 7), dpi=150)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    def _plot_one(ax, code: str, title: str):
        if code in psd_clean:
            f, p = psd_clean[code]
            ax.plot(f, 10 * np.log10(p + 1e-20), label=f"cleaned ({code}: {label_for_code(code)})")
        if raw_summary.get("available") and code in psd_raw:
            f, p = psd_raw[code]
            ax.plot(f, 10 * np.log10(p + 1e-20), label=f"raw+chanlocs ({code}: {label_for_code(code)})", alpha=0.85)
        ax.set_xlim(1, 40)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    _plot_one(ax1, "101", "HighStress_HighCog task (code 101): raw vs cleaned")
    _plot_one(ax2, "104", "LowStress_LowCog task (code 104): raw vs cleaned")
    fig.tight_layout()
    fig.savefig(compare_plot_path)
    plt.close(fig)

    compare_path = REPORTS_DIR / "P26_compare_psd_raw_vs_cleaned_101_vs_104.json"
    compare_path.write_text(json.dumps(compare, indent=2), encoding="utf-8")
    manifest["reports"]["compare_plot_101_vs_104"] = str(compare_plot_path)
    manifest["reports"]["compare_json_101_vs_104"] = str(compare_path)

    manifest_path = REPORTS_DIR / "manifest_verify_psd.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {psd_plot_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {compare_plot_path}")
    print(f"Wrote: {compare_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
