import argparse
import json
import re
from pathlib import Path

QC_RE_BAD_CHANNELS = re.compile(r"QC: (\d+) bad channels", re.IGNORECASE)
QC_RE_ASR = re.compile(r"QC: .*?(\d+\.?\d*)% ASR-repaired", re.IGNORECASE)
QC_RE_ICS_REMOVED = re.compile(r"QC: .*?(\d+) ICs removed", re.IGNORECASE)
BAD_CHANNELS_LIST_RE = re.compile(r"bad channels identified:\s*(.+)$", re.IGNORECASE)
STATS_RE = re.compile(r"Stats (?:after|post-subcomp|after reref|at save|after ASR|after AMICA): min=([^\s]+) max=([^\s]+) mean=([^\s]+) std=([^\s]+)", re.IGNORECASE)
LL_TRACE_RE = re.compile(r"iter\s+(\d+)\s+->\s+LL\s*=\s*([\d\.\-]+)")


def parse_processing_log(text: str):
    qc = {
        "bad_channels_count": None,
        "bad_channels": [],
        "asr_repaired_percent": None,
        "ics_removed": None,
        "stats": {},
        "amica_ll_trace": [],
    }

    # First pass: QC line
    for line in text.splitlines():
        m = QC_RE_BAD_CHANNELS.search(line)
        if m:
            qc["bad_channels_count"] = int(m.group(1))
        m = QC_RE_ASR.search(line)
        if m:
            qc["asr_repaired_percent"] = float(m.group(1))
        m = QC_RE_ICS_REMOVED.search(line)
        if m:
            qc["ics_removed"] = int(m.group(1))
        m = BAD_CHANNELS_LIST_RE.search(line)
        if m:
            # split on commas and clean
            chans = [c.strip() for c in m.group(1).split(',') if c.strip()]
            qc["bad_channels"] = chans
        m = STATS_RE.search(line)
        if m:
            # label by first token after 'Stats '
            # e.g., "after ASR" or "after reref" or "at save"
            label_match = re.search(r"Stats\s+([^:]+):", line)
            label = label_match.group(1).strip() if label_match else "unknown"
            qc["stats"][label] = {
                "min": float(m.group(1)),
                "max": float(m.group(2)),
                "mean": float(m.group(3)),
                "std": float(m.group(4)),
            }
        m = LL_TRACE_RE.search(line)
        if m:
            qc["amica_ll_trace"].append({"iter": int(m.group(1)), "ll": float(m.group(2))})

    return qc


def main():
    parser = argparse.ArgumentParser(description="Emit QC JSON from processing log")
    parser.add_argument("log_path", type=Path, help="Path to Pxx_processing_log.txt")
    parser.add_argument("out_path", type=Path, nargs="?", help="Output QC JSON path (default: output/cleaned_eeg/qc/Pxx_qc.json)")
    args = parser.parse_args()

    log_text = args.log_path.read_text(encoding="utf-8", errors="ignore")
    qc = parse_processing_log(log_text)

    # default output location next to cleaned_eeg/qc
    if not args.out_path:
        pid = args.log_path.stem.split("_")[0]  # e.g., P01_processing_log -> P01
        out_dir = Path("output/cleaned_eeg/qc")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pid}_qc.json"
    else:
        out_path = args.out_path

    out_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(f"QC JSON written: {out_path}")


if __name__ == "__main__":
    main()
