from __future__ import annotations

import csv
import shutil
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_RAW_SET = PROJECT_ROOT / "share" / "P26_package_2026-02-03" / "raw" / "P26_raw_chanlocs.set"
SRC_RAW_FDT = PROJECT_ROOT / "share" / "P26_package_2026-02-03" / "raw" / "P26_raw_chanlocs.fdt"
SRC_CLEAN_SET = PROJECT_ROOT / "output" / "cleaned_eeg" / "P26_cleaned.set"
SRC_CLEAN_FDT = PROJECT_ROOT / "output" / "cleaned_eeg" / "P26_cleaned.fdt"

PKG_ROOT = PROJECT_ROOT / "share" / "P26_minimal_package_2026-02-03"
PKG_RAW = PKG_ROOT / "raw"
PKG_CLEAN = PKG_ROOT / "cleaned"
PKG_LABELS = PKG_ROOT / "labels"

CONDITIONS_YAML = PROJECT_ROOT / "config" / "conditions.yaml"


def copy_no_overwrite(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    if dst.exists():
        raise FileExistsError(f"Destination already exists (refusing to overwrite): {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_export_event_labels(yaml_path: Path) -> dict[str, int]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    mapping = data.get("export_event_labels", {})
    if not isinstance(mapping, dict):
        raise ValueError("export_event_labels missing or not a mapping")
    return {str(k): int(v) for k, v in mapping.items()}


def description_for_label(label: str) -> str:
    # Keep this intentionally simple and human-readable.
    if label == "Pre_Exposure_Blank_Fixation_Cross":
        return "Start of pre-exposure fixation cross (blank)."
    if label == "Pre_Exposure_Room_Fixation_Cross":
        return "Start of pre-exposure fixation cross (in-room)."
    if label == "Post_Exposure_Blank_Fixation_Cross":
        return "Start of post-exposure fixation cross (blank)."
    if label == "Post_Exposure_Room_Fixation_Cross":
        return "Start of post-exposure fixation cross (in-room)."

    if label in {"Forest1", "Forest2", "Forest3", "Forest4"}:
        return "Start of forest scene (baseline context)."

    if label.startswith("HighStress_") and label.endswith("_Task"):
        return "Onset of high-stress arithmetic task block (task period)."
    if label.startswith("LowStress_") and label.endswith("_Task"):
        return "Onset of low-stress arithmetic task block (task period)."

    if label == "HighStress_LowCog_Task":
        return "Onset of high-stress, low-cognitive-load task block (addition)."
    if label == "LowStress_LowCog_Task":
        return "Onset of low-stress, low-cognitive-load task block (addition)."

    if label == "Response_Correct":
        return "Correct response marker (one per response event)."
    if label == "Response_Incorrect":
        return "Incorrect response marker (one per response event)."

    return "Event marker (see label name)."


def write_event_labels(export_map: dict[str, int]) -> None:
    # CSV
    csv_path = PKG_LABELS / "event_labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "label", "description"])
        w.writeheader()
        for label, code in sorted(export_map.items(), key=lambda kv: (int(kv[1]), str(kv[0]))):
            w.writerow({"code": code, "label": label, "description": description_for_label(label)})

    # Markdown (for quick reading)
    md_path = PKG_LABELS / "event_labels.md"
    lines = [
        "# Event Labels (P26)",
        "", 
        "EEG event markers are stored in the EEGLAB `.set` files as annotations whose `description` is the numeric code.",
        "", 
        "## Mapping", 
        "", 
        "| Code | Label | Description |",
        "|---:|---|---|",
    ]
    for label, code in sorted(export_map.items(), key=lambda kv: (int(kv[1]), str(kv[0]))):
        desc = description_for_label(label)
        lines.append(f"| {code} | {label} | {desc} |")

    lines += [
        "",
        "## Notes",
        "- Some task labels intentionally share the same code (e.g., `...1022_Task` and `...2043_Task`).",
        "- If you need to disambiguate those, you must use task metadata (not included in this minimal package).",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme() -> None:
    readme = PKG_ROOT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# P26 Minimal Share Package",
                "",
                "This package contains only what’s needed to test an affective-state framework:",
                "- Raw EEG (EEGLAB): `raw/`",
                "- Cleaned EEG (EEGLAB): `cleaned/`",
                "- Event label mapping + descriptions: `labels/`",
                "",
                "## Files",
                "- Raw: `raw/P26_raw.set` + `raw/P26_raw.fdt`",
                "- Cleaned: `cleaned/P26_cleaned.set` + `cleaned/P26_cleaned.fdt`",
                "- Labels: `labels/event_labels.csv` and `labels/event_labels.md`",
                "",
                "## How to load",
                "### EEGLAB",
                "Open the `.set` file; EEGLAB will automatically read the paired `.fdt`.",
                "",
                "### Python (MNE)",
                "Example:",
                "- `mne.io.read_raw_eeglab('P26_cleaned.set', preload=False)`",
                "",
                "## Event codes",
                "The annotation `description` is a numeric code (string). Use the mapping in `labels/`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    PKG_RAW.mkdir(parents=True, exist_ok=True)
    PKG_CLEAN.mkdir(parents=True, exist_ok=True)
    PKG_LABELS.mkdir(parents=True, exist_ok=True)

    # Copy raw EEG (use the raw-with-events EEGLAB copy we created earlier)
    copy_no_overwrite(SRC_RAW_SET, PKG_RAW / "P26_raw.set")
    copy_no_overwrite(SRC_RAW_FDT, PKG_RAW / "P26_raw.fdt")

    # Copy cleaned EEG
    copy_no_overwrite(SRC_CLEAN_SET, PKG_CLEAN / "P26_cleaned.set")
    copy_no_overwrite(SRC_CLEAN_FDT, PKG_CLEAN / "P26_cleaned.fdt")

    export_map = load_export_event_labels(CONDITIONS_YAML)
    write_event_labels(export_map)
    write_readme()

    print(f"Built minimal package at: {PKG_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
