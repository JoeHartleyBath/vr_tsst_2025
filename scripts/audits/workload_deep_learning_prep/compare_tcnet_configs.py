"""Compare TCNet tuning artifacts vs retrain configuration.

This is a lightweight, read-only diagnostic that helps catch the most common
reasons TCNet retraining under-performs relative to Optuna tuning logs:
- Using a different tuned-params JSON than the one referenced in Stage2 tuning
- Baseline adjustment mismatch (Stage2 assumes zscore)
- Early stopping / max-epochs mismatch
- Dataset drift (missing participant epoch files)

It does NOT train a model; it just reports config/protocol deltas.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_py_constant(text: str, name: str) -> str | None:
    # Simple regex: NAME = <literal>
    m = re.search(rf"^\s*{re.escape(name)}\s*=\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip()


def _kind(d: dict[str, Any] | None) -> str:
    if not d:
        return "<missing>"
    if "best_value_macro_f1" in d and "best_arch" in d and "best_params" in d:
        return "stage2_overall_summary"
    if "best_accuracy" in d and "best_params" in d and "best_arch" in d:
        return "focused_with_arch"
    if "best_accuracy" in d and "best_params" in d:
        return "focused_params"
    return "unknown"


def _fmt(d: dict[str, Any] | None, k: str) -> str:
    if not d:
        return "<missing>"
    v = d.get(k)
    return "<missing>" if v is None else str(v)


def main() -> int:
    p = argparse.ArgumentParser(description="Compare TCNet tuning artifacts vs retrain config")
    p.add_argument(
        "--stage2_summary",
        default=str(REPO_ROOT / "results" / "stage2_overall_summary.json"),
        help="Path to Stage2 summary JSON (default: results/stage2_overall_summary.json)",
    )
    p.add_argument(
        "--focused_best",
        default=str(REPO_ROOT / "results" / "best_tcnet_focused_params.json"),
        help="Path to focused tuner best JSON (default: results/best_tcnet_focused_params.json)",
    )
    p.add_argument(
        "--stage2_doc",
        default=str(REPO_ROOT / "docs" / "TCNET_STAGE2_TUNING_RESULTS_2026-01-18.md"),
        help="Path to Stage2 results markdown (default: docs/TCNET_STAGE2_TUNING_RESULTS_2026-01-18.md)",
    )
    args = p.parse_args()

    stage2_path = Path(args.stage2_summary)
    focused_path = Path(args.focused_best)
    doc_path = Path(args.stage2_doc)

    stage2 = _load_json_if_exists(stage2_path)
    focused = _load_json_if_exists(focused_path)

    print("=" * 80)
    print("TCNet tuning artifacts")
    print("=" * 80)
    print(f"Stage2 summary: {stage2_path} ({_kind(stage2)})")
    if stage2:
        print(f"  study_name: {_fmt(stage2, 'study_name')}")
        print(f"  best_value_macro_f1: {_fmt(stage2, 'best_value_macro_f1')}")
        print(f"  seed: {_fmt(stage2, 'seed')}")
    else:
        print("  [NOTE] Not found. That file is generated (gitignored) in many setups.")

    print(f"Focused best:  {focused_path} ({_kind(focused)})")
    if focused:
        print(f"  best_accuracy: {_fmt(focused, 'best_accuracy')}")
    else:
        print("  [NOTE] Not found.")

    print("\n" + "=" * 80)
    print("Key differences that commonly explain worse retrain performance")
    print("=" * 80)

    # Stage2 doc (source of truth when JSON missing)
    if doc_path.exists():
        doc = _read_text(doc_path)
        # Pull the Stage2 best hyperparams from the markdown if possible
        # (only for quick human readability; not meant as a perfect parser)
        m = re.search(r"## Best hyperparameters \(Stage 2\)(.+?)(?:\n## |\Z)", doc, flags=re.DOTALL)
        if m:
            print("Stage2 doc lists best hyperparameters:")
            for line in m.group(1).splitlines():
                line = line.strip()
                if line.startswith("-"):
                    print(f"  {line}")
        else:
            print("[NOTE] Could not extract Stage2 best hyperparams block from doc.")
    else:
        print(f"[NOTE] Stage2 doc missing: {doc_path}")

    # Compare Stage2 vs focused if we have both JSONs
    if stage2 and focused:
        s2p = stage2.get("best_params") or {}
        fop = focused.get("best_params") or {}
        s2a = stage2.get("best_arch") or {}
        foa = focused.get("best_arch") or {}

        keys = [
            "dropout_eeg",
            "dropout_tcn",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "label_smoothing",
        ]
        arch_keys = ["F1", "D", "kernLength", "tcn_filters", "tcn_kernel", "tcn_depth"]

        print("\nJSON diff (Stage2 vs Focused):")
        for k in arch_keys:
            if s2a.get(k) != foa.get(k):
                print(f"  arch.{k}: stage2={s2a.get(k)} vs focused={foa.get(k)}")
        for k in keys:
            if s2p.get(k) != fop.get(k):
                print(f"  best_params.{k}: stage2={s2p.get(k)} vs focused={fop.get(k)}")

    print("\nBaseline adjustment: Stage2 tuning in tune_tcnet.py applies baseline_adjust='zscore' by design.")
    print("If retrain used baseline_adjust='none', expect a large drop vs Stage2 logs.")

    # Extract constants from scripts to highlight protocol mismatches
    tune_tcnet_py = REPO_ROOT / "pipelines" / "13_workload_deep_learning_training" / "tune_tcnet.py"
    train_tcnet_py = REPO_ROOT / "pipelines" / "13_workload_deep_learning_training" / "train_tcnet.py"
    focused_py = REPO_ROOT / "pipelines" / "12_workload_deep_learning_prep" / "tune_tcnet_focused.py"

    if tune_tcnet_py.exists() and train_tcnet_py.exists():
        ttxt = _read_text(tune_tcnet_py)
        rtxt = _read_text(train_tcnet_py)
        print("\nScript constants (helps spot protocol drift):")
        for name in ["TIME_POINTS", "SEED", "EARLY_STOP_PATIENCE", "MAX_EPOCHS", "BASELINE_ADJUST"]:
            tv = _extract_py_constant(ttxt, name)
            rv = _extract_py_constant(rtxt, name)
            if tv is not None or rv is not None:
                print(f"  {name}: tune_tcnet.py={tv} | train_tcnet.py={rv}")

    if focused_py.exists():
        ftxt = _read_text(focused_py)
        print("\nFocused tuner constants (often NOT comparable to Stage2 safe-CV):")
        for name in ["TIME_POINTS", "EARLY_STOP_PATIENCE", "MAX_EPOCHS"]:
            fv = _extract_py_constant(ftxt, name)
            if fv is not None:
                print(f"  {name}: tune_tcnet_focused.py={fv}")

    print("\nRecommended reproduction command (Stage2 parity):")
    print(
        "  python pipelines/13_workload_deep_learning_training/train_tcnet.py "
        "--use_tuned_params --tuned_params_path results/stage2_overall_summary.json "
        "--baseline_adjust zscore"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
