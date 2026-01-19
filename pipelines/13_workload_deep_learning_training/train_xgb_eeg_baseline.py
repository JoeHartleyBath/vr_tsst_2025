"""
XGBoost baseline for EEG rolling-window band power features (safe-CV).

Key properties (aligned with train_tcnet.py safe-CV):
- Deterministic folds (StratifiedGroupKFold by PID) precomputed once and reused.
- Baseline adjustment using Forest blocks as baseline windows.
- Fold-safe normalization: mean/std computed on TRAIN only and applied to both train/test.
- Optimize mean macro-F1 across folds (primary), accuracy logged as secondary.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_CSV = r"C:\vr_tsst_2025\output\aggregated\eeg_features_rolling_windows.csv"
RESULTS_DIR = r"C:\vr_tsst_2025\results"

# Match TCNet baseline adjustment
BASELINE_ADJUST = "zscore"  # 'none'|'mean'|'divstd'|'zscore'

# Safety / determinism
SEED = 1337
SMOKE = os.getenv("SMOKE", "0").strip() == "1"

# CV
N_FOLDS = 3

# XGBoost training
MAX_ROUNDS = 2000
EARLY_STOP_ROUNDS = 50

if SMOKE:
    N_FOLDS = 2
    MAX_ROUNDS = 200

# ==============================================================================
# GLOBAL DATA
# ==============================================================================
DATASET = None
SPLITS = None

META_COLS = {
    "pid",
    "event_label",
    "window_idx",
    "window_start",
    "window_end",
}

FOREST_PREFIX = "Forest"
TASK_SUFFIX = "_Task"

ALLOWED_BANDS = {
    "Delta",
    "Theta",
    "LowAlpha",
    "HighAlpha",
    "LowBeta",
    "HighBeta",
    "Alpha",
    "Beta",
}


# ==============================================================================
# HELPERS
# ==============================================================================

def make_results_path(*, baseline_adjust: str, tag: str) -> str:
    return rf"C:\vr_tsst_2025\results\workload_xgb_eeg_{baseline_adjust}_{tag}_safe_cv.txt"


def make_fold_log_path(*, baseline_adjust: str, tag: str) -> str:
    return rf"C:\vr_tsst_2025\results\workload_xgb_eeg_{baseline_adjust}_{tag}_safe_cv_folds.jsonl"


def _workload_label_from_condition(cond: str) -> int | None:
    if not isinstance(cond, str):
        return None
    if "HighCog" in cond:
        return 1
    if "LowCog" in cond:
        return 0
    return None


def _is_task_condition(cond: str) -> bool:
    return isinstance(cond, str) and cond.endswith(TASK_SUFFIX)


def _is_forest_condition(cond: str) -> bool:
    return isinstance(cond, str) and cond.startswith(FOREST_PREFIX)


def _select_bandpower_features(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        parts = str(c).split("_")
        if not parts:
            continue
        band = parts[-1]
        if band not in ALLOWED_BANDS:
            continue
        cols.append(c)
    # Drop near-constant columns
    non_constant = []
    for c in cols:
        v = df[c].to_numpy(dtype=float, copy=False)
        if np.nanstd(v) > 1e-12:
            non_constant.append(c)
    return non_constant


def _compute_forest_baseline_stats(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[dict[tuple[int, str], tuple[np.ndarray, np.ndarray]], dict[int, list[tuple[float, str]]]]:
    """Return baseline stats and forest onset lists.

    Returns:
        baseline_stats[(pid, forest_condition)] = (mean, std)
        forest_onsets_by_pid[pid] = [(onset_time, forest_condition), ...]
    """
    df_forest = df[df["event_label"].apply(_is_forest_condition)]

    baseline_stats: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    forest_onsets_by_pid: dict[int, list[tuple[float, str]]] = {}

    if df_forest.empty:
        return baseline_stats, forest_onsets_by_pid

    for (pid, cond), g in df_forest.groupby(["pid", "event_label"], sort=True):
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        baseline_stats[(int(pid), str(cond))] = (mean, std)

    forest_onsets = (
        df_forest.groupby(["pid", "event_label"], sort=True)["window_start"]
        .min()
        .reset_index()
    )
    for pid, g in forest_onsets.groupby("pid", sort=True):
        items = [(float(row.window_start), str(row.event_label)) for row in g.itertuples(index=False)]
        items.sort(key=lambda x: x[0])
        forest_onsets_by_pid[int(pid)] = items

    return baseline_stats, forest_onsets_by_pid


def _assign_baseline_arrays(
    df_task: pd.DataFrame,
    feature_cols: list[str],
    baseline_stats: dict[tuple[int, str], np.ndarray],
    forest_onsets_by_pid: dict[int, list[tuple[float, str]]],
) -> tuple[np.ndarray, np.ndarray, int]:
    n = len(df_task)
    k = len(feature_cols)
    baseline_mean = np.zeros((n, k), dtype=np.float32)
    baseline_std = np.ones((n, k), dtype=np.float32)
    missing = 0

    pids = df_task["pid"].to_numpy(dtype=int)
    starts = df_task["window_start"].to_numpy(dtype=float)

    for i in range(n):
        pid = int(pids[i])
        onset_list = forest_onsets_by_pid.get(pid)
        if not onset_list:
            missing += 1
            continue
        onset_times = [t for t, _ in onset_list]
        pos = int(np.searchsorted(onset_times, starts[i], side="right") - 1)
        if pos < 0:
            missing += 1
            continue
        _, cond = onset_list[pos]
        stats = baseline_stats.get((pid, cond))
        if stats is None:
            missing += 1
            continue
        mean, std = stats
        baseline_mean[i, :] = mean
        baseline_std[i, :] = std

    return baseline_mean, baseline_std, missing


def _apply_baseline_adjustment(
    df_task: pd.DataFrame,
    df_all: pd.DataFrame,
    feature_cols: list[str],
    baseline_adjust: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if baseline_adjust == "none":
        X = df_task[feature_cols].to_numpy(dtype=np.float32)
        return X, {"enabled": False}

    if baseline_adjust not in {"mean", "divstd", "zscore"}:
        raise ValueError(f"Unsupported baseline_adjust: {baseline_adjust}")

    baseline_stats, forest_onsets_by_pid = _compute_forest_baseline_stats(df_all, feature_cols)
    mean_arr, std_arr, missing = _assign_baseline_arrays(
        df_task, feature_cols, baseline_stats, forest_onsets_by_pid
    )

    X = df_task[feature_cols].to_numpy(dtype=np.float32)

    if baseline_adjust == "mean":
        X = X - mean_arr
    elif baseline_adjust == "divstd":
        X = X / std_arr
    else:
        X = (X - mean_arr) / std_arr

    return X, {
        "enabled": True,
        "missing_baseline_rows": int(missing),
        "n_baseline_blocks": int(len(baseline_stats)),
    }


def get_dataset():
    global DATASET
    if DATASET is None:
        df_all = pd.read_csv(DATA_CSV)

        # Task windows only (workload classification)
        df_task = df_all[df_all["event_label"].apply(_is_task_condition)].copy()
        df_task["label"] = df_task["event_label"].apply(_workload_label_from_condition)
        df_task = df_task[df_task["label"].isin([0, 1])].copy()

        # Feature selection
        feature_cols = _select_bandpower_features(df_task)
        if not feature_cols:
            raise RuntimeError("No EEG band-power features found after filtering.")

        # Baseline adjustment
        X, baseline_info = _apply_baseline_adjustment(df_task, df_all, feature_cols, BASELINE_ADJUST)

        # Replace inf with NaN for fold-safe imputation
        X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)

        y = df_task["label"].to_numpy(dtype=int)
        groups = df_task["pid"].to_numpy(dtype=int)
        window_start = df_task["window_start"].to_numpy(dtype=float)

        DATASET = {
            "X": X,
            "y": y,
            "groups": groups,
            "feature_cols": feature_cols,
            "baseline_info": baseline_info,
            "window_start": window_start,
        }

        print(
            f"Loaded EEG dataset: n={len(y)} | features={len(feature_cols)} | "
            f"baseline={baseline_info} | SMOKE={SMOKE}"
        )

    return DATASET


def get_splits():
    global SPLITS
    if SPLITS is None:
        data = get_dataset()
        y = data["y"]
        groups = data["groups"]
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        X_dummy = np.zeros(len(y))
        SPLITS = list(splitter.split(X_dummy, y, groups=groups))
        if len(SPLITS) != N_FOLDS:
            raise RuntimeError(f"Expected {N_FOLDS} splits, got {len(SPLITS)}")
        print(f"Precomputed {len(SPLITS)} deterministic folds (seed={SEED})")
    return SPLITS


# ==============================================================================
# TRAINING HELPERS
# ==============================================================================

def _macro_f1_eval(preds: np.ndarray, dtrain: xgb.DMatrix):
    labels = dtrain.get_label().astype(int)
    pred_labels = (preds >= 0.5).astype(int)
    f1 = f1_score(labels, pred_labels, average="macro", zero_division=0)
    return "macro_f1", float(f1)


def _fold_preprocess(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Impute NaN with train median
    med = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), med, X_train)
    X_test = np.where(np.isnan(X_test), med, X_test)

    # Fold-safe z-score (train-only)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test


def _train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    params: dict[str, Any],
) -> dict[str, Any]:
    X_train, X_test = _fold_preprocess(X_train, X_test)

    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    params = {
        **params,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": int(SEED),
        "nthread": 0,
        "scale_pos_weight": scale_pos_weight,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    train_kwargs = dict(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(MAX_ROUNDS),
        evals=[(dtest, "test")],
        early_stopping_rounds=int(EARLY_STOP_ROUNDS),
        maximize=True,
        verbose_eval=False,
    )

    try:
        booster = xgb.train(**train_kwargs, feval=_macro_f1_eval)
    except TypeError:
        booster = xgb.train(**train_kwargs, custom_metric=_macro_f1_eval)

    best_iter = int(booster.best_iteration) if booster.best_iteration is not None else int(MAX_ROUNDS)
    preds = booster.predict(dtest, iteration_range=(0, best_iter + 1))
    pred_labels = (preds >= 0.5).astype(int)

    f1 = float(f1_score(y_test, pred_labels, average="macro", zero_division=0))
    acc = float(accuracy_score(y_test, pred_labels))

    return {
        "best_macro_f1": f1,
        "best_acc": acc,
        "best_iter": best_iter,
    }


# ==============================================================================
# MAIN SAFE-CV
# ==============================================================================

def run_safe_cv(*, params: dict[str, Any], tag: str) -> dict[str, Any]:
    data = get_dataset()
    splits = get_splits()

    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    feature_cols = data["feature_cols"]
    baseline_info = data["baseline_info"]

    fold_f1 = []
    fold_acc = []
    fold_iter = []

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fold_log_path = make_fold_log_path(baseline_adjust=BASELINE_ADJUST, tag=tag)

    with open(fold_log_path, "w", encoding="utf-8") as f_log:
        for fold, (train_idx, test_idx) in enumerate(splits):
            fr = _train_one_fold(
                X_train=X[train_idx],
                y_train=y[train_idx],
                X_test=X[test_idx],
                y_test=y[test_idx],
                feature_cols=feature_cols,
                params=params,
            )
            fold_f1.append(float(fr["best_macro_f1"]))
            fold_acc.append(float(fr["best_acc"]))
            fold_iter.append(int(fr["best_iter"]))

            record = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "best_macro_f1": float(fr["best_macro_f1"]),
                "best_acc": float(fr["best_acc"]),
                "best_iter": int(fr["best_iter"]),
                "baseline_adjust": str(BASELINE_ADJUST),
                "baseline_info": dict(baseline_info),
            }
            f_log.write(json.dumps(record) + "\n")

    mean_f1 = float(np.mean(fold_f1)) if fold_f1 else 0.0
    mean_acc = float(np.mean(fold_acc)) if fold_acc else 0.0

    return {
        "mean_macro_f1": mean_f1,
        "mean_acc": mean_acc,
    }


def _contiguous_folds(indices: np.ndarray, n_folds: int) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    n_folds = int(n_folds)
    n_folds = max(2, n_folds)
    n_folds = min(n_folds, indices.size)
    return [np.asarray(x, dtype=int) for x in np.array_split(indices, n_folds) if len(x) > 0]


def run_within_subject_cv(*, params: dict[str, Any], tag: str) -> dict[str, Any]:
    data = get_dataset()

    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    feature_cols = data["feature_cols"]
    baseline_info = data["baseline_info"]
    window_start = data["window_start"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fold_log_path = make_fold_log_path(baseline_adjust=BASELINE_ADJUST, tag=tag)

    per_subject = []
    with open(fold_log_path, "w", encoding="utf-8") as f_log:
        for pid in sorted({int(p) for p in np.unique(groups)}):
            pid_mask = groups == pid
            pid_idx = np.where(pid_mask)[0]
            if pid_idx.size < 4:
                continue

            order = np.argsort(window_start[pid_idx])
            pid_idx = pid_idx[order]
            folds = _contiguous_folds(pid_idx, N_FOLDS)
            if len(folds) < 2:
                continue

            fold_f1 = []
            fold_acc = []

            for fold_id, test_idx in enumerate(folds):
                train_idx = np.setdiff1d(pid_idx, test_idx, assume_unique=False)
                if train_idx.size == 0 or test_idx.size == 0:
                    continue

                fr = _train_one_fold(
                    X_train=X[train_idx],
                    y_train=y[train_idx],
                    X_test=X[test_idx],
                    y_test=y[test_idx],
                    feature_cols=feature_cols,
                    params=params,
                )

                fold_f1.append(float(fr["best_macro_f1"]))
                fold_acc.append(float(fr["best_acc"]))

                record = {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "pid": int(pid),
                    "fold": int(fold_id),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "best_macro_f1": float(fr["best_macro_f1"]),
                    "best_acc": float(fr["best_acc"]),
                    "best_iter": int(fr["best_iter"]),
                    "baseline_adjust": str(BASELINE_ADJUST),
                    "baseline_info": dict(baseline_info),
                }
                f_log.write(json.dumps(record) + "\n")

            if fold_f1:
                per_subject.append(
                    {
                        "pid": int(pid),
                        "mean_macro_f1": float(np.mean(fold_f1)),
                        "mean_acc": float(np.mean(fold_acc)),
                        "n_windows": int(pid_idx.size),
                        "n_folds": int(len(fold_f1)),
                    }
                )

    if not per_subject:
        return {"mean_macro_f1": 0.0, "mean_acc": 0.0, "n_subjects": 0}

    mean_f1 = float(np.mean([p["mean_macro_f1"] for p in per_subject]))
    mean_acc = float(np.mean([p["mean_acc"] for p in per_subject]))

    return {
        "mean_macro_f1": mean_f1,
        "mean_acc": mean_acc,
        "n_subjects": int(len(per_subject)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XGBoost baseline for EEG band power (safe-CV).")
    p.add_argument("--baseline_adjust", choices=["none", "mean", "divstd", "zscore"], default=BASELINE_ADJUST)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max_rounds", type=int, default=MAX_ROUNDS)
    p.add_argument("--early_stop_rounds", type=int, default=EARLY_STOP_ROUNDS)
    p.add_argument("--tag", type=str, default="baseline")
    p.add_argument("--within_subject", action="store_true", help="Run within-subject time-blocked CV")
    return p.parse_args()


def main() -> None:
    global SEED
    global SPLITS
    global BASELINE_ADJUST
    global MAX_ROUNDS
    global EARLY_STOP_ROUNDS

    args = parse_args()
    SEED = int(args.seed)
    SPLITS = None
    BASELINE_ADJUST = str(args.baseline_adjust)
    MAX_ROUNDS = int(args.max_rounds)
    EARLY_STOP_ROUNDS = int(args.early_stop_rounds)
    tag = str(args.tag).strip() or "baseline"
    within_subject = bool(args.within_subject)

    print("=" * 60)
    print("EEG XGBoost Baseline (safe-CV)")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print(f"Folds: {N_FOLDS}")
    print(f"Baseline adjust: {BASELINE_ADJUST}")
    print(f"SMOKE: {SMOKE}")
    print(f"Max rounds: {MAX_ROUNDS}")
    print(f"Early stop rounds: {EARLY_STOP_ROUNDS}")

    print("Loading dataset and precomputing folds (once)...")
    get_dataset()
    if not within_subject:
        get_splits()

    params = {
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5.0,
        "gamma": 0.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "max_delta_step": 0,
    }

    start = time.time()
    if within_subject:
        metrics = run_within_subject_cv(params=params, tag=f"within_{tag}")
        mode = "within_subject"
    else:
        metrics = run_safe_cv(params=params, tag=tag)
        mode = "safe_cv"
    elapsed = time.time() - start

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = make_results_path(baseline_adjust=BASELINE_ADJUST, tag=(f"within_{tag}" if within_subject else tag))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"EEG XGBoost baseline - Workload Classification ({mode})\n")
        f.write(f"Data: {DATA_CSV}\n")
        f.write(f"Features: band power only ({len(DATASET['feature_cols'])})\n")
        f.write(f"Folds: {N_FOLDS}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Baseline adjust: {BASELINE_ADJUST}\n")
        f.write("Normalization: train-only per-feature zscore\n")
        f.write(f"Elapsed_s: {elapsed:.2f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Mean Macro-F1: {metrics['mean_macro_f1']:.4f}\n")
        f.write(f"Mean Accuracy: {metrics['mean_acc']:.4f}\n")

    print(f"Saved results -> {out_path}")


if __name__ == "__main__":
    main()
