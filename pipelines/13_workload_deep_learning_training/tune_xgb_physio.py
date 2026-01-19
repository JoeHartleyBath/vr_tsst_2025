"""Staged Optuna tuning for physio-only XGBoost workload classifier (safe-CV).

Key properties (aligned with tune_tcnet.py safe-CV):
- Deterministic folds (StratifiedGroupKFold by PID) precomputed once and reused.
- Baseline adjustment (z-score) using Forest blocks as baseline windows.
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
import optuna
import pandas as pd
import xgboost as xgb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_CSV = r"C:\vr_tsst_2025\output\aggregated\physio_features_rolling_windows.csv"
RESULTS_DIR = r"C:\vr_tsst_2025\results"
DEFAULT_DB_FILE = os.path.join(RESULTS_DIR, "optuna_physio_xgb_staged.db")
DEFAULT_STUDY_NAME_PREFIX = "physio_xgb_staged_tuning"

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
    "Participant_ID",
    "Condition",
    "Window_Index",
    "Window_Start",
    "Window_End",
}

FOREST_PREFIX = "Forest"
TASK_SUFFIX = "_Task"


# ==============================================================================
# DATA HELPERS
# ==============================================================================

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


def _select_autonomic_features(df: pd.DataFrame) -> list[str]:
    allowed_prefixes = (
        "Polar_HeartRate_",
        "Shimmer_D36A_GSR_",
        "Foveal_",
        "Inter_Blink_",
        "Current_Blink_",
        "Full_Pupil_",
    )

    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if "LowVar_Flag" in c:
            continue
        if not any(c.startswith(pfx) for pfx in allowed_prefixes):
            continue
        cols.append(c)

    # Keep only numeric columns
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    # Drop near-constant columns
    non_constant = []
    for c in numeric_cols:
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
    df_forest = df[df["Condition"].apply(_is_forest_condition)]

    baseline_stats: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    forest_onsets_by_pid: dict[int, list[tuple[float, str]]] = {}

    if df_forest.empty:
        return baseline_stats, forest_onsets_by_pid

    for (pid, cond), g in df_forest.groupby(["Participant_ID", "Condition"], sort=True):
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        baseline_stats[(int(pid), str(cond))] = (mean, std)

    forest_onsets = (
        df_forest.groupby(["Participant_ID", "Condition"], sort=True)["Window_Start"]
        .min()
        .reset_index()
    )
    for pid, g in forest_onsets.groupby("Participant_ID", sort=True):
        items = [(float(row.Window_Start), str(row.Condition)) for row in g.itertuples(index=False)]
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

    pids = df_task["Participant_ID"].to_numpy(dtype=int)
    starts = df_task["Window_Start"].to_numpy(dtype=float)

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
        df_task = df_all[df_all["Condition"].apply(_is_task_condition)].copy()
        df_task["label"] = df_task["Condition"].apply(_workload_label_from_condition)
        df_task = df_task[df_task["label"].isin([0, 1])].copy()

        # Feature selection
        feature_cols = _select_autonomic_features(df_task)
        if not feature_cols:
            raise RuntimeError("No autonomic features found after filtering.")

        # Baseline adjustment (z-score)
        X, baseline_info = _apply_baseline_adjustment(df_task, df_all, feature_cols, BASELINE_ADJUST)

        # Replace inf with NaN for fold-safe imputation
        X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)

        y = df_task["label"].to_numpy(dtype=int)
        groups = df_task["Participant_ID"].to_numpy(dtype=int)

        DATASET = {
            "X": X,
            "y": y,
            "groups": groups,
            "feature_cols": feature_cols,
            "baseline_info": baseline_info,
        }

        print(
            f"Loaded physio dataset: n={len(y)} | features={len(feature_cols)} | "
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
# OPTUNA OBJECTIVES
# ==============================================================================

def _sample_stage1_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
    }


def _bounded(value: float, lo: float, hi: float) -> tuple[float, float]:
    lo = float(lo)
    hi = float(hi)
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, 1e-8), max(hi, lo + 1e-8)


def _sample_stage2_params(trial: optuna.Trial, center: dict[str, Any]) -> dict[str, Any]:
    eta_c = float(center.get("eta", 0.1))
    eta_lo, eta_hi = _bounded(eta_c * 0.5, 0.005, min(0.5, eta_c * 1.5))

    md_c = int(center.get("max_depth", 6))
    md_lo = max(3, md_c - 2)
    md_hi = min(12, md_c + 2)

    subs_c = float(center.get("subsample", 0.8))
    subs_lo, subs_hi = _bounded(subs_c * 0.8, 0.5, min(1.0, subs_c * 1.2))

    col_c = float(center.get("colsample_bytree", 0.8))
    col_lo, col_hi = _bounded(col_c * 0.8, 0.5, min(1.0, col_c * 1.2))

    mcw_c = float(center.get("min_child_weight", 5.0))
    mcw_lo, mcw_hi = _bounded(mcw_c * 0.7, 0.5, mcw_c * 1.3)

    gamma_c = float(center.get("gamma", 0.0))
    gamma_lo, gamma_hi = _bounded(max(0.0, gamma_c - 1.0), 0.0, gamma_c + 1.0)

    lam_c = float(center.get("lambda", 1.0))
    lam_lo, lam_hi = _bounded(lam_c * 0.5, 1e-4, lam_c * 1.5)

    alpha_c = float(center.get("alpha", 1.0))
    alpha_lo, alpha_hi = _bounded(alpha_c * 0.5, 1e-4, alpha_c * 1.5)

    mds_c = int(center.get("max_delta_step", 0))
    mds_lo = max(0, mds_c - 2)
    mds_hi = min(10, mds_c + 2)

    return {
        "eta": trial.suggest_float("eta", eta_lo, eta_hi, log=True),
        "max_depth": trial.suggest_int("max_depth", md_lo, md_hi),
        "subsample": trial.suggest_float("subsample", subs_lo, subs_hi),
        "colsample_bytree": trial.suggest_float("colsample_bytree", col_lo, col_hi),
        "min_child_weight": trial.suggest_float("min_child_weight", mcw_lo, mcw_hi),
        "gamma": trial.suggest_float("gamma", gamma_lo, gamma_hi),
        "lambda": trial.suggest_float("lambda", lam_lo, lam_hi, log=True),
        "alpha": trial.suggest_float("alpha", alpha_lo, alpha_hi, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", mds_lo, mds_hi),
    }


def _run_cv(trial: optuna.Trial, params: dict[str, Any]) -> float:
    data = get_dataset()
    splits = get_splits()

    X = data["X"]
    y = data["y"]
    feature_cols = data["feature_cols"]

    fold_f1 = []
    fold_acc = []
    fold_iter = []

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

        if fold == 0:
            trial.report(float(fr["best_macro_f1"]), step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

    trial.set_user_attr("fold_best_f1", fold_f1)
    trial.set_user_attr("fold_best_acc", fold_acc)
    trial.set_user_attr("fold_best_iter", fold_iter)

    mean_f1 = float(np.mean(fold_f1)) if fold_f1 else 0.0
    mean_acc = float(np.mean(fold_acc)) if fold_acc else 0.0
    trial.set_user_attr("mean_macro_f1", mean_f1)
    trial.set_user_attr("mean_acc", mean_acc)

    return mean_f1


def objective_stage1(trial: optuna.Trial) -> float:
    trial.set_user_attr("stage", "stage1")
    params = _sample_stage1_params(trial)
    return _run_cv(trial, params)


def objective_stage2(trial: optuna.Trial, center: dict[str, Any]) -> float:
    trial.set_user_attr("stage", "stage2")
    params = _sample_stage2_params(trial, center)
    return _run_cv(trial, params)


# ==============================================================================
# OPTUNA ORCHESTRATION
# ==============================================================================

def _normalize_storage(db_path: str) -> str:
    s = str(db_path).strip()
    if s.startswith("sqlite:///"):
        return s
    p = os.path.abspath(s)
    p_norm = p.replace("\\", "/")
    return f"sqlite:///{p_norm}"


def _select_top_k(study: optuna.Study, top_k: int) -> list[optuna.trial.FrozenTrial]:
    completed = [t for t in study.trials if t.value is not None and str(t.state) == "TrialState.COMPLETE"]
    completed.sort(key=lambda t: float(t.value), reverse=True)
    return completed[: int(top_k)]


def _center_from_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "eta": float(trial.params.get("eta", 0.1)),
        "max_depth": int(trial.params.get("max_depth", 6)),
        "subsample": float(trial.params.get("subsample", 0.8)),
        "colsample_bytree": float(trial.params.get("colsample_bytree", 0.8)),
        "min_child_weight": float(trial.params.get("min_child_weight", 5.0)),
        "gamma": float(trial.params.get("gamma", 0.0)),
        "lambda": float(trial.params.get("lambda", 1.0)),
        "alpha": float(trial.params.get("alpha", 1.0)),
        "max_delta_step": int(trial.params.get("max_delta_step", 0)),
    }


def save_stage_artifacts(
    *,
    stage: str,
    study: optuna.Study,
    prefix: str,
    elapsed_s: float,
    top_k: int,
    center: dict[str, Any] | None = None,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    top_trials = _select_top_k(study, top_k)
    top_rows = []
    for t in top_trials:
        row = {
            "trial_number": int(t.number),
            "value_macro_f1": float(t.value) if t.value is not None else None,
            "params": {k: t.params.get(k) for k in sorted(t.params.keys())},
        }
        top_rows.append(row)

    data = get_dataset()
    summary = {
        "stage": stage,
        "study_name": str(study.study_name),
        "storage": str(study._storage),
        "best_value_macro_f1": float(study.best_value) if study.best_value is not None else None,
        "best_params": dict(study.best_params) if study.best_params is not None else None,
        "center": dict(center) if center is not None else None,
        "n_trials": int(len(study.trials)),
        "elapsed_s": float(elapsed_s),
        "folds": int(N_FOLDS),
        "smoke": bool(SMOKE),
        "baseline_adjust": str(BASELINE_ADJUST),
        "n_samples": int(len(data["y"])),
        "n_features": int(len(data["feature_cols"])),
        "top_k": int(top_k),
        "top_k_trials": top_rows,
    }

    out_path = os.path.join(RESULTS_DIR, f"{prefix}_{stage}_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {stage} summary -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staged Optuna tuning for physio-only XGBoost (safe-CV, macro-F1).")
    p.add_argument("--stage", choices=["1", "2", "both"], default="both")
    p.add_argument("--stage1_trials", type=int, default=100)
    p.add_argument("--stage2_trials", type=int, default=50)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--db_path", type=str, default=DEFAULT_DB_FILE)
    p.add_argument("--study_name_prefix", type=str, default=DEFAULT_STUDY_NAME_PREFIX)
    p.add_argument("--stage1_hours", type=float, default=0.0)
    p.add_argument("--stage2_hours", type=float, default=0.0)
    p.add_argument("--early_stop_rounds", type=int, default=50)
    return p.parse_args()


def run_stage1(*, storage: str, prefix: str, n_trials: int, top_k: int, seed: int, stage1_hours: float) -> optuna.Study:
    sampler = TPESampler(seed=int(seed), n_startup_trials=15)
    pruner = MedianPruner(n_startup_trials=15, n_warmup_steps=10, interval_steps=5)

    study_name = f"{prefix}_stage1"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    existing_trials = int(len(study.trials))
    remaining_trials = max(0, int(n_trials) - existing_trials)
    print(f"Stage1 study '{study_name}': existing_trials={existing_trials} | target_trials={int(n_trials)} | running_additional={remaining_trials}")

    trials_jsonl_path = os.path.join(RESULTS_DIR, f"optuna_trials_xgb_{prefix}_stage1.jsonl")

    class TrialJSONLLogger:
        def __init__(self, path: str):
            self.path = path

        def __call__(self, study, trial):
            record = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "study_name": str(study.study_name),
                "trial_number": int(trial.number),
                "state": str(trial.state),
                "value_macro_f1": float(trial.value) if trial.value is not None else None,
                "params": dict(trial.params),
                "stage": trial.user_attrs.get("stage", None),
                "fold_best_f1": trial.user_attrs.get("fold_best_f1", None),
                "fold_best_acc": trial.user_attrs.get("fold_best_acc", None),
                "fold_best_iter": trial.user_attrs.get("fold_best_iter", None),
            }
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    start = time.time()
    timeout_s = None
    if stage1_hours is not None and float(stage1_hours) > 0:
        timeout_s = float(stage1_hours) * 3600.0

    if remaining_trials > 0:
        study.optimize(
            objective_stage1,
            n_trials=int(remaining_trials),
            timeout=timeout_s,
            callbacks=[TrialJSONLLogger(trials_jsonl_path)],
            show_progress_bar=True,
            gc_after_trial=True,
        )
    else:
        print("Stage1: target already met; skipping optimize().")

    elapsed = time.time() - start
    center = _center_from_trial(study.best_trial) if study.best_trial is not None else None
    save_stage_artifacts(stage="stage1", study=study, prefix=prefix, elapsed_s=elapsed, top_k=top_k, center=center)
    print(f"Stage1 JSONL log -> {trials_jsonl_path}")
    return study


def run_stage2(*, storage: str, prefix: str, n_trials: int, top_k: int, seed: int, center: dict[str, Any], stage2_hours: float) -> optuna.Study:
    sampler = TPESampler(seed=int(seed), n_startup_trials=15)
    pruner = MedianPruner(n_startup_trials=15, n_warmup_steps=10, interval_steps=5)

    study_name = f"{prefix}_stage2"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    existing_trials = int(len(study.trials))
    remaining_trials = max(0, int(n_trials) - existing_trials)
    print(f"Stage2 study '{study_name}': existing_trials={existing_trials} | target_trials={int(n_trials)} | running_additional={remaining_trials}")

    trials_jsonl_path = os.path.join(RESULTS_DIR, f"optuna_trials_xgb_{prefix}_stage2.jsonl")

    class TrialJSONLLogger:
        def __init__(self, path: str):
            self.path = path

        def __call__(self, study, trial):
            record = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "study_name": str(study.study_name),
                "trial_number": int(trial.number),
                "state": str(trial.state),
                "value_macro_f1": float(trial.value) if trial.value is not None else None,
                "params": dict(trial.params),
                "stage": trial.user_attrs.get("stage", None),
                "fold_best_f1": trial.user_attrs.get("fold_best_f1", None),
                "fold_best_acc": trial.user_attrs.get("fold_best_acc", None),
                "fold_best_iter": trial.user_attrs.get("fold_best_iter", None),
            }
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def obj(trial: optuna.Trial) -> float:
        return objective_stage2(trial, center=center)

    start = time.time()
    timeout_s = None
    if stage2_hours is not None and float(stage2_hours) > 0:
        timeout_s = float(stage2_hours) * 3600.0

    if remaining_trials > 0:
        study.optimize(
            obj,
            n_trials=int(remaining_trials),
            timeout=timeout_s,
            callbacks=[TrialJSONLLogger(trials_jsonl_path)],
            show_progress_bar=True,
            gc_after_trial=True,
        )
    else:
        print("Stage2: target already met; skipping optimize().")

    elapsed = time.time() - start
    save_stage_artifacts(stage="stage2", study=study, prefix=prefix, elapsed_s=elapsed, top_k=top_k, center=center)
    print(f"Stage2 JSONL log -> {trials_jsonl_path}")
    return study


def main() -> None:
    global SEED
    global SPLITS
    global EARLY_STOP_ROUNDS

    args = parse_args()
    SEED = int(args.seed)
    SPLITS = None
    EARLY_STOP_ROUNDS = int(args.early_stop_rounds)

    storage = _normalize_storage(args.db_path)
    prefix = str(args.study_name_prefix).strip()
    stage = str(args.stage)

    print("=" * 60)
    print("Physio XGBoost Staged Optuna Tuning (safe-CV)")
    print("=" * 60)
    print(f"Stage: {stage}")
    print(f"Seed: {SEED}")
    print(f"Folds: {N_FOLDS}")
    print(f"Baseline adjust: {BASELINE_ADJUST}")
    print(f"Storage: {storage}")
    print(f"Study prefix: {prefix}")
    print(f"SMOKE: {SMOKE}")
    print(f"Early stop rounds: {EARLY_STOP_ROUNDS}")

    print("Loading dataset and precomputing folds (once)...")
    get_dataset()
    get_splits()

    stage1_study = None
    if stage in {"1", "both"}:
        print("\n--- STAGE 1 ---")
        stage1_study = run_stage1(
            storage=storage,
            prefix=prefix,
            n_trials=int(args.stage1_trials),
            top_k=int(args.top_k),
            seed=SEED,
            stage1_hours=float(args.stage1_hours),
        )

    if stage in {"2", "both"}:
        print("\n--- STAGE 2 ---")
        if stage1_study is None:
            stage1_study = optuna.load_study(study_name=f"{prefix}_stage1", storage=storage)
        if stage1_study.best_trial is None:
            raise RuntimeError("No completed Stage 1 trials found; cannot run Stage 2")

        center = _center_from_trial(stage1_study.best_trial)
        _ = run_stage2(
            storage=storage,
            prefix=prefix,
            n_trials=int(args.stage2_trials),
            top_k=int(args.top_k),
            seed=SEED,
            center=center,
            stage2_hours=float(args.stage2_hours),
        )


if __name__ == "__main__":
    main()
