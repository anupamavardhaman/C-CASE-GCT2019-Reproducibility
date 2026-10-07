#!/usr/bin/env python3
"""
A1 Ablation: Stacked Ensemble without Clustering
=================================================

A1:
    Linear Regression + Random Forest + XGBoost
                    |
                    v
              XGBoost Meta Learner

Cluster context is NOT used.

The script:
- uses development data only;
- uses the same three outer temporal folds as A0/C-CASE;
- creates leakage-free temporal inner OOF predictions;
- trains separate CPU and memory base learners;
- trains separate CPU and memory XGBoost meta learners;
- refits base learners on each complete outer-training period;
- evaluates on the corresponding outer-validation period;
- reports MSE, RMSE, MAE and R²;
- reports mean ± SD across the three outer folds;
- saves final outer-fold predictions for reproducibility.

Run from the repository root:
    python src/ablation/ablation_2_stacking_no_clustering.py

Required checkpoint:
    checkpoints/ccase_step15.pkl

Generate it first with:
    python src/preprocess_data.py
"""

from pathlib import Path
import gc

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor


# ============================================================================
# REPOSITORY PATHS
# ============================================================================

# <repository_root>/src/ablation/ablation_2_stacking_no_clustering.py
ROOT_DIR = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "ccase_step15.pkl"
OUTPUT_DIR = ROOT_DIR / "results" / "ablation"


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
N_INNER_SPLITS = 3

# Exact same outer temporal folds used by A0 and the reported ablation.
OUTER_FOLDS = [
    {
        "name": "Fold1",
        "train_end": 658500000000,
        "val_start": 659100000000,
        "val_end": 1314900000000,
    },
    {
        "name": "Fold2",
        "train_end": 1314600000000,
        "val_start": 1315200000000,
        "val_end": 1971000000000,
    },
    {
        "name": "Fold3",
        "train_end": 1970700000000,
        "val_start": 1971300000000,
        "val_end": 2627100000000,
    },
]

RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": 2,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 150,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": RANDOM_STATE,
    "objective": "reg:squarederror",
}

META_XGB_PARAMS = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": RANDOM_STATE,
    "objective": "reg:squarederror",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate MSE, RMSE, MAE and R²."""
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def create_inner_temporal_folds(outer_train_time, n_splits=3):
    """
    Create temporal inner folds from the outer-training period.

    This preserves the temporal-fold construction used in the A1 notebook:
    timestamps are sorted and successive validation blocks are formed from
    later timestamps.
    """
    unique_times = np.sort(outer_train_time.unique())
    n_times = len(unique_times)

    if n_times < n_splits + 1:
        raise ValueError(
            f"Not enough unique timestamps for {n_splits} inner folds. "
            f"Found {n_times}."
        )

    fold_size = n_times // (n_splits + 1)

    if fold_size == 0:
        raise ValueError("Inner temporal fold size is zero.")

    folds = []

    for i in range(n_splits):
        train_end_idx = fold_size * (i + 1)
        val_start_idx = train_end_idx

        if i == n_splits - 1:
            val_end_idx = n_times
        else:
            val_end_idx = fold_size * (i + 2)

        inner_train_times = unique_times[:train_end_idx]
        inner_val_times = unique_times[val_start_idx:val_end_idx]

        if len(inner_train_times) == 0:
            raise RuntimeError(
                f"Inner fold {i + 1} has no training timestamps."
            )

        if len(inner_val_times) == 0:
            raise RuntimeError(
                f"Inner fold {i + 1} has no validation timestamps."
            )

        if inner_train_times.max() >= inner_val_times.min():
            raise RuntimeError(
                f"Temporal leakage detected in inner fold {i + 1}."
            )

        folds.append((inner_train_times, inner_val_times))

    return folds


def validate_outer_fold(time_all, fold):
    """Create and validate an outer temporal split."""
    outer_train_mask = time_all <= fold["train_end"]

    outer_val_mask = (
        (time_all >= fold["val_start"])
        & (time_all <= fold["val_end"])
    )

    if not outer_train_mask.any():
        raise RuntimeError(
            f"{fold['name']}: outer training set is empty."
        )

    if not outer_val_mask.any():
        raise RuntimeError(
            f"{fold['name']}: outer validation set is empty."
        )

    train_max = time_all.loc[outer_train_mask].max()
    val_min = time_all.loc[outer_val_mask].min()

    if train_max >= val_min:
        raise RuntimeError(
            f"{fold['name']}: temporal leakage detected. "
            f"Training maximum time ({train_max}) is not earlier than "
            f"validation minimum time ({val_min})."
        )

    return outer_train_mask, outer_val_mask


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 90)
    print("A1 ABLATION — STACKED ENSEMBLE WITHOUT CLUSTERING")
    print("LR + RF + XGB -> XGB META")
    print("=" * 90)

    # ------------------------------------------------------------------------
    # STEP 1 — LOAD CHECKPOINT
    # ------------------------------------------------------------------------
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "\nRequired preprocessing checkpoint was not found.\n\n"
            f"Expected location:\n{CHECKPOINT_PATH}\n\n"
            "Run the preprocessing step first:\n\n"
            "    python src/preprocess_data.py\n\n"
            "Expected generated file:\n"
            "    checkpoints/ccase_step15.pkl\n"
        )

    print("\n" + "=" * 90)
    print("STEP 1 — LOAD CHECKPOINT")
    print("=" * 90)
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    checkpoint = joblib.load(CHECKPOINT_PATH)

    required_keys = [
        "model_data",
        "train_data",
        "feature_cols",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in checkpoint
    ]

    if missing_keys:
        raise KeyError(
            "Checkpoint is missing required keys: "
            + ", ".join(missing_keys)
        )

    model_data = checkpoint["model_data"]
    train_data = checkpoint["train_data"].copy()
    feature_cols = checkpoint["feature_cols"]

    required_columns = (
        list(feature_cols)
        + ["future_cpu", "future_memory", "time_bucket"]
    )

    missing_columns = [
        col for col in required_columns
        if col not in train_data.columns
    ]

    if missing_columns:
        raise KeyError(
            "train_data is missing required columns: "
            + ", ".join(missing_columns)
        )

    print("Checkpoint loaded successfully.")
    print(
        "Model data shape:",
        getattr(model_data, "shape", "not available"),
    )
    print("Development data shape:", train_data.shape)
    print("Number of features:", len(feature_cols))

    # ------------------------------------------------------------------------
    # STEP 2 — DEVELOPMENT DATA ONLY
    # ------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("STEP 2 — PREPARE DEVELOPMENT DATA")
    print("=" * 90)

    X_all = train_data[feature_cols]
    y_cpu_all = train_data["future_cpu"]
    y_memory_all = train_data["future_memory"]
    time_all = train_data["time_bucket"]

    if X_all.isnull().any().any():
        raise ValueError(
            "Feature matrix contains missing values. "
            "Check preprocessing before running A1."
        )

    if y_cpu_all.isnull().any():
        raise ValueError("CPU target contains missing values.")

    if y_memory_all.isnull().any():
        raise ValueError("Memory target contains missing values.")

    if time_all.isnull().any():
        raise ValueError("time_bucket contains missing values.")

    print("X shape:", X_all.shape)
    print("CPU target shape:", y_cpu_all.shape)
    print("Memory target shape:", y_memory_all.shape)

    a1_results = []
    prediction_records = []

    # ------------------------------------------------------------------------
    # STEP 3 — OUTER FOLD LOOP
    # ------------------------------------------------------------------------
    for fold in OUTER_FOLDS:
        fold_name = fold["name"]

        print("\n" + "=" * 90)
        print(f"{fold_name} — A1 LR + RF + XGB -> XGB META")
        print("NO CLUSTER CONTEXT")
        print("=" * 90)

        # ================================================================
        # 3.1 OUTER TRAIN / VALIDATION SPLIT
        # ================================================================
        outer_train_mask, outer_val_mask = validate_outer_fold(
            time_all,
            fold,
        )

        X_outer_train = X_all.loc[outer_train_mask]
        X_outer_val = X_all.loc[outer_val_mask]

        y_cpu_outer_train = y_cpu_all.loc[outer_train_mask]
        y_cpu_outer_val = y_cpu_all.loc[outer_val_mask]

        y_memory_outer_train = y_memory_all.loc[outer_train_mask]
        y_memory_outer_val = y_memory_all.loc[outer_val_mask]

        time_outer_train = time_all.loc[outer_train_mask]
        time_outer_val = time_all.loc[outer_val_mask]

        print("Outer training samples :", len(X_outer_train))
        print("Outer validation samples:", len(X_outer_val))
        print(
            "Training time range:",
            time_outer_train.min(),
            "->",
            time_outer_train.max(),
        )
        print(
            "Validation time range:",
            time_outer_val.min(),
            "->",
            time_outer_val.max(),
        )

        # ================================================================
        # 3.2 INNER TEMPORAL FOLDS
        # ================================================================
        inner_folds = create_inner_temporal_folds(
            time_outer_train,
            n_splits=N_INNER_SPLITS,
        )

        # OOF predictions are stored against original DataFrame indices.
        oof = pd.DataFrame(index=X_outer_train.index)

        for column in [
            "cpu_lr",
            "cpu_rf",
            "cpu_xgb",
            "memory_lr",
            "memory_rf",
            "memory_xgb",
        ]:
            oof[column] = np.nan

        # ================================================================
        # 3.3 INNER OOF LOOP
        # ================================================================
        for inner_no, (
            inner_train_times,
            inner_val_times,
        ) in enumerate(inner_folds, start=1):

            print("\n" + "-" * 75)
            print(f"{fold_name} — Inner Fold {inner_no}")
            print("-" * 75)

            inner_train_mask = time_outer_train.isin(
                inner_train_times
            )
            inner_val_mask = time_outer_train.isin(
                inner_val_times
            )

            X_inner_train = X_outer_train.loc[inner_train_mask]
            X_inner_val = X_outer_train.loc[inner_val_mask]

            if X_inner_train.empty or X_inner_val.empty:
                raise RuntimeError(
                    f"{fold_name}, inner fold {inner_no}: "
                    "empty training or validation data."
                )

            inner_train_max = time_outer_train.loc[
                inner_train_mask
            ].max()
            inner_val_min = time_outer_train.loc[
                inner_val_mask
            ].min()

            if inner_train_max >= inner_val_min:
                raise RuntimeError(
                    f"{fold_name}, inner fold {inner_no}: "
                    "temporal leakage detected."
                )

            y_cpu_inner_train = y_cpu_outer_train.loc[
                X_inner_train.index
            ]
            y_memory_inner_train = y_memory_outer_train.loc[
                X_inner_train.index
            ]

            print("Inner train:", len(X_inner_train))
            print("Inner validation:", len(X_inner_val))

            # ------------------------------------------------------------
            # CPU BASE LEARNERS
            # ------------------------------------------------------------
            print("Training CPU base learners...")

            cpu_lr = LinearRegression()
            cpu_rf = RandomForestRegressor(**RF_PARAMS)
            cpu_xgb = XGBRegressor(**XGB_PARAMS)

            cpu_lr.fit(
                X_inner_train,
                y_cpu_inner_train,
            )
            cpu_rf.fit(
                X_inner_train,
                y_cpu_inner_train,
            )
            cpu_xgb.fit(
                X_inner_train,
                y_cpu_inner_train,
            )

            cpu_lr_pred = cpu_lr.predict(X_inner_val)
            cpu_rf_pred = cpu_rf.predict(X_inner_val)
            cpu_xgb_pred = cpu_xgb.predict(X_inner_val)

            # ------------------------------------------------------------
            # MEMORY BASE LEARNERS
            # ------------------------------------------------------------
            print("Training Memory base learners...")

            memory_lr = LinearRegression()
            memory_rf = RandomForestRegressor(**RF_PARAMS)
            memory_xgb = XGBRegressor(**XGB_PARAMS)

            memory_lr.fit(
                X_inner_train,
                y_memory_inner_train,
            )
            memory_rf.fit(
                X_inner_train,
                y_memory_inner_train,
            )
            memory_xgb.fit(
                X_inner_train,
                y_memory_inner_train,
            )

            memory_lr_pred = memory_lr.predict(X_inner_val)
            memory_rf_pred = memory_rf.predict(X_inner_val)
            memory_xgb_pred = memory_xgb.predict(X_inner_val)

            # ------------------------------------------------------------
            # STORE OOF PREDICTIONS BY ORIGINAL INDEX
            # ------------------------------------------------------------
            oof.loc[
                X_inner_val.index, "cpu_lr"
            ] = cpu_lr_pred

            oof.loc[
                X_inner_val.index, "cpu_rf"
            ] = cpu_rf_pred

            oof.loc[
                X_inner_val.index, "cpu_xgb"
            ] = cpu_xgb_pred

            oof.loc[
                X_inner_val.index, "memory_lr"
            ] = memory_lr_pred

            oof.loc[
                X_inner_val.index, "memory_rf"
            ] = memory_rf_pred

            oof.loc[
                X_inner_val.index, "memory_xgb"
            ] = memory_xgb_pred

            print(
                "OOF predictions stored:",
                len(X_inner_val),
            )

            del (
                cpu_lr,
                cpu_rf,
                cpu_xgb,
                memory_lr,
                memory_rf,
                memory_xgb,
                cpu_lr_pred,
                cpu_rf_pred,
                cpu_xgb_pred,
                memory_lr_pred,
                memory_rf_pred,
                memory_xgb_pred,
                X_inner_train,
                X_inner_val,
                y_cpu_inner_train,
                y_memory_inner_train,
            )

            gc.collect()

        # ================================================================
        # 3.4 OOF VALIDATION
        # ================================================================
        print("\n" + "=" * 75)
        print(f"{fold_name} — OOF VALIDATION CHECK")
        print("=" * 75)

        oof_columns = [
            "cpu_lr",
            "cpu_rf",
            "cpu_xgb",
            "memory_lr",
            "memory_rf",
            "memory_xgb",
        ]

        print("OOF shape:", oof.shape)
        print("\nMissing values in OOF columns:")
        print(oof[oof_columns].isna().sum())

        complete_oof = oof.dropna(
            subset=oof_columns
        )

        print("Complete OOF rows:", len(complete_oof))

        if complete_oof.empty:
            raise RuntimeError(
                f"{fold_name}: OOF dataset is empty. Do not continue."
            )

        # ================================================================
        # 3.5 META FEATURES — EXACTLY THREE BASE PREDICTIONS
        # ================================================================
        # A1 intentionally excludes cluster/context variables.
        cpu_meta_X = complete_oof[
            ["cpu_lr", "cpu_rf", "cpu_xgb"]
        ].to_numpy(dtype=np.float32)

        memory_meta_X = complete_oof[
            ["memory_lr", "memory_rf", "memory_xgb"]
        ].to_numpy(dtype=np.float32)

        if cpu_meta_X.shape[1] != 3:
            raise RuntimeError(
                "A1 CPU meta-feature matrix must have exactly 3 columns."
            )

        if memory_meta_X.shape[1] != 3:
            raise RuntimeError(
                "A1 memory meta-feature matrix must have exactly 3 columns."
            )

        cpu_meta_y = y_cpu_outer_train.loc[
            complete_oof.index
        ].to_numpy()

        memory_meta_y = y_memory_outer_train.loc[
            complete_oof.index
        ].to_numpy()

        print("CPU meta shape:", cpu_meta_X.shape)
        print("Memory meta shape:", memory_meta_X.shape)

        # ================================================================
        # 3.6 TRAIN META LEARNERS
        # ================================================================
        print("\nTraining CPU XGBoost meta learner...")

        cpu_meta_model = XGBRegressor(
            **META_XGB_PARAMS
        )
        cpu_meta_model.fit(
            cpu_meta_X,
            cpu_meta_y,
        )

        print("Training Memory XGBoost meta learner...")

        memory_meta_model = XGBRegressor(
            **META_XGB_PARAMS
        )
        memory_meta_model.fit(
            memory_meta_X,
            memory_meta_y,
        )

        # ================================================================
        # 3.7 REFIT BASE LEARNERS ON COMPLETE OUTER TRAINING
        # ================================================================
        print(
            "\nRefitting base learners on complete outer training..."
        )

        final_cpu_lr = LinearRegression()
        final_cpu_rf = RandomForestRegressor(**RF_PARAMS)
        final_cpu_xgb = XGBRegressor(**XGB_PARAMS)

        final_cpu_lr.fit(
            X_outer_train,
            y_cpu_outer_train,
        )
        final_cpu_rf.fit(
            X_outer_train,
            y_cpu_outer_train,
        )
        final_cpu_xgb.fit(
            X_outer_train,
            y_cpu_outer_train,
        )

        final_memory_lr = LinearRegression()
        final_memory_rf = RandomForestRegressor(**RF_PARAMS)
        final_memory_xgb = XGBRegressor(**XGB_PARAMS)

        final_memory_lr.fit(
            X_outer_train,
            y_memory_outer_train,
        )
        final_memory_rf.fit(
            X_outer_train,
            y_memory_outer_train,
        )
        final_memory_xgb.fit(
            X_outer_train,
            y_memory_outer_train,
        )

        # ================================================================
        # 3.8 OUTER VALIDATION BASE PREDICTIONS
        # ================================================================
        print(
            "Generating outer validation base predictions..."
        )

        cpu_val_lr = final_cpu_lr.predict(X_outer_val)
        cpu_val_rf = final_cpu_rf.predict(X_outer_val)
        cpu_val_xgb = final_cpu_xgb.predict(X_outer_val)

        memory_val_lr = final_memory_lr.predict(X_outer_val)
        memory_val_rf = final_memory_rf.predict(X_outer_val)
        memory_val_xgb = final_memory_xgb.predict(X_outer_val)

        # ================================================================
        # 3.9 FINAL A1 META INPUT
        # ================================================================
        cpu_outer_meta_X = np.column_stack(
            [
                cpu_val_lr,
                cpu_val_rf,
                cpu_val_xgb,
            ]
        ).astype(np.float32)

        memory_outer_meta_X = np.column_stack(
            [
                memory_val_lr,
                memory_val_rf,
                memory_val_xgb,
            ]
        ).astype(np.float32)

        if cpu_outer_meta_X.shape[1] != 3:
            raise RuntimeError(
                "Outer CPU A1 meta input must have exactly 3 columns."
            )

        if memory_outer_meta_X.shape[1] != 3:
            raise RuntimeError(
                "Outer memory A1 meta input must have exactly 3 columns."
            )

        print(
            "Outer CPU meta input:",
            cpu_outer_meta_X.shape,
        )
        print(
            "Outer Memory meta input:",
            memory_outer_meta_X.shape,
        )

        # ================================================================
        # 3.10 FINAL A1 PREDICTIONS
        # ================================================================
        cpu_final_pred = cpu_meta_model.predict(
            cpu_outer_meta_X
        )

        memory_final_pred = memory_meta_model.predict(
            memory_outer_meta_X
        )

        # ================================================================
        # 3.11 METRICS
        # ================================================================
        cpu_metrics = calculate_metrics(
            y_cpu_outer_val,
            cpu_final_pred,
        )

        memory_metrics = calculate_metrics(
            y_memory_outer_val,
            memory_final_pred,
        )

        a1_results.append(
            {
                "Fold": fold_name,
                "CPU_MSE": cpu_metrics["MSE"],
                "CPU_RMSE": cpu_metrics["RMSE"],
                "CPU_MAE": cpu_metrics["MAE"],
                "CPU_R2": cpu_metrics["R2"],
                "Memory_MSE": memory_metrics["MSE"],
                "Memory_RMSE": memory_metrics["RMSE"],
                "Memory_MAE": memory_metrics["MAE"],
                "Memory_R2": memory_metrics["R2"],
                "Train_Size": len(X_outer_train),
                "Validation_Size": len(X_outer_val),
                "Train_Max_Time": time_outer_train.max(),
                "Validation_Min_Time": time_outer_val.min(),
                "Complete_OOF_Size": len(complete_oof),
            }
        )

        # Save final outer-fold predictions for reproducibility.
        prediction_records.append(
            {
                "Fold": fold_name,
                "time": time_outer_val.to_numpy(),
                "y_cpu": y_cpu_outer_val.to_numpy(),
                "pred_cpu": cpu_final_pred,
                "y_memory": y_memory_outer_val.to_numpy(),
                "pred_memory": memory_final_pred,
            }
        )

        # ================================================================
        # 3.12 PRINT FOLD RESULTS
        # ================================================================
        print("\n" + "-" * 75)
        print(f"{fold_name} — A1 RESULTS")
        print("-" * 75)

        print("\nCPU UTILIZATION")
        print(f"MSE  : {cpu_metrics['MSE']:.10e}")
        print(f"RMSE : {cpu_metrics['RMSE']:.10f}")
        print(f"MAE  : {cpu_metrics['MAE']:.10f}")
        print(f"R²   : {cpu_metrics['R2']:.10f}")

        print("\nMEMORY UTILIZATION")
        print(f"MSE  : {memory_metrics['MSE']:.10e}")
        print(f"RMSE : {memory_metrics['RMSE']:.10f}")
        print(f"MAE  : {memory_metrics['MAE']:.10f}")
        print(f"R²   : {memory_metrics['R2']:.10f}")

        del (
            X_outer_train,
            X_outer_val,
            y_cpu_outer_train,
            y_cpu_outer_val,
            y_memory_outer_train,
            y_memory_outer_val,
            time_outer_train,
            time_outer_val,
            oof,
            complete_oof,
            cpu_meta_X,
            memory_meta_X,
            cpu_meta_y,
            memory_meta_y,
            cpu_meta_model,
            memory_meta_model,
            final_cpu_lr,
            final_cpu_rf,
            final_cpu_xgb,
            final_memory_lr,
            final_memory_rf,
            final_memory_xgb,
            cpu_val_lr,
            cpu_val_rf,
            cpu_val_xgb,
            memory_val_lr,
            memory_val_rf,
            memory_val_xgb,
            cpu_outer_meta_X,
            memory_outer_meta_X,
            cpu_final_pred,
            memory_final_pred,
        )

        gc.collect()

    # ------------------------------------------------------------------------
    # STEP 4 — RESULTS DATAFRAME
    # ------------------------------------------------------------------------
    a1_results_df = pd.DataFrame(a1_results)

    if len(a1_results_df) != len(OUTER_FOLDS):
        raise RuntimeError(
            "A1 did not produce results for all outer folds."
        )

    # ------------------------------------------------------------------------
    # STEP 5 — MEAN ± SD
    # ------------------------------------------------------------------------
    metric_definitions = [
        ("CPU MSE", "CPU_MSE"),
        ("CPU RMSE", "CPU_RMSE"),
        ("CPU MAE", "CPU_MAE"),
        ("CPU R²", "CPU_R2"),
        ("Memory MSE", "Memory_MSE"),
        ("Memory RMSE", "Memory_RMSE"),
        ("Memory MAE", "Memory_MAE"),
        ("Memory R²", "Memory_R2"),
    ]

    a1_summary = pd.DataFrame(
        [
            {
                "Metric": label,
                "Mean": a1_results_df[column].mean(),
                "SD": a1_results_df[column].std(ddof=1),
            }
            for label, column in metric_definitions
        ]
    )

    # ------------------------------------------------------------------------
    # STEP 6 — SAVE RESULTS
    # ------------------------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    a1_results_path = (
        OUTPUT_DIR
        / "A1_CORRECTED_LR_RF_XGB_Meta_outer_fold_results.csv"
    )

    a1_summary_path = (
        OUTPUT_DIR
        / "A1_CORRECTED_LR_RF_XGB_Meta_mean_sd.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / "A1_CORRECTED_LR_RF_XGB_Meta_predictions.npz"
    )

    a1_results_df.to_csv(
        a1_results_path,
        index=False,
    )

    a1_summary.to_csv(
        a1_summary_path,
        index=False,
    )

    # Validation-fold lengths can differ, so store each fold as an object
    # array inside the compressed NPZ file.
    np.savez_compressed(
        predictions_path,
        fold_names=np.array(
            [item["Fold"] for item in prediction_records],
            dtype=object,
        ),
        times=np.array(
            [item["time"] for item in prediction_records],
            dtype=object,
        ),
        y_cpu=np.array(
            [item["y_cpu"] for item in prediction_records],
            dtype=object,
        ),
        pred_cpu=np.array(
            [item["pred_cpu"] for item in prediction_records],
            dtype=object,
        ),
        y_memory=np.array(
            [item["y_memory"] for item in prediction_records],
            dtype=object,
        ),
        pred_memory=np.array(
            [item["pred_memory"] for item in prediction_records],
            dtype=object,
        ),
    )

    # ------------------------------------------------------------------------
    # STEP 7 — FINAL REPORT
    # ------------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("A1 — STACKED ENSEMBLE WITHOUT CLUSTERING")
    print("=" * 90)
    print(a1_results_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("A1 — MEAN ± SD")
    print("=" * 90)
    print(a1_summary.to_string(index=False))

    print("\n" + "=" * 90)
    print("FILES SAVED")
    print("=" * 90)
    print(f"Fold results : {a1_results_path}")
    print(f"Mean ± SD    : {a1_summary_path}")
    print(f"Predictions  : {predictions_path}")

    print("\nA1 Stacked Ensemble without Clustering completed successfully.")


if __name__ == "__main__":
    main()
