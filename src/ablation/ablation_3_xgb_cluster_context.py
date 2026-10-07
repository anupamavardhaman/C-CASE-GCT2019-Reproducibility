#!/usr/bin/env python3
"""
A3 Ablation: XGBoost + Cluster Context (No Meta Learner)

This script reproduces the A3 ablation from the supplied notebook.

A3:
    Original workload features + one-hot K-Means cluster context
    -> XGBoost
    No stacking / meta learner.

Important:
- Uses development data only from the preprocessing checkpoint.
- Uses the same three chronological outer folds as A0/A1.
- K-Means is fitted independently inside each outer training fold.
- StandardScaler for K-Means is fitted only on the corresponding
  outer training fold and then applied to that fold's validation data.
- The XGBoost models are trained separately for CPU and memory targets.
- Results are saved under results/ablation/.

The supplied A3 notebook uses N_CLUSTERS=6. This script preserves that
setting exactly. If the manuscript's final C-CASE configuration uses a
different K, update N_CLUSTERS consistently across the controlled
ablation experiments and document the change.
"""

from pathlib import Path
import gc

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ---------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "ccase_step15.pkl"
OUTPUT_DIR = ROOT_DIR / "results" / "ablation"


# ---------------------------------------------------------------------
# Reproducibility / experiment configuration
# ---------------------------------------------------------------------
RANDOM_STATE = 42
N_CLUSTERS = 6

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

KMEANS_PARAMS = {
    "n_clusters": N_CLUSTERS,
    "random_state": RANDOM_STATE,
    "n_init": 10,
}


def calculate_metrics(y_true, y_pred):
    """Return the four metrics reported for the ablation."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def create_cluster_context(cluster_labels, n_clusters=N_CLUSTERS):
    """Create one-hot cluster-context features."""
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)

    if np.any(cluster_labels < 0) or np.any(cluster_labels >= n_clusters):
        raise ValueError("K-Means produced an invalid cluster label.")

    context = np.zeros(
        (len(cluster_labels), n_clusters),
        dtype=np.float32,
    )
    context[np.arange(len(cluster_labels)), cluster_labels] = 1.0
    return context


def validate_checkpoint(checkpoint):
    """Validate the minimum checkpoint structure required by A3."""
    required_keys = {
        "model_data",
        "train_data",
        "feature_cols",
    }
    missing = required_keys.difference(checkpoint.keys())

    if missing:
        raise KeyError(
            "Checkpoint is missing required keys: "
            + ", ".join(sorted(missing))
        )

    train_data = checkpoint["train_data"]
    feature_cols = checkpoint["feature_cols"]

    required_columns = set(feature_cols) | {
        "future_cpu",
        "future_memory",
        "time_bucket",
    }
    missing_columns = required_columns.difference(train_data.columns)

    if missing_columns:
        raise KeyError(
            "Development data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if len(feature_cols) == 0:
        raise ValueError("feature_cols is empty.")

    return train_data, list(feature_cols)


def validate_outer_split(time_all, fold):
    """Validate chronological separation for one outer fold."""
    train_mask = time_all <= fold["train_end"]
    val_mask = (
        (time_all >= fold["val_start"])
        & (time_all <= fold["val_end"])
    )

    if not train_mask.any():
        raise ValueError(f"{fold['name']}: outer training set is empty.")

    if not val_mask.any():
        raise ValueError(f"{fold['name']}: outer validation set is empty.")

    train_max = time_all.loc[train_mask].max()
    val_min = time_all.loc[val_mask].min()

    if train_max >= val_min:
        raise ValueError(
            f"{fold['name']}: temporal leakage detected. "
            f"max(train_time)={train_max}, min(val_time)={val_min}."
        )

    return train_mask, val_mask


def main():
    """Run the A3 outer-fold experiment."""
    print("=" * 90)
    print("A3 — XGBoost + Cluster Context")
    print("NO META LEARNER")
    print("=" * 90)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run the preprocessing pipeline first and place "
            "ccase_step15.pkl in the repository's checkpoints/ directory."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = joblib.load(CHECKPOINT_PATH)
    train_data, feature_cols = validate_checkpoint(checkpoint)

    print(f"Checkpoint loaded: {CHECKPOINT_PATH}")
    print("Development data shape:", train_data.shape)
    print("Number of original features:", len(feature_cols))
    print("Number of clusters:", N_CLUSTERS)

    # Development data only. The untouched final 20% is not used here.
    X_all = train_data[feature_cols].copy()
    y_cpu_all = train_data["future_cpu"].copy()
    y_memory_all = train_data["future_memory"].copy()
    time_all = train_data["time_bucket"].copy()

    if X_all.isna().any().any():
        raise ValueError("NaN values detected in A3 feature columns.")
    if y_cpu_all.isna().any() or y_memory_all.isna().any():
        raise ValueError("NaN values detected in A3 target columns.")
    if time_all.isna().any():
        raise ValueError("NaN values detected in time_bucket.")

    # Keep the original index so predictions can be traced to observations.
    a3_results = []
    prediction_records = []
    cluster_records = []

    for fold in OUTER_FOLDS:
        fold_name = fold["name"]

        print("\n" + "=" * 90)
        print(f"{fold_name} — A3 XGBoost + Cluster Context")
        print("NO META LEARNER")
        print("=" * 90)

        train_mask, val_mask = validate_outer_split(time_all, fold)

        X_train = X_all.loc[train_mask].copy()
        X_val = X_all.loc[val_mask].copy()

        y_cpu_train = y_cpu_all.loc[train_mask]
        y_cpu_val = y_cpu_all.loc[val_mask]

        y_memory_train = y_memory_all.loc[train_mask]
        y_memory_val = y_memory_all.loc[val_mask]

        train_times = time_all.loc[train_mask]
        val_times = time_all.loc[val_mask]

        print("Training samples :", len(X_train))
        print("Validation samples:", len(X_val))
        print(
            "Training time range:",
            train_times.min(),
            "→",
            train_times.max(),
        )
        print(
            "Validation time range:",
            val_times.min(),
            "→",
            val_times.max(),
        )

        # -------------------------------------------------------------
        # K-Means: fit ONLY on outer training data
        # -------------------------------------------------------------
        print("\nFitting StandardScaler for K-Means...")
        cluster_scaler = StandardScaler()

        X_train_cluster = cluster_scaler.fit_transform(X_train)
        X_val_cluster = cluster_scaler.transform(X_val)

        print(f"Fitting K-Means K={N_CLUSTERS}...")
        kmeans = KMeans(**KMEANS_PARAMS)

        train_clusters = kmeans.fit_predict(X_train_cluster)
        val_clusters = kmeans.predict(X_val_cluster)

        print("\nTraining cluster distribution:")
        print(
            pd.Series(train_clusters)
            .value_counts(sort=False)
            .sort_index()
            .to_string()
        )

        print("\nValidation cluster distribution:")
        print(
            pd.Series(val_clusters)
            .value_counts(sort=False)
            .sort_index()
            .to_string()
        )

        # -------------------------------------------------------------
        # One-hot cluster context
        # -------------------------------------------------------------
        train_context = create_cluster_context(
            train_clusters,
            N_CLUSTERS,
        )
        val_context = create_cluster_context(
            val_clusters,
            N_CLUSTERS,
        )

        X_train_original = X_train.to_numpy(dtype=np.float32)
        X_val_original = X_val.to_numpy(dtype=np.float32)

        X_train_a3 = np.hstack(
            [X_train_original, train_context]
        )
        X_val_a3 = np.hstack(
            [X_val_original, val_context]
        )

        print("\nA3 training feature shape:", X_train_a3.shape)
        print("A3 validation feature shape:", X_val_a3.shape)

        # -------------------------------------------------------------
        # CPU XGBoost
        # -------------------------------------------------------------
        print("\nTraining A3 CPU XGBoost...")
        cpu_model = XGBRegressor(**XGB_PARAMS)
        cpu_model.fit(X_train_a3, y_cpu_train)

        cpu_pred = cpu_model.predict(X_val_a3)

        # -------------------------------------------------------------
        # Memory XGBoost
        # -------------------------------------------------------------
        print("Training A3 Memory XGBoost...")
        memory_model = XGBRegressor(**XGB_PARAMS)
        memory_model.fit(X_train_a3, y_memory_train)

        memory_pred = memory_model.predict(X_val_a3)

        # -------------------------------------------------------------
        # Metrics
        # -------------------------------------------------------------
        cpu_metrics = calculate_metrics(y_cpu_val, cpu_pred)
        memory_metrics = calculate_metrics(y_memory_val, memory_pred)

        a3_results.append(
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
            }
        )

        # Save validation predictions for independent evaluation.
        for idx, timestamp, y_cpu, p_cpu, y_mem, p_mem, c in zip(
            X_val.index,
            val_times.to_numpy(),
            y_cpu_val.to_numpy(),
            cpu_pred,
            y_memory_val.to_numpy(),
            memory_pred,
            val_clusters,
        ):
            prediction_records.append(
                {
                    "Fold": fold_name,
                    "Index": idx,
                    "time_bucket": timestamp,
                    "true_cpu": y_cpu,
                    "pred_cpu": p_cpu,
                    "true_memory": y_mem,
                    "pred_memory": p_mem,
                    "cluster": int(c),
                }
            )

        # Save cluster assignments as an auditable artifact.
        for idx, timestamp, cluster in zip(
            X_val.index,
            val_times.to_numpy(),
            val_clusters,
        ):
            cluster_records.append(
                {
                    "Fold": fold_name,
                    "Index": idx,
                    "time_bucket": timestamp,
                    "cluster": int(cluster),
                }
            )

        print("\n" + "-" * 70)
        print(f"{fold_name} — A3 RESULTS")
        print("-" * 70)
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

        # Release large fold-specific objects.
        del (
            X_train,
            X_val,
            y_cpu_train,
            y_cpu_val,
            y_memory_train,
            y_memory_val,
            train_times,
            val_times,
            X_train_cluster,
            X_val_cluster,
            cluster_scaler,
            kmeans,
            train_clusters,
            val_clusters,
            train_context,
            val_context,
            X_train_original,
            X_val_original,
            X_train_a3,
            X_val_a3,
            cpu_model,
            memory_model,
            cpu_pred,
            memory_pred,
        )
        gc.collect()

    # -----------------------------------------------------------------
    # Fold results
    # -----------------------------------------------------------------
    results_df = pd.DataFrame(a3_results)

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "CPU MSE",
                "CPU RMSE",
                "CPU MAE",
                "CPU R²",
                "Memory MSE",
                "Memory RMSE",
                "Memory MAE",
                "Memory R²",
            ],
            "Mean": [
                results_df["CPU_MSE"].mean(),
                results_df["CPU_RMSE"].mean(),
                results_df["CPU_MAE"].mean(),
                results_df["CPU_R2"].mean(),
                results_df["Memory_MSE"].mean(),
                results_df["Memory_RMSE"].mean(),
                results_df["Memory_MAE"].mean(),
                results_df["Memory_R2"].mean(),
            ],
            "SD": [
                results_df["CPU_MSE"].std(ddof=1),
                results_df["CPU_RMSE"].std(ddof=1),
                results_df["CPU_MAE"].std(ddof=1),
                results_df["CPU_R2"].std(ddof=1),
                results_df["Memory_MSE"].std(ddof=1),
                results_df["Memory_RMSE"].std(ddof=1),
                results_df["Memory_MAE"].std(ddof=1),
                results_df["Memory_R2"].std(ddof=1),
            ],
        }
    )

    predictions_df = pd.DataFrame(prediction_records)
    clusters_df = pd.DataFrame(cluster_records)

    # -----------------------------------------------------------------
    # Save reproducibility artifacts
    # -----------------------------------------------------------------
    results_path = (
        OUTPUT_DIR / "A3_XGB_ClusterContext_outer_fold_results.csv"
    )
    summary_path = (
        OUTPUT_DIR / "A3_XGB_ClusterContext_mean_sd.csv"
    )
    predictions_path = (
        OUTPUT_DIR / "A3_XGB_ClusterContext_predictions.csv"
    )
    clusters_path = (
        OUTPUT_DIR / "A3_XGB_ClusterContext_validation_clusters.csv"
    )

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    clusters_df.to_csv(clusters_path, index=False)

    print("\n" + "=" * 90)
    print("A3 — XGBoost + Cluster Context")
    print("NO META LEARNER")
    print("=" * 90)

    print("\nOuter-fold results:")
    print(results_df.to_string(index=False))

    print("\nMean ± SD:")
    print(summary_df.to_string(index=False))

    print("\nSaved artifacts:")
    print(results_path)
    print(summary_path)
    print(predictions_path)
    print(clusters_path)


if __name__ == "__main__":
    main()
