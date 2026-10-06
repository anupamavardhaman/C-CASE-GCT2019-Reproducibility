#!/usr/bin/env python3
"""
C-CASE reproducibility preprocessing pipeline.

This script is derived from the verified preprocessing workflow in:
    instance_usage_DS__C_CASE(2).ipynb

It reproduces the data preparation required before the C-CASE model:
    1. Load the specified Google Cluster Trace 2019 InstanceUsage shard.
    2. Keep complete 300-second observations with alloc_collection_id == 0.
    3. Aggregate CPU and memory at machine + 5-minute level.
    4. Remove machine_id == -1.
    5. Build continuous 5-minute sequences (minimum 12 observations).
    6. Create 10 historical CPU/memory lags.
    7. Create 5-minute-ahead CPU/memory targets.
    8. Create leakage-safe deltas, rolling statistics, ratio and cyclic time features.
    9. Perform a chronological 80:20 train/test split.
   10. Fit StandardScaler on training data only.
   11. Select K using Elbow and Silhouette on training data only.
   12. Fit final K-Means on training data only and assign test clusters by predict().
   13. Save a reproducibility checkpoint containing the exact objects used later.

Important:
- The raw Google trace is NOT redistributed by this repository.
- The Step-15 preprocessing checkpoint uses K=6, matching the verified
  preprocessing notebook. The final Table-4 training script independently
  refits K=3 for the C-CASE comparison experiment.
"""

import argparse
import gzip
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
FORECAST_HORIZON = 1                 # one subsequent 5-minute observation
TRAIN_RATIO = 0.80
MIN_SEQUENCE_LENGTH = 12
LAG_COUNT = 10
CLUSTER_SAMPLE_SIZE = 100_000
K_VALUES = [2, 3, 4, 5, 6, 7, 8]
DEFAULT_K = 6

FEATURE_COLS = [
    "cpu_usage",
    "memory_usage",
    *[f"cpu_lag_{i}" for i in range(1, LAG_COUNT + 1)],
    *[f"memory_lag_{i}" for i in range(1, LAG_COUNT + 1)],
    "cpu_delta",
    "memory_delta",
    "rolling_cpu_mean",
    "rolling_cpu_std",
    "rolling_memory_mean",
    "rolling_memory_std",
    "cpu_mem_ratio",
    "time_sin",
    "time_cos",
]

TARGET_COLS = ["future_cpu", "future_memory"]

CLUSTER_FEATURES = [
    "cpu_usage",
    "memory_usage",
    *[f"cpu_lag_{i}" for i in range(1, LAG_COUNT + 1)],
    *[f"memory_lag_{i}" for i in range(1, LAG_COUNT + 1)],
    "cpu_delta",
    "memory_delta",
    "rolling_cpu_mean",
    "rolling_cpu_std",
    "rolling_memory_mean",
    "rolling_memory_std",
    "cpu_mem_ratio",
]


def load_instance_usage(file_path: Path) -> pd.DataFrame:
    """Load the specified InstanceUsage JSON.GZ shard."""
    rows = []

    with gzip.open(file_path, "rt") as f:
        for line in f:
            record = json.loads(line)
            avg = record.get("average_usage", {})

            rows.append(
                {
                    "start_time": int(record["start_time"]),
                    "end_time": int(record["end_time"]),
                    "collection_id": int(record["collection_id"]),
                    "instance_index": int(record["instance_index"]),
                    "machine_id": int(record["machine_id"]),
                    "alloc_collection_id": (
                        int(record["alloc_collection_id"])
                        if record.get("alloc_collection_id") not in [None, ""]
                        else None
                    ),
                    "cpu_usage": avg.get("cpus"),
                    "memory_usage": avg.get("memory"),
                }
            )

    df = pd.DataFrame(rows)

    # Match the notebook's duration calculation.
    df["duration_seconds"] = (
        df["end_time"] - df["start_time"]
    ) / 1_000_000

    # Match the notebook's 5-minute bucket construction.
    df["time_bucket"] = (
        (df["start_time"] // (300 * 1_000_000))
        * (300 * 1_000_000)
    )

    return df


def build_continuous_machine_series(df: pd.DataFrame) -> pd.DataFrame:
    """Create machine-level 5-minute series and retain continuous sequences."""
    # Complete 300-second, non-allocated observations.
    df_primary = df[
        (df["duration_seconds"] == 300)
        & (df["alloc_collection_id"] == 0)
    ].copy()

    # Machine + 5-minute aggregation.
    machine_ts = (
        df_primary
        .groupby(["machine_id", "time_bucket"], as_index=False)
        .agg(
            cpu_usage=("cpu_usage", "sum"),
            memory_usage=("memory_usage", "sum"),
            instance_count=("instance_index", "count"),
        )
        .sort_values(["machine_id", "time_bucket"])
        .reset_index(drop=True)
    )

    # Remove unknown/special machine ID.
    machine_ts_clean = machine_ts[
        machine_ts["machine_id"] != -1
    ].copy()

    machine_ts_clean = (
        machine_ts_clean
        .sort_values(["machine_id", "time_bucket"])
        .reset_index(drop=True)
    )

    # Identify exactly consecutive 5-minute observations.
    machine_ts_clean["time_gap_sec"] = (
        machine_ts_clean
        .groupby("machine_id")["time_bucket"]
        .diff()
        / 1_000_000
    )

    machine_ts_clean["is_consecutive"] = (
        machine_ts_clean["time_gap_sec"] == 300
    )

    # New sequence whenever the gap is not exactly 300 seconds.
    machine_ts_clean["sequence_id"] = (
        ~machine_ts_clean["is_consecutive"]
    ).groupby(
        machine_ts_clean["machine_id"]
    ).cumsum()

    sequence_summary = (
        machine_ts_clean
        .groupby(["machine_id", "sequence_id"])
        .agg(
            start_time=("time_bucket", "min"),
            end_time=("time_bucket", "max"),
            observations=("time_bucket", "size"),
        )
        .reset_index()
    )

    valid_sequences = sequence_summary[
        sequence_summary["observations"] >= MIN_SEQUENCE_LENGTH
    ][["machine_id", "sequence_id"]]

    machine_ts_continuous = machine_ts_clean.merge(
        valid_sequences,
        on=["machine_id", "sequence_id"],
        how="inner",
    )

    return (
        machine_ts_continuous
        .sort_values(["machine_id", "sequence_id", "time_bucket"])
        .reset_index(drop=True)
    )


def engineer_features(machine_ts_continuous: pd.DataFrame) -> pd.DataFrame:
    """Create the exact 31-feature forecasting representation."""
    data = machine_ts_continuous.copy()
    group_cols = ["machine_id", "sequence_id"]

    # Ten previous 5-minute observations.
    for lag in range(1, LAG_COUNT + 1):
        data[f"cpu_lag_{lag}"] = (
            data.groupby(group_cols)["cpu_usage"].shift(lag)
        )
        data[f"memory_lag_{lag}"] = (
            data.groupby(group_cols)["memory_usage"].shift(lag)
        )

    # One-step-ahead = 5-minute forecasting target.
    data["future_cpu"] = (
        data.groupby(group_cols)["cpu_usage"]
        .shift(-FORECAST_HORIZON)
    )
    data["future_memory"] = (
        data.groupby(group_cols)["memory_usage"]
        .shift(-FORECAST_HORIZON)
    )

    data["target_time"] = (
        data.groupby(group_cols)["time_bucket"]
        .shift(-FORECAST_HORIZON)
    )

    data["target_gap_sec"] = (
        data["target_time"] - data["time_bucket"]
    ) / 1_000_000

    # Cyclic time-of-day features.
    elapsed_seconds = data["time_bucket"] / 1_000_000
    seconds_in_day = 24 * 60 * 60
    time_of_day_seconds = elapsed_seconds % seconds_in_day

    data["time_sin"] = np.sin(
        2 * np.pi * time_of_day_seconds / seconds_in_day
    )
    data["time_cos"] = np.cos(
        2 * np.pi * time_of_day_seconds / seconds_in_day
    )

    # Changes relative to the previous observation.
    data["cpu_delta"] = (
        data["cpu_usage"] - data["cpu_lag_1"]
    )
    data["memory_delta"] = (
        data["memory_usage"] - data["memory_lag_1"]
    )

    # Leakage-safe rolling statistics:
    # current observation is excluded by shift(1).
    data["rolling_cpu_mean"] = (
        data.groupby(group_cols)["cpu_usage"]
        .transform(
            lambda x: x.shift(1).rolling(
                5, min_periods=5
            ).mean()
        )
    )

    data["rolling_cpu_std"] = (
        data.groupby(group_cols)["cpu_usage"]
        .transform(
            lambda x: x.shift(1).rolling(
                5, min_periods=5
            ).std()
        )
    )

    data["rolling_memory_mean"] = (
        data.groupby(group_cols)["memory_usage"]
        .transform(
            lambda x: x.shift(1).rolling(
                5, min_periods=5
            ).mean()
        )
    )

    data["rolling_memory_std"] = (
        data.groupby(group_cols)["memory_usage"]
        .transform(
            lambda x: x.shift(1).rolling(
                5, min_periods=5
            ).std()
        )
    )

    # CPU-to-memory relationship.
    data["cpu_mem_ratio"] = np.where(
        data["memory_usage"] != 0,
        data["cpu_usage"] / data["memory_usage"],
        np.nan,
    )

    # Remove non-finite values before the final model dataset.
    data = data.replace([np.inf, -np.inf], np.nan)

    required_cols = FEATURE_COLS + TARGET_COLS

    # Require complete history, complete targets and complete context.
    data = (
        data
        .dropna(subset=required_cols)
        .sort_values(
            ["time_bucket", "machine_id", "sequence_id"]
        )
        .reset_index(drop=True)
    )

    return data


def chronological_split(
    model_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Create the verified chronological 80:20 split.

    The verified notebook determines the training row boundary first:
        n_train_target = int(n_total * 0.80)

    It then uses the time bucket at that row as the cutoff and assigns
    complete time buckets to either the development set or the final test
    set. This reproduces the verified split of:

        total = 1,419,259
        train = 1,135,395
        test  =   283,864

    for the official instance-usage shard used in the study.
    """
    model_data = (
        model_data
        .sort_values(["time_bucket", "machine_id", "sequence_id"])
        .reset_index(drop=True)
    )

    n_total = len(model_data)
    n_train_target = int(n_total * TRAIN_RATIO)

    if n_train_target <= 0 or n_train_target >= n_total:
        raise ValueError("Invalid chronological split.")

    cutoff_time = int(
        model_data.loc[n_train_target, "time_bucket"]
    )

    train_data = model_data[
        model_data["time_bucket"] < cutoff_time
    ].copy()

    test_data = model_data[
        model_data["time_bucket"] >= cutoff_time
    ].copy()

    if train_data.empty or test_data.empty:
        raise RuntimeError("Chronological split produced an empty partition.")

    if train_data["time_bucket"].max() >= test_data["time_bucket"].min():
        raise RuntimeError("Temporal leakage detected at train/test boundary.")

    return (
        train_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
        cutoff_time,
    )


def fit_scaler(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on training data only."""
    X_train = train_data[FEATURE_COLS].copy()
    X_test = test_data[FEATURE_COLS].copy()

    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled_array,
        columns=FEATURE_COLS,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled_array,
        columns=FEATURE_COLS,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, scaler


def select_k(
    X_train_cluster: pd.DataFrame,
    k_values: list[int],
) -> tuple[dict, dict]:
    """Evaluate candidate K values using training data only."""
    sample_size = min(CLUSTER_SAMPLE_SIZE, len(X_train_cluster))

    sample = X_train_cluster.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    inertias = {}
    silhouettes = {}

    for k in k_values:
        km = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        labels = km.fit_predict(sample)

        inertias[k] = float(km.inertia_)
        silhouettes[k] = float(
            silhouette_score(sample, labels)
        )

    return inertias, silhouettes


def fit_final_kmeans(
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    scaler: StandardScaler,
    k_final: int,
):
    """Fit final K-Means on training data only and assign test clusters."""
    X_train_cluster = X_train_scaled[CLUSTER_FEATURES].copy()
    X_test_cluster = X_test_scaled[CLUSTER_FEATURES].copy()

    kmeans_final = KMeans(
        n_clusters=k_final,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    kmeans_final.fit(X_train_cluster)

    train_clusters = kmeans_final.labels_
    test_clusters = kmeans_final.predict(X_test_cluster)

    train_data_clustered = X_train_cluster.copy()
    test_data_clustered = X_test_cluster.copy()

    # Convert centroids back to original units for interpretation.
    centroids_original = pd.DataFrame(
        kmeans_final.cluster_centers_,
        columns=CLUSTER_FEATURES,
    )

    for col in CLUSTER_FEATURES:
        feature_index = FEATURE_COLS.index(col)
        centroids_original[col] = (
            centroids_original[col] * scaler.scale_[feature_index]
            + scaler.mean_[feature_index]
        )

    return (
        kmeans_final,
        train_clusters,
        test_clusters,
        centroids_original,
        train_data_clustered,
        test_data_clustered,
    )


def run(args):
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input shard not found: {input_path}"
        )

    if args.k < 2:
        raise ValueError("K must be at least 2.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("C-CASE REPRODUCIBILITY PREPROCESSING")
    print("=" * 80)
    print("Input:", input_path)
    print("Final K:", args.k)
    print("Random seed:", RANDOM_STATE)

    # ------------------------------------------------------------------
    # 1. Load specified GCT 2019 InstanceUsage shard
    # ------------------------------------------------------------------
    df = load_instance_usage(input_path)

    print("\nRaw records:", len(df))
    print("Unique machines:", df["machine_id"].nunique())

    # ------------------------------------------------------------------
    # 2. Build continuous machine-level workload series
    # ------------------------------------------------------------------
    machine_ts_continuous = build_continuous_machine_series(df)

    print(
        "\nContinuous-sequence observations:",
        len(machine_ts_continuous),
    )
    print(
        "Machines:",
        machine_ts_continuous["machine_id"].nunique(),
    )
    print(
        "Sequences:",
        machine_ts_continuous[
            ["machine_id", "sequence_id"]
        ].drop_duplicates().shape[0],
    )

    # ------------------------------------------------------------------
    # 3. Feature engineering and target creation
    # ------------------------------------------------------------------
    model_data = engineer_features(machine_ts_continuous)

    if len(FEATURE_COLS) != 31:
        raise RuntimeError(
            f"Expected 31 features, found {len(FEATURE_COLS)}."
        )

    if not set(FEATURE_COLS).issubset(model_data.columns):
        raise RuntimeError("Feature engineering did not create all features.")

    # Target horizon audit.
    bad_target_gaps = model_data.loc[
        model_data["target_gap_sec"] != 300,
        "target_gap_sec",
    ]

    if len(bad_target_gaps) > 0:
        raise RuntimeError(
            "Found target gaps other than exactly 300 seconds."
        )

    # Feature leakage audit.
    forbidden = {
        "future_cpu",
        "future_memory",
        "target_time",
        "target_gap_sec",
        "machine_id",
        "sequence_id",
        "time_bucket",
    }

    leakage_features = set(FEATURE_COLS).intersection(forbidden)

    if leakage_features:
        raise RuntimeError(
            f"Forbidden leakage/identifier columns in features: "
            f"{sorted(leakage_features)}"
        )

    # ------------------------------------------------------------------
    # 4. Chronological 80:20 split
    # ------------------------------------------------------------------
    train_data, test_data, cutoff_time = chronological_split(model_data)

    print("\nChronological split")
    print("Training rows:", len(train_data))
    print("Testing rows :", len(test_data))
    print(
        "Training %:",
        round(len(train_data) / len(model_data) * 100, 3),
    )
    print(
        "Testing % :",
        round(len(test_data) / len(model_data) * 100, 3),
    )
    print("Training end:", train_data["time_bucket"].max())
    print("Testing start:", test_data["time_bucket"].min())

    # Verified dimensions for the exact GCT 2019 shard used in the study.
    # The check is informative rather than a hard failure so the script can
    # still be reused with another shard.
    if len(model_data) == 1_419_259:
        if len(train_data) != 1_135_395 or len(test_data) != 283_864:
            raise RuntimeError(
                "Verified GCT 2019 split mismatch: expected "
                "1,135,395 training rows and 283,864 test rows, but found "
                f"{len(train_data):,} and {len(test_data):,}."
            )
        print(
            "Verified split check: PASS "
            "(1,135,395 train / 283,864 test)"
        )

    # ------------------------------------------------------------------
    # 5. Training-only scaling
    # ------------------------------------------------------------------
    X_train_scaled, X_test_scaled, scaler = fit_scaler(
        train_data,
        test_data,
    )

    if scaler.n_samples_seen_ != len(train_data):
        raise RuntimeError(
            "Scaler sample count does not match training observations."
        )

    # ------------------------------------------------------------------
    # 6. Training-only clustering
    # ------------------------------------------------------------------
    inertias, silhouettes = select_k(
        X_train_scaled[CLUSTER_FEATURES],
        K_VALUES,
    )

    print("\nK selection")
    for k in K_VALUES:
        print(
            f"K={k}: "
            f"inertia={inertias[k]:.6f}, "
            f"silhouette={silhouettes[k]:.6f}"
        )

    (
        kmeans_final,
        train_clusters,
        test_clusters,
        centroids_original,
        _,
        _,
    ) = fit_final_kmeans(
        X_train_scaled,
        X_test_scaled,
        scaler,
        args.k,
    )

    train_data = train_data.copy()
    test_data = test_data.copy()

    train_data["cluster"] = train_clusters
    test_data["cluster"] = test_clusters

    # Cluster profiles used in the paper/reviewer response.
    cluster_profile_train = (
        train_data
        .groupby("cluster")
        .agg(
            observations=("cluster", "size"),
            cpu_mean=("cpu_usage", "mean"),
            cpu_std=("cpu_usage", "std"),
            memory_mean=("memory_usage", "mean"),
            memory_std=("memory_usage", "std"),
            cpu_delta_mean=("cpu_delta", "mean"),
            cpu_delta_std=("cpu_delta", "std"),
            memory_delta_mean=("memory_delta", "mean"),
            memory_delta_std=("memory_delta", "std"),
            cpu_mem_ratio_mean=("cpu_mem_ratio", "mean"),
            cpu_mem_ratio_std=("cpu_mem_ratio", "std"),
        )
        .sort_index()
    )

    cluster_profile_train["percentage"] = (
        cluster_profile_train["observations"]
        / len(train_data)
        * 100
    )

    # ------------------------------------------------------------------
    # 7. Reproducibility checkpoint
    # ------------------------------------------------------------------
    checkpoint = {
        "model_data": model_data,
        "train_data": train_data,
        "test_data": test_data,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "kmeans_final": kmeans_final,
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "cluster_features": CLUSTER_FEATURES,
        "train_clusters": train_clusters,
        "test_clusters": test_clusters,
        "cluster_profile_train": cluster_profile_train,
        "centroids_original": centroids_original,
        "k_selection_inertia": inertias,
        "k_selection_silhouette": silhouettes,
        "k_final": args.k,
        "random_state": RANDOM_STATE,
        "forecast_horizon": FORECAST_HORIZON,
        "train_ratio": TRAIN_RATIO,
        "min_sequence_length": MIN_SEQUENCE_LENGTH,
        "source_file": str(input_path),
        "train_cutoff_time": cutoff_time,
    }

    joblib.dump(
        checkpoint,
        output_path,
        compress=3,
    )

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print("Checkpoint:", output_path)
    print("Final K:", args.k)
    print("Training rows:", len(train_data))
    print("Testing rows :", len(test_data))
    print("Feature count:", len(FEATURE_COLS))
    print("Train clusters:", train_data["cluster"].value_counts().sort_index().to_dict())
    print("Test clusters :", test_data["cluster"].value_counts().sort_index().to_dict())
    print("\nCluster profiles:")
    print(cluster_profile_train.round(6))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the C-CASE preprocessing and clustering checkpoint."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to instance_usage-000000000000.json.gz",
    )

    parser.add_argument(
        "--output",
        default="checkpoints/ccase_step15.pkl",
        help="Output checkpoint path.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Final number of preprocessing clusters. Default: 6.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
