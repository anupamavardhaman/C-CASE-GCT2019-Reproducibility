#!/usr/bin/env python3
"""
C-CASE Table 4 training and evaluation.

Source notebook:
    K=3_Table_4_final_results_of_all_models.ipynb

Architecture reproduced from the notebook:
    - load Step 15 checkpoint
    - use global LR, RF and XGBoost base learners
    - K-Means K=3 fitted on development data only
    - cluster labels converted to 3 one-hot context variables
    - 3-fold temporal OOF predictions
    - XGBoost meta learner for no-clustering stacking
    - XGBoost meta learner for C-CASE:
          3 base predictions + 3 cluster-context variables = 6 features
    - refit global base models on the complete development set
    - evaluate all models on the same untouched test set
    - save final prediction arrays and Table 4 metrics
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


RANDOM_STATE = 42
K = 3
N_SPLITS = 3
EXPECTED_TEST_SIZE = 283864

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

META_PARAMS = {
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


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ccase_step15.pkl",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # STEP 1 — LOAD CHECKPOINT AND PREPARE DATA
    # ============================================================
    print("=" * 70)
    print("STEP 1 - LOAD CHECKPOINT AND PREPARE DEVELOPMENT/TEST DATA")
    print("=" * 70)

    checkpoint = joblib.load(args.checkpoint)

    train_data = checkpoint["train_data"].copy()
    test_data = checkpoint["test_data"].copy()
    feature_cols = checkpoint["feature_cols"]

    TARGET_CPU = "future_cpu"
    TARGET_MEMORY = "future_memory"

    X_dev = train_data[feature_cols].copy()
    y_dev_cpu = train_data[TARGET_CPU].values
    y_dev_mem = train_data[TARGET_MEMORY].values

    X_test = test_data[feature_cols].copy()
    y_test_cpu = test_data[TARGET_CPU].values
    y_test_mem = test_data[TARGET_MEMORY].values

    print("Development data shape:", train_data.shape)
    print("Test data shape       :", test_data.shape)
    print("Number of features    :", len(feature_cols))

    print("\n===== DIMENSION CHECK =====")
    print("X_dev      :", X_dev.shape)
    print("y_dev_cpu  :", y_dev_cpu.shape)
    print("y_dev_mem  :", y_dev_mem.shape)
    print("X_test     :", X_test.shape)
    print("y_test_cpu :", y_test_cpu.shape)
    print("y_test_mem :", y_test_mem.shape)

    assert len(X_test) == EXPECTED_TEST_SIZE, (
        f"Expected {EXPECTED_TEST_SIZE} test rows but found "
        f"{len(X_test)}"
    )

    assert X_dev.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0

    # The checkpoint contains a training-fitted scaler, but the uploaded
    # Table-4 notebook refits StandardScaler here. This line reproduces
    # the notebook's actual Table-4 procedure.
    print("\nFitting StandardScaler on development data only...")

    scaler_final = StandardScaler()

    X_dev_scaled = scaler_final.fit_transform(X_dev)
    X_test_scaled = scaler_final.transform(X_test)

    print("X_dev_scaled :", X_dev_scaled.shape)
    print("X_test_scaled:", X_test_scaled.shape)

    # ------------------------------------------------------------
    # K=3 is intentionally refitted here, independently of the
    # K=6 clustering stored in the Step 15 checkpoint.
    # ------------------------------------------------------------
    print("\n===== K-MEANS =====")
    print("K =", K)
    print("Random state =", RANDOM_STATE)

    kmeans_final = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    cluster_dev = kmeans_final.fit_predict(X_dev_scaled)
    cluster_test = kmeans_final.predict(X_test_scaled)

    cluster_dev_oh = np.eye(K)[cluster_dev]
    cluster_test_oh = np.eye(K)[cluster_test]

    print("\nDevelopment cluster distribution:")
    print(pd.Series(cluster_dev).value_counts().sort_index())

    print("\nTest cluster distribution:")
    print(pd.Series(cluster_test).value_counts().sort_index())

    print("\nCluster context development:", cluster_dev_oh.shape)
    print("Cluster context test       :", cluster_test_oh.shape)

    # Save the preprocessing objects used by Table 4.
    joblib.dump(
        {
            "scaler": scaler_final,
            "kmeans": kmeans_final,
            "feature_cols": feature_cols,
            "cluster_dev": cluster_dev,
            "cluster_test": cluster_test,
        },
        os.path.join(args.output_dir, "final_preprocessing.pkl"),
    )

    # ============================================================
    # STEP 2B — LEAKAGE-FREE TEMPORAL OOF PREDICTIONS
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2B - LEAKAGE-FREE TEMPORAL OOF PREDICTIONS")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    n_dev = len(X_dev_scaled)

    oof_cpu_lr = np.full(n_dev, np.nan)
    oof_cpu_rf = np.full(n_dev, np.nan)
    oof_cpu_xgb = np.full(n_dev, np.nan)

    oof_mem_lr = np.full(n_dev, np.nan)
    oof_mem_rf = np.full(n_dev, np.nan)
    oof_mem_xgb = np.full(n_dev, np.nan)

    for fold, (train_idx, val_idx) in enumerate(
        tscv.split(X_dev_scaled), start=1
    ):
        print("\n" + "=" * 60)
        print(f"OOF FOLD {fold}")
        print("=" * 60)
        print("Training rows  :", len(train_idx))
        print("Validation rows:", len(val_idx))

        X_inner_train = X_dev_scaled[train_idx]
        X_inner_val = X_dev_scaled[val_idx]

        # CPU base learners
        lr_cpu = LinearRegression()
        rf_cpu = RandomForestRegressor(**RF_PARAMS)
        xgb_cpu = XGBRegressor(**XGB_PARAMS)

        lr_cpu.fit(X_inner_train, y_dev_cpu[train_idx])
        rf_cpu.fit(X_inner_train, y_dev_cpu[train_idx])
        xgb_cpu.fit(X_inner_train, y_dev_cpu[train_idx])

        oof_cpu_lr[val_idx] = lr_cpu.predict(X_inner_val)
        oof_cpu_rf[val_idx] = rf_cpu.predict(X_inner_val)
        oof_cpu_xgb[val_idx] = xgb_cpu.predict(X_inner_val)

        # Memory base learners
        lr_mem = LinearRegression()
        rf_mem = RandomForestRegressor(**RF_PARAMS)
        xgb_mem = XGBRegressor(**XGB_PARAMS)

        lr_mem.fit(X_inner_train, y_dev_mem[train_idx])
        rf_mem.fit(X_inner_train, y_dev_mem[train_idx])
        xgb_mem.fit(X_inner_train, y_dev_mem[train_idx])

        oof_mem_lr[val_idx] = lr_mem.predict(X_inner_val)
        oof_mem_rf[val_idx] = rf_mem.predict(X_inner_val)
        oof_mem_xgb[val_idx] = xgb_mem.predict(X_inner_val)

        print("Fold completed.")

    oof_mask = (
        ~np.isnan(oof_cpu_lr)
        & ~np.isnan(oof_cpu_rf)
        & ~np.isnan(oof_cpu_xgb)
        & ~np.isnan(oof_mem_lr)
        & ~np.isnan(oof_mem_rf)
        & ~np.isnan(oof_mem_xgb)
    )

    print("\n" + "=" * 60)
    print("OOF SUMMARY")
    print("=" * 60)
    print("Total development rows      :", n_dev)
    print("Rows with OOF predictions   :", oof_mask.sum())
    print("Rows without OOF predictions:", (~oof_mask).sum())

    # ============================================================
    # STEP 2C — META-TRAINING DATASETS
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2C - CREATE META-TRAINING DATASETS")
    print("=" * 70)

    meta_cpu_base = np.column_stack([
        oof_cpu_lr[oof_mask],
        oof_cpu_rf[oof_mask],
        oof_cpu_xgb[oof_mask],
    ])

    meta_mem_base = np.column_stack([
        oof_mem_lr[oof_mask],
        oof_mem_rf[oof_mask],
        oof_mem_xgb[oof_mask],
    ])

    meta_cluster = cluster_dev_oh[oof_mask]

    meta_cpu_no_cluster = meta_cpu_base.copy()
    meta_mem_no_cluster = meta_mem_base.copy()

    meta_cpu_ccase = np.column_stack([
        meta_cpu_base,
        meta_cluster,
    ])

    meta_mem_ccase = np.column_stack([
        meta_mem_base,
        meta_cluster,
    ])

    meta_y_cpu = y_dev_cpu[oof_mask]
    meta_y_mem = y_dev_mem[oof_mask]

    assert meta_cpu_no_cluster.shape[1] == 3
    assert meta_mem_no_cluster.shape[1] == 3
    assert meta_cpu_ccase.shape[1] == 6
    assert meta_mem_ccase.shape[1] == 6

    print("No-clustering CPU meta features :", meta_cpu_no_cluster.shape)
    print("No-clustering Memory meta features:", meta_mem_no_cluster.shape)
    print("C-CASE CPU meta features        :", meta_cpu_ccase.shape)
    print("C-CASE Memory meta features     :", meta_mem_ccase.shape)

    # ============================================================
    # STEP 2D — TRAIN META LEARNERS
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2D - TRAIN META LEARNERS")
    print("=" * 70)

    meta_cpu_no_cluster_model = XGBRegressor(**META_PARAMS)
    meta_mem_no_cluster_model = XGBRegressor(**META_PARAMS)

    meta_cpu_ccase_model = XGBRegressor(**META_PARAMS)
    meta_mem_ccase_model = XGBRegressor(**META_PARAMS)

    meta_cpu_no_cluster_model.fit(
        meta_cpu_no_cluster, meta_y_cpu
    )
    meta_mem_no_cluster_model.fit(
        meta_mem_no_cluster, meta_y_mem
    )
    meta_cpu_ccase_model.fit(
        meta_cpu_ccase, meta_y_cpu
    )
    meta_mem_ccase_model.fit(
        meta_mem_ccase, meta_y_mem
    )

    print("All four meta learners completed.")

    # ============================================================
    # STEP 2E — FINAL GLOBAL BASE MODELS
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2E - TRAIN FINAL GLOBAL BASE MODELS")
    print("=" * 70)

    final_lr_cpu = LinearRegression()
    final_lr_mem = LinearRegression()

    final_rf_cpu = RandomForestRegressor(**RF_PARAMS)
    final_rf_mem = RandomForestRegressor(**RF_PARAMS)

    final_xgb_cpu = XGBRegressor(**XGB_PARAMS)
    final_xgb_mem = XGBRegressor(**XGB_PARAMS)

    final_lr_cpu.fit(X_dev_scaled, y_dev_cpu)
    final_lr_mem.fit(X_dev_scaled, y_dev_mem)

    final_rf_cpu.fit(X_dev_scaled, y_dev_cpu)
    final_rf_mem.fit(X_dev_scaled, y_dev_mem)

    final_xgb_cpu.fit(X_dev_scaled, y_dev_cpu)
    final_xgb_mem.fit(X_dev_scaled, y_dev_mem)

    print("Final global base models trained.")
    print("The 20% chronological test data was not used for training.")

    # ============================================================
    # STEP 2F — FINAL TEST PREDICTIONS
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2F - FINAL TEST PREDICTION")
    print("=" * 70)

    test_cpu_lr = final_lr_cpu.predict(X_test_scaled)
    test_mem_lr = final_lr_mem.predict(X_test_scaled)

    test_cpu_rf = final_rf_cpu.predict(X_test_scaled)
    test_mem_rf = final_rf_mem.predict(X_test_scaled)

    test_cpu_xgb = final_xgb_cpu.predict(X_test_scaled)
    test_mem_xgb = final_xgb_mem.predict(X_test_scaled)

    # ------------------------------------------------------------
    # Stacked ensemble without clustering
    # ------------------------------------------------------------
    test_cpu_meta_no_cluster = np.column_stack([
        test_cpu_lr,
        test_cpu_rf,
        test_cpu_xgb,
    ])

    test_mem_meta_no_cluster = np.column_stack([
        test_mem_lr,
        test_mem_rf,
        test_mem_xgb,
    ])

    test_cpu_stacked = (
        meta_cpu_no_cluster_model.predict(
            test_cpu_meta_no_cluster
        )
    )

    test_mem_stacked = (
        meta_mem_no_cluster_model.predict(
            test_mem_meta_no_cluster
        )
    )

    # ------------------------------------------------------------
    # Proposed C-CASE
    # ------------------------------------------------------------
    test_cpu_meta_ccase = np.column_stack([
        test_cpu_lr,
        test_cpu_rf,
        test_cpu_xgb,
        cluster_test_oh,
    ])

    test_mem_meta_ccase = np.column_stack([
        test_mem_lr,
        test_mem_rf,
        test_mem_xgb,
        cluster_test_oh,
    ])

    test_cpu_ccase = (
        meta_cpu_ccase_model.predict(
            test_cpu_meta_ccase
        )
    )

    test_mem_ccase = (
        meta_mem_ccase_model.predict(
            test_mem_meta_ccase
        )
    )

    # Common test-set assertions.
    assert len(y_test_cpu) == EXPECTED_TEST_SIZE
    assert len(y_test_mem) == EXPECTED_TEST_SIZE

    for arr in [
        test_cpu_lr, test_cpu_rf, test_cpu_xgb,
        test_cpu_stacked, test_cpu_ccase,
        test_mem_lr, test_mem_rf, test_mem_xgb,
        test_mem_stacked, test_mem_ccase,
    ]:
        assert len(arr) == EXPECTED_TEST_SIZE

    assert test_cpu_meta_ccase.shape == (
        EXPECTED_TEST_SIZE, 6
    )
    assert test_mem_meta_ccase.shape == (
        EXPECTED_TEST_SIZE, 6
    )

    print("Same test rows used for all models:", EXPECTED_TEST_SIZE)

    # ============================================================
    # STEP 2F-5 — SAVE FINAL PREDICTIONS
    # ============================================================
    prediction_df = pd.DataFrame({
        "y_test_cpu": y_test_cpu,
        "y_test_memory": y_test_mem,
        "LR_cpu": test_cpu_lr,
        "LR_memory": test_mem_lr,
        "RF_cpu": test_cpu_rf,
        "RF_memory": test_mem_rf,
        "XGB_cpu": test_cpu_xgb,
        "XGB_memory": test_mem_xgb,
        "Stacked_cpu": test_cpu_stacked,
        "Stacked_memory": test_mem_stacked,
        "CCASE_cpu": test_cpu_ccase,
        "CCASE_memory": test_mem_ccase,
        "test_cluster": cluster_test,
    })

    prediction_path = os.path.join(
        args.output_dir,
        "Table4_Final_Test_Predictions.csv",
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    print("\nPrediction file saved:")
    print(prediction_path)

    # ============================================================
    # STEP 3A — FINAL TABLE 4 METRICS
    # ============================================================
    results = []

    model_predictions = [
        (
            "Linear Regression",
            test_cpu_lr,
            test_mem_lr,
        ),
        (
            "Random Forest",
            test_cpu_rf,
            test_mem_rf,
        ),
        (
            "XGBoost",
            test_cpu_xgb,
            test_mem_xgb,
        ),
        (
            "Stacked Ensemble (No Clustering)",
            test_cpu_stacked,
            test_mem_stacked,
        ),
        (
            "Proposed C-CASE",
            test_cpu_ccase,
            test_mem_ccase,
        ),
    ]

    for name, cpu_pred, mem_pred in model_predictions:
        cpu = calculate_metrics(y_test_cpu, cpu_pred)
        memory = calculate_metrics(y_test_mem, mem_pred)

        results.append({
            "Model": name,
            "CPU MSE": cpu["MSE"],
            "CPU RMSE": cpu["RMSE"],
            "CPU MAE": cpu["MAE"],
            "CPU R2": cpu["R2"],
            "Memory MSE": memory["MSE"],
            "Memory RMSE": memory["RMSE"],
            "Memory MAE": memory["MAE"],
            "Memory R2": memory["R2"],
        })

    table4 = pd.DataFrame(results)

    print("\n" + "=" * 130)
    print("FINAL TABLE 4 RESULTS")
    print("=" * 130)
    print(
        table4.to_string(
            index=False,
            float_format=lambda x: f"{x:.10e}",
        )
    )

    # ============================================================
    # STEP 3C — REVIEWER 1 CONSISTENCY CHECK
    # ============================================================
    cpu_variance = np.var(y_test_cpu, ddof=0)
    memory_variance = np.var(y_test_mem, ddof=0)

    check_results = []

    for _, row in table4.iterrows():
        cpu_implied_variance = (
            row["CPU MSE"] / (1 - row["CPU R2"])
        )
        memory_implied_variance = (
            row["Memory MSE"] / (1 - row["Memory R2"])
        )

        check_results.append({
            "Model": row["Model"],
            "CPU Actual Variance": cpu_variance,
            "CPU Implied Variance": cpu_implied_variance,
            "Memory Actual Variance": memory_variance,
            "Memory Implied Variance": memory_implied_variance,
        })

    variance_check = pd.DataFrame(check_results)

    # ============================================================
    # STEP 3D — SAVE TABLE 4
    # ============================================================
    table4_path = os.path.join(
        args.output_dir,
        "Table4_Final_Common_Test_Comparison.csv",
    )

    table4.to_csv(table4_path, index=False)

    variance_path = os.path.join(
        args.output_dir,
        "Table4_Reviewer_Consistency_Check.csv",
    )

    variance_check.to_csv(variance_path, index=False)

    print("\nResults saved:")
    print(prediction_path)
    print(table4_path)
    print(variance_path)
    print(
        os.path.join(
            args.output_dir,
            "final_preprocessing.pkl",
        )
    )

    print("\n" + "=" * 70)
    print("C-CASE TRAINING COMPLETED")
    print("=" * 70)
    print("K-Means K              :", K)
    print("Random state           :", RANDOM_STATE)
    print("Temporal OOF folds     :", N_SPLITS)
    print("Base models            : LR + RF + XGB")
    print("Cluster-specific models: NO")
    print("C-CASE meta-features   : 3 predictions + 3 cluster variables")
    print("Final test set         :", len(y_test_cpu), "observations")


if __name__ == "__main__":
    main()
