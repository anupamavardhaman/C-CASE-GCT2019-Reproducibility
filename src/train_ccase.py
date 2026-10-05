"""
C-CASE Training and Evaluation
================================

Reproducibility implementation corresponding to:
K=3_Table_4_final_results_of_all_models(2).ipynb

Modelling approach:
- K-Means clustering with K=3
- Global Linear Regression, Random Forest and XGBoost base learners
- 3-fold temporal OOF predictions
- Cluster one-hot context added only to the C-CASE meta learner
- XGBoost meta learner
- Final base models refitted on the complete 80% development data
- Evaluation on the untouched chronological 20% test set

IMPORTANT:
This implementation does NOT train separate base models for C0, C1 and C2.
Cluster labels are used as contextual meta-features only.

The numerical modelling logic follows the supplied K=3 Table 4 notebook.
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from xgboost import XGBRegressor


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_CHECKPOINT = "checkpoints/ccase_step15_k3.pkl"
DEFAULT_OUTPUT_DIR = "results"

TARGET_CPU = "future_cpu"
TARGET_MEMORY = "future_memory"

K = 3
RANDOM_STATE = 42
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


# ============================================================
# MODEL FACTORIES
# ============================================================

def create_base_models():
    """Create one LR, RF and XGB base model."""
    lr = LinearRegression()
    rf = RandomForestRegressor(**RF_PARAMS)
    xgb = XGBRegressor(**XGB_PARAMS)
    return lr, rf, xgb


def create_meta_model():
    """Create the XGBoost meta-regressor."""
    return XGBRegressor(**META_PARAMS)


# ============================================================
# DATA PREPARATION
# ============================================================

def load_and_prepare(checkpoint_path):
    """
    Load the preprocessing checkpoint and reproduce the Table 4
    notebook's scaling and K=3 clustering step.

    The supplied notebook uses train_data/test_data and refits the
    StandardScaler and K-Means using the development data only.
    """
    print("=" * 70)
    print("STEP 1 - LOAD CHECKPOINT AND PREPARE DATA")
    print("=" * 70)

    checkpoint = joblib.load(checkpoint_path)

    train_data = checkpoint["train_data"].copy()
    test_data = checkpoint["test_data"].copy()
    feature_cols = checkpoint["feature_cols"]

    print("Development data shape:", train_data.shape)
    print("Test data shape       :", test_data.shape)
    print("Number of features    :", len(feature_cols))

    X_dev = train_data[feature_cols].copy()
    y_dev_cpu = train_data[TARGET_CPU].values
    y_dev_mem = train_data[TARGET_MEMORY].values

    X_test = test_data[feature_cols].copy()
    y_test_cpu = test_data[TARGET_CPU].values
    y_test_mem = test_data[TARGET_MEMORY].values

    assert len(X_test) == EXPECTED_TEST_SIZE, (
        f"Expected {EXPECTED_TEST_SIZE} test rows, found {len(X_test)}"
    )

    assert len(X_test) == len(y_test_cpu)
    assert len(X_test) == len(y_test_mem)

    assert X_dev.isna().sum().sum() == 0, "Missing values in development features."
    assert X_test.isna().sum().sum() == 0, "Missing values in test features."

    # Same as supplied Table 4 notebook:
    # scaler fitted on development data only.
    scaler_final = StandardScaler()
    X_dev_scaled = scaler_final.fit_transform(X_dev)
    X_test_scaled = scaler_final.transform(X_test)

    # Same K=3 clustering used in the supplied notebook.
    kmeans_final = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    cluster_dev = kmeans_final.fit_predict(X_dev_scaled)
    cluster_test = kmeans_final.predict(X_test_scaled)

    # Three one-hot cluster context variables.
    cluster_dev_oh = np.eye(K)[cluster_dev]
    cluster_test_oh = np.eye(K)[cluster_test]

    print("\n===== K-MEANS =====")
    print("K =", K)
    print("Random state =", RANDOM_STATE)

    print("\nDevelopment cluster distribution:")
    print(pd.Series(cluster_dev).value_counts().sort_index())

    print("\nTest cluster distribution:")
    print(pd.Series(cluster_test).value_counts().sort_index())

    print("\nCluster context development:", cluster_dev_oh.shape)
    print("Cluster context test       :", cluster_test_oh.shape)

    return {
        "train_data": train_data,
        "test_data": test_data,
        "feature_cols": feature_cols,
        "X_dev_scaled": X_dev_scaled,
        "X_test_scaled": X_test_scaled,
        "y_dev_cpu": y_dev_cpu,
        "y_dev_mem": y_dev_mem,
        "y_test_cpu": y_test_cpu,
        "y_test_mem": y_test_mem,
        "cluster_dev": cluster_dev,
        "cluster_test": cluster_test,
        "cluster_dev_oh": cluster_dev_oh,
        "cluster_test_oh": cluster_test_oh,
        "scaler": scaler_final,
        "kmeans": kmeans_final,
    }


# ============================================================
# TEMPORAL OOF PREDICTIONS
# ============================================================

def generate_oof_predictions(data):
    """Generate 3-fold chronological OOF predictions for LR/RF/XGB."""

    print("\n" + "=" * 70)
    print("STEP 2B - LEAKAGE-FREE TEMPORAL OOF PREDICTIONS")
    print("=" * 70)

    X_dev_scaled = data["X_dev_scaled"]
    y_dev_cpu = data["y_dev_cpu"]
    y_dev_mem = data["y_dev_mem"]

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

        X_train_inner = X_dev_scaled[train_idx]
        X_val_inner = X_dev_scaled[val_idx]

        # ---------------- CPU ----------------
        lr_cpu, rf_cpu, xgb_cpu = create_base_models()

        lr_cpu.fit(X_train_inner, y_dev_cpu[train_idx])
        rf_cpu.fit(X_train_inner, y_dev_cpu[train_idx])
        xgb_cpu.fit(X_train_inner, y_dev_cpu[train_idx])

        oof_cpu_lr[val_idx] = lr_cpu.predict(X_val_inner)
        oof_cpu_rf[val_idx] = rf_cpu.predict(X_val_inner)
        oof_cpu_xgb[val_idx] = xgb_cpu.predict(X_val_inner)

        # ---------------- MEMORY ----------------
        lr_mem, rf_mem, xgb_mem = create_base_models()

        lr_mem.fit(X_train_inner, y_dev_mem[train_idx])
        rf_mem.fit(X_train_inner, y_dev_mem[train_idx])
        xgb_mem.fit(X_train_inner, y_dev_mem[train_idx])

        oof_mem_lr[val_idx] = lr_mem.predict(X_val_inner)
        oof_mem_rf[val_idx] = rf_mem.predict(X_val_inner)
        oof_mem_xgb[val_idx] = xgb_mem.predict(X_val_inner)

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

    assert oof_mask.sum() > 0

    return {
        "oof_cpu_lr": oof_cpu_lr,
        "oof_cpu_rf": oof_cpu_rf,
        "oof_cpu_xgb": oof_cpu_xgb,
        "oof_mem_lr": oof_mem_lr,
        "oof_mem_rf": oof_mem_rf,
        "oof_mem_xgb": oof_mem_xgb,
        "oof_mask": oof_mask,
    }


# ============================================================
# META-TRAINING DATA
# ============================================================

def create_meta_datasets(data, oof):
    """Create no-clustering and C-CASE meta-training matrices."""

    print("\n" + "=" * 70)
    print("STEP 2C - CREATE META-TRAINING DATASETS")
    print("=" * 70)

    mask = oof["oof_mask"]
    cluster_dev_oh = data["cluster_dev_oh"]

    meta_cpu_base = np.column_stack([
        oof["oof_cpu_lr"][mask],
        oof["oof_cpu_rf"][mask],
        oof["oof_cpu_xgb"][mask],
    ])

    meta_mem_base = np.column_stack([
        oof["oof_mem_lr"][mask],
        oof["oof_mem_rf"][mask],
        oof["oof_mem_xgb"][mask],
    ])

    meta_cluster = cluster_dev_oh[mask]

    # Non-clustered stacked ensemble:
    meta_cpu_no_cluster = meta_cpu_base.copy()
    meta_mem_no_cluster = meta_mem_base.copy()

    # C-CASE:
    # 3 base predictions + 3 K-Means context variables.
    meta_cpu_ccase = np.column_stack([
        meta_cpu_base,
        meta_cluster,
    ])

    meta_mem_ccase = np.column_stack([
        meta_mem_base,
        meta_cluster,
    ])

    meta_y_cpu = data["y_dev_cpu"][mask]
    meta_y_mem = data["y_dev_mem"][mask]

    print("No-clustering CPU meta features :", meta_cpu_no_cluster.shape)
    print("No-clustering Memory meta features:", meta_mem_no_cluster.shape)
    print("C-CASE CPU meta features        :", meta_cpu_ccase.shape)
    print("C-CASE Memory meta features     :", meta_mem_ccase.shape)

    assert meta_cpu_no_cluster.shape[1] == 3
    assert meta_mem_no_cluster.shape[1] == 3
    assert meta_cpu_ccase.shape[1] == 6
    assert meta_mem_ccase.shape[1] == 6

    return {
        "meta_cpu_no_cluster": meta_cpu_no_cluster,
        "meta_mem_no_cluster": meta_mem_no_cluster,
        "meta_cpu_ccase": meta_cpu_ccase,
        "meta_mem_ccase": meta_mem_ccase,
        "meta_y_cpu": meta_y_cpu,
        "meta_y_mem": meta_y_mem,
    }


# ============================================================
# META LEARNERS
# ============================================================

def train_meta_learners(meta):
    """Train the no-clustering and C-CASE XGBoost meta learners."""

    print("\n" + "=" * 70)
    print("STEP 2D - TRAIN META LEARNERS")
    print("=" * 70)

    models = {}

    # No-clustering stacked ensemble
    models["meta_cpu_no_cluster"] = create_meta_model()
    models["meta_mem_no_cluster"] = create_meta_model()

    models["meta_cpu_no_cluster"].fit(
        meta["meta_cpu_no_cluster"],
        meta["meta_y_cpu"],
    )

    models["meta_mem_no_cluster"].fit(
        meta["meta_mem_no_cluster"],
        meta["meta_y_mem"],
    )

    # Proposed C-CASE
    models["meta_cpu_ccase"] = create_meta_model()
    models["meta_mem_ccase"] = create_meta_model()

    models["meta_cpu_ccase"].fit(
        meta["meta_cpu_ccase"],
        meta["meta_y_cpu"],
    )

    models["meta_mem_ccase"].fit(
        meta["meta_mem_ccase"],
        meta["meta_y_mem"],
    )

    print("All four meta learners completed.")

    return models


# ============================================================
# FINAL BASE MODELS
# ============================================================

def train_final_base_models(data):
    """Refit global LR/RF/XGB on all development data."""

    print("\n" + "=" * 70)
    print("STEP 2E - TRAIN FINAL GLOBAL BASE MODELS")
    print("=" * 70)

    X_dev_scaled = data["X_dev_scaled"]
    y_dev_cpu = data["y_dev_cpu"]
    y_dev_mem = data["y_dev_mem"]

    final = {}

    final["lr_cpu"], final["rf_cpu"], final["xgb_cpu"] = create_base_models()
    final["lr_mem"], final["rf_mem"], final["xgb_mem"] = create_base_models()

    final["lr_cpu"].fit(X_dev_scaled, y_dev_cpu)
    final["rf_cpu"].fit(X_dev_scaled, y_dev_cpu)
    final["xgb_cpu"].fit(X_dev_scaled, y_dev_cpu)

    final["lr_mem"].fit(X_dev_scaled, y_dev_mem)
    final["rf_mem"].fit(X_dev_scaled, y_dev_mem)
    final["xgb_mem"].fit(X_dev_scaled, y_dev_mem)

    print("Final global base models trained.")
    print("The 20% chronological test data was not used for training.")

    return final


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

def generate_test_predictions(data, meta_models, final_models):
    """Generate baseline, stacked and C-CASE predictions."""

    print("\n" + "=" * 70)
    print("STEP 2F - FINAL TEST PREDICTION")
    print("=" * 70)

    X_test_scaled = data["X_test_scaled"]
    cluster_test_oh = data["cluster_test_oh"]

    test_cpu_lr = final_models["lr_cpu"].predict(X_test_scaled)
    test_mem_lr = final_models["lr_mem"].predict(X_test_scaled)

    test_cpu_rf = final_models["rf_cpu"].predict(X_test_scaled)
    test_mem_rf = final_models["rf_mem"].predict(X_test_scaled)

    test_cpu_xgb = final_models["xgb_cpu"].predict(X_test_scaled)
    test_mem_xgb = final_models["xgb_mem"].predict(X_test_scaled)

    # No-clustering stacked ensemble
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

    test_cpu_stacked = meta_models["meta_cpu_no_cluster"].predict(
        test_cpu_meta_no_cluster
    )

    test_mem_stacked = meta_models["meta_mem_no_cluster"].predict(
        test_mem_meta_no_cluster
    )

    # C-CASE: add K=3 cluster context
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

    test_cpu_ccase = meta_models["meta_cpu_ccase"].predict(
        test_cpu_meta_ccase
    )

    test_mem_ccase = meta_models["meta_mem_ccase"].predict(
        test_mem_meta_ccase
    )

    test_size = len(data["y_test_cpu"])

    assert test_size == EXPECTED_TEST_SIZE
    assert test_cpu_meta_ccase.shape == (test_size, 6)
    assert test_mem_meta_ccase.shape == (test_size, 6)

    print("Same test rows used for all models:", test_size)

    return {
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
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2


def calculate_table4(data, predictions):
    """Calculate MSE, RMSE, MAE and R2 directly from test arrays."""

    y_cpu = data["y_test_cpu"]
    y_mem = data["y_test_mem"]

    model_pairs = [
        ("Linear Regression", "LR_cpu", "LR_memory"),
        ("Random Forest", "RF_cpu", "RF_memory"),
        ("XGBoost", "XGB_cpu", "XGB_memory"),
        ("Stacked Ensemble (No Clustering)", "Stacked_cpu", "Stacked_memory"),
        ("Proposed C-CASE", "CCASE_cpu", "CCASE_memory"),
    ]

    rows = []

    for name, cpu_key, mem_key in model_pairs:
        cpu = calculate_metrics(y_cpu, predictions[cpu_key])
        mem = calculate_metrics(y_mem, predictions[mem_key])

        rows.append({
            "Model": name,
            "CPU MSE": cpu[0],
            "CPU RMSE": cpu[1],
            "CPU MAE": cpu[2],
            "CPU R2": cpu[3],
            "Memory MSE": mem[0],
            "Memory RMSE": mem[1],
            "Memory MAE": mem[2],
            "Memory R2": mem[3],
        })

    return pd.DataFrame(rows)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(data, predictions, table4, output_dir, scaler, kmeans):
    os.makedirs(output_dir, exist_ok=True)

    # Same ground-truth/prediction structure used for reviewer verification.
    prediction_df = pd.DataFrame({
        "y_test_cpu": data["y_test_cpu"],
        "y_test_memory": data["y_test_mem"],
        "LR_cpu": predictions["LR_cpu"],
        "LR_memory": predictions["LR_memory"],
        "RF_cpu": predictions["RF_cpu"],
        "RF_memory": predictions["RF_memory"],
        "XGB_cpu": predictions["XGB_cpu"],
        "XGB_memory": predictions["XGB_memory"],
        "Stacked_cpu": predictions["Stacked_cpu"],
        "Stacked_memory": predictions["Stacked_memory"],
        "CCASE_cpu": predictions["CCASE_cpu"],
        "CCASE_memory": predictions["CCASE_memory"],
        "test_cluster": data["cluster_test"],
    })

    prediction_path = os.path.join(
        output_dir, "Table4_Final_Test_Predictions.csv"
    )
    prediction_df.to_csv(prediction_path, index=False)

    table4_path = os.path.join(
        output_dir, "Table4_Final_Common_Test_Comparison.csv"
    )
    table4.to_csv(table4_path, index=False)

    # Reviewer 1 consistency check:
    cpu_variance = np.var(data["y_test_cpu"], ddof=0)
    memory_variance = np.var(data["y_test_mem"], ddof=0)

    check_results = []

    for _, row in table4.iterrows():
        check_results.append({
            "Model": row["Model"],
            "CPU Actual Variance": cpu_variance,
            "CPU Implied Variance": (
                row["CPU MSE"] / (1 - row["CPU R2"])
                if row["CPU R2"] != 1 else np.nan
            ),
            "Memory Actual Variance": memory_variance,
            "Memory Implied Variance": (
                row["Memory MSE"] / (1 - row["Memory R2"])
                if row["Memory R2"] != 1 else np.nan
            ),
        })

    variance_check = pd.DataFrame(check_results)

    variance_path = os.path.join(
        output_dir, "Table4_Reviewer_Consistency_Check.csv"
    )
    variance_check.to_csv(variance_path, index=False)

    # Save the fitted preprocessing objects used by this training run.
    preprocessing_path = os.path.join(
        output_dir, "final_preprocessing.pkl"
    )
    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": kmeans,
            "K": K,
            "random_state": RANDOM_STATE,
            "feature_cols": data["feature_cols"],
        },
        preprocessing_path,
    )

    print("\nResults saved:")
    print(" ", prediction_path)
    print(" ", table4_path)
    print(" ", variance_path)
    print(" ", preprocessing_path)

    return prediction_df, variance_check


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the K=3 C-CASE model."
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to ccase_step15_k3.pkl generated by preprocess_data.py.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for predictions, metrics and preprocessing objects.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run src/preprocess_data.py first."
        )

    data = load_and_prepare(args.checkpoint)

    oof = generate_oof_predictions(data)

    meta = create_meta_datasets(data, oof)

    meta_models = train_meta_learners(meta)

    final_models = train_final_base_models(data)

    predictions = generate_test_predictions(
        data,
        meta_models,
        final_models,
    )

    table4 = calculate_table4(data, predictions)

    print("\n" + "=" * 130)
    print("FINAL TABLE 4 RESULTS")
    print("=" * 130)
    print(
        table4.to_string(
            index=False,
            float_format=lambda x: f"{x:.10e}",
        )
    )

    save_results(
        data,
        predictions,
        table4,
        args.output_dir,
        data["scaler"],
        data["kmeans"],
    )

    print("\n" + "=" * 70)
    print("C-CASE TRAINING COMPLETED")
    print("=" * 70)
    print("K-Means K              :", K)
    print("Random state           :", RANDOM_STATE)
    print("Temporal OOF folds     :", N_SPLITS)
    print("Base models             : LR + RF + XGB")
    print("Cluster-specific models : NO")
    print("C-CASE meta-features   : 3 predictions + 3 cluster variables")
    print("Final test set         :", EXPECTED_TEST_SIZE, "observations")


if __name__ == "__main__":
    main()
