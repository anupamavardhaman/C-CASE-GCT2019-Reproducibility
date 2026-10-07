# ============================================================
# ABLATION A0: GLOBAL XGBOOST
# Same outer folds as Stage 4B
# No clustering
# No stacking
# No meta learner
# ============================================================

import os
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# 1. LOAD STAGE 4B CHECKPOINT
# ============================================================

CHECKPOINT_PATH = (
    "/content/drive/MyDrive/C_CASE/checkpoints/ccase_step15.pkl"
)

checkpoint = joblib.load(CHECKPOINT_PATH)

model_data = checkpoint["model_data"]
feature_cols = checkpoint["feature_cols"]

print("Checkpoint loaded successfully.")
print("Model data shape:", model_data.shape)
print("Number of features:", len(feature_cols))


# ============================================================
# 2. USE ONLY DEVELOPMENT DATA
#    Same data used for Stage 4B outer CV
# ============================================================

train_data = checkpoint["train_data"].copy()

print("\nDevelopment data shape:", train_data.shape)

# IMPORTANT:
# We DO NOT use the final 20% test_data here.
# The final test set remains untouched.


# ============================================================
# 3. VERIFY FEATURES
# ============================================================

print("\nFeature columns:")
for i, col in enumerate(feature_cols, 1):
    print(f"{i:2d}. {col}")

X_all = train_data[feature_cols]
y_cpu_all = train_data["future_cpu"]
y_memory_all = train_data["future_memory"]
time_all = train_data["time_bucket"]

print("\nX shape:", X_all.shape)
print("CPU target shape:", y_cpu_all.shape)
print("Memory target shape:", y_memory_all.shape)


# ============================================================
# 4. EXACT SAME STAGE 4B OUTER FOLDS
# ============================================================

outer_folds = [
    {
        "name": "Fold1",
        "train_end": 658500000000,
        "val_start": 659100000000,
        "val_end": 1314900000000
    },
    {
        "name": "Fold2",
        "train_end": 1314600000000,
        "val_start": 1315200000000,
        "val_end": 1971000000000
    },
    {
        "name": "Fold3",
        "train_end": 1970700000000,
        "val_start": 1971300000000,
        "val_end": 2627100000000
    }
]


# ============================================================
# 5. SAME XGBOOST HYPERPARAMETERS AS STAGE 4B
# ============================================================

xgb_params = {
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
    "random_state": 42
}


# ============================================================
# 6. FUNCTION TO CALCULATE METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


# ============================================================
# 7. STORAGE FOR RESULTS
# ============================================================

results = []

predictions = {}


# ============================================================
# 8. RUN EXACT SAME THREE OUTER FOLDS
# ============================================================

for fold in outer_folds:

    fold_name = fold["name"]

    print("\n" + "=" * 70)
    print(f"{fold_name} — A0 GLOBAL XGBOOST")
    print("=" * 70)

    # --------------------------------------------------------
    # Outer training data
    # --------------------------------------------------------

    train_mask = (
        time_all <= fold["train_end"]
    )

    # --------------------------------------------------------
    # Outer validation data
    # --------------------------------------------------------

    val_mask = (
        (time_all >= fold["val_start"]) &
        (time_all <= fold["val_end"])
    )

    X_train = X_all.loc[train_mask]
    X_val = X_all.loc[val_mask]

    y_cpu_train = y_cpu_all.loc[train_mask]
    y_cpu_val = y_cpu_all.loc[val_mask]

    y_memory_train = y_memory_all.loc[train_mask]
    y_memory_val = y_memory_all.loc[val_mask]

    print("Training samples :", len(X_train))
    print("Validation samples:", len(X_val))

    # --------------------------------------------------------
    # Check that folds match Stage 4B
    # --------------------------------------------------------

    print(
        "Training time range:",
        time_all.loc[train_mask].min(),
        "→",
        time_all.loc[train_mask].max()
    )

    print(
        "Validation time range:",
        time_all.loc[val_mask].min(),
        "→",
        time_all.loc[val_mask].max()
    )


    # ========================================================
    # CPU XGBOOST
    # ========================================================

    print("\nTraining CPU XGBoost...")

    cpu_model = XGBRegressor(
        **xgb_params
    )

    cpu_model.fit(
        X_train,
        y_cpu_train
    )

    cpu_pred = cpu_model.predict(X_val)

    cpu_metrics = calculate_metrics(
        y_cpu_val,
        cpu_pred
    )


    # ========================================================
    # MEMORY XGBOOST
    # ========================================================

    print("Training Memory XGBoost...")

    memory_model = XGBRegressor(
        **xgb_params
    )

    memory_model.fit(
        X_train,
        y_memory_train
    )

    memory_pred = memory_model.predict(X_val)

    memory_metrics = calculate_metrics(
        y_memory_val,
        memory_pred
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({
        "Fold": fold_name,

        "CPU_MSE": cpu_metrics["MSE"],
        "CPU_RMSE": cpu_metrics["RMSE"],
        "CPU_MAE": cpu_metrics["MAE"],
        "CPU_R2": cpu_metrics["R2"],

        "Memory_MSE": memory_metrics["MSE"],
        "Memory_RMSE": memory_metrics["RMSE"],
        "Memory_MAE": memory_metrics["MAE"],
        "Memory_R2": memory_metrics["R2"]
    })


    # ========================================================
    # STORE PREDICTIONS
    # ========================================================

    predictions[fold_name] = {
        "cpu_true": y_cpu_val.to_numpy(),
        "cpu_pred": cpu_pred,
        "memory_true": y_memory_val.to_numpy(),
        "memory_pred": memory_pred
    }


    # ========================================================
    # PRINT FOLD RESULTS
    # ========================================================

    print("\nCPU RESULTS")
    print("-" * 40)

    print(f"MSE  : {cpu_metrics['MSE']:.10e}")
    print(f"RMSE : {cpu_metrics['RMSE']:.10f}")
    print(f"MAE  : {cpu_metrics['MAE']:.10f}")
    print(f"R²   : {cpu_metrics['R2']:.10f}")

    print("\nMEMORY RESULTS")
    print("-" * 40)

    print(f"MSE  : {memory_metrics['MSE']:.10e}")
    print(f"RMSE : {memory_metrics['RMSE']:.10f}")
    print(f"MAE  : {memory_metrics['MAE']:.10f}")
    print(f"R²   : {memory_metrics['R2']:.10f}")


    # --------------------------------------------------------
    # Memory cleanup
    # --------------------------------------------------------

    del (
        X_train,
        X_val,
        y_cpu_train,
        y_cpu_val,
        y_memory_train,
        y_memory_val,
        cpu_model,
        memory_model,
        cpu_pred,
        memory_pred
    )

    gc.collect()


# ============================================================
# 9. CONVERT RESULTS TO DATAFRAME
# ============================================================

a0_results = pd.DataFrame(results)

print("\n\n" + "=" * 80)
print("A0 — GLOBAL XGBOOST RESULTS")
print("=" * 80)

display(a0_results)


# ============================================================
# 10. MEAN ± SD ACROSS OUTER FOLDS
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "CPU MSE",
        "CPU RMSE",
        "CPU MAE",
        "CPU R²",
        "Memory MSE",
        "Memory RMSE",
        "Memory MAE",
        "Memory R²"
    ],

    "Mean": [
        a0_results["CPU_MSE"].mean(),
        a0_results["CPU_RMSE"].mean(),
        a0_results["CPU_MAE"].mean(),
        a0_results["CPU_R2"].mean(),

        a0_results["Memory_MSE"].mean(),
        a0_results["Memory_RMSE"].mean(),
        a0_results["Memory_MAE"].mean(),
        a0_results["Memory_R2"].mean()
    ],

    "SD": [
        a0_results["CPU_MSE"].std(ddof=1),
        a0_results["CPU_RMSE"].std(ddof=1),
        a0_results["CPU_MAE"].std(ddof=1),
        a0_results["CPU_R2"].std(ddof=1),

        a0_results["Memory_MSE"].std(ddof=1),
        a0_results["Memory_RMSE"].std(ddof=1),
        a0_results["Memory_MAE"].std(ddof=1),
        a0_results["Memory_R2"].std(ddof=1)
    ]
})

print("\n" + "=" * 80)
print("A0 — MEAN ± SD")
print("=" * 80)

display(summary)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

OUTPUT_DIR = "/content/drive/MyDrive/C_CASE/ablation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

a0_csv = os.path.join(
    OUTPUT_DIR,
    "A0_Global_XGBoost_outer_fold_results.csv"
)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "A0_Global_XGBoost_mean_sd.csv"
)

a0_results.to_csv(
    a0_csv,
    index=False
)

summary.to_csv(
    summary_csv,
    index=False
)

print("\nResults saved:")
print(a0_csv)
print(summary_csv)
