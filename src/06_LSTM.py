# ============================================================
# CONTROLLED LSTM BASELINE
# Google Cluster Trace 2019
#
# Same experimental setup as C-CASE:
#   - Same checkpoint / dataset
#   - Same chronological 80:20 split
#   - 10 historical 5-minute observations
#   - 50-minute history
#   - 5-minute prediction horizon
#   - CPU + Memory prediction
#   - MSE, RMSE, MAE, R2
# ============================================================

import os
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

CHECKPOINT_PATH = (
    "/content/drive/MyDrive/C_CASE/checkpoints/"
    "ccase_step15.pkl"
)

OUTPUT_DIR = (
    "/content/drive/MyDrive/C_CASE/"
    "LSTM_baseline"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("CONTROLLED LSTM BASELINE")
print("=" * 70)

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))


# ============================================================
# 2. LOAD THE SAME C-CASE CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = joblib.load(CHECKPOINT_PATH)

train_data = checkpoint["train_data"].copy()
test_data = checkpoint["test_data"].copy()

feature_cols = checkpoint["feature_cols"]

print("Development rows :", len(train_data))
print("Test rows        :", len(test_data))
print("Feature count    :", len(feature_cols))

# Same test size used in your C-CASE Table 4
assert len(test_data) == 283864, (
    f"Expected 283,864 test rows, "
    f"found {len(test_data)}"
)

print("\n✓ Same 283,864 test observations loaded.")


# ============================================================
# 3. TARGETS
# ============================================================

TARGETS = [
    "future_cpu",
    "future_memory"
]

y_train_raw = train_data[TARGETS].values.astype("float32")
y_test_raw  = test_data[TARGETS].values.astype("float32")

print("\nTraining targets:", y_train_raw.shape)
print("Testing targets :", y_test_raw.shape)


# ============================================================
# 4. TEMPORAL INPUT
#
# 10 previous observations + current observation
#
# Order:
# t-10, t-9, ..., t-2, t-1, t
#
# Each timestep contains:
#   CPU
#   Memory
# ============================================================

CPU_HISTORY = [
    "cpu_lag_10",
    "cpu_lag_9",
    "cpu_lag_8",
    "cpu_lag_7",
    "cpu_lag_6",
    "cpu_lag_5",
    "cpu_lag_4",
    "cpu_lag_3",
    "cpu_lag_2",
    "cpu_lag_1",
    "cpu_usage"
]

MEM_HISTORY = [
    "memory_lag_10",
    "memory_lag_9",
    "memory_lag_8",
    "memory_lag_7",
    "memory_lag_6",
    "memory_lag_5",
    "memory_lag_4",
    "memory_lag_3",
    "memory_lag_2",
    "memory_lag_1",
    "memory_usage"
]

print("\nLSTM historical sequence:")
print("Number of time steps:", len(CPU_HISTORY))
print("History covered     : 50 minutes")
print("Forecast horizon    : 5 minutes")


# ============================================================
# 5. CONTEXT FEATURES
#
# These are the same engineered/context features used by
# the C-CASE preprocessing.
# ============================================================

CONTEXT_FEATURES = [
    "cpu_delta",
    "memory_delta",
    "rolling_cpu_mean",
    "rolling_cpu_std",
    "rolling_memory_mean",
    "rolling_memory_std",
    "cpu_mem_ratio",
    "time_sin",
    "time_cos"
]

print("\nContext features:", len(CONTEXT_FEATURES))
print(CONTEXT_FEATURES)


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    CPU_HISTORY
    + MEM_HISTORY
    + CONTEXT_FEATURES
    + TARGETS
)

missing_train = [
    c for c in required_columns
    if c not in train_data.columns
]

missing_test = [
    c for c in required_columns
    if c not in test_data.columns
]

if missing_train:
    raise ValueError(
        "Missing training columns: "
        + str(missing_train)
    )

if missing_test:
    raise ValueError(
        "Missing testing columns: "
        + str(missing_test)
    )

print("\n✓ All required columns are available.")


# ============================================================
# 7. CREATE TEMPORAL INPUT ARRAYS
# ============================================================

def create_lstm_sequence(data):

    cpu = data[CPU_HISTORY].to_numpy(
        dtype=np.float32
    )

    memory = data[MEM_HISTORY].to_numpy(
        dtype=np.float32
    )

    # Shape:
    # (samples, 11, 2)

    sequence = np.stack(
        [cpu, memory],
        axis=-1
    )

    context = data[
        CONTEXT_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    return sequence, context


print("\nCreating LSTM sequences...")

X_train_seq, X_train_context = (
    create_lstm_sequence(train_data)
)

X_test_seq, X_test_context = (
    create_lstm_sequence(test_data)
)

print("\nSequence shapes:")
print("X_train_seq     :", X_train_seq.shape)
print("X_test_seq      :", X_test_seq.shape)

print("\nContext shapes:")
print("X_train_context :", X_train_context.shape)
print("X_test_context  :", X_test_context.shape)


# Expected:
# train: (development_rows, 11, 2)
# test : (283864, 11, 2)


# ============================================================
# 8. LEAKAGE-SAFE SCALING
#
# Fit ONLY on development data.
# ============================================================

print("\nScaling inputs and targets...")


# ------------------------------------------------------------
# Sequence scaling
# ------------------------------------------------------------

cpu_scaler = StandardScaler()
memory_scaler = StandardScaler()

# Flatten CPU historical values
cpu_train_values = X_train_seq[:, :, 0].reshape(-1, 1)
mem_train_values = X_train_seq[:, :, 1].reshape(-1, 1)

cpu_scaler.fit(cpu_train_values)
memory_scaler.fit(mem_train_values)

X_train_seq[:, :, 0] = (
    cpu_scaler.transform(
        cpu_train_values
    ).reshape(
        X_train_seq.shape[0],
        X_train_seq.shape[1]
    )
)

X_test_seq[:, :, 0] = (
    cpu_scaler.transform(
        X_test_seq[:, :, 0].reshape(-1, 1)
    ).reshape(
        X_test_seq.shape[0],
        X_test_seq.shape[1]
    )
)

X_train_seq[:, :, 1] = (
    memory_scaler.transform(
        mem_train_values
    ).reshape(
        X_train_seq.shape[0],
        X_train_seq.shape[1]
    )
)

X_test_seq[:, :, 1] = (
    memory_scaler.transform(
        X_test_seq[:, :, 1].reshape(-1, 1)
    ).reshape(
        X_test_seq.shape[0],
        X_test_seq.shape[1]
    )
)


# ------------------------------------------------------------
# Context scaling
# ------------------------------------------------------------

context_scaler = StandardScaler()

X_train_context = context_scaler.fit_transform(
    X_train_context
).astype("float32")

X_test_context = context_scaler.transform(
    X_test_context
).astype("float32")


# ------------------------------------------------------------
# Target scaling
# ------------------------------------------------------------

target_scaler = StandardScaler()

y_train = target_scaler.fit_transform(
    y_train_raw
).astype("float32")

y_test = target_scaler.transform(
    y_test_raw
).astype("float32")


print("✓ Scalers fitted using development data only.")


# ============================================================
# 9. FINAL ARRAY CHECK
# ============================================================

print("\n===== ARRAY CHECK =====")

print("Sequence training :", X_train_seq.shape)
print("Sequence testing  :", X_test_seq.shape)

print("Context training  :", X_train_context.shape)
print("Context testing   :", X_test_context.shape)

print("Target training   :", y_train.shape)
print("Target testing    :", y_test.shape)

assert X_train_seq.ndim == 3
assert X_test_seq.ndim == 3

assert X_train_seq.shape[1] == 11
assert X_train_seq.shape[2] == 2

assert X_test_seq.shape[0] == 283864

assert not np.isnan(X_train_seq).any()
assert not np.isnan(X_test_seq).any()

assert not np.isnan(X_train_context).any()
assert not np.isnan(X_test_context).any()

print("\n✓ Data checks passed.")


# ============================================================
# 10. BUILD LSTM MODEL
# ============================================================

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Concatenate
)

print("\nBuilding LSTM model...")


# Temporal input
sequence_input = Input(
    shape=(11, 2),
    name="historical_cpu_memory"
)

x1 = LSTM(
    32,
    return_sequences=False,
    name="lstm_layer"
)(sequence_input)

x1 = Dropout(
    0.10
)(x1)


# Context input
context_input = Input(
    shape=(len(CONTEXT_FEATURES),),
    name="context_features"
)

x2 = Dense(
    16,
    activation="relu"
)(context_input)


# Combine temporal + context information
combined = Concatenate()([
    x1,
    x2
])

combined = Dense(
    32,
    activation="relu"
)(combined)

combined = Dropout(
    0.10
)(combined)

output = Dense(
    2,
    name="cpu_memory_output"
)(combined)


lstm_model = Model(
    inputs=[
        sequence_input,
        context_input
    ],
    outputs=output
)


lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)


lstm_model.summary()


# ============================================================
# 11. TRAIN LSTM
#
# Small number of epochs for fast execution.
# Chronological validation: shuffle=False
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LSTM")
print("=" * 70)

BATCH_SIZE = 4096
EPOCHS = 5


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=1,
        restore_best_weights=True
    )
]


history = lstm_model.fit(
    [
        X_train_seq,
        X_train_context
    ],
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.10,
    shuffle=False,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# 12. PREDICT ON THE SAME UNTOUCHED TEST SET
# ============================================================

print("\n" + "=" * 70)
print("LSTM TEST PREDICTION")
print("=" * 70)

y_pred_scaled = lstm_model.predict(
    [
        X_test_seq,
        X_test_context
    ],
    batch_size=BATCH_SIZE,
    verbose=1
)


# Convert predictions back to original units
y_pred = target_scaler.inverse_transform(
    y_pred_scaled
)


print("\nPrediction shape:", y_pred.shape)

assert y_pred.shape == (
    283864,
    2
)


# ============================================================
# 13. EVALUATION FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


# ============================================================
# 14. CPU RESULTS
# ============================================================

cpu_results = calculate_metrics(
    y_test_raw[:, 0],
    y_pred[:, 0]
)

memory_results = calculate_metrics(
    y_test_raw[:, 1],
    y_pred[:, 1]
)


print("\n" + "=" * 70)
print("LSTM BASELINE RESULTS")
print("=" * 70)

print("\nCPU")
print("-" * 40)

for metric, value in cpu_results.items():
    print(f"{metric:<6}: {value:.10f}")


print("\nMemory")
print("-" * 40)

for metric, value in memory_results.items():
    print(f"{metric:<6}: {value:.10f}")


# ============================================================
# 15. FINAL RESULT TABLE
# ============================================================

lstm_results = pd.DataFrame([
    {
        "Model": "LSTM",
        "Target": "CPU",
        **cpu_results
    },
    {
        "Model": "LSTM",
        "Target": "Memory",
        **memory_results
    }
])

print("\n" + "=" * 70)
print("FINAL LSTM RESULTS")
print("=" * 70)

print(
    lstm_results.to_string(
        index=False
    )
)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "LSTM_test_results.csv"
)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "LSTM_test_predictions.csv"
)

model_path = os.path.join(
    OUTPUT_DIR,
    "LSTM_model.keras"
)


lstm_results.to_csv(
    results_path,
    index=False
)


predictions_df = pd.DataFrame({
    "actual_cpu": y_test_raw[:, 0],
    "predicted_cpu": y_pred[:, 0],
    "actual_memory": y_test_raw[:, 1],
    "predicted_memory": y_pred[:, 1]
})

predictions_df.to_csv(
    predictions_path,
    index=False
)


lstm_model.save(
    model_path
)


print("\nFiles saved:")
print(results_path)
print(predictions_path)
print(model_path)


# ============================================================
# 17. FINAL CONTROLLED-EXPERIMENT CHECK
# ============================================================

print("\n" + "=" * 70)
print("CONTROLLED EXPERIMENT VERIFICATION")
print("=" * 70)

print("Dataset                 : Google Cluster Trace 2019")
print("Historical observations : 10 previous observations")
print("History                 : 50 minutes")
print("Prediction horizon      : 5 minutes")
print("Split                   : Chronological 80:20")
print("Development rows       :", len(train_data))
print("Test rows              :", len(test_data))
print("CPU target             : future_cpu")
print("Memory target          : future_memory")
print("Test observations      :", len(y_test_raw))
print("Metrics                : MSE, RMSE, MAE, R2")

print("\n✓ LSTM controlled baseline completed.")
