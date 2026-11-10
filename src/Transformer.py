# ============================================================
# CONTROLLED TRANSFORMER BASELINE
# Google Cluster Trace 2019
#
# Same experimental setup as C-CASE and LSTM:
#
#   Dataset          : Google Cluster Trace 2019
#   Split            : Chronological 80:20
#   Lookback         : 10 observations = 50 minutes
#   Sequence         : t-10 ... t = 11 time steps
#   Horizon          : 1 observation = 5 minutes ahead
#   Targets          : CPU + Memory
#   Metrics           : MSE, RMSE, MAE, R2
#   Random seed       : 42
#
# IMPORTANT:
# Transformer does NOT use:
#   - C-CASE cluster labels
#   - C-CASE OOF predictions
#   - C-CASE meta-learner
#
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

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Concatenate,
    Add
)

# ============================================================
# 1. RANDOM SEED
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 2. PATHS
# ============================================================

CHECKPOINT_PATH = (
    "/content/drive/MyDrive/C_CASE/checkpoints/"
    "ccase_step15.pkl"
)

OUTPUT_DIR = (
    "/content/drive/MyDrive/C_CASE/"
    "Transformer_baseline"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("CONTROLLED TRANSFORMER BASELINE")
print("=" * 75)

print("TensorFlow version :", tf.__version__)
print("GPU available      :", tf.config.list_physical_devices("GPU"))

# ============================================================
# 3. LOAD THE SAME C-CASE CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = joblib.load(CHECKPOINT_PATH)

train_data = checkpoint["train_data"].copy()
test_data = checkpoint["test_data"].copy()

feature_cols = checkpoint["feature_cols"]

print("\nDevelopment rows :", len(train_data))
print("Test rows        :", len(test_data))
print("Feature count    :", len(feature_cols))

# Expected final test size from the controlled experiment
assert len(test_data) == 283864, (
    f"Expected 283,864 test rows, "
    f"but found {len(test_data)}"
)

# ============================================================
# 4. TARGET VARIABLES
# ============================================================

TARGETS = [
    "future_cpu",
    "future_memory"
]

y_train_raw = (
    train_data[TARGETS]
    .values
    .astype("float32")
)

y_test_raw = (
    test_data[TARGETS]
    .values
    .astype("float32")
)

print("\nTarget shape:")
print("y_train:", y_train_raw.shape)
print("y_test :", y_test_raw.shape)

# ============================================================
# 5. TEMPORAL INPUT FEATURES
#
# 10 historical observations + current observation
#
# t-10, t-9, ..., t-1, t
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

# ============================================================
# 6. CONTEXT FEATURES
#
# Same contextual information used in the LSTM baseline.
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

# ============================================================
# 7. CHECK REQUIRED COLUMNS
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

print("\nAll required columns are available.")

# ============================================================
# 8. CREATE TRANSFORMER SEQUENCES
#
# Shape:
#     samples × 11 time steps × 2 resources
#
# Resource 1 = CPU
# Resource 2 = Memory
# ============================================================

def create_transformer_sequence(data):

    cpu = data[
        CPU_HISTORY
    ].to_numpy(
        dtype=np.float32
    )

    memory = data[
        MEM_HISTORY
    ].to_numpy(
        dtype=np.float32
    )

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


X_train_seq, X_train_context = (
    create_transformer_sequence(
        train_data
    )
)

X_test_seq, X_test_context = (
    create_transformer_sequence(
        test_data
    )
)

print("\nTransformer input shapes:")
print("X_train_seq     :", X_train_seq.shape)
print("X_test_seq      :", X_test_seq.shape)
print("X_train_context :", X_train_context.shape)
print("X_test_context  :", X_test_context.shape)

# Expected:
# X_train_seq = (development_rows, 11, 2)
# X_test_seq  = (283864, 11, 2)

assert X_train_seq.shape[1:] == (11, 2)
assert X_test_seq.shape[1:] == (11, 2)

# ============================================================
# 9. LEAKAGE-SAFE SCALING
#
# IMPORTANT:
# All scalers are fitted ONLY on development data.
#
# Test data is transformed using development-fitted scalers.
# ============================================================

print("\nScaling temporal CPU and memory features...")

cpu_scaler = StandardScaler()
memory_scaler = StandardScaler()

# -------------------------
# CPU scaler
# -------------------------

cpu_train_values = (
    X_train_seq[:, :, 0]
    .reshape(-1, 1)
)

cpu_scaler.fit(
    cpu_train_values
)

X_train_seq[:, :, 0] = (
    cpu_scaler
    .transform(cpu_train_values)
    .reshape(
        X_train_seq.shape[0],
        X_train_seq.shape[1]
    )
)

X_test_seq[:, :, 0] = (
    cpu_scaler
    .transform(
        X_test_seq[:, :, 0]
        .reshape(-1, 1)
    )
    .reshape(
        X_test_seq.shape[0],
        X_test_seq.shape[1]
    )
)

# -------------------------
# Memory scaler
# -------------------------

memory_train_values = (
    X_train_seq[:, :, 1]
    .copy()
    .reshape(-1, 1)
)

memory_scaler.fit(
    memory_train_values
)

X_train_seq[:, :, 1] = (
    memory_scaler
    .transform(memory_train_values)
    .reshape(
        X_train_seq.shape[0],
        X_train_seq.shape[1]
    )
)

X_test_seq[:, :, 1] = (
    memory_scaler
    .transform(
        X_test_seq[:, :, 1]
        .reshape(-1, 1)
    )
    .reshape(
        X_test_seq.shape[0],
        X_test_seq.shape[1]
    )
)

# ============================================================
# 10. SCALE CONTEXT FEATURES
# ============================================================

print("Scaling context features...")

context_scaler = StandardScaler()

X_train_context = (
    context_scaler
    .fit_transform(
        X_train_context
    )
    .astype("float32")
)

X_test_context = (
    context_scaler
    .transform(
        X_test_context
    )
    .astype("float32")
)

# ============================================================
# 11. SCALE TARGETS
# ============================================================

print("Scaling targets...")

target_scaler = StandardScaler()

y_train = (
    target_scaler
    .fit_transform(
        y_train_raw
    )
    .astype("float32")
)

y_test = (
    target_scaler
    .transform(
        y_test_raw
    )
    .astype("float32")
)

print("Scaling completed.")

# ============================================================
# 12. TRANSFORMER ENCODER BLOCK
# ============================================================

def transformer_encoder(
    inputs,
    head_size=16,
    num_heads=2,
    ff_dim=64,
    dropout=0.10
):

    # --------------------------------------------------------
    # Multi-Head Self Attention
    # --------------------------------------------------------

    x = LayerNormalization(
        epsilon=1e-6
    )(inputs)

    attention_output = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(
        x,
        x
    )

    x = Add()([
        inputs,
        attention_output
    ])

    # --------------------------------------------------------
    # Feed Forward Network
    # --------------------------------------------------------

    y = LayerNormalization(
        epsilon=1e-6
    )(x)

    y = Dense(
        ff_dim,
        activation="relu"
    )(y)

    y = Dropout(
        dropout
    )(y)

    y = Dense(
        x.shape[-1]
    )(y)

    y = Dropout(
        dropout
    )(y)

    output = Add()([
        x,
        y
    ])

    return output


# ============================================================
# 13. BUILD TRANSFORMER MODEL
# ============================================================

print("\nBuilding Transformer model...")

# ------------------------------------------------------------
# Temporal sequence input
# Shape = 11 time steps × 2 resources
# ------------------------------------------------------------

sequence_input = Input(
    shape=(11, 2),
    name="historical_cpu_memory"
)

# Project CPU + memory into Transformer embedding dimension

x = Dense(
    32,
    name="input_projection"
)(sequence_input)

# ------------------------------------------------------------
# Positional information
#
# The sequence contains ordered observations:
# t-10 ... t
#
# A small trainable positional embedding is used.
# ------------------------------------------------------------

positions = tf.range(
    start=0,
    limit=11,
    delta=1
)

position_embedding = tf.keras.layers.Embedding(
    input_dim=11,
    output_dim=32,
    name="position_embedding"
)(positions)

x = x + position_embedding

# ------------------------------------------------------------
# Transformer Encoder
# ------------------------------------------------------------

x = transformer_encoder(
    x,
    head_size=16,
    num_heads=2,
    ff_dim=64,
    dropout=0.10
)

# ------------------------------------------------------------
# Convert sequence representation to fixed vector
# ------------------------------------------------------------

x = GlobalAveragePooling1D(
    name="temporal_pooling"
)(x)

x = Dropout(
    0.10
)(x)

# ============================================================
# 14. CONTEXT BRANCH
# ============================================================

context_input = Input(
    shape=(len(CONTEXT_FEATURES),),
    name="context_features"
)

context_x = Dense(
    16,
    activation="relu",
    name="context_dense"
)(context_input)

context_x = Dropout(
    0.10
)(context_x)

# ============================================================
# 15. COMBINE TEMPORAL + CONTEXT INFORMATION
# ============================================================

combined = Concatenate(
    name="temporal_context_fusion"
)([
    x,
    context_x
])

combined = Dense(
    32,
    activation="relu",
    name="fusion_dense"
)(combined)

combined = Dropout(
    0.10
)(combined)

# ============================================================
# 16. OUTPUT
#
# Output 1 = future CPU
# Output 2 = future Memory
# ============================================================

output = Dense(
    2,
    name="cpu_memory_output"
)(combined)

# ============================================================
# 17. CREATE MODEL
# ============================================================

transformer_model = Model(
    inputs=[
        sequence_input,
        context_input
    ],
    outputs=output,
    name="Controlled_Transformer"
)

# ============================================================
# 18. COMPILE
# ============================================================

transformer_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse"
)

print("\nModel summary:")
transformer_model.summary()

# ============================================================
# 19. TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 4096
EPOCHS = 5

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=1,
        restore_best_weights=True
    )
]

print("\nTraining settings:")
print("Batch size :", BATCH_SIZE)
print("Epochs     :", EPOCHS)
print("Shuffle    : False")
print("Validation : Last 10% of development data")

# ============================================================
# 20. TRAIN TRANSFORMER
#
# shuffle=False preserves chronological order.
# validation_split=0.10 uses the final 10% of development
# observations as validation data.
# ============================================================

print("\n" + "=" * 75)
print("STARTING TRANSFORMER TRAINING")
print("=" * 75)

history = transformer_model.fit(
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

print("\nTransformer training completed.")

# ============================================================
# 21. PREDICTION ON UNTOUCHED TEST SET
# ============================================================

print("\n" + "=" * 75)
print("PREDICTING ON UNTOUCHED TEST SET")
print("=" * 75)

y_pred_scaled = transformer_model.predict(
    [
        X_test_seq,
        X_test_context
    ],
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\nScaled prediction shape:")
print(y_pred_scaled.shape)

# ============================================================
# 22. CONVERT PREDICTIONS BACK TO ORIGINAL SCALE
# ============================================================

y_pred = (
    target_scaler
    .inverse_transform(
        y_pred_scaled
    )
)

print("Original-scale prediction shape:")
print(y_pred.shape)

# ============================================================
# 23. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mse
    )

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
# 24. CPU METRICS
# ============================================================

cpu_results = calculate_metrics(
    y_test_raw[:, 0],
    y_pred[:, 0]
)

# ============================================================
# 25. MEMORY METRICS
# ============================================================

memory_results = calculate_metrics(
    y_test_raw[:, 1],
    y_pred[:, 1]
)

# ============================================================
# 26. CREATE RESULTS TABLE
# ============================================================

transformer_results = pd.DataFrame([

    {
        "Model": "Transformer",
        "Target": "CPU",
        **cpu_results
    },

    {
        "Model": "Transformer",
        "Target": "Memory",
        **memory_results
    }

])

print("\n" + "=" * 75)
print("TRANSFORMER TEST RESULTS")
print("=" * 75)

print(
    transformer_results
    .to_string(index=False)
)

# ============================================================
# 27. SAVE RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "Transformer_test_results.csv"
)

transformer_results.to_csv(
    results_path,
    index=False
)

# ============================================================
# 28. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({

    "actual_cpu":
        y_test_raw[:, 0],

    "predicted_cpu":
        y_pred[:, 0],

    "actual_memory":
        y_test_raw[:, 1],

    "predicted_memory":
        y_pred[:, 1]

})

predictions_path = os.path.join(
    OUTPUT_DIR,
    "Transformer_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

# ============================================================
# 29. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_path = os.path.join(
    OUTPUT_DIR,
    "Transformer_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

# ============================================================
# 30. SAVE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "Transformer_model.keras"
)

transformer_model.save(
    model_path
)

# ============================================================
# 31. SAVE SCALERS
# ============================================================

scalers = {
    "cpu_scaler": cpu_scaler,
    "memory_scaler": memory_scaler,
    "context_scaler": context_scaler,
    "target_scaler": target_scaler
}

scalers_path = os.path.join(
    OUTPUT_DIR,
    "Transformer_scalers.pkl"
)

joblib.dump(
    scalers,
    scalers_path
)

# ============================================================
# 32. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("TRANSFORMER BASELINE COMPLETED")
print("=" * 75)

print("\nDevelopment rows :", len(train_data))
print("Test rows        :", len(test_data))
print("Sequence length   :", 11)
print("Historical window :", "50 minutes")
print("Forecast horizon  :", "5 minutes")
print("Targets           :", "CPU + Memory")
print("Random seed       :", SEED)

print("\nFiles saved:")

print(
    "\nResults:",
    results_path
)

print(
    "Predictions:",
    predictions_path
)

print(
    "Training history:",
    history_path
)

print(
    "Model:",
    model_path
)

print(
    "Scalers:",
    scalers_path
)

print("\n" + "=" * 75)
