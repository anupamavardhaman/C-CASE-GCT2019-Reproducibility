import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# ============================================================
# STATISTICAL SIGNIFICANCE TEST
# Random Forest vs Proposed C-CASE
# ============================================================

# ------------------------------------------------------------
# 1. Load the common test-set predictions
# ------------------------------------------------------------

pred_path = "/content/drive/MyDrive/C_CASE/final_table4/Table4_Final_Test_Predictions.csv"

pred_df = pd.read_csv(pred_path)

print("=" * 75)
print("STATISTICAL SIGNIFICANCE TEST")
print("Random Forest vs Proposed C-CASE")
print("=" * 75)

print("Number of test observations:", len(pred_df))


# ============================================================
# 2. Calculate absolute prediction errors
# ============================================================

# CPU
rf_cpu_error = np.abs(
    pred_df["y_test_cpu"] - pred_df["RF_cpu"]
)

ccase_cpu_error = np.abs(
    pred_df["y_test_cpu"] - pred_df["CCASE_cpu"]
)

# Memory
rf_memory_error = np.abs(
    pred_df["y_test_memory"] - pred_df["RF_memory"]
)

ccase_memory_error = np.abs(
    pred_df["y_test_memory"] - pred_df["CCASE_memory"]
)


# ============================================================
# 3. Wilcoxon Signed-Rank Test — CPU
# ============================================================

cpu_stat, cpu_p = wilcoxon(
    ccase_cpu_error,
    rf_cpu_error,
    alternative="two-sided",
    method="auto"
)


# ============================================================
# 4. Wilcoxon Signed-Rank Test — Memory
# ============================================================

memory_stat, memory_p = wilcoxon(
    ccase_memory_error,
    rf_memory_error,
    alternative="two-sided",
    method="auto"
)


# ============================================================
# 5. Error differences
# ============================================================

cpu_difference = ccase_cpu_error - rf_cpu_error
memory_difference = ccase_memory_error - rf_memory_error


# ============================================================
# 6. Rank-biserial effect size
# ============================================================

def rank_biserial_effect(x, y):

    d = np.asarray(x) - np.asarray(y)

    # Remove zero differences
    d = d[d != 0]

    if len(d) == 0:
        return 0.0

    ranks = pd.Series(np.abs(d)).rank().to_numpy()

    positive_ranks = ranks[d > 0].sum()
    negative_ranks = ranks[d < 0].sum()

    return (
        positive_ranks - negative_ranks
    ) / (
        positive_ranks + negative_ranks
    )


cpu_effect = rank_biserial_effect(
    ccase_cpu_error,
    rf_cpu_error
)

memory_effect = rank_biserial_effect(
    ccase_memory_error,
    rf_memory_error
)


# ============================================================
# 7. Print detailed results
# ============================================================

print("\n" + "=" * 75)
print("CPU UTILIZATION")
print("=" * 75)

print(f"RF Mean Absolute Error       : {rf_cpu_error.mean():.10e}")
print(f"C-CASE Mean Absolute Error   : {ccase_cpu_error.mean():.10e}")

print(f"RF Median Absolute Error     : {np.median(rf_cpu_error):.10e}")
print(f"C-CASE Median Absolute Error : {np.median(ccase_cpu_error):.10e}")

print(f"Median Error Difference      : {np.median(cpu_difference):.10e}")

print(f"Wilcoxon Statistic           : {cpu_stat:.4f}")
print(f"Wilcoxon p-value             : {cpu_p:.10e}")
print(f"Rank-biserial Effect Size    : {cpu_effect:.6f}")


print("\n" + "=" * 75)
print("MEMORY UTILIZATION")
print("=" * 75)

print(f"RF Mean Absolute Error       : {rf_memory_error.mean():.10e}")
print(f"C-CASE Mean Absolute Error   : {ccase_memory_error.mean():.10e}")

print(f"RF Median Absolute Error     : {np.median(rf_memory_error):.10e}")
print(f"C-CASE Median Absolute Error : {np.median(ccase_memory_error):.10e}")

print(f"Median Error Difference      : {np.median(memory_difference):.10e}")

print(f"Wilcoxon Statistic           : {memory_stat:.4f}")
print(f"Wilcoxon p-value             : {memory_p:.10e}")
print(f"Rank-biserial Effect Size    : {memory_effect:.6f}")


# ============================================================
# 8. Statistical decision
# ============================================================

alpha = 0.05

print("\n" + "=" * 75)
print("STATISTICAL DECISION (α = 0.05)")
print("=" * 75)

if cpu_p < alpha:
    print("CPU    : Significant difference (p < 0.05)")
else:
    print("CPU    : No significant difference (p >= 0.05)")

if memory_p < alpha:
    print("Memory : Significant difference (p < 0.05)")
else:
    print("Memory : No significant difference (p >= 0.05)")


# ============================================================
# 9. Identify which model has lower error
# ============================================================

print("\n" + "=" * 75)
print("ERROR COMPARISON")
print("=" * 75)

if ccase_cpu_error.mean() < rf_cpu_error.mean():
    print("CPU    : C-CASE has lower mean absolute error.")
else:
    print("CPU    : Random Forest has lower mean absolute error.")

if ccase_memory_error.mean() < rf_memory_error.mean():
    print("Memory : C-CASE has lower mean absolute error.")
else:
    print("Memory : Random Forest has lower mean absolute error.")
