# C-CASE-GCT2019-Reproducibility

Reproducibility repository for the **Cluster-Based Context-Aware Stacked Ensemble (C-CASE)** framework for multi-resource cloud workload forecasting using Google Cluster Data 2019.

## 1. Overview

This repository contains the implementation and experimental artifacts associated with the research study:

**A Cluster-Based Context-Aware Machine Learning Framework for Multi-Resource Cloud Workload Forecasting**

The proposed C-CASE framework forecasts CPU and memory utilization from cloud workload traces using a stacked ensemble learning approach with workload-context information obtained through K-Means clustering.

The repository is intended to support reproducibility of the reported experiments and evaluation results.

---

## 2. Dataset

The experiments use the **Google Cluster Data 2019 (GCT 2019)** workload trace, specifically the instance usage records containing CPU and memory utilization information.

The dataset is publicly available from Google Cluster Data.

Raw Google Cluster Data files are **not redistributed in this repository**. Instead, the repository provides the procedure and configuration required to obtain the required data from the original source.

See:

```text
data/README.md

## Experimental Protocol

Dataset:
Google Cluster Trace 2019 instance usage data.

Temporal resolution:
5 minutes.

Lookback:
10 observations (50 minutes).

Forecast horizon:
1 observation (5 minutes ahead).

Train-test split:
Chronological 80:20 split.

Training observations:
1,135,395.

Test observations:
283,864.

Input features:
31 engineered features.

Target variables:
Future CPU utilization and future memory utilization.

Scaling:
StandardScaler fitted only on the development/training data.

Clustering:
K-Means with K=3 for the final C-CASE experiment.

K-Means random state:
42.

Base learners:
Linear Regression,
Random Forest,
XGBoost.

Stacking:
3-fold temporal out-of-fold predictions.

Meta learner:
XGBoost.

C-CASE meta features:
3 base-model predictions + 3 cluster-context variables.

Final evaluation:
All models are evaluated on the same untouched
chronological test set containing 283,864 observations.



## Reproducing the C-CASE Results

### 1. Clone the repository

git clone https://github.com/anupamavardhaman/C-CASE-GCT2019-Reproducibility.git

cd C-CASE-GCT2019-Reproducibility

### 2. Create a Python environment

python -m venv .venv

Windows:
.venv\Scripts\activate

Linux/macOS:
source .venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Download the Google Cluster Trace 2019 data

python src/download_data.py

The raw dataset is downloaded to:

data/raw/instance_usage-000000000000.json.gz

The raw dataset is not committed to this repository.

### 5. Run preprocessing

python src/preprocess_data.py \
    --input data/raw/instance_usage-000000000000.json.gz \
    --output checkpoints/ccase_step15.pkl

Expected output:

Total observations: 1,419,259
Training observations: 1,135,395
Testing observations: 283,864

The preprocessing checkpoint uses K=6 for the Step-15
preprocessing workflow.

### 6. Train and evaluate C-CASE

python src/train_ccase.py \
    --checkpoint checkpoints/ccase_step15.pkl \
    --output-dir results

The final training script independently fits K=3 for the
C-CASE Table 4 experiment.

### 7. Output files

The following files are generated:

results/Table4_Final_Test_Predictions.csv

results/Table4_Final_Common_Test_Comparison.csv

results/Table4_Reviewer_Consistency_Check.csv
