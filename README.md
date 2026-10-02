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
