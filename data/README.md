# Dataset Information

## Google Cluster Data 2019

The experiments in this repository use the **Google Cluster Data 2019 (GCT 2019)** workload trace.

The specific data used in the experiment are the **instance usage records** containing CPU and memory utilization information.

The raw Google Cluster Data files are not redistributed in this repository. Users should obtain the data directly from the original Google Cluster Data source.

---

## Data File Used

The experiment uses the following Google Cluster Data 2019 instance usage file:

```text
instance_usage-000000000000.json.gz

The following command was used to download the required file:

```bash
wget -c \
https://storage.googleapis.com/clusterdata_2019_a/instance_usage-000000000000.json.gz \
-O "data/raw/instance_usage-000000000000.json.gz"
```
