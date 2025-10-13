import json
import os
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

NUM_SEED = 3  # we run 3 seeds for each task

import json
import os
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

NUM_SEED = 3  # expected number of runs per task


def aggregate_metrics(log_dir, metrics, task_name, key_col="mean"):
    """
    Parameters:
        log_dir (str): log directory containing subdirectories for each seed run
        metrics (list): metrics to aggregate
        task_name (str): task name
        key_col (str): which column in progression_task, default 'mean'
    """
    collected = {m: [] for m in metrics}

    for dirpath, _, filenames in os.walk(log_dir):
        try:
            if task_name == "progression_task" and "total_metrics.csv" in filenames:
                # progression_task: CSV with metrics in index, "mean" column
                csv_path = os.path.join(dirpath, "total_metrics.csv")
                df = pd.read_csv(csv_path, index_col=0)
                for m in metrics:
                    if m in df.index and key_col in df.columns:
                        collected[m].append(df.loc[m, key_col])

            elif task_name == "mortality" and "total_metrics.csv" in filenames:
                # mortality: CSV with metrics as columns
                csv_path = os.path.join(dirpath, "total_metrics.csv")
                df = pd.read_csv(csv_path)
                for m in metrics:
                    if m in df.columns:
                        collected[m].extend(df[m].values.tolist())

            elif task_name == "length_of_stay" and "total_metrics.json" in filenames:
                # los: JSON file with metrics
                json_path = os.path.join(dirpath, "total_metrics.json")
                with open(json_path, "r") as f:
                    data = json.load(f)
                for m in metrics:
                    if m in data:
                        collected[m].append(data[m])

        except Exception as e:
            print(f"Error reading in {dirpath}: {e}")

    metric_df = pd.DataFrame(data=collected)

    if len(metric_df) != NUM_SEED:
        raise ValueError(f"{task_name}: expected {NUM_SEED} runs, but got {len(metric_df)}")

    mean = metric_df.mean()
    std = metric_df.std()

    print(f"\n{task_name} results:")
    for col in metric_df.columns:
        print(f"{col}: {mean[col]:.4f} ± {std[col]:.4f}")


if __name__ == "__main__":
    # progression_task
    aggregate_metrics(
        log_dir="./logs/test_run/disease_progression",
    
        metrics=['roc_auc', 'pr_auc_macro', 'accuracy', 'avg_precision_macro', 'avg_recall_macro', 'f1_macro'],
        task_name="progression_task"
    )

    # length_of_stay
    aggregate_metrics(
        log_dir="./logs/test_run/length_of_stay",
        
        metrics=['roc_auc', 'pr_auc_macro', 'accuracy', 'avg_precision_macro', 'avg_recall_macro', 'f1_macro', 'kappa'],
        task_name="length_of_stay"
    )

    # mortality
    aggregate_metrics(
        log_dir="./logs/test_run/mortality",
    
        metrics=['pr_auc', 'roc_auc'],
        task_name="mortality"
    )
