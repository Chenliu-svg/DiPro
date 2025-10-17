import json
import os
import pandas as pd
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings("ignore")

NUM_SEED = 3  # expected number of runs per task

Task_Name_2_Metrics = {
    "disease_progression": ['avg_precision_macro', 'avg_recall_macro', 'f1_macro',  'pr_auc_macro','roc_auc'],
    "length_of_stay": ['kappa', 'avg_precision_macro', 'avg_recall_macro', 'f1_macro','roc_auc', 'pr_auc_macro', 'accuracy' ],
    "mortality": ['pr_auc', 'roc_auc']
}

def aggregate_metrics(logdir, metrics, task_name, key_col="mean"):
    """
    Parameters:
        logdir (str): log directory containing subdirectories for each seed run
        metrics (list): metrics to aggregate
        task_name (str): task name
        key_col (str): which column in disease_progression, default 'mean'
    """
    collected = {m: [] for m in metrics}

    for dirpath, _, filenames in tqdm(os.walk(logdir), desc=f"Processing {task_name} folders"):
        try:
            if task_name == "disease_progression" and "total_metrics.csv" in filenames:
                # disease_progression: CSV with metrics in index, "mean" column
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

    # using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="log directory containing subdirectories for each seed run",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=["disease_progression", "length_of_stay", "mortality", "all"],
        help="task name",
    )
    args = parser.parse_args()
    logdir = args.logdir
    task_name = args.task_name

    aggregate_metrics(logdir, Task_Name_2_Metrics[task_name], task_name)
    
