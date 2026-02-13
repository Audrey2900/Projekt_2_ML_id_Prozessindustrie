import pandas as pd
import numpy as np
import logging, sys
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logging.info("Importing torch...")
import torch
logging.info("Torch imported.")
logging.info("CUDA available? %s", torch.cuda.is_available())
import time

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# -----------------------
# Config
# -----------------------
DATA_PATH = "../data/SmA-Four-Tank-Batch-Process_V2.csv"
SEP = ";"
INDEX_COL = "timestamp"
RANDOM_STATE = 42

dataset_sizes = [100, 1000, 5000]
steps_to_run = [1, 3, 7, 8]

percentile_threshold = 90
n_permutations = 10

device = "cuda"
n_estimators = 4


# -----------------------
# Load once
# -----------------------
logger.info("Loading dataset...")
df_full = pd.read_csv(DATA_PATH, sep=SEP, index_col=INDEX_COL)

df_full["DeviationID ValueY"] = np.where(
    df_full["DeviationID ValueY"] == 1, False, True
)

logger.info("Dataset loaded successfully.")


all_metrics_rows = []

for step in steps_to_run:
    logger.info(f"========== Starting Step {step} ==========")

    df = df_full[df_full["CuStepNo ValueY"] == step].copy()

    if df.empty:
        logger.warning(f"Step {step}: No data found. Skipping.")
        continue

    logger.info(f"Step {step}: {len(df)} rows available.")

    step_metrics_rows = []

    for dataset_size in dataset_sizes:
        start_time = time.time()

        logger.info(
            f"[Step {step}] Processing dataset_size = {dataset_size}"
        )

        n_anom = int(0.1 * dataset_size)
        n_norm = dataset_size - n_anom

        df_anom_pool = df[df["DeviationID ValueY"] == True]
        df_norm_pool = df[df["DeviationID ValueY"] == False]

        if len(df_anom_pool) < n_anom or len(df_norm_pool) < n_norm:
            logger.warning(
                f"[Step {step} | Size {dataset_size}] Skipping: "
                f"Need anom={n_anom} (have {len(df_anom_pool)}), "
                f"normal={n_norm} (have {len(df_norm_pool)})"
            )
            continue

        rs = RANDOM_STATE

        logger.info(
            f"[Step {step} | Size {dataset_size}] Sampling data..."
        )

        df_anom = df_anom_pool.sample(n=n_anom, random_state=rs)
        df_norm = df_norm_pool.sample(n=n_norm, random_state=rs)

        df_all = pd.concat([df_anom, df_norm], axis=0).sample(
            frac=1, random_state=rs
        )

        X_cols = [c for c in df_all.columns if c != "DeviationID ValueY"]
        y_col = "DeviationID ValueY"

        X = df_all[X_cols].copy()
        y = df_all[y_col].astype(bool).copy()

        const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        X = X.drop(columns=const_cols)

        logger.info(
            f"[Step {step} | Size {dataset_size}] "
            f"{X.shape[1]} features after dropping constants."
        )

        X_tensor = torch.from_numpy(X.to_numpy()).float()

        logger.info(
            f"[Step {step} | Size {dataset_size}] Initializing model..."
        )

        clf = TabPFNClassifier(device=device, n_estimators=n_estimators)
        reg = TabPFNRegressor(device=device, n_estimators=n_estimators)
        model_unsupervised = TabPFNUnsupervisedModel(
            tabpfn_clf=clf, tabpfn_reg=reg
        )

        logger.info(
            f"[Step {step} | Size {dataset_size}] Fitting model..."
        )
        model_unsupervised.fit(X_tensor)

        logger.info(
            f"[Step {step} | Size {dataset_size}] Computing outlier scores..."
        )
        scores = model_unsupervised.outliers(
            X_tensor, n_permutations=n_permutations
        )

        threshold = np.percentile(scores, percentile_threshold)
        y_pred = scores > threshold
        y_true = y.to_numpy()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        logger.info(
            f"[Step {step} | Size {dataset_size}] "
            f"Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )

        # Save predictions
        results = pd.DataFrame(
            {
                "CuStepNo ValueY": step,
                "dataset_size": dataset_size,
                "y_true": y_true,
                "y_pred": y_pred,
                "score": scores,
            },
            index=df_all.index,
        )
        results.index.name = "timestamp"
        results_filename = f"results/predictions/results_step{step}_size{dataset_size}.csv"
        results.to_csv(results_filename)

        logger.info(
            f"[Step {step} | Size {dataset_size}] Saved predictions to {results_filename}"
        )

        row = {
            "CuStepNo ValueY": step,
            "dataset_size": dataset_size,
            "n_features": X.shape[1],
            "n_anom": int(y_true.sum()),
            "n_normal": int((~y_true).sum()),
            "threshold_percentile": percentile_threshold,
            "threshold_value": float(threshold),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        step_metrics_rows.append(row)
        all_metrics_rows.append(row)

        elapsed = time.time() - start_time
        logger.info(
            f"[Step {step} | Size {dataset_size}] Finished in {elapsed:.2f} seconds"
        )

    step_metrics_df = pd.DataFrame(step_metrics_rows).sort_values("dataset_size")
    step_metrics_filename = f"results/metrics_step{step}.csv"
    step_metrics_df.to_csv(step_metrics_filename, index=False)

    logger.info(f"Saved step metrics to {step_metrics_filename}")
    logger.info(f"========== Finished Step {step} ==========")


metrics_df = pd.DataFrame(all_metrics_rows).sort_values(
    ["CuStepNo ValueY", "dataset_size"]
)
metrics_df.to_csv("results/metrics_overall_by_step.csv", index=False)

logger.info("Saved overall metrics to metrics_overall_by_step.csv")
logger.info("All processing complete.")
print(metrics_df)
