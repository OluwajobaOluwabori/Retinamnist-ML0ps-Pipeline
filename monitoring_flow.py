import os
import datetime as dt

import torch
import pandas as pd
import psycopg
import torch.nn as nn
from prefect import flow, task, get_run_logger
from medmnist import INFO, RetinaMNIST
from evidently import Report, Dataset, DataDefinition
from torchvision import models, transforms
from torch.utils.data import DataLoader
from evidently.metrics import (
    ValueDrift,
    MissingValueCount,
    DriftedColumnsCount
)

# ---------------------------
# Config
# ---------------------------

DRIFT_THRESHOLD = 0.3  # if dataset drift is detected (True), alert/retrain
BATCH_SIZE = 64
MODEL_PATH = "resnet50_retinamnist.pth"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

CONNECTION_STRING = "host=localhost port=5432 user=postgres password=example"
CONNECTION_STRING_DB = CONNECTION_STRING + " dbname=test"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS retina_metrics (
    ts TIMESTAMP,
    dataset_drift BOOLEAN,
    share_drifted_columns FLOAT,
    n_drifted_columns INT,
    sample_size INT
);
"""

# ---------------------------
# Torch model (match your training code)
# ---------------------------

# Load model
# model = models.resnet50()
# model.fc = nn.Linear(model.fc.in_features, 5)
# model.load_state_dict(torch.load("resnet50_retinamnist.pth",map_location=torch.device('cpu')))
# model.eval()


# ---------------------------
# Helpers
# ---------------------------


def _predict_dataset(model, dataset, device):
    """Run inference and return predictions + labels as numpy arrays."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.squeeze().cpu().numpy()
            out = model(X)
            pred = out.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            targets.append(y)
    return (
        pd.Series(torch.concat([torch.tensor(p) for p in preds]).numpy()),
        pd.Series(torch.concat([torch.tensor(t) for t in targets]).numpy()),
    )


def _df_from_preds(preds, targets):
    return pd.DataFrame({"prediction": preds, "target": targets})


def _extract_drift_info(report_obj: dict):
    """Safely pull drift flags and counters from Evidently report.as_object()."""
    # The structure of as_object() can change; this function is defensive.
    dataset_drift = None
    share = None
    n_drifted = None

    try:
        # Find the DataDriftPreset block
        for m in report_obj.get("metrics", []):
            header = m.get("metric", {}).get("display_name", "")
            # This name may differ slightly by version; adjust if needed
            if "Data Drift" in header or "DataDrift" in header:
                res = m.get("result", {})
                dataset_drift = res.get("dataset_drift", None)
                share = res.get("share_drifted_columns", None)
                n_drifted = res.get("number_of_drifted_columns", None)
                break
    except Exception:
        pass

    return dataset_drift, share, n_drifted


# ---------------------------
# Prefect tasks
# ---------------------------


@task
def prep_db():
    # with psycopg.connect(CONNECTION_STRING, autocommit=True) as conn:
    #     with conn.cursor() as cur:
    #         cur.execute(CREATE_TABLE_SQL)
    with psycopg.connect(CONNECTION_STRING, autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect(CONNECTION_STRING_DB) as conn:
            conn.execute(CREATE_TABLE_SQL)


@task
def load_model(device="cpu"):
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(
        torch.load("resnet50_retinamnist.pth", map_location=torch.device('cpu'))
    )
    model.eval()

    return model


@task
def load_reference_data():
    # Use train split as reference
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    ref_ds = RetinaMNIST(split="train", download=True, transform=transform)
    return ref_ds


@task
def load_current_data():
    # Simulate "production batch" with val split (or test split)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    curr_ds = RetinaMNIST(split="val", download=True, transform=transform)
    return curr_ds


@task
def build_reference_frame(model, ref_ds, device="cpu"):
    preds, labels = _predict_dataset(model, ref_ds, device)
    return _df_from_preds(preds, labels)


@task
def build_current_frame(model, curr_ds, device="cpu"):
    preds, labels = _predict_dataset(model, curr_ds, device)
    return _df_from_preds(preds, labels)


@task
def run_evidently(reference_df: pd.DataFrame, current_df: pd.DataFrame):

    # Wrap in Evidently Dataset/DataDefinition for new API style
    data_def = DataDefinition(
        numerical_columns=['prediction'],  # none
        categorical_columns=[],  # none
        # target_column="target",
        # prediction_column="prediction"
    )

    reference_data = Dataset.from_pandas(reference_df, data_definition=data_def)
    current_data = Dataset.from_pandas(current_df, data_definition=data_def)

    report = Report(
        metrics=[
            ValueDrift(column='prediction'),
            DriftedColumnsCount(),
            MissingValueCount(column='prediction'),
        ]
    )
    run = report.run(reference_data=reference_data, current_data=current_data)
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(REPORTS_DIR, f"evidently_report_{ts}.html")
    json_path = os.path.join(REPORTS_DIR, f"evidently_report_{ts}.json")
    result = run.dict()

    run.save_html(html_path)
    run.save_json(json_path)

    logger = get_run_logger()

    logger.info(f"Evidently report summary: {result}")
    return result, html_path


@task
def persist_metrics_to_db(result: dict):
    logger = get_run_logger()

    dataset_drift, share, n_drifted = _extract_drift_info(result)
    sample_size = result.get("meta", {}).get("current", {}).get("number_of_rows", None)

    logger.info(
        f"Drift: {dataset_drift}, share_drifted={share}, n_drifted={n_drifted}, n={sample_size}"
    )

    with psycopg.connect(CONNECTION_STRING_DB, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO retina_metrics (ts, dataset_drift, share_drifted_columns, n_drifted_columns, sample_size) VALUES (%s, %s, %s, %s, %s)",
                (dt.datetime.utcnow(), dataset_drift, share, n_drifted, sample_size),
            )
    return dataset_drift, share


@task
def maybe_alert_or_retrain(dataset_drift: bool, share: float):
    logger = get_run_logger()
    should_trigger = dataset_drift or (share is not None and share > DRIFT_THRESHOLD)

    if should_trigger:
        logger.warning(
            "🚨 Drift threshold exceeded! Triggering conditional workflow..."
        )

    else:
        logger.info("✅ No drift action required.")


# ---------------------------
# Prefect flow
# ---------------------------


@flow(name="retina-monitoring-flow")
def monitoring_flow(device: str = "cpu"):
    prep_db()

    model = load_model(device=device)
    ref_ds = load_reference_data()
    curr_ds = load_current_data()

    ref_df = build_reference_frame(model, ref_ds, device=device)
    curr_df = build_current_frame(model, curr_ds, device=device)

    result, html_report = run_evidently(ref_df, curr_df)
    drift_flag, share = persist_metrics_to_db(result)
    maybe_alert_or_retrain(drift_flag, share)

    return html_report


if __name__ == "__main__":
    monitoring_flow(device="cpu")
