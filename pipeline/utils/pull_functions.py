from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import os
import json

from config.logging import get_logger

logger = get_logger(__name__)

def pull_from_bq(df, context):
    client = context["client"]
    log_path = context["log_path"]
    data_dir = context["data_dir"]
    dataset = context["dataset"]
    start_date = context["start_date"]  # datetime.date object

    # --- Load downloaded tables log
    if os.path.exists(log_path):
        with open(log_path) as f:
            downloaded = {line.strip() for line in f if line.strip()}
    else:
        downloaded = set()

    # --- Format start_date as YYYYMMDD string
    start_date_str = start_date.strftime("%Y%m%d")

    # --- Get all existing tables
    query = f"""
    SELECT table_id
    FROM `{dataset}.__TABLES__`
    WHERE table_id LIKE 'events_%'
    """
    tables = [row.table_id for row in client.query(query)]

    # --- Filter tables by start_date
    tables = [t for t in tables if t.split("_")[1] >= start_date_str]

    # --- Determine which tables are new
    new_tables = [t for t in tables if t not in downloaded]

    if not new_tables:
        logger.info("No new tables to process.")

    for table_name in new_tables:
        logger.info(f"Processing {table_name}...")

        df = client.query(f"SELECT * FROM `{dataset}.{table_name}`").to_dataframe()

        # --- Example transformation (replace with yours)
        df["source_table"] = table_name

        # --- Save as parquet
        path = os.path.join(data_dir, f"{table_name}.parquet")
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)
        logger.info(f"Fetched {df.shape[0]} rows from {table_name}")

        # --- Update log
        with open(log_path, "a") as f:
            f.write(table_name + "\n")

    # --- Combine all Parquets for report
    logger.info(f"Merging data...")

    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} does not exist.")
        all_data = pd.DataFrame()
    else:
        parquet_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".parquet")
        ]
        
        if not parquet_files:
            logger.warning(f"No parquet files found in {data_dir}.")
            all_data = pd.DataFrame()
        else:
            dfs = [pd.read_parquet(f) for f in parquet_files]
            all_data = pd.concat(dfs, ignore_index=True)

    all_data = normalize_bq_types(all_data)

    sample = all_data["event_params"].iloc[0] if not all_data.empty else None
    if sample is not None and not isinstance(sample, (list, dict)):
        logger.warning(f"event_params type={type(sample)} — normalization may have failed")

    return all_data



def normalize_bq_types(df):
    for col in ["event_params", "user_properties", "items", "item_params"]:
        if col not in df.columns:
            continue
        def fix(x):
            if isinstance(x, str):
                try: return json.loads(x)
                except: return None
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x
        df[col] = df[col].apply(fix)
    return df