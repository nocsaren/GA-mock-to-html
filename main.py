from config import settings
from config.logging import get_logger

from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import date

import argparse
import os
import json
import pandas as pd

from pipeline.utils.main_functions import ensure_directories
from pipeline.utils.staging import (run_pipeline, PIPELINE_STAGES)
from pipeline.utils.dataframes import create_dataframes

from pipeline.utils.calculate_kpis import calculate_kpis

from pipeline.utils.reporting import generate_report


logger = get_logger(__name__)

logger.info('Starting pipeline...')


# --- Ensure directories exist
logger.info("Validating folder structure...")

folders_to_create = []

for name in dir(settings):
    if name.isupper():
        value = getattr(settings, name)
        if isinstance(value, str) and (value.startswith("./") or value.startswith("/")):
            # If value ends with a known file extension, take dirname; else, take the path itself
            if os.path.splitext(value)[1]:  # has extension → file path
                folders_to_create.append(os.path.dirname(value))
            else:  # likely already a folder
                folders_to_create.append(value)
    
ensure_directories(folders_to_create)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Override data directory")
    parser.add_argument("--log_path", help="Override log path")
    parser.add_argument("--csv_dir", help="Override CSV directory")
    parser.add_argument("--report_path", help="Override report path")
    return parser.parse_args()

logger.info("Initializing BigQuery client...")

# for local dev, use key file
# NEEDS A PROPER key.json FILE FROM A GOOGLE SERVICE ACCOUNT

# KEY_PATH = "./keys/key.json"
# credentials = service_account.Credentials.from_service_account_file(KEY_PATH)

# for GitHub Actions

creds_dict = json.loads(os.environ["BQ_SERVICE_ACCOUNT"])
credentials = service_account.Credentials.from_service_account_info(creds_dict)


logger.info("BigQuery client initialized.")


df = pd.DataFrame()
df_sessions = pd.DataFrame()


logger.info("Starting data pull...")
if __name__ == "__main__":

    args = parse_args()

    # Override settings if CLI args are provided
    data_dir = args.data_dir if args.data_dir else settings.DATA_DIR
    log_path = args.log_path if args.log_path else settings.LOG_PATH
    csv_dir = args.csv_dir if args.csv_dir else settings.CSV_DIR
    report_path = args.report_path if args.report_path else settings.REPORT_PATH
    
    client = bigquery.Client(credentials=credentials)

    context = {
        "client": client,
        "log_path": settings.LOG_PATH,
        "data_dir": settings.DATA_DIR,
        'csv_dir': settings.CSV_DIR,
        "dataset": settings.DATASET,
        'start_date': settings.START_DATE,
        'report_path': settings.REPORT_PATH
    }
    

    # pull data and run through pipeline
    df = run_pipeline(df=df, context=context)
    logger.info("Data pipeline executed successfully.")
    

    logger.info("Generating dataframes...")
    dfs = create_dataframes(df=df)
    logger.info("Dataframes generated successfully.")

    logger.info("Calculating KPIs...")

    kpis = calculate_kpis(df=df, dict=dfs)
    
    # df.to_csv(os.path.join(settings.CSV_DIR, "processed_data.csv"), index=False)

    # for name, dataframe in dfs.items():
    #    dataframe.to_csv(os.path.join(settings.CSV_DIR, f"{name}_data.csv"), index=False)
    
    logger.info("Data pipeline complete. Processed data saved.")
    generate_report(df=df, dfs_dict = dfs, kpis = kpis, context = context)
