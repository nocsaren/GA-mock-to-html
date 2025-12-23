import os
import pandas as pd
from config.logging import get_logger
from pathlib import Path

logger = get_logger(__name__)

def ensure_directories(directories):
    """
    Ensure that the specified directories exist.
    Paths are resolved relative to the current working directory (repo root).
    """
    try:
        for d in directories:
            path = Path(d).resolve()
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory exists or created: {path}")
        logger.info("All folder structures validated.")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        exit(1)


def filter_events_by_date(df, context):
    """Filter events in the DataFrame to only include those on or after start_date (UTC)."""
    # Convert date to pandas Timestamp (assumed UTC)

    start_date = context["start_date"]
    start_dt = pd.Timestamp(start_date, tz='UTC')

    # Convert to UNIX microseconds
    start_ts = int(start_dt.value // 10**3)

    # Filter
    filtered_df = df[df['event_timestamp'] >= start_ts]

    logger.info(
        f"Filtered events from {len(df)} to {len(filtered_df)} based on start date {start_dt}."
    )
    return filtered_df
