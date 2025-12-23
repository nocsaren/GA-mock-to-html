from config.logging import get_logger
from pipeline.utils.split_functions import (df_by_sessions,
                                            df_by_users,
                                            df_by_questions,
                                            df_by_ads,
                                            df_by_date,
                                            df_technical_events,
                                            df_most_active_by_time)
import pandas as pd

logger = get_logger(__name__)

def create_dataframes(df: pd.DataFrame):
    """Generate actual dataframes from a single source df."""
    dataframes = {
        "by_sessions": df_by_sessions(df),
        "by_users": df_by_users(df),
        "by_questions": df_by_questions(df),
        "by_ads": df_by_ads(df),
        "by_date": df_by_date(df),
        "technical_events": df_technical_events(df),
        "most_active_by_time": df_most_active_by_time(df),
    }
    logger.info("All split dataframes successfully created.")
    return dataframes
