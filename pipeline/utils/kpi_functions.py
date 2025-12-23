from config.logging import get_logger
import pandas as pd



logger = get_logger(__name__)

def retention_rate(df: pd.DataFrame, days: int =1) -> float:
    """
    Calculate the retention rate for users after a specified number of days.

    Args:
        df (pd.DataFrame): DataFrame containing user event data with 'user_pseudo_id' and 'event_date' columns.
        days (int): Number of days after which to calculate retention.
    Returns:
        float: Retention rate as a percentage.
    """
    first_event = (
        df.groupby('user_pseudo_id')['event_date']
        .min()
        .reset_index()
        .rename(columns={'event_date': 'first_event_date'})
    )

    df = df.merge(first_event, on='user_pseudo_id', how='left')
    df['days_since_first_event'] = (df['event_date'] - df['first_event_date']).dt.days

    retained_users = (
        df[df['days_since_first_event'] == days]['user_pseudo_id']
        .nunique()
    )

    total_users = first_event['user_pseudo_id'].nunique()

    retention_rate = (retained_users / total_users) * 100 if total_users > 0 else 0.0

    return round(retention_rate, 2)