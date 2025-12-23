import pandas as pd
import numpy as np
from config.logging import get_logger

logger = get_logger(__name__)

def transform_datetime_fields(df: pd.DataFrame, context=None) -> pd.DataFrame:
    """Clean and transform timestamp and date/time-related columns."""
    
    # Drop redundant or conflicting columns
    df = df.drop(columns=['event_date'], errors='ignore')
    
    # Define timestamp conversions (unit in microseconds unless noted)
    time_fields = {
        'event_datetime': ('event_timestamp', 'us'),
        'event_previous_datetime': ('event_previous_timestamp', 'us'),
        'event_first_touch_datetime': ('user_first_touch_timestamp', 'us'),
        'user__first_open_datetime': ('user__first_open_time', 'ms'),
        
    }
    
    for new_col, (src_col, unit) in time_fields.items():
        df[new_col] = pd.to_datetime(df[src_col], unit=unit, utc=True)
    
    # Time delta in seconds
    df['time_delta'] = (
        df['event_datetime'] - df['event_previous_datetime']
    ).dt.total_seconds()
    
    # Derive normalized date/time components
    for base in ['event', 'event_previous', 'event_first_touch', 'user__first_open']:
        df[f'{base}_date'] = df[f'{base}_datetime'].dt.normalize()
        df[f'{base}_time'] = df[f'{base}_datetime'].dt.time

    # Unit conversions and renames
    df['device__time_zone_offset_hours'] = df['device__time_zone_offset_seconds'] / 3600
    df['event_params__engagement_time_seconds'] = df['event_params__engagement_time_msec'] / 1000
    df['event_server_delay_seconds'] = df['event_server_timestamp_offset'] / 1000
    df['event_params__time_spent_seconds'] = df.get('event_params__time_spent')

    logger.info("Date/time cleanup and transformation complete.")
    return df




def add_time_based_features(df: pd.DataFrame, context=None) -> pd.DataFrame:
    """Add time-based features derived from event_datetime and time zone offset."""
    
    # Ensure datetime and offset fields exist
    if 'event_datetime' not in df or 'device__time_zone_offset_hours' not in df:
        raise KeyError("Required columns 'event_datetime' or 'device__time_zone_offset_hours' missing.")
    
    # Weekday name (ordered categorical)
    df['ts_weekday'] = pd.Categorical(
        df['event_datetime'].dt.day_name(),
        categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        ordered=True
    )

    # Local timestamp and derived hour
    df['ts_local_time'] = df['event_datetime'] + pd.to_timedelta(
        df['device__time_zone_offset_hours'].fillna(0), unit='h'
    )
    df['ts_hour'] = df['ts_local_time'].dt.hour

    # Named daypart (Turkish)
    def _daytime_label(hour: int) -> str:
        if hour < 6 or hour > 22:
            return 'Gece'
        elif hour < 11:
            return 'Sabah'
        elif hour < 14:
            return 'Öğle'
        elif hour < 17:
            return 'Öğleden Sonra'
        else:
            return 'Akşam'

    df['ts_daytime_named'] = df['ts_hour'].apply(_daytime_label)

    # Weekend / weekday flag (Turkish)
    df['ts_is_weekend'] = df['ts_weekday'].astype(str).apply(
        lambda x: 'Hafta Sonu' if x in ['Saturday', 'Sunday'] else 'Hafta İçi'
    )

    # Convert weekday to string for consistency
    df['ts_weekday'] = df['ts_weekday'].astype(str)

    logger.info("Time-based features added successfully.")
    return df


def add_durations(df: pd.DataFrame, context=None) -> pd.DataFrame:
    
    # Ensure event_datetime is in datetime format
    df['event_datetime'] = pd.to_datetime(df['event_datetime'])

    # Firebase assigns these events at the last session's end, hence the messy logic
    exclude_session_end_events = ['app_remove', 'app_update', 'app_clear_data']

    # Calculate session duration (ignoring session-end events)
    df['session_duration_seconds'] = (
        df.groupby(['user_pseudo_id', 'event_params__ga_session_id'])
        .apply(
            lambda g: (
                g.loc[~g['event_name'].isin(exclude_session_end_events), 'event_datetime'].max()
                - g.loc[g['event_name'] == 'session_start', 'event_datetime'].min()
            ).total_seconds()
            if not g.loc[g['event_name'] == 'session_start'].empty else np.nan
        )
        .reindex(df.set_index(['user_pseudo_id', 'event_params__ga_session_id']).index)
        .round(3)
        .values
    )
    df['session_duration_minutes'] = (df['session_duration_seconds'] / 60).round(2)
    df['session_duration_hours'] = (df['session_duration_seconds'] / 3600).round(3)
    # Session start and end times
    df['session_start_time'] = df.groupby(['user_pseudo_id', 'event_params__ga_session_id'])['event_datetime'].transform('min')
    df['session_end_time'] = df.groupby(['user_pseudo_id', 'event_params__ga_session_id'])['event_datetime'].transform('max')
    logger.info(f"Session IDs assigned and durations calculated for {df['event_params__ga_session_id'].nunique()} unique sessions.")

    # Calculate event duration within sessions
    df = df.sort_values(by=['user_pseudo_id', 'event_params__ga_session_id', 'event_datetime'])
    df['event_duration_seconds'] = df.groupby(['user_pseudo_id', 'event_params__ga_session_id'])['event_datetime'].shift(-1) - df['event_datetime']
    df['event_duration_seconds'] = df['event_duration_seconds'].dt.total_seconds().fillna(0).round(3)
    logger.info("Event durations within sessions calculated successfully.")
    return df


