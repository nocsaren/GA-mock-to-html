from config.logging import get_logger

import pandas as pd
import numpy as np

from pipeline.utils.kpi_functions import retention_rate

logger = get_logger(__name__)

def calculate_kpis (df: pd.DataFrame, dict):
    
    df_by_ads = dict['by_ads']
    df_by_sessions = dict['by_sessions']
    df_by_users = dict['by_users']
    df_by_questions = dict['by_questions']
    df_by_date = dict['by_date']
    df_technical_events = dict['technical_events']

    session_start_date = pd.DataFrame()
    session_start_date['date'] = df['session_start_time'].dt.normalize()
    # Precompute shared quantities
    start_date = df['event_date'].min()
    end_date = df['event_date'].max()
    day_count = df['event_date'].nunique()

    user_count = df_by_users['user_pseudo_id'].nunique()
    session_count = df_by_sessions['event_params__ga_session_id'].nunique()

    ad_count = (df_by_ads['event_name'] == 'Ad Rewarded').sum()
    exception_count = df_technical_events[df_technical_events['event_name'] == 'App Exception'].shape[0]
    ad_fail_count = df_technical_events[df_technical_events['event_name'] == 'Ad Load Failed'].shape[0]

    session_start_day_count = session_start_date['date'].nunique()

    # Avoid division issues
    safe = lambda num, den: round(num / den, 2) if den else 0

    kpis_df = pd.DataFrame([{
        'From': start_date.strftime('%d.%m.%Y'),
        'To': end_date.strftime('%d.%m.%Y'),
        'Total Days': (end_date - start_date).days,

        'Total Users': user_count,
        'Users per Day': safe(user_count, day_count),

        'Total Sessions': session_count,
        'Sessions per Day': safe(session_count, session_start_day_count),
        'Sessions per User': safe(session_count, user_count),

        'Average Session Duration': round(df_by_sessions['session_duration_seconds'].mean(skipna=True) / 60, 2)
            if not df_by_sessions.empty else 0,

        'Total Ads Viewed': ad_count,
        'Ads per User': safe(ad_count, user_count),
        'Ads per User per Day': safe(ad_count, user_count * day_count),
        'Ads per Session': safe(ad_count, session_count),

        '1-Day Retention %': retention_rate(df=df, days=1),
        '7-Day Retention %': retention_rate(df=df, days=7),
        '30-Day Retention %': retention_rate(df=df, days=30),

        'App Exceptions': exception_count,
        'App Exceptions per Session': safe(exception_count, session_count),

        'Ad Load Failures per Session': safe(ad_fail_count, session_count),

        'Tutorial Completion %': safe(
            df_by_users[df_by_users['tutorial_completed'] == True]['user_pseudo_id'].nunique(),
            user_count
        ) * 100,
    }])

    logger.info("✅ KPIs calculated.")

    return kpis_df.iloc[0].to_dict()
