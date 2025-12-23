import pandas as pd

from pipeline.utils.utils import summarize_gold # summarize_energy
from config.logging import get_logger

logger = get_logger(__name__)

def df_by_sessions(df: pd.DataFrame) -> pd.DataFrame:
    session_groups = ['event_params__ga_session_id', 'user_pseudo_id']

    df = df[df['session_duration_seconds'] > 15]

    # --- Base sessions: all unique session-user pairs ---
    base_sessions = (
        df[session_groups].drop_duplicates().reset_index(drop=True)
    )

    # --- Session duration ---
    session_duration = (
        df.groupby(session_groups, as_index=False)['session_duration_seconds']
          .mean()
          .round(2)
    )
    session_duration['passed_10_min'] = session_duration['session_duration_seconds'] >= 600
    # --- Session start time ---
    session_start = (
        df.loc[df['event_name'] == 'Session Started']
          .groupby(session_groups, as_index=False)['session_start_time']
          .min()
    )

    # --- Question Started metrics ---
    q_started = df.loc[df['event_name'] == 'Question Started']
    qs_metrics = (
        q_started.groupby(session_groups, as_index=False)
        .agg(
            customer_character_count=('event_params__character_name', 'nunique'),
            character_list=('event_params__character_name',
                            lambda x: [v for v in x.dropna().tolist()]),
            average_tier=('event_params__current_tier', 'mean'),
        )
    )

    # --- Question Completed metrics ---
    q_completed = df.loc[df['event_name'] == 'Question Completed']
    qc_metrics = (
        q_completed.groupby(session_groups, as_index=False)
        .agg(average_wrong_answers=('event_params__answered_wrong', 'mean'))
    )

    # --- Wheel metrics ---
    wheel = (
        df.groupby(session_groups, as_index=False)['event_params__mini_game_ri']
          .agg(
              Wheel_Impression=lambda x: (x == 'Daily Spin').sum(),
              Wheel_Skips=lambda x: (x == 'spin_skipped').sum(),
          )
          .assign(Wheel_Spins=lambda d: d['Wheel_Impression'] - d['Wheel_Skips'])
    )

    # --- Ads watched ---
    ads = (
        df.groupby(session_groups, as_index=False)['event_name']
          .agg(Ads_Watched_Count=lambda x: (x == 'Ad Rewarded').sum())
    )

    # --- In-game currency (custom summarizer) ---
    gold = df.groupby(session_groups, as_index=False).apply(summarize_gold)
    if isinstance(gold.index, pd.MultiIndex):
        gold = gold.reset_index(drop=True)

    # --- Consumables purchased ---
    consumable = (
        df.loc[df['event_params__spent_to'] == 'Consumable Item']
          .groupby(session_groups, as_index=False)['shop_consumable_item']
          .agg(
              Potions_Bought=lambda x: (x == 'Potion').sum(),
              Incenses_Bought=lambda x: (x == 'Incense').sum(),
              Amulets_Bought=lambda x: (x == 'Amulet').sum(),
          )
    )

    # --- Energy spent ---
    energy = (
        df.loc[df['event_params__spent_to'].isin(['Cauldron', 'AliCin', 'Coffee'])]
          .groupby(session_groups, as_index=False)['event_params__spent_to']
          .agg(
              AliCin_Used=lambda x: (x == 'AliCin').sum(),
              Cauldron_Used=lambda x: (x == 'Cauldron').sum(),
              Coffee_Used=lambda x: (x == 'Coffee').sum(),
          )
    )
    skip_events = ['User Engagement', 
                   'Screen Viewed', 
                   'Earned Virtual Currency', 
                   'Firebase Campaign',
                   'App Removed', 
                   'App Data Cleared', 
                   'App Updated', 
                   'Starting Currencies'
                   ]

    df_l_sorted = df.sort_values(['event_datetime'], ascending=False)

    def pick_last_valid(g):
        non_skip = g[~g['event_name'].isin(skip_events)]
        row = non_skip.iloc[0] if len(non_skip) > 0 else g.iloc[0]
        return pd.DataFrame({
            'event_params__ga_session_id': [row['event_params__ga_session_id']],
            'user_pseudo_id': [row['user_pseudo_id']],
            'last_event_name': [row['event_name']],
            'last_event_time': [row['event_datetime']],
        })

    session_last_event = (
        df_l_sorted
        .groupby(session_groups, group_keys=False)
        .apply(pick_last_valid)
        .reset_index(drop=True)
)
    # --- Merge everything ---
    result = (
        base_sessions
        .merge(session_duration, on=session_groups, how='left')
        .merge(session_start, on=session_groups, how='left')
        .merge(qs_metrics, on=session_groups, how='left')
        .merge(qc_metrics, on=session_groups, how='left')
        .merge(wheel, on=session_groups, how='left')
        .merge(ads, on=session_groups, how='left')
        .merge(gold, on=session_groups, how='left')
        .merge(consumable, on=session_groups, how='left')
        .merge(energy, on=session_groups, how='left')
        .merge(session_last_event, on=session_groups, how='left')
        .fillna({
            'average_tier': 0,
            'average_wrong_answers': 0,
            'Ads_Watched_Count': 0,
            'Wheel_Impression': 0,
            'Wheel_Skips': 0,
            'Wheel_Spins': 0,
            'Potions_Bought': 0,
            'Incenses_Bought': 0,
            'Amulets_Bought': 0,
            'AliCin_Used': 0,
            'Cauldron_Used': 0,
            'Coffee_Used': 0,
        })
    )

    # --- Derived metric ---
    result['bought_new_customer'] = (
        result['customer_character_count'].fillna(0).astype(int) // 3
    )

    logger.info(
        f"Session-level dataframe created with {result.shape[0]} records "
        f"and {result.shape[1]} columns."
    )
    return result



def df_by_users(df: pd.DataFrame) -> pd.DataFrame:
    user_groups = ['user_pseudo_id']

    # Events to exclude from last_event_date computation
    exclude_last_events = ['App Removed', 
                           'App Data Cleared', 
                           'App Updated', 
                           'User Engagement', 
                           'Screen Viewed', 
                           'Firebase Campaign',
                           'Starting Currencies'
                           ]
    df_no_end = df[~df['event_name'].isin(exclude_last_events)]

    # --- Precompute masks for event counts ---
    is_ad = df['event_name'] == 'Ad Impression'
    is_question = df['event_name'] == 'Question Completed'

    # --- Base user-level aggregations ---
    
    df['session_duration_minutes'] = df['session_duration_seconds'] / 60
    
    user_df = (
        df.groupby(user_groups, as_index=False)
        .agg(
            first_event_date=('event_date', 'min'),
            total_sessions=('event_params__ga_session_id', 'nunique'),
            total_characters_opened=('event_params__character_name', 'nunique'),
            total_playtime_minutes=('session_duration_minutes', 'sum'),
            country=('geo__country', 'first'),
            install_source=('app_info__install_source', 'first'),
            operating_system=('device__operating_system', 'first'),
            operating_system_version=('device__operating_system_version', 'first'),
            is_limited_ad_tracking=('device__is_limited_ad_tracking', 'first'),
            device_language=('device__language', 'first'),
            version=('app_info__version', 'last'),
        )
    )
    user_df['total_playtime_minutes'] = user_df['total_playtime_minutes'].round(2)


    # --- Wheel metrics ---
    wheel = (
        df.groupby(user_groups, as_index=False)['event_params__mini_game_ri']
          .agg(
              Wheel_Impression=lambda x: (x == 'Daily Spin').sum(),
              Wheel_Skips=lambda x: (x == 'spin_skipped').sum(),
          )
          .assign(Wheel_Spins=lambda d: d['Wheel_Impression'] - d['Wheel_Skips'])
    )
    # --- Count event-based actions per user ---
    counts_df = (
        pd.concat(
            [
                df[user_groups],
                pd.DataFrame({
                    'total_ads_watched': (df['event_name'] == 'Ad Rewarded').astype(int),
                    'total_questions_answered': (df['event_name'] == 'Question Completed').astype(int),
                    'game_ended': (df['event_name'] == 'Game Ended').astype(int),
                    'tutorial_completed': (df['event_params__tutorial_video'] == 'tutorial_video').astype(int),
                    'session_started': (df['event_name'] == 'Session Started').astype(int),
                }, index=df.index),
            ],
            axis=1
        )
        .groupby(user_groups, as_index=False)
        .sum(numeric_only=True)
    )
    counts_df['second_session'] = (counts_df['session_started'] > 1).astype(int)
    counts_df = counts_df.drop(columns='session_started')

    # --- Merge base aggregates + counts + last_event_date ---

    last_event = (
        df_no_end
        .sort_values(['event_date'])
        .drop_duplicates(subset=user_groups, keep='last')
        [[*user_groups, 'event_date', 'event_name']]
        .rename(columns={'event_date': 'last_event_date',
                        'event_name': 'last_event_name'})
    )


    user_df = (
        user_df
        .merge(counts_df, on=user_groups, how='left')
        .merge(last_event, on=user_groups, how='left')
        .merge(wheel, on=user_groups, how='left')
        .fillna({
            'total_ads_watched': 0,
            'total_questions_answered': 0,
            'game_ended': 0,
        })
    )

    user_df['passed_10_min'] = (user_df['total_playtime_minutes'] >= 10).astype(int)

    logger.info(
        f"User-level dataframe created with {user_df.shape[0]} records and "
        f"{user_df.shape[1]} columns."
    )

    return user_df


def df_by_questions(df: pd.DataFrame) -> pd.DataFrame:
    question_groups = ['question_address', 
                       'event_params__character_name', 
                       'event_params__current_tier', 
                       'event_params__current_question_index',
                       'event_params__ga_session_id']

    # Boolean masks
    masks = {
        'question_started': df['event_name'].eq('Question Started'),
        'potions_bought':   df['shop_consumable_item'].eq('Potion'),
        'incense_bought':   df['shop_consumable_item'].eq('Incense'),
        'amulet_bought':    df['shop_consumable_item'].eq('Amulet'),
        'alicin_used':      df['event_params__spent_to'].eq('AliCin'),
        'coffee_used':      df['event_params__spent_to'].eq('Coffee'),
        'cauldron_used':    df['event_params__spent_to'].eq('Cauldron'),
        'scroll_opened':   (df['event_name'].eq('Menu Opened')) & (df['event_params__menu_name'].eq('Scroll Menu')),
        'answered_correct': df['event_name'].eq('Question Completed'),
        'answered_wrong':   df['event_params__answered_wrong'].fillna(0),
        'ads_watched':     df['event_name'].eq('Ad Rewarded').fillna(0),
    }

    # Convert masks to DataFrame of ints
    temp = pd.DataFrame({k: v.astype(int) if v.dtype == bool else v for k, v in masks.items()})
    temp = temp.reset_index(drop=True)

    # Combine and aggregate
    question_df = (
        pd.concat([df[question_groups].reset_index(drop=True), temp], axis=1)
        .groupby(question_groups, as_index=False)
        .sum()
    )
    question_df['wrong_answer_ratio'] = (
        question_df['answered_wrong'] / question_df['question_started'].replace(0, pd.NA)
    ).fillna(0).round(3)

    question_df['ads_watch_ratio'] = (
        question_df['ads_watched'] / question_df['question_started'].replace(0, pd.NA)
    ).fillna(0).round(3)

    question_df['alicin_use_ratio'] = (
        question_df['alicin_used'] / question_df['question_started'].replace(0, pd.NA)
    ).fillna(0).round(3)

    question_df['coffee_use_ratio'] = (
        question_df['coffee_used'] / question_df['question_started'].replace(0, pd.NA)
    ).fillna(0).round(3)

    question_df['cauldron_use_ratio'] = (
        question_df['cauldron_used'] / question_df['question_started'].replace(0, pd.NA)
    ).fillna(0).round(3)

    question_df['scroll_use_ratio'] = (
        question_df['scroll_opened'] / question_df['question_started'].replace(0, pd.NA)
    ).fillna(0).round(3)



    logger.info(
        f"Question-level dataframe created with {question_df.shape[0]} records and "
        f"{question_df.shape[1]} columns."
    )

    return question_df


def df_by_date(df: pd.DataFrame) -> pd.DataFrame:
    
    date_df = (
        df.groupby(['event_date'])
        .agg(
            weekday=('ts_weekday', 'first'),
            unique_users=('user_pseudo_id', 'nunique'),
            new_users=('event_name', lambda x: (x == 'First Open').sum()),
            android_users=('device__operating_system', lambda x: (x == 'ANDROID').sum()),
            ios_users=('device__operating_system', lambda x: (x == 'IOS').sum()),
            uninstall_count=('event_name', lambda x: (x == 'App Removed').sum()),
            unique_sessions=('event_params__ga_session_id', 'nunique'),
            ads_watched=('event_name', lambda x: (x == 'Ad Rewarded').sum()),
            questions_started=('event_name', lambda x: (x == 'Question Started').sum()),
            questions_completed=('event_name', lambda x: (x == 'Question Completed').sum()),
        )
        .reset_index()
    )

    ads_network_df = (
        df.groupby(['event_date', 'event_params__ad_network'])
        .size()
        .unstack(fill_value=0)
        .add_prefix('nwk_')
        .reset_index()
    )

    ads_unit_df = (
        df.groupby(['event_date', 'event_params__ad_unit_id'])
        .size()
        .unstack(fill_value=0)
        .add_prefix('unt_')
        .reset_index()
    )

    ads_instance_df = (
        df.groupby(['event_date', 'event_params__ad_instance'])
        .size()
        .unstack(fill_value=0)
        .add_prefix('ins_')
        .reset_index()
    )

    result = (
        date_df
        .merge(ads_network_df, on='event_date', how='left')
        .merge(ads_unit_df, on='event_date', how='left')
        .merge(ads_instance_df, on='event_date', how='left')
        .fillna(0)
    )

    logger.info(
        f"Date-level dataframe created with {result.shape[0]} records and "
        f"{result.shape[1]} columns."
    )
    return result

def df_technical_events(df: pd.DataFrame) -> pd.DataFrame:
    # Sort by user, session, and event time
    df = df.sort_values(['user_pseudo_id', 
                         'event_params__ga_session_id', 
                         'event_datetime', 
                         'app_info__version', 
                         'device__mobile_marketing_name',
                         'device__operating_system_version'])

    # Create previous event column within the same session
    df['prev_event_name'] = df.groupby(['user_pseudo_id', 'event_params__ga_session_id'])['event_name'].shift(1)
    df['prev_event_menu'] = df.groupby(['user_pseudo_id', 'event_params__ga_session_id'])['event_params__menu_name'].shift(1) if 'event_params__menu_name' in df.columns else pd.NA

    # Filter technical events
    tech_events = df[df['event_name'].isin(['App Exception', 'Ad Load Failed'])].copy()
    logger.info(
        f"Technical events dataframe created with {tech_events.shape[0]} records and "
        f"{tech_events.shape[1]} columns."
    )
    # Keep relevant columns
    return tech_events[['event_datetime',
                        'event_name', 
                        'user_pseudo_id', 
                        'event_params__ga_session_id',
                        'app_info__version',
                        'device__mobile_marketing_name',
                        'device__operating_system_version',
                        'prev_event_name',
                        'prev_event_menu',
                        'event_params__ad_network',
                        'event_params__ad_instance',
                        'event_params__ad_id',
                        'event_params__ad_error_code',
                        'event_server_delay_seconds' 
                        ]]

def df_most_active_by_time(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    
    session_groups = ['event_params__ga_session_id', 'user_pseudo_id']
    # --- Session duration ---      
    session_duration = (
        df.groupby(session_groups, as_index=False)['session_duration_seconds']
          .mean()
          .query('session_duration_seconds > 10')
          .round(2)
    )

    top_sessions = (
        session_duration
        .groupby('user_pseudo_id', as_index=False)
        .sum()
        .nlargest(top_n, 'session_duration_seconds')
        .sort_values('session_duration_seconds', ascending=False)
    )

    return top_sessions

def df_by_ads(df: pd.DataFrame) -> pd.DataFrame:

    ad_related_mask = df['event_name'].isin(['Ad Loaded', 'Ad Closed', 'Ad Displayed', 'Ad Rewarded', 'Ad Load Failed', 'Ad Clicked'])
    ads = df[ad_related_mask].copy()

    columns = [
        'event_datetime',
        'event_params__ga_session_id',
        'event_name',
        'event_params__ad_id',
        'event_params__ad_unit_id',
        'event_params__ad_network',
        'event_params__ad_placement',
        'event_params__ad_reward_type',
        'event_params__ad_instance',
        'event_params__ad_error_code',
        'event_params__character_name',
        'event_params__current_tier',
        'event_params__current_question_index',
        'question_address',
        'ts_weekday',
        'ts_daytime_named',
        'app_info__version',
        'geo__country',
        'device__operating_system',
        'event_server_delay_seconds',
    ]

    ads = ads[columns]

    fill_na = ['event_params__ad_network', 'event_params__ad_placement', 'event_params__ad_reward_type', 'event_params__ad_instance']
    ads[fill_na] = ads[fill_na].fillna('Unknown/Missing')


    logger.info(
        f"Rewarded ads dataframe created with {ads.shape[0]} records and "
        f"{ads.shape[1]} columns."
    )
    return ads