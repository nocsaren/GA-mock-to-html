import pandas as pd

from config.logging import get_logger

logger = get_logger(__name__)

def forward_fill_progress(df: pd.DataFrame, context=None) -> pd.DataFrame:

    # Infer and forward-fill the character name, current tier, and current question index within each session
    # Step 1: Sort chronologically within sessions
    df_sorted = df.sort_values(by=['user_pseudo_id', 'event_params__ga_session_id', 'event_datetime'])
    # Step 2: Forward-fill the relevant columns per user-session group
    cols_to_fill = [
        'event_params__character_name',
        'event_params__current_tier',
        'event_params__current_qi',
    ]
    df_sorted[cols_to_fill] = (
        df_sorted
        .groupby(['user_pseudo_id', 'event_params__ga_session_id'])[cols_to_fill]
        .ffill()
    )
    df.loc[df_sorted.index, cols_to_fill] = df_sorted[cols_to_fill]
    logger.info(f"Character names, tiers, and question indices forward-filled for {df['event_params__ga_session_id'].nunique()} unique sessions.")
    return df


def question_cumulative_qi(df: pd.DataFrame, context=None) -> pd.DataFrame:
# Calculate cumulative question index
    df['cumulative_question_index'] = df['event_params__current_question_index'].copy()
    df['cumulative_question_index'] = pd.to_numeric(df['cumulative_question_index'], errors='coerce')
    # Tier 2
    df.loc[(df['event_params__current_tier'] == 2) & (df['event_params__character_name'] == 't'), 'cumulative_question_index'] += 12
    df.loc[(df['event_params__current_tier'] == 2) & (df['event_params__character_name'] != 't'), 'cumulative_question_index'] += 16
    # Tier 3
    df.loc[(df['event_params__current_tier'] == 3) & (df['event_params__character_name'] == 't'), 'cumulative_question_index'] += 24
    df.loc[(df['event_params__current_tier'] == 3) & (df['event_params__character_name'] != 't'), 'cumulative_question_index'] += 28
    # Tier 4
    df.loc[(df['event_params__current_tier'] == 4) & (df['event_params__character_name'] == 't'), 'cumulative_question_index'] += 36
    df.loc[(df['event_params__current_tier'] == 4) & (df['event_params__character_name'] != 't'), 'cumulative_question_index'] += 40
    # NaNs
    df.loc[df['event_params__current_tier'].isna(), 'cumulative_question_index'] = pd.NA
    logger.info(f"Cumulative question index calculated for {df['cumulative_question_index'].notna().sum()} rows.")
    return df

def mini_game_features(df: pd.DataFrame, context=None) -> pd.DataFrame:
    # Split 'event_params_mini_game_ri' maze_hand_* into columns
    # e.g 'maze_hand_WomanHandTwo_maze_level_3'
    # Column to process
    col = 'event_params__mini_game_ri'
    # Filter rows starting with 'maze_hand'
    mask = df[col].str.startswith('maze_hand', na=False)
    # Split the matching rows by underscore
    parts = df.loc[mask, col].str.split('_', expand=True)
    # Extract Gender and Hand using the updated regex
    gender_hand = parts[2].str.extract(r'(?P<Gender>Woman|Man)Hand(?P<Hand>\w+)')
    # Extract Level (assumed to be in the last part)
    levels = parts[5]
    # Create new columns with extracted data
    df.loc[mask, 'maze_gender'] = gender_hand['Gender']
    df.loc[mask, 'maze_hand'] = gender_hand['Hand']
    df.loc[mask, 'maze_level'] = levels
    logger.info(f"Extracted maze hand data for {mask.sum()} rows.")
    return df

def mini_game_reward_split(df: pd.DataFrame, context=None) -> pd.DataFrame:
    # Split event_params_mini_game_ri buff_* into columns
    # e.g. 'buff_IncreaseXEnergy_gift_True_gold_False'
    # Column to process
    col = 'event_params__mini_game_ri'
    # Filter rows starting with 'buff'
    mask = df[col].str.startswith('buff', na=False)
    # Split the matching rows by underscore
    parts = df.loc[mask, col].str.split('_', expand=True)
    # Extract Buff Type and Level
    buff_type = parts[2].str.extract(r'(?P<BuffType>\w+)')
    # Extract Buff Gift and Gold status
    buff_gift = parts[3].str.extract(r'(?P<BuffGift>\w+)')
    buff_gold = parts[5].str.extract(r'(?P<BuffGold>\w+)')
    # Create new columns with extracted data
    df.loc[mask, 'buff_type'] = buff_type['BuffType']
    df.loc[mask, 'buff_gift'] = buff_gift['BuffGift'].str.lower() == 'true'
    df.loc[mask, 'buff_gold'] = buff_gold['BuffGold'].str.lower() == 'true'
    logger.info(f"Extracted buff data for {mask.sum()} rows.")
    return df

def mini_game_buffs(df: pd.DataFrame, context=None) -> pd.DataFrame:
    # Split event_params_mini_game_ri earned_buff_* into columns
    # e.g. 'earned_buff_GiveXCharacter'
    # Column to process
    col = 'event_params__mini_game_ri'
    # Filter rows starting with 'earned_buff'
    mask = df[col].str.startswith('earned_buff', na=False)
    # Split the matching rows by underscore
    parts = df.loc[mask, col].str.split('_', expand=True)
    # Extract Buff Type
    buff_type = parts[2].str.extract(r'(?P<BuffType>\w+)')
    # Create new columns with extracted data
    df.loc[mask, 'earned_buff_type'] = buff_type['BuffType']
    logger.info(f"Extracted earned buff data for {mask.sum()} rows.")

    return df

def mini_game_dolls(df: pd.DataFrame, context=None) -> pd.DataFrame:
    # Split event_params__spent_to doll values into columns
    # e.g. 'erjohndoll'
    # Column to process
    col = 'event_params__spent_to'
    # Filter rows including string 'doll'
    mask = df[col].str.contains('doll', na=False)
    # Split the string by name and doll
    parts = df.loc[mask, col].str.split('doll', expand=True)
    # Extract the doll name
    df.loc[mask, 'doll_name'] = parts[0].str.strip()  # Get the name before 'doll'
    # Rewrite the 'event_params__spent_to' column to just the doll name
    df.loc[mask, col] = 'Doll'
    logger.info(f"Extracted doll data for {mask.sum()} rows.")
    return df

def currency_define_permanent(df: pd.DataFrame, context=None) -> pd.DataFrame:
    # Write event_params_spent_to permanent shop item values into shop_permanent_item
    # list of possible values: dreamcatcher, catcollar, library1, library2, bugspray, schedule
    # Column to process
    col = 'event_params__spent_to'
    # Filter rows including values from the list
    mask = df[col].str.contains('dreamcatcher|catcollar|library1|library2|bugspray|schedule|crystal|horseshoe', na=False)
    # Create a new column for the shop permanent item
    df.loc[mask, 'shop_permanent_item'] = df.loc[mask, col].str.extract(r'(dreamcatcher|catcollar|library1|library2|bugspray|schedule|crystal|horseshoe)')[0]
    # Rewrite the 'event_params__spent_to' column to just the item name
    df.loc[mask, col] = 'Permanent Item'
    logger.info(f"Extracted permanent shop item data for {mask.sum()} rows.")
    return df

def currency_define_consumable(df: pd.DataFrame, context=None) -> pd.DataFrame:
    # Write event_params_spent_to consumable shop item values into shop_consumable_item
    # list of possible values: potion, ıncense, amulet, incense
    # Column to process
    col = 'event_params__spent_to'
    # Filter rows including values from the list
    mask = df[col].str.contains('potion|ıncense|amulet|incense', na=False)
    # Create a new column for the shop consumable item
    df.loc[mask, 'shop_consumable_item'] = df.loc[mask, col].str.extract(r'(potion|ıncense|amulet|incense)')[0]
    # Rewrite the 'event_params__spent_to' column to just the item name
    df.loc[mask, col] = 'Consumable Item'
    logger.info(f"Extracted consumable shop item data for {mask.sum()} rows.")
    return df


def currency_define_board(df: pd.DataFrame, context=None) -> pd.DataFrame:
    col = 'event_params__spent_to'
    # Filter rows that are not in the known categories
    mask = (~df[col].isin(['Doll', 'Crystal Ball', 'Permanent Item', 'Consumable Item'])) & \
        (df['event_params__where_its_spent'].isin(['board', 'board_item']))
    # Create a new column for the board item
    df.loc[mask, 'board_item'] = df.loc[mask, col]
    # Rewrite the 'event_params__spent_to' column to just the item name
    df.loc[mask, col] = 'Board Item'
    logger.info(f"Extracted board item data for {mask.sum()} rows.")
    return df

def currency_define_keys(df: pd.DataFrame, context=None) -> pd.DataFrame:
    df.loc[df['event_params__spent_to'] == 'key', 'event_params__spent_to'] = 'Key'
    logger.info(f"Defined keys in spent_to for {df['event_params__spent_to'].eq('Key').sum()} rows.")
    return df



def question_addressable_index(df: pd.DataFrame, context=None) -> pd.DataFrame:
    df['question_address'] = df['event_params__character_name'] + ' - T: ' + df['event_params__current_tier'].astype(str) + ' - Q: ' + df['event_params__current_question_index'].astype(str)
    logger.info(f"Question address created for {df['question_address'].notna().sum()} rows.")
    return df

def question_answer_wrong_zeros(df: pd.DataFrame, context=None) -> pd.DataFrame:
    mask = (df['event_name'] == 'Question Completed') & (df['event_params__answered_wrong'].isna())
    df.loc[mask, 'event_params__answered_wrong'] = 0
    logger.info("Filled NaN values in 'event_params__answered_wrong' with 0 for 'Question Completed' events.")
    return df
