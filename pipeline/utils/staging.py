import pandas as pd
from config.logging import get_logger

logger = get_logger(__name__)



PIPELINE_STAGES = [
    ("pull_from_bq", "Pull data from BigQuery"),
    ("flatten_dataframe", "Flatten dataframe"),
    ("dots_to_underscores", "Convert dots in column names to underscores"),
    ("transform_datetime_fields", "Transform datetime fields"),
    ("add_time_based_features", "Add time-based features"),
    ("add_durations", "Add durations"),
    ("forward_fill_progress", "Forward-fill progress"),
    ("question_index_cleanup", "Clean up question index"),
    ("calculate_cumulative_qi", "Calculate cumulative question index"),
    ("mini_game_features", "Engineer mini-game features"),
    ("mini_game_reward_split", "Split mini-game reward info"),
    ("mini_game_buffs", "Engineer mini-game buff features"),
    ("mini_game_dolls", "Engineer mini-game doll features"),
    ("currency_define_permanent", "Define permanent currency"),
    ("currency_define_consumable", "Define consumable currency")
]

def run_pipeline(df: pd.DataFrame, context: dict) -> pd.DataFrame:
    from pipeline.utils.main_functions import filter_events_by_date
    from pipeline.utils.lists_and_maps import map_of_maps
    from pipeline.utils.pull_functions import pull_from_bq
    from pipeline.utils.flattening_functions import flatten_dataframe
    from pipeline.utils.time_and_date_functions import (
        transform_datetime_fields,
        add_time_based_features,
        add_durations
    )

       
    from pipeline.utils.feature_engineering import (              
        forward_fill_progress,
        question_cumulative_qi,
        mini_game_features,
        mini_game_reward_split,
        mini_game_buffs,
        mini_game_dolls,
        currency_define_permanent,
        currency_define_consumable,
        currency_define_board,
        currency_define_keys,
        question_addressable_index,
        question_answer_wrong_zeros
    )
    from pipeline.utils.cleaning_functions import (
        question_index_cleanup,
        dots_to_underscores,
        apply_value_maps)


    # Put the stage functions in a list in the right order
    stages = [
        pull_from_bq,
        filter_events_by_date,
        flatten_dataframe,
        dots_to_underscores,
        transform_datetime_fields,
        add_time_based_features,
        add_durations,
        forward_fill_progress,
        question_index_cleanup,
        question_cumulative_qi,
        mini_game_features,
        mini_game_reward_split,
        mini_game_buffs,
        mini_game_dolls,
        currency_define_permanent,
        currency_define_consumable,
        currency_define_board,
        currency_define_keys,
        apply_value_maps,
        question_addressable_index,
        question_answer_wrong_zeros
    ]

    for stage in stages:
        stage_name = stage.__name__
        logger.info(f"Running {stage_name}...")

        # Each stage accepts df and context
        df = stage(df=df, context=context)

        logger.info(f"{stage_name} done.")

    return df
