'''
This module contains lists and maps used for organizing and displaying data in the application.
designed to be used in conjunction with the main application logic.
'''


# This is a list and order of the column names that I used for conveniancy in EDA and LookerStudio. 

df_column_names_map = {

    # 🎯 Event Core Info
    'event_name': 'event_name',
    'event_bundle_sequence_id': 'event_batch_id',
    'user_pseudo_id': 'user_pseudo_id',
    'stream_id': 'stream_identifier',
    'platform': 'platform',
    'is_active_user': 'is_active_user',
    'batch_event_index': 'event_index_in_batch',

    # ⏱️ Event Timing
    'event_datetime': 'event_datetime',
    'event_date': 'event_date',
    'event_time': 'event_time',
    'event_previous_date': 'previous_event_date',
    'event_previous_time': 'previous_event_time',
    'event_first_touch_date': 'first_touch_date',
    'event_first_touch_time': 'first_touch_time',
    'time_delta': 'time_since_previous_event',
    'ts_weekday': 'weekday',
    'ts_local_time': 'local_time',
    'ts_hour': 'hour_of_day',
    'ts_daytime_named': 'time_of_day',
    'ts_is_weekend': 'is_weekend',
    'device__time_zone_offset_hours': 'time_zone_offset_hours',
    'event_server_delay_seconds': 'server_delay_seconds',

    # 🧭 Session Tracking
    'event_params__ga_session_id': 'session_id',
    'event_params__ga_session_number': 'session_number',
    'event_params__firebase_event_origin': 'firebase_event_origin',
    'event_params__engaged_session_event': 'engaged_session_event',
    'event_params__session_engaged': 'session_was_engaged',
    'user__ga_session_number': 'user_session_count',
    'user__ga_session_id': 'user_session_id',
    'inferred_session_id': 'inferred_session_id',
    'session_duration_seconds': 'session_duration_seconds',
    'session_duration_minutes': 'session_duration_minutes',
    'session_duration_hours': 'session_duration_hours',
    'session_start_time': 'session_start_time',
    'session_end_time': 'session_end_time',

    # 📺 Screen & Navigation
    'event_params__firebase_screen_id': 'screen_id',
    'event_params__firebase_screen_class': 'screen_class',
    'event_params__menu_name': 'menu_name',
    'event_params__entrances': 'entrances_count',

    # 📱 Device Info
    'device__category': 'device_type',
    'device__mobile_brand_name': 'mobile_brand',
    'device__mobile_model_name': 'mobile_model',
    'device__mobile_os_hardware_model': 'device_hardware_model',
    'device__operating_system': 'operating_system',
    'device__operating_system_version': 'os_version',
    'device__advertising_id': 'advertising_id',
    'device__language': 'device_language',
    'device__is_limited_ad_tracking': 'ad_tracking_limited',

    # 🌍 Geolocation
    'geo__city': 'city',
    'geo__country': 'country',
    'geo__continent': 'continent',
    'geo__region': 'region',
    'geo__sub_continent': 'subcontinent',
    'geo__metro': 'metro_area',

    # 📲 App Metadata
    'app_info__id': 'app_id',
    'app_info__version': 'app_version',
    'app_info__install_store': 'install_store',
    'app_info__firebase_app_id': 'firebase_app_id',
    'app_info__install_source': 'app_install_source',

    # 🔒 Privacy Settings
    'privacy_info__analytics_storage': 'consent_analytics_storage',
    'privacy_info__ads_storage': 'consent_ads_storage',

    # 📈 Ads & Monetization
#    'event_params__ad_platform': 'ad_platform',
    'event_params__ad_shown_where': 'ad_shown_in',
    'event_params__ad_unit_id': 'ad_unit_id',
    'event_params__ad_network': 'ad_network',
#    'event_params__ad_format': 'ad_format',
    'event_params__firebase_conversion': 'conversion_event',

    # 🛠️ Debug / Quality
    'event_params__firebase_error': 'firebase_error',
    'event_params__fatal': 'fatal_error',
    'event_params__timestamp': 'raw_timestamp',

    # ⏳ Engagement & Duration
    'event_params__engagement_time_seconds': 'engagement_time_seconds',
    'event_params__time_spent_seconds': 'time_spent_on_activity_seconds',

    # 🧠 Gameplay (Maze & Buffs)
    'maze_gender': 'maze_gender',
    'maze_hand': 'maze_hand',
    'maze_level': 'maze_level',
    'buff_type': 'buff_type',
    'buff_gift': 'buff_gift',
    'buff_gold': 'buff_gold',
    'earned_buff_type': 'earned_buff_type',

    # 🎮 Game Event Data
    'event_params__current_qi': 'original_qi',
    'question_address': 'question_address',
    'event_params__character_name': 'character_name',
    'event_params__current_tier': 'current_tier',
    'event_params__current_question_index': 'current_question_index',
    'event_params__current_qi': 'original_qi',
    'cumulative_question_index': 'cumulative_question_index',
    'question_address': 'question_address',
    'event_params__mini_game_ri': 'event_params__mini_game_ri',
    'event_params__mini_game_name': 'mini_game_name',
    'event_params__answered_wrong': 'answered_incorrectly',

    # 💰 Currency Events
    'event_params__where_its_earned': 'where_currency_was_earned',
    'event_params__currency_name': 'currency_name',
    'event_params__earned_amount': 'currency_earned',
    'event_params__how_its_earned': 'how_currency_was_earned',
    'event_params__spent_amount': 'currency_spent',
    'event_params__where_its_spent': 'where_currency_was_spent',
    'event_params__spent_to': 'spent_on',

    # 🛒 Shop
    'doll_name': 'doll_name',
    'spent_in_crystal': 'spent_in_crystal',
    'shop_permanent_item': 'permanent_item',
    'shop_consumable_item': 'consumable_item',
    'board_item': 'board_item',

    # 🔄 User Lifecycle
    'user__first_open_time': 'user_first_open_time',
    'user__first_open_date': 'user_first_open_date',
    'event_params__previous_first_open_count': 'previous_first_open_count',

}




columns_to_drop = ['event_timestamp',
                    'event_previous_timestamp', 
                    'user_first_touch_timestamp', 
                    'device__time_zone_offset_seconds', 
                    'event_params__engagement_time_msec',
                    'event_previous_datetime',
                    'event_params__time_spent',
                    'event_first_touch_datetime',
                    'user__first_open_datetime',
                    'event_value_in_usd', 
                    'user_id', 
                    'batch_page_id',	
                    'batch_ordering_id', 
                    'privacy_info__uses_transient_token',
                    'user_ltv',
                    'device__mobile_marketing_name', 
                    'device__vendor_id',
                    'device__browser', 
                    'device__browser_version', 
                    'device__web_info', 
                    'event_dimensions',
                    'traffic_source__name', 
                    'traffic_source__medium',	
                    'traffic_source__source',	
                    'ecommerce', 
                    'event_server_timestamp_offset',
                    'event_params__update_with_analytics', 
                    'event_params__system_app_update',
                    'collected_traffic_source',
                    'event_params__system_app'
]



# These dictionaries map specific VALUES to their display names.
# They are used to convert raw values in the data to more user-friendly names.


event_name_map = {'ad_clicked': 'Ad Clicked',
                  'app_remove': 'App Removed',
                  'first_open': 'First Open',
                  'menu_closed': 'Menu Closed',
                  'menu_opened': 'Menu Opened',
                  'screen_view': 'Screen Viewed',
                  'ad_impression': 'Ad Impression',
                  'session_start': 'Session Started',
                  'app_clear_data': 'App Data Cleared',
                  'user_engagement': 'User Engagement',
                  'question_started': 'Question Started',
                  'mini_game_started': 'Mini-game Started',
                  'question_completed': 'Question Completed',
                  'mini_game_completed': 'Mini-game Completed',
                  'earn_virtual_currency': 'Earned Virtual Currency',
                  'spend_virtual_currency': 'Spent Virtual Currency',
                  'mini_game_failed': 'Mini-game Failed',
                  'app_exception': 'App Exception',
                  'app_update': 'App Updated',
                  'ad_loaded': 'Ad Loaded',
                  'ad_load_failed': 'Ad Load Failed',
                  'game_ended': 'Game Ended',
                  'start_currencies': 'Starting Currencies',
                  'video_watched': 'Video Watched',
                  'ad_rewarded': 'Ad Rewarded',
                  'ad_displayed': 'Ad Displayed',
                  'ad_closed': 'Ad Closed',
                  'firebase_campaign': 'Firebase Campaign',
                    }



event_params__mini_game_ri_map = {'stone_game': 'Stone Game',                                 
                                  'cauldron_game': 'Cauldron Game',
                                  'coffee_game': 'Coffee Game',
                                  'card_game': 'Card Game',
                                  'daily_spin': 'Daily Spin',
                                  'completed': 'Completed',
                                  'failed': 'Failed',
                                  'gold_500': 'Gold 500',
                                      }

event_params__menu_name_map = {'Scroll Menu' : 'Scroll Menu',
                               'crystal_menu' : 'Crystal Menu',
                               'crystal_aliginn_menu' : 'Crystal Alignin Menu',
                               'wanna_play_menu' : 'Wanna Play Menu',
                               'shop_menu' : 'Shop Menu',
                               'board_menu' : 'Board Menu',
                               'crystal_character_menu' : 'Crystal Character Menu',
                               'energy_gold_exchange' : 'Energy Gold Exchange',
                               'crystal_cauldron_menu' : 'Crystal Cauldron Menu',
                               'crystal_energy_menu' : 'Crystal Energy Menu',
                               'crystal_coffee_menu' : 'Crystal Coffee Menu',
                               'wheel_of_fortune' : 'Wheel of Fortune',
                               'scroll_menu' : 'Scroll Menu',
                                    }

event_params__spent_to_map = {'cauldron_item' : 'Cauldron',
                              'aliginn_item' : 'AliCin',
                              'coffee_item' : 'Coffee'
                              }
 
event_params__character_name_map = { 'aturtle': 'A Turtle',
                                     'littlea': 'Little A',
                                     'sinnct': 'Sinnct',
                                     'obviousjoe': 'Obvious Joe',
                                     'erjohn': 'ER John',
                                     'billy': 'Billy',
                                     'maydenis': 'Maydenis',
                                     'almiralotus': 'Almira Lotus',
                                     't': 'T',
                                     'mrspearl': 'Mrs. Pearl',
                                     'therock': 'The Rock',
                                     'army': 'Army',
                                     'ladydodo': 'Lady Dodo',
                                     'dlion': 'D Lion',
                                     'frenchie': 'Frenchie',
                                     'mi': 'Mi',
                                     'biga': 'Big A',
                                     'cjay': 'C Jay',
                                     'mo': 'Mo',
                                     'mryogurt': "Mr. Yogurt",
                                     'crystalraw': "Crystal Raw",
                                     'aisha': "Aisha",
                                     'mustafa': "Mustafa",
                                     'fathergold': "Father Gold",
                                     'tracy': "Tracy",
                                     'whymargaret': "Why Margaret",
                                     'suedoluni': "Sue Doluni",
                                     'joe': 'Obvious Joe',
                                        }

event_params__mini_game_name_map = { 'stone_mini_game': 'Stone Game',
                                     'cauldron_mini_game': 'Cauldron Game',
                                     'star_mini_game': 'Star Game',
                                     'maze_mini_game': 'Maze Game',
                                     'coffee_mini_game': 'Coffee Game',
                                     'card_mini_game': 'Card Game',
                                     'wheel_of_fortune': 'Wheel of Fortune',
                                     'voodoo_mini_game': 'Voodoo Game',
                                     'catch_up_cauldron': 'Catch Up Cauldron',
                                     'catch_up_coffee': 'Catch Up Coffee',
                                        }





event_params__where_its_earned_map = { 'mini_game': 'Mini-game',
                                       'question': 'Question',
                                       'wanna_play': 'Wanna Play',
                                       'energy_gold_exchange': 'Energy Gold Exchange',
}



event_params__currency_name_map = { 'gold': 'Gold',
                                    'energy': 'Energy'
}


event_params__how_its_earned_map = { 'combo': 'Combo',
                                     'normal': 'Normal',
                                     'mini_game_completed': 'Mini-game Completed',
                                     'star_mini_game': 'Star Game',
                                     'CauldronMiniGame': 'Cauldron Game',
                                     'CoffeeMiniGame': 'Coffee Game',
                                     'card_mini_game': 'Card Game',
                                     'maze_mini_game': 'Maze Game',
                                     'mini_game_failed': 'Mini-game Failed',
}
                                    
event_params__where_its_spent_map = { 'shop': 'Shop',
                                      'board': 'Board',
                                      'crystal': 'Crystal',
}

event_params__ad_shown_where_map = { 'crystal_character_ad': 'Crystal Character',
                                     'crystal_energy_ad': 'Crystal Energy',
                                     'wheel_of_fortune_ad': 'Wheel of Fortune',
                                     'ad_shown_where': 'ADSHOWNWHERE',
                                     'wanna_play_ad': 'Wanna Play',
                                     'EnergyGoldExchangeAd': 'Energy Gold Exchange'
}


shop_consumable_item_map = {'potion': 'Potion',
                            'incense': 'Incense',
                            'amulet': 'Amulet',
                            'ıncense': 'Incense',
}
                            
shop_permanent_item_map = {'dreamcatcher': 'Dreamcatcher',
                           'catcollar': 'Cat Collar',
                           'library1': 'Library 1',
                           'library2': 'Library 2',
                           'bugspray': 'Bug Spray',
                           'schedule': 'Schedule',
                           'crystal': 'Crystal',
                           'horseshoe': 'Horseshoe'
}

ts_weekday_map = {'Monday': 'Pazartesi',
                  'Tuesday': 'Salı',
                  'Wednesday': 'Çarşamba',
                  'Thursday': 'Perşembe',
                  'Friday': 'Cuma',
                  'Saturday': 'Cumartesi',
                  'Sunday': 'Pazar'
}


map_of_maps = {
    'event_name': event_name_map,
    'event_params__mini_game_ri': event_params__mini_game_ri_map,
    'event_params__menu_name': event_params__menu_name_map,
    'event_params__character_name': event_params__character_name_map,
    'event_params__mini_game_name': event_params__mini_game_name_map,
    'event_params__where_its_earned': event_params__where_its_earned_map,
    'event_params__currency_name': event_params__currency_name_map,
    'event_params__how_its_earned': event_params__how_its_earned_map,
    'event_params__where_its_spent': event_params__where_its_spent_map,
    'event_params__ad_shown_where': event_params__ad_shown_where_map,
    'doll_name' : event_params__character_name_map,
    'event_params__spent_to': event_params__spent_to_map,
    'shop_consumable_item': shop_consumable_item_map,
    'shop_permanent_item': shop_permanent_item_map,
    'ts_weekday': ts_weekday_map,
}

# Dataframe Splits

df_splits = {
    'sessions_df' : ['user_pseudo_id', 
                     'event_date',
                     'event_datetime', 
                     'inferred_session_id',
                     'session_duration_seconds',
                     'session_duration_minutes',
                     'session_duration_hours',
                     'session_start_time',
                     'time_of_day',
                     'weekday',
                     'session_end_time',
                     'event_name',
                     'question_address',
                     'mini_game_name',
                     'ad_unit_id',
                     'menu_name',
                     'country',
                     'app_version'
    ],
    'players_df' :[
                    # Original dataframe columns
                    'event_name', 
                    'user_pseudo_id',
                    'event_date',  
                    'inferred_session_id',
                    'country',
                    'app_version',

                    # Aggregated user metrics
                    'first_seen',              # user's earliest event datetime
                    'last_seen',               # user's latest event datetime
                    'total_sessions',          # count of unique sessions per user
                    'total_events',            # total event count per user

                    # Calculated durations and dates
                    'lifetime_days',           # days between first_seen and last_seen
                    'days_since_last_seen',    # days since user's last activity

                    # Boolean flags (churn/retention/activity)
                    'is_churned',              # churned if no activity in 14+ days
                    'is_retained_1d',          # retained at least 0 days (exists)
                    'is_retained_7d',          # retained 7 or more days
                    'is_retained_30d',         # retained 30 or more days
                    'active_days',             # number of unique active days (normalized)
                    'is_active_1d',            # active 2+ unique days
                    'is_active_7d',            # active 7+ unique days
                    'is_active_30d',           # active 30+ unique days
                    'is_active_yesterday',     # active within last 24 hours

                    # Categorical status label
                    'user_status'              # status label based on activity ("Bırakmış", "Aktif", "Yeni", "dormant")
                    ],
    'character_df' : ['user_pseudo_id',
                      'event_date',
                      'event_name',
                      'current_tier',
                      'character_name',
                      'cumulative_question_index',
                      'app_version'
                      ],
    'questions_df' : ['user_pseudo_id', 
                      'event_date',
                      'event_name',
                      'answered_incorrectly',
                      'question_address',
                      'character_name',
                      'current_tier',
                      'current_question_index',
                      'cumulative_question_index',
                      'app_version'
    ],
    'mini_games_df' : ['user_pseudo_id',
                       'event_date',
                       'event_datetime',
                       'mini_game_name',
                       'event_name',
                       'time_spent_on_activity_seconds',
                       'app_version'
    ],
    'crystal_and_energy_df' : ['user_pseudo_id',
                               'event_name',
                               'event_date',
                               'spent_in_crystal',
                               'question_address',
                               'inferred_session_id',
                               'where_currency_was_earned',
                               'how_currency_was_earned',
 #                              'ad_shown_in',
                               'currency_earned',
                               'currency_name',
                               'currency_spent',
                               'cumulative_question_index',
                               'where_currency_was_spent',
                               'current_tier',
                               'current_question_index',
                               'app_version'

    ],
    'shop_and_gold_df' : ['user_pseudo_id',
                          'event_name',
                          'event_date', 
                          'inferred_session_id',
                          'currency_name',
                          'currency_earned',
                          'currency_spent',
                          'where_currency_was_spent',
                          'how_currency_was_earned',
                          'where_currency_was_earned',
                          'permanent_item',
                          'consumable_item',
                          'board_item',
                          'doll_name',
                          'question_address',
                          'character_name',
                          'current_tier',
                          'current_question_index', 
                          'spent_on',
                          'app_version'                    
    ],
    'ads_df' : ['user_pseudo_id', 
                'event_name', 
                'event_date',
                'event_datetime',
                'inferred_session_id',
#                'ad_platform', 
#                'ad_shown_in', 
                'ad_unit_id', 
                'ad_network', 
#                'ad_format', 
                'conversion_event', 
                'question_address',
                'cumulative_question_index',
                'app_version'

    ], 
    'technical_df' : ['user_pseudo_id',
                      'event_name',
                      'event_date',
                      'event_datetime',
                      'inferred_session_id',
                      'server_delay_seconds',
                      'platform',
                      'device_type',
                      'operating_system',
                      'os_version',
                      'device_language',
                      'app_version',
                      'fatal_error',
    ]
}


df_filters = {
    'sessions_df': lambda df: df['inferred_session_id'].notna(),
    'players_df': lambda df: df['user_pseudo_id'].notna(),
    'character_df': lambda df: df['character_name'].notna(),
    'questions_df': lambda df: df['current_question_index'].notna(),
    'mini_games_df': lambda df: df['mini_game_name'].notna(), 
    'crystal_and_energy_df': lambda df: (df['currency_name'] == 'Energy') | (df['spent_in_crystal'].notna()),
    'shop_and_gold_df': lambda df: (df['currency_name'] == 'Gold'),
    'ads_df': lambda df: df['ad_unit_id'].notna(),
}
