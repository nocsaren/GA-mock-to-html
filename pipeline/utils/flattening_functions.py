import pandas as pd
import json
from config.logging import get_logger

logger = get_logger(__name__)

def flatten_dataframe(df, context=None):
    """
    Flattens a DataFrame by expanding nested dictionaries in each row.
    
    Args:
    - df: The input pandas DataFrame with potential nested dictionary columns.
    
    Returns:
    - A new pandas DataFrame with flattened structure.
    """
    return pd.DataFrame([flatten_row(row) for _, row in df.iterrows()])



def flatten_extract_params(params_list):
    """
    Converts a list of Firebase-style param dicts into a flat {key: value} dict.
    Prioritizes value types: string > int > float > double.
    Example input: [{'key': 'foo', 'value': {'string_value': 'bar'}}]
    Returns: {'foo': 'bar'}    
    """
    result = {}

    try: 
        if isinstance(params_list, list):
            for param in params_list:
                key = param.get('key')
                val_dict = param.get('value', {})
                # Prioritize types: string > int > float > double
                val = (
                    val_dict.get('string_value') or
                    val_dict.get('int_value') or
                    val_dict.get('float_value') or
                    val_dict.get('double_value')
                )
                result[key] = val
        return result
    except Exception as e:
        logger.error(f"Error flattening params: {e}")
        return {}


def flatten_nested_column(row, col_name, flat, default_keys=None):
    """
    Flattens a nested column in the row if it's a dictionary. If the data is missing
    (None or NaN), it handles it by updating with `None` for default keys.
    
    Args:
    - row: The row of data (as a pandas Series)
    - col_name: The name of the column to flatten
    - flat: The dictionary where flattened data will be updated
    - default_keys: A list of keys to set `None` if the column data is missing (optional)
    
    Returns:
    - None: This function updates the `flat` dictionary in place.
    """
    # Get the column value
    col_data = row.get(col_name, None)  # Get the data for the given column
    
    # Check if the data is a dictionary
    try:
        if isinstance(col_data, dict):
            # Flatten the dictionary and add it to the 'flat' dictionary
            flat.update({f'{col_name}.{k}': v for k, v in col_data.items()})
        elif col_data is None or isinstance(col_data, float):  # Handle NaN or missing data
            # If it's None or NaN, we update with default keys set to None
            if default_keys is not None:
                flat.update({f'{col_name}.{k}': None for k in default_keys})
            else:
                flat.update({f'{col_name}': None})  # Add the key with None if no default keys
    except Exception as e:
        logger.error(f"Error flattening column {col_name}: {e}")
        if default_keys is not None:
            flat.update({f'{col_name}.{k}': None for k in default_keys})
        else:
            flat.update({f'{col_name}': None})  # Add the key with None if no default keys


def flatten_row(row):
    # Helper to parse JSON strings if necessary
    def parse_json(value):
        if isinstance(value, str):
            try:
                return json.loads(value)  # Convert string to dictionary
            except json.JSONDecodeError:
                return value  # Return as is if it's not a valid JSON string
        return value

    # Base-level fields
    flat = {
        'event_date': row.get('event_date'),
        'event_timestamp': row.get('event_timestamp'),
        'event_name': row.get('event_name'),
        'event_previous_timestamp': row.get('event_previous_timestamp'),
        'event_value_in_usd': row.get('event_value_in_usd'),        
        'event_bundle_sequence_id': row.get('event_bundle_sequence_id'),        
        'event_server_timestamp_offset': row.get('event_server_timestamp_offset'),        
        'user_id': row.get('user_id'),
        'user_pseudo_id': row.get('user_pseudo_id'),
        'user_first_touch_timestamp': row.get('user_first_touch_timestamp'),
        'stream_id': row.get('stream_id'),
        'platform': row.get('platform'),
        'is_active_user': row.get('is_active_user'),
        'batch_event_index': row.get('batch_event_index'),
        'batch_page_id': row.get('batch_page_id'),
        'batch_ordering_id': row.get('batch_ordering_id'),
        'batch_page_id': row.get('batch_page_id')
    }

    # Parse nested columns if they're in string format
    row['device'] = parse_json(row.get('device', {}))
    row['geo'] = parse_json(row.get('geo', {}))
    row['app_info'] = parse_json(row.get('app_info', {}))
    row['traffic_source'] = parse_json(row.get('traffic_source', {}))
    row['user_ltv'] = parse_json(row.get('user_ltv', {}))

    # Nested: event_params
    event_params = flatten_extract_params(row.get('event_params', []))
    flat.update({f'event_params.{k}': v for k, v in event_params.items()})
    
    # Nested: user_properties
    user_props = flatten_extract_params(row.get('user_properties', []))
    flat.update({f'user.{k}': v for k, v in user_props.items()})

    # Nested: privacy_info (flat dict)
    privacy = row.get('privacy_info', {})
    flat.update({f'privacy_info.{k}': v for k, v in privacy.items()})

    # Nested: user_ltv
    flatten_nested_column(row, 'user_ltv', flat, default_keys=None)  # Provide keys if needed
    
    # Nested: device
    device = row.get('device', {})
    flat.update({f'device.{k}': v for k, v in device.items()})

    # Nested: geo
    geo = row.get('geo', {})
    flat.update({f'geo.{k}': v for k, v in geo.items()})

    # Nested: app_info
    app_info = row.get('app_info', {})
    flat.update({f'app_info.{k}': v for k, v in app_info.items()})

    # Nested: event_dimensions
    flatten_nested_column(row, 'event_dimensions', flat, default_keys=None)  # Provide keys if needed

    # Nested: traffic_source
    traffic_source = row.get('traffic_source', {})
    flat.update({f'traffic_source.{k}': v for k, v in traffic_source.items()}),
    
    # Nested: ecommerce
    flatten_nested_column(row, 'ecommerce', flat, default_keys=None)  # Provide keys if needed

    # Nested: items
    flatten_nested_column(row, 'items', flat, default_keys=None)  # Provide keys if needed

    # Nested: item_params
    item_params = flatten_extract_params(row.get('item_params', []))
    flat.update({f'item_params.{k}': v for k, v in item_params.items()})

    # Nested: collected_traffic_source
    flatten_nested_column(row, 'collected_traffic_source', flat, default_keys=None)  # Provide keys if needed

    return flat