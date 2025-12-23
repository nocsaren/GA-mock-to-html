import pandas as pd


def summarize_gold(g):

    ''' Summarize gold currency metrics for a session 
    1. gold_starting: Total gold at session start
    2. gold_gained: Total gold earned during session
    3. gold_spent: Total gold spent during session
    4. gold_delta: Net change in gold (gained - spent)
    '''

    gold_starting = 0

    try:
        if (
            isinstance(g, pd.DataFrame)
            and 'event_name' in g.columns
            and 'event_params__gold' in g.columns
        ):
            subset = g.loc[g['event_name'] == 'start_currencies', 'event_params__gold']
            # Handle non-numeric values safely
            gold_starting = pd.to_numeric(subset, errors='coerce').sum()
    except Exception:
        gold_starting = 0

    gold_gained = g.loc[
        (g['event_name'] == 'Earned Virtual Currency') &
        (g['event_params__currency_name'] == 'Gold'),
        'event_params__earned_amount'
    ].sum()
    gold_spent = g.loc[
        (g['event_name'] == 'Spent Virtual Currency') & 
        (g['event_params__currency_name'] == 'Gold'),
        'event_params__spent_amount'
    ].sum()
    gold_delta = gold_gained - gold_spent

    is_depted_for_doll = int((gold_spent > (gold_starting + gold_gained)) and (gold_spent >= 2000))

    return pd.Series({
        'gold_starting': gold_starting,
        'gold_gained': gold_gained,
        'gold_spent': gold_spent,
        'gold_delta': gold_delta,
        'is_depted_for_doll': is_depted_for_doll
    })

""" 

I DONT NEED THIS YET

def summarize_energy(e):

    ''' Summarize energy currency metrics for a session 
    1. energy_starting: Total energy at session start
    2. energy_gained: Total energy earned during session
    3. energy_spent: Total energy spent during session
    4. energy_delta: Net change in energy (gained - spent)
    '''

#    energy_starting = e.loc[e['event_name'] == 'start_currencies', 'event_params__energy'].sum()
    energy_gained = e.loc[
        (e['event_name'] == 'Earned Virtual Currency') &
        (e['event_params__currency_name'] == 'Energy'),
        'event_params__earned_amount'
    ].sum()
    energy_spent = e.loc[
        (e['event_name'] == 'Spent Virtual Currency') & 
        (e['event_params__currency_name'] == 'Energy'),
        'event_params__spent_amount'
    ].sum()
    energy_delta = energy_gained - energy_spent

    return pd.Series({
#        'energy_starting': energy_starting,
        'energy_gained': energy_gained,
        'energy_spent': energy_spent,
        'energy_delta': energy_delta
    })
    
"""