import pandas as pd
import plotly.graph_objects as go


from config.logging import get_logger

import datetime as dt

logger = get_logger(__name__)

def create_wrong_answers_heatmap(df: pd.DataFrame):

    # --- constants controlling auto sizing ---
    row_px = 35          # pixels per row
    col_px = 25          # pixels per column
    header_px = 100      # space for x-axis labels
    left_px = 100        # space for y-axis labels

    min_height = 400
    max_height = 1800
    min_width = 700
    max_width = 1800

    # ------------------ aggregation ------------------
    heat = (
        df.groupby([
            'event_params__character_name',
            'event_params__current_tier',
            'event_params__current_question_index'
        ])['wrong_answer_ratio']
        .mean()
        .reset_index()
    )

    pivot = heat.pivot_table(
        index='event_params__character_name',
        columns=['event_params__current_tier', 'event_params__current_question_index'],
        values='wrong_answer_ratio',
        fill_value=0
    )

    pivot = pivot.sort_index(axis=1, level=[0, 1])
    pivot = pivot.reindex(sorted(pivot.index, key=lambda x: str(x).lower()), axis=0)

    # Build axis labels
    tiers = [f"{t}" for t, q in pivot.columns]
    questions = [f"{q}" for t, q in pivot.columns]

    # ------------------ auto figure size ------------------
    n_rows = len(pivot.index)
    n_cols = len(pivot.columns)

    height = header_px + n_rows * row_px
    height = max(min_height, min(height, max_height))

    width = left_px + n_cols * col_px
    width = max(min_width, min(width, max_width))

    # ------------------ figure ------------------
    fig = go.Figure(
        data=go.Heatmap(            
            z=pivot.values,
            x=[tiers, questions],
            y=pivot.index.tolist(),
            colorscale='Oranges',
            showscale=False,
            hovertemplate=(
                "Character: %{y}<br>"
                "Tier: %{x[0]}<br>"
                "Question: %{x[1]}<br>"
                "Wrong Ratio: %{z:.2f}<extra></extra>"
            )
        )
    )

    # force all y labels to show
    fig.update_yaxes(
        autorange='reversed',
        automargin=True,
        tickmode='array',
        tickvals=pivot.index.tolist(),
        ticktext=pivot.index.tolist(),
        tickfont=dict(size=12)
    )

    fig.update_layout(
        title = 'Wrong Answers per Question Heatmap',        
        xaxis=dict(type='multicategory'),
        yaxis=dict(title='Character'),
        xaxis_title='Tier - Question',
        height=height,
        width=width,
        autosize=False,
        margin=dict(l=left_px, r=40, t=160, b=80)
    )

    fig.update_xaxes(side='top')

    return fig.to_html(full_html=False, include_plotlyjs='cdn')




def create_users_per_day_chart(df: pd.DataFrame):
    """
    Create a line chart showing the number of unique users per day.
    """

    fig = go.Figure(
        data=go.Scatter(
            x=df['event_date'],
            y=df['unique_users'],
            mode='lines+markers',
            line=dict(color='blue'),
            marker=dict(size=6)
        )
    )
    fig.update_layout(
        title='Unique Users Per Day',
        xaxis_title='Date',
        yaxis_title='Number of Unique Users',
        xaxis=dict(tickformat='%Y-%m-%d'),
        yaxis=dict(dtick=1)
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_cumulative_users_chart(df: pd.DataFrame):
    """
    Create an area chart showing cumulative unique users over time.
    """

    fig = go.Figure(
        data=go.Scatter(
            x=df['event_date'],
            y=df['unique_users'].cumsum(),
            mode='lines',
            fill='tozeroy',
            line=dict(color='green')
        )
    )
    fig.update_layout(
        title='Cumulative Unique Users Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative Unique Users',
        xaxis=dict(tickformat='%Y-%m-%d'),
        yaxis=dict(dtick=5)
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_sessions_per_day_chart(df: pd.DataFrame): 
    """
    Create a line chart showing the number of sessions per day.
    """

    fig = go.Figure(
        data=go.Scatter(
            x=df['event_date'],
            y=df['unique_sessions'],
            mode='lines+markers',
            line=dict(color='purple'),
            marker=dict(size=6)
        )
    )
    fig.update_layout(
        title='Sessions Per Day',
        xaxis_title='Date',
        yaxis_title='Number of Sessions',
        xaxis=dict(tickformat='%Y-%m-%d'),
        yaxis=dict(dtick=1)
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_session_duration_histogram(df: pd.DataFrame):
    """
    Create a histogram showing the distribution of session durations.
    """

    fig = go.Figure(
        data=go.Histogram(
            x=df['session_duration_seconds'] / 60,
            marker=dict(color='orange')
        )
    )
    fig.update_layout(
        title='Session Duration Distribution',
        xaxis_title='Session Duration (minutes)',
        yaxis_title='Count of Sessions'
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_ads_per_question_heatmap(df: pd.DataFrame):

    # --- constants controlling auto sizing ---
    row_px = 35          # pixels per row
    col_px = 25          # pixels per column
    header_px = 100      # space for x-axis labels
    left_px = 100        # space for y-axis labels

    min_height = 400
    max_height = 1800
    min_width = 700
    max_width = 1800

    # ------------------ aggregation ------------------
    heat = (
        df.groupby([
            'event_params__character_name',
            'event_params__current_tier',
            'event_params__current_question_index'
        ])['ads_watch_ratio']
        .mean()
        .reset_index()
    )

    pivot = heat.pivot_table(
        index='event_params__character_name',
        columns=['event_params__current_tier', 'event_params__current_question_index'],
        values='ads_watch_ratio',
        fill_value=0
    )

    pivot = pivot.sort_index(axis=1, level=[0, 1])
    pivot = pivot.reindex(sorted(pivot.index, key=lambda x: str(x).lower()), axis=0)

    # Build axis labels
    tiers = [f"{t}" for t, q in pivot.columns]
    questions = [f"{q}" for t, q in pivot.columns]

    # ------------------ auto figure size ------------------
    n_rows = len(pivot.index)
    n_cols = len(pivot.columns)

    height = header_px + n_rows * row_px
    height = max(min_height, min(height, max_height))

    width = left_px + n_cols * col_px
    width = max(min_width, min(width, max_width))

    # ------------------ figure ------------------
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[tiers, questions],
            y=pivot.index.tolist(),
            colorscale='Oranges',
            showscale=False,
            hovertemplate=(
                "Character: %{y}<br>"
                "Tier: %{x[0]}<br>"
                "Question: %{x[1]}<br>"
                "Ad Watch Ratio: %{z:.2f}<extra></extra>"
            )
        )
    )

    # force all y labels to show
    fig.update_yaxes(
        autorange='reversed',
        automargin=True,
        tickmode='array',
        tickvals=pivot.index.tolist(),
        ticktext=pivot.index.tolist(),
        tickfont=dict(size=12)
    )

    fig.update_layout(
        title = 'Ads Watched per Question Heatmap',
        xaxis=dict(type='multicategory'),
        yaxis=dict(title='Character'),
        xaxis_title='Tier - Question',
        height=height,
        width=width,
        autosize=False,
        margin=dict(l=left_px, r=40, t=160, b=80)
    )

    fig.update_xaxes(side='top')

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_item_per_question_heatmap(item, df: pd.DataFrame):
    item_ratio = item + '_use_ratio'
    # --- constants controlling auto sizing ---
    row_px = 35          # pixels per row
    col_px = 25          # pixels per column
    header_px = 100      # space for x-axis labels
    left_px = 100        # space for y-axis labels

    min_height = 400
    max_height = 1800
    min_width = 700
    max_width = 1800

    # ------------------ aggregation ------------------
    heat = (
        df.groupby([
            'event_params__character_name',
            'event_params__current_tier',
            'event_params__current_question_index'
        ])[item_ratio]
        .mean()
        .reset_index()
    )

    pivot = heat.pivot_table(
        index='event_params__character_name',
        columns=['event_params__current_tier', 'event_params__current_question_index'],
        values=item_ratio,
        fill_value=0
    )

    pivot = pivot.sort_index(axis=1, level=[0, 1])
    pivot = pivot.reindex(sorted(pivot.index, key=lambda x: str(x).lower()), axis=0)

    # Build axis labels
    tiers = [f"{t}" for t, q in pivot.columns]
    questions = [f"{q}" for t, q in pivot.columns]

    # ------------------ auto figure size ------------------
    n_rows = len(pivot.index)
    n_cols = len(pivot.columns)

    height = header_px + n_rows * row_px
    height = max(min_height, min(height, max_height))

    width = left_px + n_cols * col_px
    width = max(min_width, min(width, max_width))

    # ------------------ figure ------------------
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[tiers, questions],
            y=pivot.index.tolist(),
            colorscale='Oranges',
            showscale=False,
            hovertemplate=(
                "Character: %{y}<br>"
                "Tier: %{x[0]}<br>"
                "Question: %{x[1]}<br>"
                f"{item_ratio}: "
                "%{z:.2f}<extra></extra>"
            )
        )
    )

    # force all y labels to show
    fig.update_yaxes(
        autorange='reversed',
        automargin=True,
        tickmode='array',
        tickvals=pivot.index.tolist(),
        ticktext=pivot.index.tolist(),
        tickfont=dict(size=12)
    )

    fig.update_layout(
        title = f'{item.capitalize()} Use per Question Heatmap',
        xaxis=dict(type='multicategory'),
        yaxis=dict(title='Character'),
        xaxis_title='Tier - Question',
        height=height,
        width=width,
        autosize=False,
        margin=dict(l=left_px, r=40, t=160, b=80)
    )

    fig.update_xaxes(side='top')

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_ads_per_day_chart(df: pd.DataFrame):
    """
    Create a line chart showing the number of ads viewed per day.
    """

    fig = go.Figure(
        data=go.Scatter(
            x=df['event_date'],
            y=df['ads_watched'],
            mode='lines+markers',
            line=dict(color='red'),
            marker=dict(size=6)
        )
    )
    fig.update_layout(
        title='Ads Viewed Per Day',
        xaxis_title='Date',
        yaxis_title='Number of Ads Viewed',
        xaxis=dict(tickformat='%Y-%m-%d'),
        yaxis=dict(dtick=1)
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')

import datetime as dt
import pandas as pd
import plotly.graph_objects as go

def create_user_last_event_chart(df: pd.DataFrame, threshold: int = 7):
    """
    Create a histogram of the last event users generated, for users inactive > threshold days.
    """
    if df.empty:
        # return an empty figure HTML if no data
        fig = go.Figure()
        fig.update_layout(title="No data")
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Ensure last_event_date is datetime
    last_dates = pd.to_datetime(df['last_event_date'], errors='coerce')

    # make everything UTC
    last_dates = last_dates.dt.tz_convert('UTC')

    now = pd.Timestamp.now(tz='UTC').normalize()

    # compute age
    age = now - last_dates.dt.normalize()


    # Mask for users inactive more than threshold days
    mask = age > pd.Timedelta(days=threshold)
    df = df.loc[mask].copy()

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f'No users inactive for more than {threshold} days')
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Count events (ordered by count)
    counts = df['last_event_name'].value_counts()

    fig = go.Figure(
        data=go.Bar(
            x=counts.index,
            y=counts.values,
            marker=dict(color='cyan')
        )
    )
    fig.update_layout(
        title=f'Last Event Generated by Users Inactive for More Than {threshold} Days',
        xaxis_title='Event Name',
        yaxis_title='Number of Users'
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_session_last_event_chart(df: pd.DataFrame):
    """
    Create a histogram of the last event sessions had, for sessions inactive > threshold days.
    """
    if df.empty:
        # return an empty figure HTML if no data
        fig = go.Figure()
        fig.update_layout(title="No data")
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Count events (ordered by count)
    counts = df['last_event_name'].value_counts()

    fig = go.Figure(
        data=go.Bar(
            x=counts.index,
            y=counts.values,
            marker=dict(color='magenta')
        )
    )
    fig.update_layout(
        title=f'Last Event in Sessions',
        xaxis_title='Event Name',
        yaxis_title='Number of Sessions'
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_new_users_per_day_chart(df: pd.DataFrame):
    """
    Create a line chart showing the number of new users per day.
    """
    fig = go.Figure(
        data=go.Scatter(
            x=df['event_date'],
            y=df['new_users'],
            mode='lines+markers',
            line=dict(color='orange'),
            marker=dict(size=6)
        )
    )
    fig.update_layout(
        title='New Users Per Day',
        xaxis_title='Date',
        yaxis_title='Number of New Users',
        xaxis=dict(tickformat='%Y-%m-%d'),
        yaxis=dict(dtick=1)
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')