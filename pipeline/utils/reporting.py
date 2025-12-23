from jinja2 import Environment, FileSystemLoader
import pandas as pd
import plotly.express as px
from pathlib import Path
from config.logging import get_logger
import os
import numpy as np

logger = get_logger(__name__)

from pipeline.utils.plotting.plot_functions import (create_wrong_answers_heatmap, 
                                             create_users_per_day_chart, 
                                             create_cumulative_users_chart,
                                             create_session_duration_histogram,
                                             create_sessions_per_day_chart,
                                             create_ads_per_question_heatmap,
                                             create_item_per_question_heatmap,
                                             create_ads_per_day_chart,
                                             create_user_last_event_chart,
                                             create_session_last_event_chart,
                                             create_new_users_per_day_chart
                                             )


def generate_report(df, dfs_dict, kpis, context):
    """
    Generate HTML report pages in the folder specified by context["report_path"].
    """
    # --- Paths ---
    output_path = Path(context["report_path"]).resolve()
    output_path.mkdir(parents=True, exist_ok=True)  # ensure folder exists

    # --- Extract dfs ---
    df_by_ads = dfs_dict.get('by_ads')
    df_by_sessions = dfs_dict.get('by_sessions')
    df_by_users = dfs_dict.get('by_users')
    df_by_questions = dfs_dict.get('by_questions')
    df_by_date = dfs_dict.get('by_date')
    df_technical_events = dfs_dict.get('technical_events')

    item_list = ['alicin', 'coffee', 'cauldron', 'scroll']

    # --- Visualizations ---
    questions_heatmap = create_wrong_answers_heatmap(df_by_questions)
    ads_per_question_heatmap = create_ads_per_question_heatmap(df_by_questions)
    users_per_day_chart = create_users_per_day_chart(df_by_date)
    cumulative_users_chart = create_cumulative_users_chart(df_by_date)
    session_duration_histogram = create_session_duration_histogram(df_by_sessions)
    item_histograms = [create_item_per_question_heatmap(item, df_by_questions) for item in item_list]
    ads_per_day_chart = create_ads_per_day_chart(df_by_date)
    sessions_per_day_chart = create_sessions_per_day_chart(df_by_date)
    user_last_event_chart = create_user_last_event_chart(df_by_users)
    session_last_event_chart = create_session_last_event_chart(df_by_sessions)
    new_users_per_day_chart = create_new_users_per_day_chart(df_by_date)

    # --- Jinja2 setup ---
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader('templates'))

    def render_page(template_name, **data):
        template = env.get_template(template_name)
        return template.render(**data)

    pages = {
        "index.html": (
            "main_template.html",
            dict(title="Main", kpis=kpis)
        ),
        "users.html": (
            "users_template.html",
            dict(
                title="Users",
                users_chart=users_per_day_chart,
                new_users_per_day_chart=new_users_per_day_chart,
                user_last_event_chart=user_last_event_chart,
                new_users=(
                    df_by_users.sort_values("first_event_date", ascending=False)
                               .head(100)
                               .to_dict(orient="records")
                ),
                users_cols=list(df_by_users.columns),
                kpis=kpis
            )
        ),
        "sessions.html": (
            "sessions_template.html",
            dict(
                title="Sessions",
                sessions=df_by_sessions,
                kpis=kpis,
                session_duration_histogram=session_duration_histogram,
                sessions_per_day_chart=sessions_per_day_chart,
                session_last_event_chart=session_last_event_chart
            )
        ),
        "questions.html": (
            "questions_template.html",
            dict(
                title="Questions",
                questions_heatmap=questions_heatmap,
                ads_per_question_heatmap=ads_per_question_heatmap,
                item_histograms=item_histograms,
                kpis=kpis
            )
        ),
        "ads.html": (
            "ads_template.html",
            dict(
                title="Ads",
                ads=df_by_ads,
                kpis=kpis,
                ads_per_day_chart=ads_per_day_chart
            )
        ),
        "technical.html": (
            "technical_template.html",
            dict(
                title="Technical",
                technical_df=(
                    df_technical_events.sort_values("event_datetime", ascending=False)
                                       .head(100)
                                       .to_dict(orient="records")
                ),
                technical_cols=list(df_technical_events.columns),
                kpis=kpis
            )
        ),
    }

    # --- Write pages ---
    for output_filename, (template_name, page_context) in pages.items():
        html = render_page(template_name, **page_context)
        page_path = output_path / output_filename
        page_path.write_text(html, encoding="utf-8")

    logger.info(f"Report generated at {output_path}")
    print(f"Report generated at {output_path}")  # optional for Actions logs
