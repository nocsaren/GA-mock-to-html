from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import json

from .config_model import MockConfig
from .schema_utils import ensure_columns, read_csv_header


SKIP_LAST_EVENTS = {
    "User Engagement",
    "Screen Viewed",
    "Earned Virtual Currency",
    "Firebase Campaign",
    "App Removed",
    "App Data Cleared",
    "App Updated",
    "Starting Currencies",
}

TURKISH_WEEKDAYS = {
    0: "Pazartesi",
    1: "Salı",
    2: "Çarşamba",
    3: "Perşembe",
    4: "Cuma",
    5: "Cumartesi",
    6: "Pazar",
}


def _daytime_named(hour: int) -> str:
    if 0 <= hour <= 5:
        return "Gece"
    if 6 <= hour <= 11:
        return "Sabah"
    if 12 <= hour <= 17:
        return "Öğle"
    return "Akşam"


def _is_weekend(weekday: int) -> str:
    return "Hafta Sonu" if weekday >= 5 else "Hafta İçi"


def _random_hex(rng: np.random.Generator, n: int = 32) -> str:
    alphabet = np.array(list("0123456789abcdef"))
    return "".join(rng.choice(alphabet, size=n))


def _param(key: str, value) -> dict:
    """Create a GA4 BigQuery-style param dict: {key, value:{string_value|int_value|double_value}}."""
    payload: dict[str, object] = {}
    if value is None or (isinstance(value, float) and np.isnan(value)):
        payload["string_value"] = None
    elif isinstance(value, bool):
        payload["string_value"] = "true" if value else "false"
    elif isinstance(value, (int, np.integer)):
        payload["int_value"] = int(value)
    elif isinstance(value, (float, np.floating)):
        payload["double_value"] = float(value)
    else:
        payload["string_value"] = str(value)

    return {"key": key, "value": payload}


def _maybe_write_parquet(df: pd.DataFrame, path: Path) -> bool:
    """Write parquet if pyarrow is available; return True if written."""
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return False

    df.to_parquet(path, index=False)
    return True


def _build_raw_events(cfg: MockConfig) -> pd.DataFrame:
    """Build a DataFrame that matches the *raw* BigQuery pull shape (pre-flatten)."""
    rng = np.random.default_rng(cfg.seed)

    start = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=cfg.days - 1)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")

    rows: list[dict] = []
    event_bundle_seq = 1

    # generate users
    for _u in range(cfg.users):
        user_id = None
        user_pseudo_id = _random_hex(rng, 32)
        stream_id = str(rng.integers(10_000_000_000, 99_999_999_999))
        platform = str(rng.choice(cfg.operating_systems))  # e.g. ANDROID / IOS
        is_active_user = True

        country = str(rng.choice(cfg.countries))
        app_version_start = str(rng.choice(cfg.app_versions))
        app_version_now = str(rng.choice(cfg.app_versions))

        # times
        first_open_day = int(rng.integers(0, cfg.days))
        first_open_dt = start + pd.Timedelta(days=first_open_day) + pd.Timedelta(minutes=int(rng.integers(0, 1440)))
        user_first_touch_ts_us = int(first_open_dt.value // 1_000)  # microseconds

        # user_properties: first_open_time (ms)
        user_properties = [
            _param("first_open_time", int(first_open_dt.value // 1_000_000)),
            _param("ga_session_number", int(rng.integers(1, 8))),
        ]

        # base nested structs
        device = {
            "category": "mobile",
            "operating_system": platform,
            "operating_system_version": str(rng.choice(["16", "17", "18", "Android 15", "Android 16"])),
            "language": str(rng.choice(["en-us", "tr-tr"])),
            "is_limited_ad_tracking": str(rng.choice(["Yes", "No"])),
            "time_zone_offset_seconds": int(rng.choice([-18000, 0, 10800])),
            "mobile_brand_name": str(rng.choice(["Samsung", "Google", "Apple"])),
            "mobile_model_name": str(rng.choice(["Galaxy", "Pixel", "iPhone"])),
            "mobile_marketing_name": str(rng.choice(["Galaxy S24", "Pixel 8", "iPhone 15"])) ,
        }
        geo = {
            "country": country,
            "continent": "Americas" if country == "United States" else "Europe",
            "city": str(rng.choice(["San Antonio", "New York", "İstanbul", "Ankara", None])),
            "region": str(rng.choice(["Texas", "NY", "Marmara", None])),
        }
        app_info = {
            "version": app_version_now,
            "install_source": str(rng.choice(["com.android.vending", "apps.apple.com", None])),
            "id": "com.GlyphexGames.EmojiOracle",
        }
        privacy_info = {
            "ads_storage": str(rng.choice(["Yes", "No"])),
            "analytics_storage": str(rng.choice(["Yes", "No"])),
            "uses_transient_token": "No",
        }
        traffic_source = {
            "name": None,
            "medium": None,
            "source": None,
        }

        # vocab-driven values
        unlocked_characters = rng.choice(
            cfg.vocab.characters,
            size=min(len(cfg.vocab.characters), int(rng.integers(1, len(cfg.vocab.characters) + 1))),
            replace=False,
        )

        # sessions per user
        sessions = int(max(1, rng.poisson(lam=max(cfg.avg_sessions_per_user, 0.1))))
        for s in range(sessions):
            ga_session_id = int(rng.integers(1_000_000_000, 9_999_999_999))
            ga_session_number = int(s + 1)

            session_day = int(rng.integers(first_open_day, cfg.days))
            session_start = start + pd.Timedelta(days=session_day) + pd.Timedelta(minutes=int(rng.integers(0, 1440)))

            def add_event(event_name: str, dt: pd.Timestamp, params: list[dict] | None = None, extra: dict | None = None):
                nonlocal event_bundle_seq
                ts_us = int(dt.value // 1_000)
                prev_us = int((dt - pd.Timedelta(seconds=int(rng.integers(0, 60)))).value // 1_000)
                base_params = [
                    _param("ga_session_id", ga_session_id),
                    _param("ga_session_number", ga_session_number),
                ]
                if params:
                    base_params.extend(params)

                row = {
                    "event_date": dt.strftime("%Y%m%d"),
                    "event_timestamp": ts_us,
                    "event_name": event_name,
                    "event_previous_timestamp": prev_us,
                    "event_value_in_usd": None,
                    "event_bundle_sequence_id": event_bundle_seq,
                    "event_server_timestamp_offset": int(rng.integers(0, 5000)),
                    "user_id": user_id,
                    "user_pseudo_id": user_pseudo_id,
                    "user_first_touch_timestamp": user_first_touch_ts_us,
                    "stream_id": stream_id,
                    "platform": platform,
                    "is_active_user": is_active_user,
                    "batch_event_index": int(rng.integers(1, 5)),
                    "batch_page_id": None,
                    "batch_ordering_id": None,

                    "device": device,
                    "geo": geo,
                    "app_info": app_info,
                    "traffic_source": traffic_source,
                    "privacy_info": privacy_info,
                    "user_ltv": {},

                    "event_params": base_params,
                    "user_properties": user_properties,
                    "items": [],
                    "item_params": [],
                    "event_dimensions": {},
                    "ecommerce": {},
                    "collected_traffic_source": {},
                }
                if extra:
                    row.update(extra)

                rows.append(row)
                event_bundle_seq += 1

            # session_start event
            add_event(
                "session_start",
                session_start,
                params=[
                    _param("firebase_event_origin", "auto"),
                    _param("session_engaged", str(rng.choice(["0", "1"]))),
                ],
            )

            # first_open (once per user, roughly)
            if s == 0:
                add_event(
                    "first_open",
                    first_open_dt,
                    params=[
                        _param("pp_accepted", rng.random() < 0.85),
                        _param("video_start", rng.random() < 0.75),
                        _param("video_finished", rng.random() < 0.55),
                        _param("previous_first_open_count", 0),
                    ],
                )

            # per-session gameplay events
            n_questions = int(rng.integers(1, 8))
            for _q in range(n_questions):
                dt_q = session_start + pd.Timedelta(seconds=int(rng.integers(5, 400)))
                character = str(rng.choice(unlocked_characters))
                tier = int(rng.choice(cfg.tiers))
                current_qi = int(rng.integers(1, cfg.questions_per_tier + 1))

                add_event(
                    "question_started",
                    dt_q,
                    params=[
                        _param("character_name", character),
                        _param("current_tier", tier),
                        _param("current_qi", current_qi),
                    ],
                )

                add_event(
                    "question_completed",
                    dt_q + pd.Timedelta(seconds=3),
                    params=[
                        _param("character_name", character),
                        _param("current_tier", tier),
                        _param("current_qi", current_qi),
                        _param("answered_wrong", int(rng.integers(0, 3))),
                    ],
                )

                # ad_rewarded
                if rng.random() < 0.25:
                    add_event(
                        "ad_rewarded",
                        dt_q + pd.Timedelta(seconds=2),
                        params=[
                            _param("ad_network", str(rng.choice(["admob", "unity", "ironSource"]))),
                            _param("ad_unit_id", str(rng.choice(["rewarded_1", "rewarded_2"]))),
                            _param("ad_instance", str(rng.choice(["instance_a", "instance_b"]))),
                            _param("ad_id", _random_hex(rng, 12)),
                            _param("character_name", character),
                            _param("current_tier", tier),
                            _param("current_qi", current_qi),
                        ],
                    )

                # menu_opened scroll
                if rng.random() < 0.15:
                    add_event(
                        "menu_opened",
                        dt_q + pd.Timedelta(seconds=1),
                        params=[
                            _param("menu_name", cfg.vocab.scroll_menu_name),
                            _param("character_name", character),
                            _param("current_tier", tier),
                            _param("current_qi", current_qi),
                        ],
                    )

                # spend_virtual_currency (energy item usage)
                if rng.random() < 0.18:
                    spent_to = str(rng.choice([cfg.vocab.alicin_name, cfg.vocab.coffee_name, cfg.vocab.cauldron_name]))
                    add_event(
                        "spend_virtual_currency",
                        dt_q + pd.Timedelta(seconds=4),
                        params=[
                            _param("currency_name", "Gold"),
                            _param("spent_amount", float(int(rng.integers(10, 120)))),
                            _param("where_its_spent", str(rng.choice(["board", "board_item", "shop"]))),
                            _param("spent_to", spent_to),
                            _param("character_name", character),
                            _param("current_tier", tier),
                            _param("current_qi", current_qi),
                        ],
                    )

                # spend_virtual_currency (consumable)
                if rng.random() < 0.10:
                    cons = str(rng.choice([cfg.vocab.potion_name, cfg.vocab.incense_name, cfg.vocab.amulet_name]))
                    add_event(
                        "spend_virtual_currency",
                        dt_q + pd.Timedelta(seconds=5),
                        params=[
                            _param("currency_name", "Gold"),
                            _param("spent_amount", float(int(rng.integers(100, 500)))),
                            _param("where_its_spent", "shop"),
                            _param("spent_to", "Consumable Item"),
                            _param("character_name", character),
                            _param("current_tier", tier),
                            _param("current_qi", current_qi),
                        ],
                        extra={"shop_consumable_item": cons},
                    )

            # occasional technical
            if rng.random() < 0.03:
                add_event(
                    "ad_load_failed",
                    session_start + pd.Timedelta(seconds=6),
                    params=[
                        _param("ad_error_code", str(rng.choice(["0", "1", "2", "timeout"]))),
                        _param("ad_network", str(rng.choice(["admob", "unity"]))),
                        _param("ad_instance", str(rng.choice(["instance_a", "instance_b"]))),
                        _param("ad_id", _random_hex(rng, 12)),
                    ],
                )

            if rng.random() < 0.01:
                add_event(
                    "app_exception",
                    session_start + pd.Timedelta(seconds=7),
                    params=[
                        _param("fatal", rng.random() < 0.5),
                        _param("firebase_error", "NullPointer"),
                    ],
                )

            if rng.random() < 0.04:
                add_event("app_remove", session_start + pd.Timedelta(minutes=5))

    return pd.DataFrame(rows)


def _build_events(cfg: MockConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    start = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=cfg.days - 1)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")

    rows: list[dict] = []

    session_id_counter = 10_000
    for _ in range(cfg.users):
        user_id = _random_hex(rng, 32)
        country = rng.choice(cfg.countries)
        os_name = rng.choice(cfg.operating_systems)
        start_version = rng.choice(cfg.app_versions)

        # sessions per user (at least 1)
        sessions = int(max(1, rng.poisson(lam=max(cfg.avg_sessions_per_user, 0.1))))

        # Conversion-like booleans at user-level
        pp_ok = rng.random() < 0.85
        video_start = rng.random() < 0.75
        video_finished = video_start and (rng.random() < 0.7)

        tutorial_completed = rng.random() < 0.55

        first_open_day = int(rng.integers(0, cfg.days))
        first_open_dt = start + pd.Timedelta(days=first_open_day) + pd.Timedelta(minutes=int(rng.integers(0, 1200)))

        rows.append({
            "event_name": "First Open",
            "event_datetime": first_open_dt,
            "user_pseudo_id": user_id,
            "event_params__ga_session_id": None,
            "app_info__version": start_version,
            "geo__country": country,
            "device__operating_system": os_name,
            "event_params__pp_accepted": "true" if pp_ok else "false",
            "event_params__video_start": "true" if video_start else "false",
            "event_params__video_finished": "true" if video_finished else "false",
            "event_params__tutorial_video": "tutorial_video" if tutorial_completed else None,
        })

        # Give each user a character progression set
        unlocked_characters = rng.choice(cfg.vocab.characters, size=min(len(cfg.vocab.characters), int(rng.integers(1, len(cfg.vocab.characters) + 1))), replace=False)

        for _s in range(sessions):
            session_id_counter += 1
            session_id = session_id_counter

            session_day = int(rng.integers(first_open_day, cfg.days))
            session_start = start + pd.Timedelta(days=session_day) + pd.Timedelta(minutes=int(rng.integers(0, 1440)))
            session_end = session_start + pd.Timedelta(seconds=int(rng.integers(30, 900)))

            # Session start
            rows.append({
                "event_name": "Session Started",
                "event_datetime": session_start,
                "user_pseudo_id": user_id,
                "event_params__ga_session_id": session_id,
                "app_info__version": rng.choice(cfg.app_versions),
                "geo__country": country,
                "device__operating_system": os_name,
                "event_params__entered": "true" if (rng.random() < 0.6) else "false",
                "event_params__shown": "true" if (rng.random() < 0.5) else "false",
                "event_params__opened": "true" if (rng.random() < 0.4) else "false",
                "event_params__return": "true" if (rng.random() < 0.25) else "false",
                "event_params__closed": "true" if (rng.random() < 0.35) else "false",
                "event_params__drag": "true" if (rng.random() < 0.45) else "false",
            })

            # starting currencies
            rows.append({
                "event_name": "Starting Currencies",
                "event_datetime": session_start + pd.Timedelta(seconds=1),
                "user_pseudo_id": user_id,
                "event_params__ga_session_id": session_id,
                "geo__country": country,
                "device__operating_system": os_name,
                "app_info__version": rng.choice(cfg.app_versions),
                "event_params__gold": float(int(rng.integers(0, 1200))),
            })

            # question loop
            n_questions = int(rng.integers(1, 8))
            for q in range(n_questions):
                event_t = session_start + pd.Timedelta(seconds=int(rng.integers(5, max(10, int((session_end - session_start).total_seconds())))))
                character = str(rng.choice(unlocked_characters))
                tier = int(rng.choice(cfg.tiers))
                qi = int(rng.integers(1, cfg.questions_per_tier + 1))

                rows.append({
                    "event_name": "Question Started",
                    "event_datetime": event_t,
                    "user_pseudo_id": user_id,
                    "event_params__ga_session_id": session_id,
                    "geo__country": country,
                    "device__operating_system": os_name,
                    "app_info__version": rng.choice(cfg.app_versions),
                    "event_params__character_name": character,
                    "event_params__current_tier": tier,
                    "event_params__current_question_index": qi,
                })

                rows.append({
                    "event_name": "Question Completed",
                    "event_datetime": event_t + pd.Timedelta(seconds=3),
                    "user_pseudo_id": user_id,
                    "event_params__ga_session_id": session_id,
                    "geo__country": country,
                    "device__operating_system": os_name,
                    "app_info__version": rng.choice(cfg.app_versions),
                    "event_params__character_name": character,
                    "event_params__current_tier": tier,
                    "event_params__current_question_index": qi,
                    "event_params__answered_wrong": int(rng.integers(0, 3)),
                })

                # optional ad rewarded
                if rng.random() < 0.25:
                    rows.append({
                        "event_name": "Ad Rewarded",
                        "event_datetime": event_t + pd.Timedelta(seconds=2),
                        "user_pseudo_id": user_id,
                        "event_params__ga_session_id": session_id,
                        "geo__country": country,
                        "device__operating_system": os_name,
                        "app_info__version": rng.choice(cfg.app_versions),
                        "event_params__character_name": character,
                        "event_params__current_tier": tier,
                        "event_params__current_question_index": qi,
                        "event_params__ad_network": str(rng.choice(["admob", "unity", "ironSource", None])),
                        "event_params__ad_unit_id": str(rng.choice(["rewarded_1", "rewarded_2", None])),
                        "event_params__ad_instance": str(rng.choice(["instance_a", "instance_b", None])),
                        "event_params__ad_id": _random_hex(rng, 12),
                    })

                # optional scroll menu
                if rng.random() < 0.15:
                    rows.append({
                        "event_name": "Menu Opened",
                        "event_datetime": event_t + pd.Timedelta(seconds=1),
                        "user_pseudo_id": user_id,
                        "event_params__ga_session_id": session_id,
                        "geo__country": country,
                        "device__operating_system": os_name,
                        "app_info__version": rng.choice(cfg.app_versions),
                        "event_params__menu_name": cfg.vocab.scroll_menu_name,
                        "event_params__character_name": character,
                        "event_params__current_tier": tier,
                        "event_params__current_question_index": qi,
                    })

                # optional energy item usage (spent_to)
                if rng.random() < 0.18:
                    spent_to = rng.choice([
                        cfg.vocab.alicin_name,
                        cfg.vocab.coffee_name,
                        cfg.vocab.cauldron_name,
                    ])
                    rows.append({
                        "event_name": "Spent Virtual Currency",
                        "event_datetime": event_t + pd.Timedelta(seconds=4),
                        "user_pseudo_id": user_id,
                        "event_params__ga_session_id": session_id,
                        "geo__country": country,
                        "device__operating_system": os_name,
                        "app_info__version": rng.choice(cfg.app_versions),
                        "event_params__currency_name": "Gold",
                        "event_params__spent_amount": float(int(rng.integers(10, 120))),
                        "event_params__where_its_spent": str(rng.choice(["board", "board_item", "shop"])) ,
                        "event_params__spent_to": spent_to,
                        "event_params__character_name": character,
                        "event_params__current_tier": tier,
                        "event_params__current_question_index": qi,
                    })

                # optional consumable purchase
                if rng.random() < 0.10:
                    cons = rng.choice([cfg.vocab.potion_name, cfg.vocab.incense_name, cfg.vocab.amulet_name])
                    rows.append({
                        "event_name": "Spent Virtual Currency",
                        "event_datetime": event_t + pd.Timedelta(seconds=5),
                        "user_pseudo_id": user_id,
                        "event_params__ga_session_id": session_id,
                        "geo__country": country,
                        "device__operating_system": os_name,
                        "app_info__version": rng.choice(cfg.app_versions),
                        "event_params__currency_name": "Gold",
                        "event_params__spent_amount": float(int(rng.integers(100, 500))),
                        "event_params__where_its_spent": "shop",
                        "event_params__spent_to": "Consumable Item",
                        "shop_consumable_item": cons,
                        "event_params__character_name": character,
                        "event_params__current_tier": tier,
                        "event_params__current_question_index": qi,
                    })

            # wheel interactions
            if rng.random() < 0.25:
                rows.append({
                    "event_name": "Mini-game Started",
                    "event_datetime": session_start + pd.Timedelta(seconds=2),
                    "user_pseudo_id": user_id,
                    "event_params__ga_session_id": session_id,
                    "geo__country": country,
                    "device__operating_system": os_name,
                    "app_info__version": rng.choice(cfg.app_versions),
                    "event_params__mini_game_ri": cfg.vocab.wheel_impression_ri,
                })
                if rng.random() < 0.3:
                    rows.append({
                        "event_name": "Mini-game Completed",
                        "event_datetime": session_start + pd.Timedelta(seconds=3),
                        "user_pseudo_id": user_id,
                        "event_params__ga_session_id": session_id,
                        "geo__country": country,
                        "device__operating_system": os_name,
                        "app_info__version": rng.choice(cfg.app_versions),
                        "event_params__mini_game_ri": cfg.vocab.wheel_skip_ri,
                    })

            # technical noise
            if rng.random() < 0.03:
                rows.append({
                    "event_name": "Ad Load Failed",
                    "event_datetime": session_start + pd.Timedelta(seconds=6),
                    "user_pseudo_id": user_id,
                    "event_params__ga_session_id": session_id,
                    "geo__country": country,
                    "device__operating_system": os_name,
                    "app_info__version": rng.choice(cfg.app_versions),
                    "device__mobile_marketing_name": str(rng.choice(["Pixel", "iPhone", "Galaxy"])) ,
                    "device__operating_system_version": str(rng.choice(["16", "17", "18", "Android 15"])) ,
                    "event_params__ad_error_code": str(rng.choice(["0", "1", "2", "timeout"])) ,
                    "event_server_delay_seconds": float(rng.random() * 2),
                })

            if rng.random() < 0.01:
                rows.append({
                    "event_name": "App Exception",
                    "event_datetime": session_start + pd.Timedelta(seconds=7),
                    "user_pseudo_id": user_id,
                    "event_params__ga_session_id": session_id,
                    "geo__country": country,
                    "device__operating_system": os_name,
                    "app_info__version": rng.choice(cfg.app_versions),
                    "device__mobile_marketing_name": str(rng.choice(["Pixel", "iPhone", "Galaxy"])) ,
                    "device__operating_system_version": str(rng.choice(["16", "17", "18", "Android 15"])) ,
                    "event_server_delay_seconds": float(rng.random() * 5),
                })

            # optional game ended / uninstall
            if rng.random() < 0.15:
                rows.append({
                    "event_name": "Game Ended",
                    "event_datetime": session_end - pd.Timedelta(seconds=3),
                    "user_pseudo_id": user_id,
                    "event_params__ga_session_id": session_id,
                    "geo__country": country,
                    "device__operating_system": os_name,
                    "app_info__version": rng.choice(cfg.app_versions),
                })

            if rng.random() < 0.04:
                rows.append({
                    "event_name": "App Removed",
                    "event_datetime": session_end,
                    "user_pseudo_id": user_id,
                    "event_params__ga_session_id": session_id,
                    "geo__country": country,
                    "device__operating_system": os_name,
                    "app_info__version": rng.choice(cfg.app_versions),
                })

    df = pd.DataFrame(rows)

    # timestamps + date/time features
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], utc=True)
    df = df.sort_values(["user_pseudo_id", "event_datetime"]).reset_index(drop=True)
    # microseconds since epoch
    df["event_timestamp"] = (df["event_datetime"].astype("int64") // 1_000).astype("int64")

    df["event_date"] = df["event_datetime"].dt.normalize()
    df["event_time"] = df["event_datetime"].dt.time

    df["ts_weekday"] = df["event_datetime"].dt.weekday.map(TURKISH_WEEKDAYS)
    df["ts_hour"] = df["event_datetime"].dt.hour
    df["ts_daytime_named"] = df["ts_hour"].apply(_daytime_named)
    df["ts_is_weekend"] = df["event_datetime"].dt.weekday.apply(_is_weekend)

    # session features
    session_groups = ["user_pseudo_id", "event_params__ga_session_id"]
    df["session_start_time"] = df.groupby(session_groups)["event_datetime"].transform("min")
    df["session_end_time"] = df.groupby(session_groups)["event_datetime"].transform("max")
    df["session_duration_seconds"] = (
        (df["session_end_time"] - df["session_start_time"]).dt.total_seconds().fillna(0)
    )
    df["session_duration_minutes"] = df["session_duration_seconds"] / 60

    # question address
    def _qa(row):
        c = row.get("event_params__character_name")
        t = row.get("event_params__current_tier")
        q = row.get("event_params__current_question_index")
        if pd.isna(c) or pd.isna(t) or pd.isna(q):
            return pd.NA
        return f"{c} - T: {int(t)} - Q: {int(q)}"

    df["question_address"] = df.apply(_qa, axis=1)

    # cumulative question index (mirrors feature_engineering.py intent)
    qi = pd.to_numeric(df.get("event_params__current_question_index"), errors="coerce")
    tier = pd.to_numeric(df.get("event_params__current_tier"), errors="coerce")
    char = df.get("event_params__character_name")
    df["cumulative_question_index"] = qi

    # offsets by tier, special vs non-special character
    special = cfg.vocab.special_character_for_offsets
    is_special = char.astype(str) == str(special)

    # tier 2
    df.loc[(tier == 2) & is_special, "cumulative_question_index"] = qi + 12
    df.loc[(tier == 2) & (~is_special), "cumulative_question_index"] = qi + 16
    # tier 3
    df.loc[(tier == 3) & is_special, "cumulative_question_index"] = qi + 24
    df.loc[(tier == 3) & (~is_special), "cumulative_question_index"] = qi + 28
    # tier 4
    df.loc[(tier == 4) & is_special, "cumulative_question_index"] = qi + 36
    df.loc[(tier == 4) & (~is_special), "cumulative_question_index"] = qi + 40

    return df


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer_f = pd.to_numeric(numer, errors="coerce").astype("float64")
    denom_f = pd.to_numeric(denom, errors="coerce").astype("float64")
    denom_f = denom_f.mask(denom_f == 0)
    return (numer_f / denom_f).fillna(0).round(3)


def _df_by_questions(cfg: MockConfig, df: pd.DataFrame) -> pd.DataFrame:
    gcols = [
        "question_address",
        "event_params__character_name",
        "event_params__current_tier",
        "event_params__current_question_index",
        "event_params__ga_session_id",
    ]

    base = df[gcols].copy()

    v = cfg.vocab
    temp = pd.DataFrame({
        "question_started": df["event_name"].eq("Question Started").astype(int),
        "potions_bought": df.get("shop_consumable_item").eq(v.potion_name).fillna(False).astype(int),
        "incense_bought": df.get("shop_consumable_item").eq(v.incense_name).fillna(False).astype(int),
        "amulet_bought": df.get("shop_consumable_item").eq(v.amulet_name).fillna(False).astype(int),
        "alicin_used": df.get("event_params__spent_to").eq(v.alicin_name).fillna(False).astype(int),
        "coffee_used": df.get("event_params__spent_to").eq(v.coffee_name).fillna(False).astype(int),
        "cauldron_used": df.get("event_params__spent_to").eq(v.cauldron_name).fillna(False).astype(int),
        "scroll_opened": (
            df["event_name"].eq("Menu Opened") & df.get("event_params__menu_name").eq(v.scroll_menu_name)
        ).fillna(False).astype(int),
        "answered_correct": df["event_name"].eq("Question Completed").astype(int),
        "answered_wrong": pd.to_numeric(df.get("event_params__answered_wrong"), errors="coerce").fillna(0),
        "ads_watched": df["event_name"].eq("Ad Rewarded").astype(int),
    })

    qdf = pd.concat([base.reset_index(drop=True), temp.reset_index(drop=True)], axis=1)
    qdf = qdf.groupby(gcols, as_index=False).sum()

    qdf["wrong_answer_ratio"] = _safe_ratio(qdf["answered_wrong"], qdf["question_started"])
    qdf["ads_watch_ratio"] = _safe_ratio(qdf["ads_watched"], qdf["question_started"])
    qdf["alicin_use_ratio"] = _safe_ratio(qdf["alicin_used"], qdf["question_started"])
    qdf["coffee_use_ratio"] = _safe_ratio(qdf["coffee_used"], qdf["question_started"])
    qdf["cauldron_use_ratio"] = _safe_ratio(qdf["cauldron_used"], qdf["question_started"])
    qdf["scroll_use_ratio"] = _safe_ratio(qdf["scroll_opened"], qdf["question_started"])

    return qdf


def _df_by_ads(df: pd.DataFrame) -> pd.DataFrame:
    ad_events = {
        "Ad Loaded",
        "Ad Closed",
        "Ad Displayed",
        "Ad Rewarded",
        "Ad Load Failed",
        "Ad Clicked",
    }
    ads = df[df["event_name"].isin(ad_events)].copy()

    columns = [
        "event_datetime",
        "event_params__ga_session_id",
        "event_name",
        "event_params__ad_id",
        "event_params__ad_unit_id",
        "event_params__ad_network",
        "event_params__ad_placement",
        "event_params__ad_reward_type",
        "event_params__ad_instance",
        "event_params__ad_error_code",
        "event_params__character_name",
        "event_params__current_tier",
        "event_params__current_question_index",
        "question_address",
        "ts_weekday",
        "ts_daytime_named",
        "app_info__version",
        "geo__country",
        "device__operating_system",
        "event_server_delay_seconds",
    ]
    for c in columns:
        if c not in ads.columns:
            ads[c] = None
    ads = ads[columns]

    for c in ["event_params__ad_network", "event_params__ad_placement", "event_params__ad_reward_type", "event_params__ad_instance"]:
        ads[c] = ads[c].fillna("Unknown/Missing")

    return ads


def _df_technical_events(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "user_pseudo_id",
        "event_params__ga_session_id",
        "event_datetime",
        "app_info__version",
        "device__mobile_marketing_name",
        "device__operating_system_version",
        "event_name",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    df_sorted = df.sort_values(cols)
    df_sorted["prev_event_name"] = df_sorted.groupby(["user_pseudo_id", "event_params__ga_session_id"])["event_name"].shift(1)
    df_sorted["prev_event_menu"] = df_sorted.groupby(["user_pseudo_id", "event_params__ga_session_id"])["event_params__menu_name"].shift(1) if "event_params__menu_name" in df_sorted.columns else None

    tech = df_sorted[df_sorted["event_name"].isin(["App Exception", "Ad Load Failed"])].copy()

    keep = [
        "event_datetime",
        "event_name",
        "user_pseudo_id",
        "event_params__ga_session_id",
        "app_info__version",
        "device__mobile_marketing_name",
        "device__operating_system_version",
        "prev_event_name",
        "prev_event_menu",
        "event_params__ad_network",
        "event_params__ad_instance",
        "event_params__ad_id",
        "event_params__ad_error_code",
        "event_server_delay_seconds",
    ]
    for c in keep:
        if c not in tech.columns:
            tech[c] = None
    return tech[keep]


def _df_by_date(df: pd.DataFrame) -> pd.DataFrame:
    # base
    required = [
        "event_date",
        "ts_weekday",
        "user_pseudo_id",
        "event_name",
        "device__operating_system",
        "event_params__ga_session_id",
    ]
    for c in required:
        if c not in df.columns:
            df[c] = None

    date_df = (
        df.groupby(["event_date"])
        .agg(
            weekday=("ts_weekday", "first"),
            unique_users=("user_pseudo_id", "nunique"),
            new_users=("event_name", lambda x: (x == "First Open").sum()),
            android_users=("device__operating_system", lambda x: (x == "ANDROID").sum()),
            ios_users=("device__operating_system", lambda x: (x == "IOS").sum()),
            uninstall_count=("event_name", lambda x: (x == "App Removed").sum()),
            unique_sessions=("event_params__ga_session_id", "nunique"),
            ads_watched=("event_name", lambda x: (x == "Ad Rewarded").sum()),
            questions_started=("event_name", lambda x: (x == "Question Started").sum()),
            questions_completed=("event_name", lambda x: (x == "Question Completed").sum()),
        )
        .reset_index()
    )

    # ad breakdowns
    def _breakdown(col: str, prefix: str) -> pd.DataFrame:
        if col not in df.columns:
            return pd.DataFrame(columns=["event_date"])
        return (
            df.groupby(["event_date", col])
            .size()
            .unstack(fill_value=0)
            .add_prefix(prefix)
            .reset_index()
        )

    ads_network_df = _breakdown("event_params__ad_network", "nwk_")
    ads_unit_df = _breakdown("event_params__ad_unit_id", "unt_")
    ads_instance_df = _breakdown("event_params__ad_instance", "ins_")

    result = (
        date_df
        .merge(ads_network_df, on="event_date", how="left")
        .merge(ads_unit_df, on="event_date", how="left")
        .merge(ads_instance_df, on="event_date", how="left")
        .fillna(0)
    )

    return result


def _summarize_gold(g: pd.DataFrame) -> pd.Series:
    gold_starting = 0.0
    subset = g.loc[g["event_name"] == "Starting Currencies", "event_params__gold"] if "event_params__gold" in g.columns else pd.Series(dtype=float)
    if len(subset) > 0:
        gold_starting = pd.to_numeric(subset, errors="coerce").sum()

    gold_gained = g.loc[
        (g["event_name"] == "Earned Virtual Currency") & (g.get("event_params__currency_name") == "Gold"),
        "event_params__earned_amount",
    ].sum() if "event_params__earned_amount" in g.columns else 0.0

    gold_spent = g.loc[
        (g["event_name"] == "Spent Virtual Currency") & (g.get("event_params__currency_name") == "Gold"),
        "event_params__spent_amount",
    ].sum() if "event_params__spent_amount" in g.columns else 0.0

    gold_delta = gold_gained - gold_spent
    is_depted_for_doll = int((gold_spent > (gold_starting + gold_gained)) and (gold_spent >= 2000))

    return pd.Series({
        "gold_starting": gold_starting,
        "gold_gained": gold_gained,
        "gold_spent": gold_spent,
        "gold_delta": gold_delta,
        "is_depted_for_doll": is_depted_for_doll,
    })


def _df_by_sessions(cfg: MockConfig, df: pd.DataFrame) -> pd.DataFrame:
    session_groups = ["event_params__ga_session_id", "user_pseudo_id"]

    # Filter sessions
    df_s = df[df["session_duration_seconds"] > 15].copy()
    base_sessions = df_s[session_groups].drop_duplicates().reset_index(drop=True)

    session_duration = (
        df_s.groupby(session_groups, as_index=False)["session_duration_seconds"].mean().round(2)
    )
    session_duration["passed_10_min"] = session_duration["session_duration_seconds"] >= 600

    session_start = (
        df_s.loc[df_s["event_name"] == "Session Started"]
        .groupby(session_groups, as_index=False)["session_start_time"].min()
        if "session_start_time" in df_s.columns else pd.DataFrame(columns=session_groups + ["session_start_time"])
    )

    q_started = df_s.loc[df_s["event_name"] == "Question Started"]
    qs_metrics = (
        q_started.groupby(session_groups, as_index=False)
        .agg(
            customer_character_count=("event_params__character_name", "nunique"),
            character_list=("event_params__character_name", lambda x: [v for v in x.dropna().tolist()]),
            average_tier=("event_params__current_tier", "mean"),
        )
    )

    q_completed = df_s.loc[df_s["event_name"] == "Question Completed"]
    qc_metrics = (
        q_completed.groupby(session_groups, as_index=False)
        .agg(average_wrong_answers=("event_params__answered_wrong", "mean"))
    )

    v = cfg.vocab
    wheel = (
        df_s.groupby(session_groups, as_index=False)["event_params__mini_game_ri"]
        .agg(
            Wheel_Impression=lambda x: (x == v.wheel_impression_ri).sum(),
            Wheel_Skips=lambda x: (x == v.wheel_skip_ri).sum(),
        )
        .assign(Wheel_Spins=lambda d: d["Wheel_Impression"] - d["Wheel_Skips"])
    )

    ads = (
        df_s.groupby(session_groups, as_index=False)["event_name"]
        .agg(Ads_Watched_Count=lambda x: (x == "Ad Rewarded").sum())
    )

    # gold
    gold = (
        df_s.groupby(session_groups, as_index=False)
        .apply(lambda g: _summarize_gold(g), include_groups=False)
        .reset_index(drop=True)
    )

    # consumables purchased
    consumable = pd.DataFrame(columns=session_groups)
    if "event_params__spent_to" in df_s.columns and "shop_consumable_item" in df_s.columns:
        consumable = (
            df_s.loc[df_s["event_params__spent_to"] == "Consumable Item"]
            .groupby(session_groups, as_index=False)["shop_consumable_item"]
            .agg(
                Potions_Bought=lambda x: (x == v.potion_name).sum(),
                Incenses_Bought=lambda x: (x == v.incense_name).sum(),
                Amulets_Bought=lambda x: (x == v.amulet_name).sum(),
            )
        )

    # energy spent
    energy = pd.DataFrame(columns=session_groups)
    if "event_params__spent_to" in df_s.columns:
        energy = (
            df_s.loc[df_s["event_params__spent_to"].isin([v.cauldron_name, v.alicin_name, v.coffee_name])]
            .groupby(session_groups, as_index=False)["event_params__spent_to"]
            .agg(
                AliCin_Used=lambda x: (x == v.alicin_name).sum(),
                Cauldron_Used=lambda x: (x == v.cauldron_name).sum(),
                Coffee_Used=lambda x: (x == v.coffee_name).sum(),
            )
        )

    # last event per session
    df_l_sorted = df_s.sort_values(["event_datetime"], ascending=False)

    def pick_last_valid(g: pd.DataFrame) -> pd.DataFrame:
        non_skip = g[~g["event_name"].isin(SKIP_LAST_EVENTS)]
        row = non_skip.iloc[0] if len(non_skip) > 0 else g.iloc[0]

        group_keys = g.name
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        group_map = dict(zip(session_groups, group_keys))

        return pd.DataFrame({
            "event_params__ga_session_id": [group_map.get("event_params__ga_session_id")],
            "user_pseudo_id": [group_map.get("user_pseudo_id")],
            "last_event_name": [row["event_name"]],
            "last_event_time": [row["event_datetime"]],
        })

    session_last_event = (
        df_l_sorted.groupby(session_groups, group_keys=False)
        .apply(pick_last_valid, include_groups=False)
        .reset_index(drop=True)
    )

    result = (
        base_sessions
        .merge(session_duration, on=session_groups, how="left")
        .merge(session_start, on=session_groups, how="left")
        .merge(qs_metrics, on=session_groups, how="left")
        .merge(qc_metrics, on=session_groups, how="left")
        .merge(wheel, on=session_groups, how="left")
        .merge(ads, on=session_groups, how="left")
        .merge(gold, on=session_groups, how="left")
        .merge(consumable, on=session_groups, how="left")
        .merge(energy, on=session_groups, how="left")
        .merge(session_last_event, on=session_groups, how="left")
    )

    result = result.infer_objects(copy=False).fillna({
        "average_tier": 0,
        "average_wrong_answers": 0,
        "Ads_Watched_Count": 0,
        "Wheel_Impression": 0,
        "Wheel_Skips": 0,
        "Wheel_Spins": 0,
        "Potions_Bought": 0,
        "Incenses_Bought": 0,
        "Amulets_Bought": 0,
        "AliCin_Used": 0,
        "Cauldron_Used": 0,
        "Coffee_Used": 0,
    })

    result["bought_new_customer"] = (result["customer_character_count"].fillna(0).astype(int) // 3) if "customer_character_count" in result.columns else 0

    return result


def _df_by_users(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_key = "user_pseudo_id"

    needed = [
        "event_name",
        "event_date",
        "session_duration_seconds",
        "event_params__ga_session_id",
        "event_params__character_name",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = None

    df = df.copy()
    df["session_duration_minutes"] = df["session_duration_seconds"] / 60

    user_df = (
        df.groupby(user_key, as_index=False)
        .agg(
            first_event_date=("event_date", "min"),
            total_sessions=("event_params__ga_session_id", "nunique"),
            total_characters_opened=("event_params__character_name", "nunique"),
            country=("geo__country", "first") if "geo__country" in df.columns else ("event_name", "first"),
            install_source=("app_info__install_source", "first") if "app_info__install_source" in df.columns else ("event_name", "first"),
            operating_system=("device__operating_system", "first") if "device__operating_system" in df.columns else ("event_name", "first"),
            operating_system_version=("device__operating_system_version", "first") if "device__operating_system_version" in df.columns else ("event_name", "first"),
            is_limited_ad_tracking=("device__is_limited_ad_tracking", "first") if "device__is_limited_ad_tracking" in df.columns else ("event_name", "first"),
            device_language=("device__language", "first") if "device__language" in df.columns else ("event_name", "first"),
            start_version=("app_info__version", "first") if "app_info__version" in df.columns else ("event_name", "first"),
            version=("app_info__version", "last") if "app_info__version" in df.columns else ("event_name", "last"),
        )
    )

    df_sessions = (
        df[["user_pseudo_id", "event_params__ga_session_id", "session_duration_minutes"]]
        .drop_duplicates(subset=["user_pseudo_id", "event_params__ga_session_id"])
    )

    user_playtime = (
        df_sessions.groupby(user_key, as_index=False)
        .agg(total_playtime_minutes=("session_duration_minutes", "sum"))
    )

    user_df = user_df.merge(user_playtime, on=user_key, how="left")

    def count_events(event: str) -> pd.Series:
        return df[df["event_name"] == event].groupby(user_key).size().rename(event)

    counts = pd.DataFrame({user_key: user_df[user_key]})
    for ev in ["Ad Rewarded", "Question Completed", "Game Ended", "App Removed", "Session Started"]:
        counts = counts.merge(count_events(ev), on=user_key, how="left")

    conversion_events = [
        "event_params__pp_accepted",
        "event_params__video_start",
        "event_params__video_finished",
        "event_params__entered",
        "event_params__shown",
        "event_params__opened",
        "event_params__return",
        "event_params__closed",
        "event_params__drag",
    ]

    def check_bool_event(event: str) -> pd.Series:
        if event not in df.columns:
            return pd.Series(0, index=user_df[user_key], name=event)
        s = df[event].astype(str).str.lower()
        col = s.isin(["true", "1", "yes", "y"]).astype(int)
        return col.groupby(df[user_key]).max().rename(event)

    for ev in conversion_events:
        counts = counts.merge(check_bool_event(ev), on=user_key, how="left")

    # tutorial
    if "event_params__tutorial_video" in df.columns:
        tutorials = (
            df[(df["event_params__tutorial_video"] == "tutorial_video") & (df["event_name"] == "Video Watched")]
            .groupby(user_key)
            .size()
            .rename("tutorial_completed")
        )
    else:
        tutorials = pd.Series(0, index=user_df[user_key], name="tutorial_completed")

    # welcome video played
    if "event_params__wecolme_video" in df.columns:
        wecolme_video_played = (
            (df["event_params__wecolme_video"] == "wecolme_video")
            .groupby(df[user_key])
            .any()
            .astype(int)
            .rename("wecolme_video_played")
        )
    else:
        wecolme_video_played = pd.Series(0, index=user_df[user_key], name="wecolme_video_played")

    counts = counts.merge(wecolme_video_played, on=user_key, how="left")
    counts = counts.merge(tutorials, on=user_key, how="left")

    count_cols = [c for c in counts.columns if c != user_key]
    counts[count_cols] = counts[count_cols].fillna(0).astype(int)

    # last event
    exclude_last = {
        "App Removed",
        "App Data Cleared",
        "App Updated",
        "User Engagement",
        "Screen Viewed",
        "Firebase Campaign",
        "Starting Currencies",
    }
    df_no_end = df[~df["event_name"].isin(exclude_last)]

    last_event = (
        df_no_end.sort_values("event_date")
        .drop_duplicates(subset=[user_key], keep="last")
        [[user_key, "event_date", "event_name"]]
        .rename(columns={"event_date": "last_event_date", "event_name": "last_event_name"})
    )

    user_df = user_df.merge(counts, on=user_key, how="left").merge(last_event, on=user_key, how="left")

    # derived KPI flags
    user_df["answered_first_question"] = (user_df.get("Question Completed", 0) > 0).astype(int)
    user_df["answered_second_question"] = (user_df.get("Question Completed", 0) > 1).astype(int)
    user_df["answered_third_question"] = (user_df.get("Question Completed", 0) > 2).astype(int)
    user_df["saw_mi"] = (user_df["total_characters_opened"] >= 2).astype(int)
    user_df["answered_ten_questions"] = (user_df.get("Question Completed", 0) >= 10).astype(int)
    user_df["second_session_started"] = (user_df["total_sessions"] >= 2).astype(int)
    user_df["second_day_active"] = (user_df["last_event_date"] > user_df["first_event_date"]).astype(int)

    user_df["passed_10_min"] = (user_df["total_playtime_minutes"] >= 10).astype(int)
    user_df["total_playtime_minutes"] = user_df["total_playtime_minutes"].round(2)

    boolean_cols = [
        "user_pseudo_id",
        "event_params__pp_accepted",
        "event_params__video_start",
        "event_params__video_finished",
        "event_params__entered",
        "event_params__shown",
        "event_params__opened",
        "event_params__return",
        "event_params__closed",
        "event_params__drag",
        "answered_first_question",
        "answered_second_question",
        "answered_third_question",
        "saw_mi",
        "passed_10_min",
        "answered_ten_questions",
        "second_session_started",
        "second_day_active",
        "tutorial_completed",
        "wecolme_video_played",
    ]

    user_bool_df = user_df[[c for c in boolean_cols if c in user_df.columns]].copy()
    user_bool_df["start_version"] = user_df.get("start_version")

    return user_df, user_bool_df


def _write_derived(cfg: MockConfig, out_root: Path, schema_from: Path | None) -> None:
    csv_dir = out_root / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    df = _build_events(cfg)

    dfs = {
        "by_sessions": _df_by_sessions(cfg, df),
        "by_users": _df_by_users(df)[0],
        "users_meta": _df_by_users(df)[1],
        "by_questions": _df_by_questions(cfg, df),
        "by_ads": _df_by_ads(df),
        "by_date": _df_by_date(df),
        "technical_events": _df_technical_events(df),
    }

    processed_name = "processed_data.csv"
    processed_path = csv_dir / processed_name

    desired_cols = read_csv_header(schema_from / processed_name) if schema_from is not None else None
    if desired_cols is None:
        desired_cols = [
            "event_timestamp",
            "event_name",
            "user_pseudo_id",
            "event_params__ga_session_id",
            "event_datetime",
            "event_date",
            "event_time",
            "ts_weekday",
            "ts_daytime_named",
            "ts_is_weekend",
            "app_info__version",
            "geo__country",
            "device__operating_system",
            "session_duration_seconds",
            "session_duration_minutes",
            "session_start_time",
            "session_end_time",
            "event_params__character_name",
            "event_params__current_tier",
            "event_params__current_question_index",
            "cumulative_question_index",
            "event_params__answered_wrong",
            "question_address",
            "event_params__menu_name",
            "event_params__spent_to",
            "shop_consumable_item",
            "event_params__currency_name",
            "event_params__earned_amount",
            "event_params__spent_amount",
            "event_params__gold",
            "event_params__mini_game_ri",
            "event_params__ad_network",
            "event_params__ad_unit_id",
            "event_params__ad_instance",
            "event_params__ad_id",
            "event_params__ad_error_code",
            "event_server_delay_seconds",
            "device__mobile_marketing_name",
            "device__operating_system_version",
            "event_params__pp_accepted",
            "event_params__video_start",
            "event_params__video_finished",
            "event_params__entered",
            "event_params__shown",
            "event_params__opened",
            "event_params__return",
            "event_params__closed",
            "event_params__drag",
            "event_params__tutorial_video",
        ]

    out_df = ensure_columns(df.copy(), desired_cols)
    for c in ["event_datetime", "session_start_time", "session_end_time", "event_date"]:
        if c in out_df.columns:
            out_df[c] = pd.to_datetime(out_df[c], utc=True, errors="coerce")
    out_df.to_csv(processed_path, index=False)

    for name, dfx in dfs.items():
        fname = f"{name}_data.csv"
        fpath = csv_dir / fname
        desired = read_csv_header(schema_from / fname) if schema_from is not None else None
        dfx_to_write = ensure_columns(dfx.copy(), desired) if desired is not None else dfx
        dfx_to_write.to_csv(fpath, index=False)


def _write_raw(cfg: MockConfig, out_root: Path) -> None:
    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_df = _build_raw_events(cfg)

    # Always write JSONL so nested lists/dicts survive round-trips.
    jsonl_path = raw_dir / "pulled_from_bq.jsonl"
    raw_df.to_json(jsonl_path, orient="records", lines=True)

    # Optionally write parquet if pyarrow exists.
    try:
        df_for_parquet = raw_df.copy()
        for c in ["user_ltv", "event_dimensions", "ecommerce", "collected_traffic_source"]:
            if c in df_for_parquet.columns:
                df_for_parquet[c] = df_for_parquet[c].apply(
                    lambda x: None if isinstance(x, dict) and len(x) == 0 else x
                )
        _maybe_write_parquet(df_for_parquet, raw_dir / "pulled_from_bq.parquet")
    except Exception:
        # Parquet is a best-effort convenience; JSONL is the supported raw artifact.
        pass


def generate_all(cfg: MockConfig, out_root: Path, schema_from: Path | None = None, kind: str = "raw") -> None:
    out_root = out_root.resolve()

    if kind in ("raw", "both"):
        _write_raw(cfg, out_root)
    if kind in ("derived", "both"):
        _write_derived(cfg, out_root, schema_from)

    (out_root / "config_used.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
