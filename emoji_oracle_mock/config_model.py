from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VocabConfig:
    # Character names used in event_params__character_name
    characters: list[str] = field(default_factory=lambda: ["t", "mi", "la", "so"])
    special_character_for_offsets: str = "t"

    # Energy-like items used in event_params__spent_to
    # These map to engineered columns: alicin_used/coffee_used/cauldron_used
    alicin_name: str = "AliCin"
    coffee_name: str = "Coffee"
    cauldron_name: str = "Cauldron"

    # Scroll menu naming (affects scroll_opened)
    scroll_menu_name: str = "Scroll Menu"

    # Consumables used in shop_consumable_item
    potion_name: str = "Potion"
    incense_name: str = "Incense"
    amulet_name: str = "Amulet"

    # Wheel identifiers
    wheel_impression_ri: str = "Daily Spin"
    wheel_skip_ri: str = "spin_skipped"


@dataclass
class MockConfig:
    seed: int = 7

    # How much data
    users: int = 200
    days: int = 14
    avg_sessions_per_user: float = 2.2

    # Question progression
    tiers: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    questions_per_tier: int = 12

    # Distribution / metadata
    countries: list[str] = field(default_factory=lambda: ["United States", "Türkiye"])
    app_versions: list[str] = field(default_factory=lambda: ["1.0.5", "1.0.6", "1.0.7"])
    operating_systems: list[str] = field(default_factory=lambda: ["ANDROID", "IOS"])

    vocab: VocabConfig = field(default_factory=VocabConfig)

    @staticmethod
    def load(path: Path | None) -> "MockConfig":
        if path is None:
            return MockConfig()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return MockConfig.from_dict(raw)

    @staticmethod
    def from_dict(d: dict) -> "MockConfig":
        vocab_dict = d.get("vocab", {})
        vocab = VocabConfig(**vocab_dict)

        cfg_kwargs = {k: v for k, v in d.items() if k != "vocab"}
        return MockConfig(vocab=vocab, **cfg_kwargs)
