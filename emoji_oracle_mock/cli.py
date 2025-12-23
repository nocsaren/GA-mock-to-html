from __future__ import annotations

import argparse
from pathlib import Path

from .config_model import MockConfig
from .generate import generate_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic CSV datasets for emoji-oracle-analytics.")
    parser.add_argument("--out", type=Path, default=Path("./mock/output"), help="Output folder root.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config overriding defaults.")
    parser.add_argument("--schema-from", type=Path, default=None, help="Folder containing existing CSVs to mirror headers from.")

    parser.add_argument(
        "--kind",
        choices=["raw", "derived", "both"],
        default="raw",
        help="What to generate: raw (pull_from_bq-like), derived CSV outputs, or both.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config).")
    parser.add_argument("--users", type=int, default=None, help="Number of users (overrides config).")
    parser.add_argument("--days", type=int, default=None, help="Number of days (overrides config).")

    args = parser.parse_args(argv)

    cfg = MockConfig.load(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.users is not None:
        cfg.users = args.users
    if args.days is not None:
        cfg.days = args.days

    generate_all(cfg=cfg, out_root=args.out, schema_from=args.schema_from, kind=args.kind)
    return 0
