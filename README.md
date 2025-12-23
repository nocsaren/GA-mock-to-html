# Mock data generator

Standalone generator that produces synthetic datasets.

- **Raw mode**: emits a GA4 BigQuery-like dataframe (shape compatible with what `pull_from_bq()` returns).
- **Derived mode**: emits the same CSV outputs your pipeline writes under `data/csv/`.

## Quick start

From repo root:

```bash
python -m mock.emoji_oracle_mock --out ./mock/output --kind raw --seed 7 --users 200 --days 14
```

This writes:

- `./mock/output/raw/pulled_from_bq.jsonl` (recommended; preserves nested lists/dicts)
- `./mock/output/raw/pulled_from_bq.parquet` (only if `pyarrow` is installed)

To generate the pipeline-like CSV outputs:

```bash
python -m mock.emoji_oracle_mock --out ./mock/output --kind derived
```

## Renaming characters/items

Pass a config JSON to override vocab while keeping engineered columns consistent:

```bash
python -m mock.emoji_oracle_mock --out ./mock/output --kind raw --config ./mock/emoji_oracle_mock/config/example.json
```

Notes:
- Output column names stay compatible (e.g. `alicin_use_ratio`), but the underlying *values* (e.g. the item name stored in `event_params__spent_to`) can be renamed and the counts/ratios will still line up.

## Optional schema mirroring

If you want `processed_data.csv` to have exactly the same columns as an existing run (derived mode only), provide:

```bash
python -m mock.emoji_oracle_mock --out ./mock/output --schema-from ./data/csv
```

The generator will read just the header row of existing CSVs in that folder and fill unknown columns with empty values.
