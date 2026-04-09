# L2 Microstructure MVP

Low-memory A-share Level-2 microstructure pipeline built on `Polars Lazy API + Parquet`.

This project focuses on one narrow problem: build a reproducible Level-2 factor research pipeline under hard hardware limits, then evaluate factors with cross-sectional time-series Rank IC instead of self-congratulatory PnL plots.

## Scope

- Market: A-share Shenzhen stocks
- Current research universe: 40 liquid names
- Sample window: November to December 2025
- Data types:
  - 3-second five-level snapshots
  - order records
  - trade records
- Hardware boundary: 16GB unified memory on M1 Pro

## Why This Repo Exists

The hard part is not writing factor formulas. The hard part is keeping the entire pipeline honest:

- no in-memory hoarding of raw CSV
- no hidden cross-day leakage
- no forward-looking label alignment
- no pretending asynchronous clocks form a clean cross-section

The code is split so each layer has one job:

- [`src/l2_project/ingest.py`](/Users/tong/L2_Project/src/l2_project/ingest.py): scan raw CSV, map schema, write per-symbol parquet
- [`src/l2_project/factors.py`](/Users/tong/L2_Project/src/l2_project/factors.py): build factor expressions and lazy compute graph
- [`src/l2_project/labels.py`](/Users/tong/L2_Project/src/l2_project/labels.py): align forward returns within day, forbid cross-day leakage
- [`src/l2_project/panel.py`](/Users/tong/L2_Project/src/l2_project/panel.py): assemble a unified cross-sectional panel
- [`src/l2_project/eval.py`](/Users/tong/L2_Project/src/l2_project/eval.py): compute Rank IC timeseries, summary, monthly summary, and threshold scans

## Factors

Current implemented factors:

- `micro_price_1`
- `relative_spread_1`
- `oir_1`
- `oir_5`
- `book_slope_5`
- `trade_imbalance_3s`
- `voi_1`
- `ofi_1`

Implementation details worth noting:

- `VOI` and `OFI` are explicitly decoupled.
- `VOI` is only defined when best bid and ask prices are unchanged.
- `OFI` keeps the event-flow interpretation at the best quote.
- invalid best quotes produce nulls instead of silently poisoning factors.

## Labels

Forward return labels are computed from best-quote mid price:

- `fwd_ret_3000ms`
- `fwd_ret_30000ms`

Alignment rules:

- forward match only
- same `symbol`
- same `trading_day`
- no cross-day carry

## Universe Construction

The original 50-stock pool contained phase-shifted clocks. That made many cross-sections statistically fake: same clock label, different actual phase.

The current 40-stock universe was rebuilt using two filters:

- tick-grid integrity during continuous auction
- liquidity proxy from trade and order record counts

This was the key turning point. The project stopped being a messy timestamp exercise and became a usable cross-sectional panel.

## Current Result Snapshot

The latest full rebuild uses the 40-stock Shenzhen universe for both November and December 2025.

Key evaluation outputs:

- panel: [`data/panel/snapshot_panel.parquet`](/Users/tong/L2_Project/data/panel/snapshot_panel.parquet)
- timeseries IC: [`data/eval/rank_ic_timeseries.parquet`](/Users/tong/L2_Project/data/eval/rank_ic_timeseries.parquet)
- summary: [`data/eval/rank_ic_summary.parquet`](/Users/tong/L2_Project/data/eval/rank_ic_summary.parquet)
- monthly summary: [`data/eval/rank_ic_monthly_summary.parquet`](/Users/tong/L2_Project/data/eval/rank_ic_monthly_summary.parquet)
- threshold scan: [`data/eval/rank_ic_min_cross_section_scan.parquet`](/Users/tong/L2_Project/data/eval/rank_ic_min_cross_section_scan.parquet)

Main findings from the current run:

- `micro_price_1`, `ofi_1`, and `oir_5` achieve `valid_ratio = 1.0`
- `oir_1` achieves `valid_ratio = 0.999789`
- `voi_1` remains sparse by design but still reaches `0.981874` to `0.985425` at `min_cross_section = 10`
- average cross-section size for non-`VOI` factors is about `39.7`, which is close to the full 40-stock universe
- `below_min_cross_section` has effectively disappeared for `micro_price`, `OFI`, and `OIR`

Factor effectiveness in the current run:

- `oir_1` is the strongest factor
  - `Rank IC mean = 0.219557` for `fwd_ret_3000ms`
  - `Rank IC mean = 0.105844` for `fwd_ret_30000ms`
- `oir_5` is positive and stable
  - `0.106027` for `3000ms`
  - `0.074379` for `30000ms`
- `ofi_1` is positive and stable
  - `0.054188` for `3000ms`
  - `0.065280` for `30000ms`
- `voi_1` is positive but structurally sparser
  - `0.098577` for `3000ms`
  - `0.092684` for `30000ms`
- `micro_price_1` is negative in this sample

One important point: the biggest improvement after rebuilding the universe was not factor mean IC itself. It was cross-sectional validity. In the earlier mixed-quality universe, many timestamps failed the minimum cross-section requirement. In the current 40-stock universe, most factors are computable at virtually every timestamp.

The latest codebase also includes two additional snapshot factors for the next evaluation round:

- `relative_spread_1`
- `book_slope_5`

The trade factor pipeline also supports:

- `trade_imbalance_3s`

## Reproducibility

Install dependencies in the project environment:

```bash
python -m pip install "polars>=1.14.0"
```

Rebuild the full pipeline:

```bash
python scripts/build_parquet.py
python scripts/run_factors.py --force
python scripts/run_labels.py --force
python scripts/run_panel.py --force
python scripts/run_eval.py --force
```

Run the smoke tests:

```bash
python -m unittest -v tests.test_smoke
```

## What This Repo Does Not Pretend To Be

- It is not a backtest report.
- It is not a production trading system.
- It does not claim alpha robustness outside the current universe and sample window.

What it does show is harder to fake:

- memory-aware data engineering
- explicit leakage control
- cross-sectional panel construction under imperfect L2 clocks
- factor evaluation with diagnostic transparency
