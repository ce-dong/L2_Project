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
- [`src/l2_project/trade_factors.py`](/Users/tong/L2_Project/src/l2_project/trade_factors.py): aggregate trade records onto snapshot clocks and compute trade-flow factors
- [`src/l2_project/labels.py`](/Users/tong/L2_Project/src/l2_project/labels.py): align forward returns within day, forbid cross-day leakage
- [`src/l2_project/panel.py`](/Users/tong/L2_Project/src/l2_project/panel.py): assemble a unified cross-sectional panel
- [`src/l2_project/eval.py`](/Users/tong/L2_Project/src/l2_project/eval.py): compute Rank IC timeseries, summary, monthly summary, and threshold scans
- [`src/l2_project/diagnostics.py`](/Users/tong/L2_Project/src/l2_project/diagnostics.py): measure factor coverage, invalid reasons, and symbol-level sparsity
- [`src/l2_project/robustness.py`](/Users/tong/L2_Project/src/l2_project/robustness.py): summarize session-level and month-session stability
- [`src/l2_project/grouping.py`](/Users/tong/L2_Project/src/l2_project/grouping.py): test cross-sectional quantile monotonicity and top-minus-bottom spreads
- [`src/l2_project/redundancy.py`](/Users/tong/L2_Project/src/l2_project/redundancy.py): diagnose factor overlap from cross-sectional correlation and IC-series correlation

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
- `trade_imbalance_3s` is computed from true trade records only: `price > 0`, `side in {0, 1}`, aligned to snapshot `event_time` on 3-second buckets.
- invalid best quotes produce nulls instead of silently poisoning factors.

Current presentation split:

- core factors: `oir_1`, `book_slope_5`, `trade_imbalance_3s`, `voi_1`, `relative_spread_1`
- supplementary or control factors: `oir_5`, `ofi_1`, `micro_price_1`

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
- diagnostics:
  - [`data/diagnostics/column_coverage.parquet`](/Users/tong/L2_Project/data/diagnostics/column_coverage.parquet)
  - [`data/diagnostics/symbol_coverage.parquet`](/Users/tong/L2_Project/data/diagnostics/symbol_coverage.parquet)
  - [`data/diagnostics/pair_reason_summary.parquet`](/Users/tong/L2_Project/data/diagnostics/pair_reason_summary.parquet)
- robustness:
  - [`data/robustness/rank_ic_session_summary.parquet`](/Users/tong/L2_Project/data/robustness/rank_ic_session_summary.parquet)
  - [`data/robustness/rank_ic_month_session_summary.parquet`](/Users/tong/L2_Project/data/robustness/rank_ic_month_session_summary.parquet)
  - [`data/robustness/rank_ic_stability_summary.parquet`](/Users/tong/L2_Project/data/robustness/rank_ic_stability_summary.parquet)
- grouping:
  - [`data/grouping/group_return_timeseries.parquet`](/Users/tong/L2_Project/data/grouping/group_return_timeseries.parquet)
  - [`data/grouping/group_return_summary.parquet`](/Users/tong/L2_Project/data/grouping/group_return_summary.parquet)
  - [`data/grouping/group_monotonicity_summary.parquet`](/Users/tong/L2_Project/data/grouping/group_monotonicity_summary.parquet)
- redundancy:
  - [`data/redundancy/factor_corr_summary.parquet`](/Users/tong/L2_Project/data/redundancy/factor_corr_summary.parquet)
  - [`data/redundancy/ic_corr_summary.parquet`](/Users/tong/L2_Project/data/redundancy/ic_corr_summary.parquet)
  - [`data/redundancy/factor_redundancy_recommendation.parquet`](/Users/tong/L2_Project/data/redundancy/factor_redundancy_recommendation.parquet)

Main findings from the current run:

- `micro_price_1`, `ofi_1`, `oir_5`, and `trade_imbalance_3s` achieve `valid_ratio = 1.0`
- `oir_1` achieves `valid_ratio = 0.999789`
- `book_slope_5` achieves `valid_ratio = 0.999789`
- `relative_spread_1` achieves `valid_ratio = 0.999789`
- `voi_1` remains sparse by design but still reaches `0.981874` to `0.985425` at `min_cross_section = 10`
- average cross-section size for non-`VOI` factors is about `39.7`, which is close to the full 40-stock universe
- `below_min_cross_section` has effectively disappeared for every non-`VOI` factor in the current universe

Factor effectiveness in the current run:

- `oir_1` is the strongest factor
  - `Rank IC mean = 0.219557` for `fwd_ret_3000ms`
  - `Rank IC mean = 0.105844` for `fwd_ret_30000ms`
- `book_slope_5` is the best added snapshot factor
  - `0.183290` for `3000ms`
  - `0.106717` for `30000ms`
- `oir_5` is positive and stable
  - `0.106027` for `3000ms`
  - `0.074379` for `30000ms`
- `ofi_1` is positive and stable
  - `0.054188` for `3000ms`
  - `0.065280` for `30000ms`
- `trade_imbalance_3s` is positive, dense, and stable after correcting trade-side sign mapping
  - `0.056876` for `3000ms`
  - `0.066407` for `30000ms`
  - `valid_ratio = 1.0`
  - `avg_cross_section ≈ 39.26`
- `voi_1` is positive but structurally sparser
  - `0.098577` for `3000ms`
  - `0.092684` for `30000ms`
- `relative_spread_1` is weakly positive
  - `0.012247` for `3000ms`
  - `0.013812` for `30000ms`
- `oir_5` and `ofi_1` remain positive, but are now treated as supplementary factors rather than headline factors
- `micro_price_1` is negative in this sample and is retained as a control rather than a core factor

One important point: the biggest improvement after rebuilding the universe was not factor mean IC itself. It was cross-sectional validity. In the earlier mixed-quality universe, many timestamps failed the minimum cross-section requirement. In the current 40-stock universe, most factors are computable at virtually every timestamp.

Diagnostics and robustness now make that claim explicit instead of rhetorical:

- `trade_imbalance_3s` has `coverage_ratio = 0.987924`
- `trade_imbalance_3s` session IC stays positive in both morning and afternoon sessions
- `positive_segment_ratio = 1.0` for both `3000ms` and `30000ms`
- `VOI` sparsity is still real, but it is now traceable to factor definition instead of universe clock damage

Grouping and redundancy diagnostics are now also part of the selection logic:

- the final headline set is not chosen by rank IC alone
- `oir_1`, `book_slope_5`, `trade_imbalance_3s`, and `voi_1` all pass both Rank IC and quantile-group monotonicity checks
- `relative_spread_1` is kept in the headline set as a weak but distinct liquidity-cost dimension
- `book_slope_5` is preferred over `oir_5` after redundancy diagnostics
- `ofi_1` is kept as a supplementary event-flow factor because it overlaps too much with `voi_1` in the current sample
- `micro_price_1` remains a control factor, not a headline signal

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
python scripts/run_diagnostics.py --force
python scripts/run_robustness.py --force
python scripts/run_grouping.py --force
python scripts/run_redundancy.py --force
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
