from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from l2_project.eval import (
    DEFAULT_MIN_CROSS_SECTION,
    DEFAULT_MIN_CROSS_SECTION_SCAN,
    compute_rank_ic_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Rank IC timeseries and summary reports from a panel parquet."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "panel" / "snapshot_panel.parquet",
        help="Input panel parquet path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval",
        help="Output directory for Rank IC reports.",
    )
    parser.add_argument(
        "--factor-column",
        action="append",
        default=[],
        help="Optional factor column selector. Repeatable.",
    )
    parser.add_argument(
        "--label-column",
        action="append",
        default=[],
        help="Optional label column selector. Repeatable.",
    )
    parser.add_argument(
        "--group-key",
        action="append",
        default=[],
        help="Cross-sectional grouping keys. Defaults to trading_day and event_time.",
    )
    parser.add_argument(
        "--min-cross-section",
        type=int,
        default=DEFAULT_MIN_CROSS_SECTION,
        help="Minimum valid symbols per timestamp used to compute Rank IC.",
    )
    parser.add_argument(
        "--scan-min-cross-section",
        type=int,
        action="append",
        default=[],
        help="Optional min-cross-section sweep values. Repeatable.",
    )
    parser.add_argument(
        "--disable-auction-filter",
        action="store_true",
        help="Disable the default continuous-auction time filter before evaluation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite evaluation outputs if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan_values = args.scan_min_cross_section or list(DEFAULT_MIN_CROSS_SECTION_SCAN)
    artifacts = compute_rank_ic_reports(
        args.input_path,
        args.output_root,
        factor_columns=args.factor_column or None,
        label_columns=args.label_column or None,
        group_keys=args.group_key or ("trading_day", "event_time"),
        min_cross_section=args.min_cross_section,
        min_cross_section_scan=scan_values,
        filter_auction=not args.disable_auction_filter,
        force=args.force,
    )
    print(f"built rank IC timeseries at {artifacts.timeseries_path}")
    print(f"built rank IC summary at {artifacts.summary_path}")
    print(f"built monthly rank IC summary at {artifacts.monthly_summary_path}")
    if artifacts.sweep_summary_path is not None:
        print(f"built min-cross-section scan at {artifacts.sweep_summary_path}")


if __name__ == "__main__":
    main()
