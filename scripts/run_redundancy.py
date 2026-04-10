from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from l2_project.eval import DEFAULT_MIN_CROSS_SECTION  # noqa: E402
from l2_project.redundancy import compute_redundancy_reports  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute factor redundancy diagnostics from panel and Rank IC artifacts."
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "panel" / "snapshot_panel.parquet",
        help="Input panel parquet path.",
    )
    parser.add_argument(
        "--rank-ic-timeseries-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "rank_ic_timeseries.parquet",
        help="Input Rank IC timeseries parquet path.",
    )
    parser.add_argument(
        "--rank-ic-summary-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "rank_ic_summary.parquet",
        help="Input Rank IC summary parquet path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "redundancy",
        help="Output directory for redundancy diagnostics.",
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
        help="Minimum valid symbols per timestamp used for factor-correlation diagnostics.",
    )
    parser.add_argument(
        "--disable-auction-filter",
        action="store_true",
        help="Disable the default continuous-auction filter before redundancy diagnostics.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite redundancy outputs if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = compute_redundancy_reports(
        args.panel_path,
        args.rank_ic_timeseries_path,
        args.rank_ic_summary_path,
        args.output_root,
        factor_columns=args.factor_column or None,
        label_columns=args.label_column or None,
        group_keys=args.group_key or ("trading_day", "event_time"),
        min_cross_section=args.min_cross_section,
        filter_auction=not args.disable_auction_filter,
        force=args.force,
    )
    print(f"built factor correlation timeseries at {artifacts.factor_corr_timeseries_path}")
    print(f"built factor correlation summary at {artifacts.factor_corr_summary_path}")
    print(f"built IC correlation summary at {artifacts.ic_corr_summary_path}")
    print(f"built redundancy recommendation at {artifacts.recommendation_path}")


if __name__ == "__main__":
    main()
