from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from l2_project.grouping import (  # noqa: E402
    DEFAULT_NUM_GROUPS,
    compute_grouping_reports,
)
from l2_project.eval import DEFAULT_MIN_CROSS_SECTION  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cross-sectional factor grouping reports from a panel parquet."
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
        default=PROJECT_ROOT / "data" / "grouping",
        help="Output directory for grouping reports.",
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
        help="Minimum valid symbols per timestamp before grouping.",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=DEFAULT_NUM_GROUPS,
        help="Number of cross-sectional quantile groups.",
    )
    parser.add_argument(
        "--disable-auction-filter",
        action="store_true",
        help="Disable the default continuous-auction time filter before grouping.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite grouping outputs if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = compute_grouping_reports(
        args.input_path,
        args.output_root,
        factor_columns=args.factor_column or None,
        label_columns=args.label_column or None,
        group_keys=args.group_key or ("trading_day", "event_time"),
        min_cross_section=args.min_cross_section,
        num_groups=args.num_groups,
        filter_auction=not args.disable_auction_filter,
        force=args.force,
    )
    print(f"built group return timeseries at {artifacts.timeseries_path}")
    print(f"built group return summary at {artifacts.summary_path}")
    print(
        "built group monotonicity summary at "
        f"{artifacts.monotonicity_summary_path}"
    )


if __name__ == "__main__":
    main()
