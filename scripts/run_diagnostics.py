from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from l2_project.diagnostics import compute_diagnostic_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build factor and label diagnostics from panel and rank IC timeseries parquet files."
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "panel" / "snapshot_panel.parquet",
        help="Input panel parquet path.",
    )
    parser.add_argument(
        "--timeseries-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "rank_ic_timeseries.parquet",
        help="Input rank IC timeseries parquet path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "diagnostics",
        help="Output directory for diagnostics parquet files.",
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
        "--disable-auction-filter",
        action="store_true",
        help="Disable the default continuous-auction filter for panel coverage diagnostics.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite diagnostics outputs if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = compute_diagnostic_reports(
        args.panel_path,
        args.timeseries_path,
        args.output_root,
        factor_columns=args.factor_column or None,
        label_columns=args.label_column or None,
        filter_auction=not args.disable_auction_filter,
        force=args.force,
    )
    print(f"built column coverage diagnostics at {artifacts.column_coverage_path}")
    print(f"built symbol coverage diagnostics at {artifacts.symbol_coverage_path}")
    print(f"built pair reason diagnostics at {artifacts.pair_reason_summary_path}")
    print(
        f"built intraday event-time diagnostics at "
        f"{artifacts.event_time_reason_summary_path}"
    )


if __name__ == "__main__":
    main()
