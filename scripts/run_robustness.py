from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from l2_project.robustness import compute_robustness_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build session and segment robustness reports from Rank IC timeseries."
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
        default=PROJECT_ROOT / "data" / "robustness",
        help="Output directory for robustness parquet files.",
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
        "--force",
        action="store_true",
        help="Overwrite robustness outputs if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = compute_robustness_reports(
        args.timeseries_path,
        args.output_root,
        factor_columns=args.factor_column or None,
        label_columns=args.label_column or None,
        force=args.force,
    )
    print(f"built session robustness summary at {artifacts.session_summary_path}")
    print(
        f"built month-session robustness summary at "
        f"{artifacts.month_session_summary_path}"
    )
    print(f"built stability summary at {artifacts.stability_summary_path}")


if __name__ == "__main__":
    main()
