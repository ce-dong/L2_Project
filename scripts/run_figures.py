from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from l2_project.figures import build_figure_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-style figures from L2 project artifacts."
    )
    parser.add_argument(
        "--eval-summary-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "rank_ic_summary.parquet",
        help="Input Rank IC summary parquet.",
    )
    parser.add_argument(
        "--group-summary-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "grouping" / "group_return_summary.parquet",
        help="Input grouping summary parquet.",
    )
    parser.add_argument(
        "--group-monotonicity-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "grouping" / "group_monotonicity_summary.parquet",
        help="Input grouping monotonicity summary parquet.",
    )
    parser.add_argument(
        "--redundancy-recommendation-path",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "redundancy"
        / "factor_redundancy_recommendation.parquet",
        help="Input redundancy recommendation parquet.",
    )
    parser.add_argument(
        "--coverage-summary-path",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "diagnostics"
        / "column_coverage.parquet",
        help="Input column coverage parquet.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "figures",
        help="Output directory for generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_figure_bundle(
        eval_summary_path=args.eval_summary_path,
        group_summary_path=args.group_summary_path,
        monotonicity_summary_path=args.group_monotonicity_path,
        redundancy_recommendation_path=args.redundancy_recommendation_path,
        coverage_summary_path=args.coverage_summary_path,
        output_root=args.output_root,
    )
    print(f"built {artifacts.core_rank_ic_png}")
    print(f"built {artifacts.core_monotonicity_png}")
    print(f"built {artifacts.redundancy_heatmap_png}")
    print(f"built {artifacts.coverage_validity_png}")


if __name__ == "__main__":
    main()
