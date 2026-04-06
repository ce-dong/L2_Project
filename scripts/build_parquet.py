from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from l2_project.ingest import DATASET_KINDS, DatasetKind, build_ingest_jobs, ingest_csv_jobs

DEFAULT_SOURCE_DIRS: dict[str, dict[DatasetKind, Path]] = {
    "202511": {
        "snapshot": Path("/Users/tong/data/11_snapshot"),
        "order": Path("/Users/tong/data/11_order"),
        "trade": Path("/Users/tong/data/11_trade"),
    },  
    "202512": {
        "snapshot": Path("/Users/tong/data/12_snapshot"),
        "order": Path("/Users/tong/data/12_order"),
        "trade": Path("/Users/tong/data/12_trade"),
    },
}


def _normalize_month(month: str) -> str:
    normalized = month.strip()
    if normalized in DEFAULT_SOURCE_DIRS:
        return normalized
    if normalized == "11":
        return "202511"
    if normalized == "12":
        return "202512"
    raise ValueError(f"unsupported month: {month}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw Level-2 CSV files to parquet.")
    parser.add_argument(
        "--month",
        action="append",
        default=[],
        help="Month selector: 11, 12, 202511 or 202512. Repeatable.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASET_KINDS,
        default=[],
        help="Dataset selector: snapshot, order, trade. Repeatable.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "parquet",
        help="Output parquet root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite parquet files that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    months = [_normalize_month(month) for month in args.month] if args.month else None
    datasets = tuple(args.dataset) if args.dataset else None

    jobs = build_ingest_jobs(
        source_dirs_by_month=DEFAULT_SOURCE_DIRS,
        output_root=args.output_root,
        months=months,
        datasets=datasets,
    )
    outputs = ingest_csv_jobs(jobs, force=args.force)
    print(f"built {len(outputs)} parquet files into {args.output_root}")


if __name__ == "__main__":
    main()
