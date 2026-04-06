from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from l2_project.factors import compute_snapshot_factors_to_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run snapshot factor calculations.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "parquet" / "snapshot",
        help="Input parquet root for normalized snapshot files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "factors" / "snapshot",
        help="Output parquet root for factor-enriched snapshot files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite factor parquet files that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_files = sorted(args.input_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found under {args.input_root}")

    outputs: list[Path] = []
    for source_path in parquet_files:
        relative_path = source_path.relative_to(args.input_root)
        output_path = args.output_root / relative_path
        outputs.append(
            compute_snapshot_factors_to_parquet(
                source_path,
                output_path,
                force=args.force,
            )
        )
    print(f"built {len(outputs)} factor parquet files into {args.output_root}")


if __name__ == "__main__":
    main()
