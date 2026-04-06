from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from l2_project.labels import DEFAULT_HORIZONS_MS, compute_forward_returns_to_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run forward-return label alignment.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "factors" / "snapshot",
        help="Input parquet root for snapshot or factor-enriched snapshot files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeled" / "snapshot",
        help="Output parquet root for label-enriched files.",
    )
    parser.add_argument(
        "--horizon-ms",
        type=int,
        action="append",
        default=[],
        help="Forward return horizon in milliseconds. Repeatable.",
    )
    parser.add_argument(
        "--price-column",
        type=str,
        default="mid_price_1",
        help="Base price column used to compute forward returns.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite label parquet files that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons_ms = args.horizon_ms or list(DEFAULT_HORIZONS_MS)
    parquet_files = sorted(args.input_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found under {args.input_root}")

    outputs: list[Path] = []
    for source_path in parquet_files:
        relative_path = source_path.relative_to(args.input_root)
        output_path = args.output_root / relative_path
        outputs.append(
            compute_forward_returns_to_parquet(
                source_path,
                output_path,
                horizons_ms=horizons_ms,
                price_column=args.price_column,
                force=args.force,
            )
        )
    print(f"built {len(outputs)} label parquet files into {args.output_root}")


if __name__ == "__main__":
    main()
