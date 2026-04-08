from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from l2_project.panel import build_panel_to_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble a cross-sectional panel from labeled parquet files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "labeled" / "snapshot",
        help="Input root for labeled snapshot parquet files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "panel" / "snapshot_panel.parquet",
        help="Output parquet path for the assembled panel.",
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
        "--base-column",
        action="append",
        default=[],
        help="Optional passthrough base columns beyond the panel index. Repeatable.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the panel parquet file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factor_columns = args.factor_column or None
    label_columns = args.label_column or None
    base_columns = args.base_column or None

    output_path = build_panel_to_parquet(
        args.input_root,
        args.output_path,
        factor_columns=factor_columns,
        label_columns=label_columns,
        base_columns=base_columns or ("source_month",),
        force=args.force,
    )
    print(f"built panel parquet at {output_path}")


if __name__ == "__main__":
    main()
