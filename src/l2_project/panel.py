from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from .factors import SnapshotSchemaError

PANEL_INDEX_COLUMNS = ["trading_day", "event_time", "symbol"]
DEFAULT_PANEL_BASE_COLUMNS: tuple[str, ...] = ("source_month",)
FACTOR_PREFIXES = (
    "micro_price_",
    "relative_spread_",
    "oir_",
    "book_slope_",
    "trade_imbalance_",
    "voi_",
    "ofi_",
)
LABEL_PREFIXES = ("fwd_ret_",)


def _sink_parquet_compat(frame: pl.LazyFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.sink_parquet(str(output_path), compression="zstd", statistics=True)
        return
    except AttributeError:
        pass
    except TypeError:
        try:
            frame.sink_parquet(str(output_path), compression="zstd")
            return
        except AttributeError:
            pass

    frame.collect(streaming=True).write_parquet(str(output_path), compression="zstd")


def discover_panel_inputs(input_root: Path) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"panel input root does not exist: {input_root}")
    paths = sorted(input_root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files found under {input_root}")
    return paths


def infer_factor_columns(schema: Mapping[str, pl.DataType]) -> list[str]:
    return sorted(
        column
        for column in schema
        if any(column.startswith(prefix) for prefix in FACTOR_PREFIXES)
    )


def infer_label_columns(schema: Mapping[str, pl.DataType]) -> list[str]:
    return sorted(
        column
        for column in schema
        if any(column.startswith(prefix) for prefix in LABEL_PREFIXES)
    )


def validate_panel_columns(
    schema: Mapping[str, pl.DataType],
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    base_columns: Sequence[str] = DEFAULT_PANEL_BASE_COLUMNS,
) -> tuple[list[str], list[str], list[str]]:
    missing_index = [column for column in PANEL_INDEX_COLUMNS if column not in schema]
    if missing_index:
        missing_text = ", ".join(missing_index)
        raise SnapshotSchemaError(f"missing panel index columns: {missing_text}")

    selected_factor_columns = list(factor_columns or infer_factor_columns(schema))
    selected_label_columns = list(label_columns or infer_label_columns(schema))
    selected_base_columns = list(base_columns)

    if not selected_factor_columns:
        raise SnapshotSchemaError("no factor columns were found for panel assembly")
    if not selected_label_columns:
        raise SnapshotSchemaError("no label columns were found for panel assembly")

    requested_columns = [
        *selected_base_columns,
        *selected_factor_columns,
        *selected_label_columns,
    ]
    missing_requested = [
        column for column in requested_columns if column not in schema
    ]
    if missing_requested:
        missing_text = ", ".join(missing_requested)
        raise SnapshotSchemaError(f"missing requested panel columns: {missing_text}")

    return selected_base_columns, selected_factor_columns, selected_label_columns


def build_panel_frame(
    input_paths: Sequence[Path],
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    base_columns: Sequence[str] = DEFAULT_PANEL_BASE_COLUMNS,
) -> pl.LazyFrame:
    frame = pl.scan_parquet([str(path) for path in input_paths])
    schema = frame.collect_schema()
    selected_base_columns, selected_factor_columns, selected_label_columns = (
        validate_panel_columns(
            schema,
            factor_columns=factor_columns,
            label_columns=label_columns,
            base_columns=base_columns,
        )
    )
    selected_columns = [
        *PANEL_INDEX_COLUMNS,
        *selected_base_columns,
        *selected_factor_columns,
        *selected_label_columns,
    ]
    return frame.select(selected_columns).sort(PANEL_INDEX_COLUMNS)


def build_panel_from_labeled_root(
    input_root: Path,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    base_columns: Sequence[str] = DEFAULT_PANEL_BASE_COLUMNS,
) -> pl.LazyFrame:
    input_paths = discover_panel_inputs(input_root)
    return build_panel_frame(
        input_paths,
        factor_columns=factor_columns,
        label_columns=label_columns,
        base_columns=base_columns,
    )


def build_panel_to_parquet(
    input_root: Path,
    output_path: Path,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    base_columns: Sequence[str] = DEFAULT_PANEL_BASE_COLUMNS,
    force: bool = False,
) -> Path:
    if output_path.exists() and not force:
        return output_path

    panel_frame = build_panel_from_labeled_root(
        input_root,
        factor_columns=factor_columns,
        label_columns=label_columns,
        base_columns=base_columns,
    )
    _sink_parquet_compat(panel_frame, output_path)
    return output_path
