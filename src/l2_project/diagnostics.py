from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from .eval import (
    CONTINUOUS_AUCTION_WINDOWS,
    _factor_nunique_column,
    _ic_column,
    _label_nunique_column,
    _nobs_column,
    _reason_column,
    _valid_column,
    filter_continuous_auction,
)
from .factors import SnapshotSchemaError
from .panel import infer_factor_columns, infer_label_columns

DEFAULT_COLUMN_COVERAGE_FILENAME = "column_coverage.parquet"
DEFAULT_SYMBOL_COVERAGE_FILENAME = "symbol_coverage.parquet"
DEFAULT_PAIR_REASON_FILENAME = "pair_reason_summary.parquet"
DEFAULT_EVENT_TIME_REASON_FILENAME = "event_time_reason_summary.parquet"


@dataclass(frozen=True)
class DiagnosticArtifacts:
    column_coverage_path: Path
    symbol_coverage_path: Path
    pair_reason_summary_path: Path
    event_time_reason_summary_path: Path


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


def _column_type(column_name: str) -> str:
    return "label" if column_name.startswith("fwd_ret_") else "factor"


def _pair_long_frame(
    timeseries: pl.LazyFrame,
    *,
    factor_columns: Sequence[str],
    label_columns: Sequence[str],
) -> pl.LazyFrame:
    frames: list[pl.LazyFrame] = []
    for factor_column in factor_columns:
        for label_column in label_columns:
            frames.append(
                timeseries.select(
                    [
                        pl.col("trading_day"),
                        pl.col("event_time"),
                        pl.lit(factor_column).alias("factor"),
                        pl.lit(label_column).alias("label"),
                        pl.col(_nobs_column(factor_column, label_column))
                        .cast(pl.Int64, strict=False)
                        .alias("n_obs"),
                        pl.col(_factor_nunique_column(factor_column, label_column))
                        .cast(pl.Int64, strict=False)
                        .alias("factor_n_unique"),
                        pl.col(_label_nunique_column(factor_column, label_column))
                        .cast(pl.Int64, strict=False)
                        .alias("label_n_unique"),
                        pl.col(_valid_column(factor_column, label_column))
                        .fill_null(False)
                        .alias("is_valid"),
                        pl.col(_reason_column(factor_column, label_column))
                        .cast(pl.Utf8, strict=False)
                        .alias("invalid_reason"),
                        pl.col(_ic_column(factor_column, label_column))
                        .cast(pl.Float64, strict=False)
                        .alias("rank_ic"),
                    ]
                )
            )

    if not frames:
        raise SnapshotSchemaError("no factor-label pairs were found for diagnostics")
    return pl.concat(frames, how="vertical")


def _validate_panel_columns(
    schema: Mapping[str, pl.DataType],
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    required_columns = ["trading_day", "event_time", "symbol"]
    missing_required = [column for column in required_columns if column not in schema]
    if missing_required:
        missing_text = ", ".join(missing_required)
        raise SnapshotSchemaError(f"missing panel diagnostics columns: {missing_text}")

    selected_factor_columns = list(factor_columns or infer_factor_columns(schema))
    selected_label_columns = list(label_columns or infer_label_columns(schema))
    if not selected_factor_columns:
        raise SnapshotSchemaError("no factor columns were found for diagnostics")
    if not selected_label_columns:
        raise SnapshotSchemaError("no label columns were found for diagnostics")
    return selected_factor_columns, selected_label_columns


def _validate_timeseries_columns(
    schema: Mapping[str, pl.DataType],
    *,
    factor_columns: Sequence[str],
    label_columns: Sequence[str],
) -> None:
    required_columns = ["trading_day", "event_time"]
    missing_required = [column for column in required_columns if column not in schema]
    if missing_required:
        missing_text = ", ".join(missing_required)
        raise SnapshotSchemaError(
            f"missing timeseries diagnostics columns: {missing_text}"
        )

    expected_columns: list[str] = []
    for factor_column in factor_columns:
        for label_column in label_columns:
            expected_columns.extend(
                [
                    _nobs_column(factor_column, label_column),
                    _factor_nunique_column(factor_column, label_column),
                    _label_nunique_column(factor_column, label_column),
                    _valid_column(factor_column, label_column),
                    _reason_column(factor_column, label_column),
                    _ic_column(factor_column, label_column),
                ]
            )

    missing_expected = [column for column in expected_columns if column not in schema]
    if missing_expected:
        missing_text = ", ".join(missing_expected)
        raise SnapshotSchemaError(
            f"missing timeseries diagnostics columns: {missing_text}"
        )


def build_column_coverage_summary(
    frame: pl.LazyFrame,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    filter_auction: bool = True,
) -> pl.LazyFrame:
    working = filter_continuous_auction(frame) if filter_auction else frame
    schema = working.collect_schema()
    selected_factor_columns, selected_label_columns = _validate_panel_columns(
        schema,
        factor_columns=factor_columns,
        label_columns=label_columns,
    )
    selected_columns = [*selected_factor_columns, *selected_label_columns]
    summary_frames: list[pl.LazyFrame] = []

    for column_name in selected_columns:
        column_expr = pl.col(column_name).cast(pl.Float64, strict=False)
        summary_frames.append(
            working.select(
                [
                    pl.lit(column_name).alias("column"),
                    pl.lit(_column_type(column_name)).alias("column_type"),
                    pl.len().alias("total_rows"),
                    column_expr.is_not_null().sum().alias("non_null_rows"),
                    column_expr.is_null().sum().alias("null_rows"),
                    column_expr.n_unique().alias("n_unique"),
                    column_expr.mean().alias("value_mean"),
                    column_expr.std(ddof=1).alias("value_std"),
                    column_expr.min().alias("value_min"),
                    column_expr.max().alias("value_max"),
                ]
            ).with_columns(
                (pl.col("non_null_rows") / pl.col("total_rows")).alias("coverage_ratio")
            )
        )

    return pl.concat(summary_frames, how="vertical").sort(["column_type", "column"])


def build_symbol_coverage_summary(
    frame: pl.LazyFrame,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    filter_auction: bool = True,
) -> pl.LazyFrame:
    working = filter_continuous_auction(frame) if filter_auction else frame
    schema = working.collect_schema()
    selected_factor_columns, selected_label_columns = _validate_panel_columns(
        schema,
        factor_columns=factor_columns,
        label_columns=label_columns,
    )
    group_keys = ["symbol"]
    if "source_month" in schema:
        group_keys.insert(0, "source_month")

    summary_frames: list[pl.LazyFrame] = []
    for column_name in [*selected_factor_columns, *selected_label_columns]:
        column_expr = pl.col(column_name).cast(pl.Float64, strict=False)
        summary_frames.append(
            working.group_by(group_keys)
            .agg(
                [
                    pl.len().alias("total_rows"),
                    column_expr.is_not_null().sum().alias("non_null_rows"),
                    column_expr.is_null().sum().alias("null_rows"),
                    column_expr.n_unique().alias("n_unique"),
                    column_expr.mean().alias("value_mean"),
                    column_expr.std(ddof=1).alias("value_std"),
                ]
            )
            .with_columns(
                [
                    pl.lit(column_name).alias("column"),
                    pl.lit(_column_type(column_name)).alias("column_type"),
                    (pl.col("non_null_rows") / pl.col("total_rows")).alias(
                        "coverage_ratio"
                    ),
                ]
            )
        )

    sort_columns = [*group_keys, "column_type", "column"]
    return pl.concat(summary_frames, how="vertical").sort(sort_columns)


def build_pair_reason_summary(
    timeseries: pl.LazyFrame,
    *,
    factor_columns: Sequence[str],
    label_columns: Sequence[str],
) -> pl.LazyFrame:
    schema = timeseries.collect_schema()
    _validate_timeseries_columns(
        schema,
        factor_columns=factor_columns,
        label_columns=label_columns,
    )
    long_frame = _pair_long_frame(
        timeseries,
        factor_columns=factor_columns,
        label_columns=label_columns,
    )
    return (
        long_frame.group_by(["factor", "label"])
        .agg(
            [
                pl.len().alias("groups_total"),
                pl.col("is_valid").sum().alias("groups_valid"),
                (~pl.col("is_valid")).sum().alias("groups_invalid"),
                (pl.col("invalid_reason") == "below_min_cross_section")
                .sum()
                .alias("groups_below_min_cross_section"),
                (pl.col("invalid_reason") == "factor_constant")
                .sum()
                .alias("groups_factor_constant"),
                (pl.col("invalid_reason") == "label_constant")
                .sum()
                .alias("groups_label_constant"),
                (pl.col("invalid_reason") == "both_constant")
                .sum()
                .alias("groups_both_constant"),
                (pl.col("invalid_reason") == "missing_pair_values")
                .sum()
                .alias("groups_missing_pair_values"),
                (pl.col("invalid_reason") == "nan_rank_ic")
                .sum()
                .alias("groups_nan_rank_ic"),
                pl.col("n_obs").filter(pl.col("is_valid")).mean().alias("avg_n_obs"),
                pl.col("rank_ic").mean().alias("rank_ic_mean"),
                pl.col("rank_ic").std(ddof=1).alias("rank_ic_std"),
            ]
        )
        .with_columns(
            (pl.col("groups_valid") / pl.col("groups_total")).alias("valid_ratio")
        )
        .sort(["factor", "label"])
    )


def build_event_time_reason_summary(
    timeseries: pl.LazyFrame,
    *,
    factor_columns: Sequence[str],
    label_columns: Sequence[str],
) -> pl.LazyFrame:
    schema = timeseries.collect_schema()
    _validate_timeseries_columns(
        schema,
        factor_columns=factor_columns,
        label_columns=label_columns,
    )
    long_frame = _pair_long_frame(
        timeseries,
        factor_columns=factor_columns,
        label_columns=label_columns,
    )
    return (
        long_frame.group_by(["factor", "label", "event_time"])
        .agg(
            [
                pl.len().alias("groups_total"),
                pl.col("is_valid").sum().alias("groups_valid"),
                (~pl.col("is_valid")).sum().alias("groups_invalid"),
                (pl.col("invalid_reason") == "below_min_cross_section")
                .sum()
                .alias("groups_below_min_cross_section"),
                (pl.col("invalid_reason") == "factor_constant")
                .sum()
                .alias("groups_factor_constant"),
                (pl.col("invalid_reason") == "label_constant")
                .sum()
                .alias("groups_label_constant"),
                (pl.col("invalid_reason") == "both_constant")
                .sum()
                .alias("groups_both_constant"),
                (pl.col("invalid_reason") == "missing_pair_values")
                .sum()
                .alias("groups_missing_pair_values"),
                (pl.col("invalid_reason") == "nan_rank_ic")
                .sum()
                .alias("groups_nan_rank_ic"),
                pl.col("n_obs").filter(pl.col("is_valid")).mean().alias("avg_n_obs"),
                pl.col("n_obs").filter(pl.col("is_valid")).min().alias("min_n_obs"),
                pl.col("n_obs").filter(pl.col("is_valid")).max().alias("max_n_obs"),
                pl.col("rank_ic").mean().alias("rank_ic_mean"),
            ]
        )
        .with_columns(
            (pl.col("groups_valid") / pl.col("groups_total")).alias("valid_ratio")
        )
        .sort(["factor", "label", "event_time"])
    )


def compute_diagnostic_reports(
    panel_path: Path,
    timeseries_path: Path,
    output_root: Path,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    filter_auction: bool = True,
    force: bool = False,
) -> DiagnosticArtifacts:
    column_coverage_path = output_root / DEFAULT_COLUMN_COVERAGE_FILENAME
    symbol_coverage_path = output_root / DEFAULT_SYMBOL_COVERAGE_FILENAME
    pair_reason_summary_path = output_root / DEFAULT_PAIR_REASON_FILENAME
    event_time_reason_summary_path = output_root / DEFAULT_EVENT_TIME_REASON_FILENAME

    if (
        column_coverage_path.exists()
        and symbol_coverage_path.exists()
        and pair_reason_summary_path.exists()
        and event_time_reason_summary_path.exists()
        and not force
    ):
        return DiagnosticArtifacts(
            column_coverage_path=column_coverage_path,
            symbol_coverage_path=symbol_coverage_path,
            pair_reason_summary_path=pair_reason_summary_path,
            event_time_reason_summary_path=event_time_reason_summary_path,
        )

    panel_frame = pl.scan_parquet(str(panel_path))
    panel_schema = panel_frame.collect_schema()
    selected_factor_columns, selected_label_columns = _validate_panel_columns(
        panel_schema,
        factor_columns=factor_columns,
        label_columns=label_columns,
    )
    timeseries_frame = pl.scan_parquet(str(timeseries_path))

    _sink_parquet_compat(
        build_column_coverage_summary(
            panel_frame,
            factor_columns=selected_factor_columns,
            label_columns=selected_label_columns,
            filter_auction=filter_auction,
        ),
        column_coverage_path,
    )
    _sink_parquet_compat(
        build_symbol_coverage_summary(
            panel_frame,
            factor_columns=selected_factor_columns,
            label_columns=selected_label_columns,
            filter_auction=filter_auction,
        ),
        symbol_coverage_path,
    )
    _sink_parquet_compat(
        build_pair_reason_summary(
            timeseries_frame,
            factor_columns=selected_factor_columns,
            label_columns=selected_label_columns,
        ),
        pair_reason_summary_path,
    )
    _sink_parquet_compat(
        build_event_time_reason_summary(
            timeseries_frame,
            factor_columns=selected_factor_columns,
            label_columns=selected_label_columns,
        ),
        event_time_reason_summary_path,
    )

    return DiagnosticArtifacts(
        column_coverage_path=column_coverage_path,
        symbol_coverage_path=symbol_coverage_path,
        pair_reason_summary_path=pair_reason_summary_path,
        event_time_reason_summary_path=event_time_reason_summary_path,
    )
