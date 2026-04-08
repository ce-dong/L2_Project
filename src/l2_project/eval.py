from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from .factors import SnapshotSchemaError
from .panel import (
    PANEL_INDEX_COLUMNS,
    infer_factor_columns,
    infer_label_columns,
)

DEFAULT_GROUP_KEYS: tuple[str, ...] = ("trading_day", "event_time")
DEFAULT_MIN_CROSS_SECTION = 10
DEFAULT_MIN_CROSS_SECTION_SCAN: tuple[int, ...] = (8, 10, 15, 20)
CONTINUOUS_AUCTION_WINDOWS: tuple[tuple[int, int], ...] = (
    (93_000_000, 113_000_000),
    (130_000_000, 145_700_000),
)


@dataclass(frozen=True)
class EvalArtifacts:
    timeseries_path: Path
    summary_path: Path
    monthly_summary_path: Path
    sweep_summary_path: Path | None = None


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


def _pair_key(factor_column: str, label_column: str) -> str:
    return f"{factor_column}__{label_column}"


def _ic_column(factor_column: str, label_column: str) -> str:
    return f"rank_ic__{_pair_key(factor_column, label_column)}"


def _nobs_column(factor_column: str, label_column: str) -> str:
    return f"n_obs__{_pair_key(factor_column, label_column)}"


def _factor_nunique_column(factor_column: str, label_column: str) -> str:
    return f"factor_n_unique__{_pair_key(factor_column, label_column)}"


def _label_nunique_column(factor_column: str, label_column: str) -> str:
    return f"label_n_unique__{_pair_key(factor_column, label_column)}"


def _valid_column(factor_column: str, label_column: str) -> str:
    return f"is_valid__{_pair_key(factor_column, label_column)}"


def _reason_column(factor_column: str, label_column: str) -> str:
    return f"invalid_reason__{_pair_key(factor_column, label_column)}"


def _missing_pair_reason() -> str:
    return "missing_pair_values"


def _continuous_auction_mask(
    *,
    event_time_column: str = "event_time",
    windows: Sequence[tuple[int, int]] = CONTINUOUS_AUCTION_WINDOWS,
) -> pl.Expr:
    mask: pl.Expr | None = None
    event_time = pl.col(event_time_column)
    for start_time, end_time in windows:
        window_mask = (event_time >= start_time) & (event_time <= end_time)
        mask = window_mask if mask is None else (mask | window_mask)
    if mask is None:
        raise ValueError("at least one continuous auction window is required")
    return mask


def filter_continuous_auction(
    frame: pl.LazyFrame,
    *,
    event_time_column: str = "event_time",
    windows: Sequence[tuple[int, int]] = CONTINUOUS_AUCTION_WINDOWS,
) -> pl.LazyFrame:
    return frame.filter(
        _continuous_auction_mask(
            event_time_column=event_time_column,
            windows=windows,
        )
    )


def validate_panel_schema(
    schema: Mapping[str, pl.DataType],
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> tuple[list[str], list[str]]:
    required_columns = list(PANEL_INDEX_COLUMNS)
    missing_required = [column for column in required_columns if column not in schema]
    if missing_required:
        missing_text = ", ".join(missing_required)
        raise SnapshotSchemaError(f"missing panel index columns: {missing_text}")

    missing_group_keys = [column for column in group_keys if column not in schema]
    if missing_group_keys:
        missing_text = ", ".join(missing_group_keys)
        raise SnapshotSchemaError(f"missing evaluation group keys: {missing_text}")

    selected_factor_columns = list(factor_columns or infer_factor_columns(schema))
    selected_label_columns = list(label_columns or infer_label_columns(schema))
    if not selected_factor_columns:
        raise SnapshotSchemaError("no factor columns were found for evaluation")
    if not selected_label_columns:
        raise SnapshotSchemaError("no label columns were found for evaluation")

    missing_selected = [
        column
        for column in [*selected_factor_columns, *selected_label_columns]
        if column not in schema
    ]
    if missing_selected:
        missing_text = ", ".join(missing_selected)
        raise SnapshotSchemaError(f"missing evaluation columns: {missing_text}")

    return selected_factor_columns, selected_label_columns


def validate_min_cross_section(min_cross_section: int) -> int:
    normalized = int(min_cross_section)
    if normalized <= 1:
        raise ValueError("min_cross_section must be greater than 1")
    return normalized


def validate_min_cross_section_scan(
    min_cross_sections: Sequence[int],
) -> tuple[int, ...]:
    normalized = tuple(
        sorted({validate_min_cross_section(min_cross_section) for min_cross_section in min_cross_sections})
    )
    if not normalized:
        raise ValueError("at least one min_cross_section value is required")
    return normalized


def _rank_ic_frame_for_pair(
    frame: pl.LazyFrame,
    *,
    factor_column: str,
    label_column: str,
    group_keys: Sequence[str],
    min_cross_section: int,
) -> pl.LazyFrame:
    factor_expr = pl.col(factor_column).cast(pl.Float64, strict=False).alias("__factor")
    label_expr = pl.col(label_column).cast(pl.Float64, strict=False).alias("__label")
    nobs_column = _nobs_column(factor_column, label_column)
    factor_nunique_column = _factor_nunique_column(factor_column, label_column)
    label_nunique_column = _label_nunique_column(factor_column, label_column)
    valid_column = _valid_column(factor_column, label_column)
    reason_column = _reason_column(factor_column, label_column)
    ic_column = _ic_column(factor_column, label_column)
    raw_ic_column = f"__raw_{ic_column}"

    return (
        frame.select([*group_keys, factor_expr, label_expr])
        .filter(pl.col("__factor").is_not_null() & pl.col("__label").is_not_null())
        .group_by(list(group_keys))
        .agg(
            [
                pl.len().alias(nobs_column),
                pl.col("__factor").n_unique().alias(factor_nunique_column),
                pl.col("__label").n_unique().alias(label_nunique_column),
                pl.corr(
                    pl.col("__factor").rank(method="average"),
                    pl.col("__label").rank(method="average"),
                    method="pearson",
                ).alias(raw_ic_column),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col(nobs_column) < min_cross_section)
                .then(pl.lit("below_min_cross_section"))
                .when(
                    (pl.col(factor_nunique_column) <= 1)
                    & (pl.col(label_nunique_column) <= 1)
                )
                .then(pl.lit("both_constant"))
                .when(pl.col(factor_nunique_column) <= 1)
                .then(pl.lit("factor_constant"))
                .when(pl.col(label_nunique_column) <= 1)
                .then(pl.lit("label_constant"))
                .when(pl.col(raw_ic_column).is_nan())
                .then(pl.lit("nan_rank_ic"))
                .otherwise(None)
                .alias(reason_column),
            ]
        )
        .with_columns(
            [
                pl.col(reason_column).is_null().alias(valid_column),
                pl.when(pl.col(reason_column).is_null() & pl.col(raw_ic_column).is_finite())
                .then(pl.col(raw_ic_column))
                .otherwise(None)
                .alias(ic_column),
            ]
        )
        .drop(raw_ic_column)
    )


def build_rank_ic_timeseries(
    frame: pl.LazyFrame,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    filter_auction: bool = True,
) -> pl.LazyFrame:
    working = filter_continuous_auction(frame) if filter_auction else frame
    schema = working.collect_schema()
    selected_factor_columns, selected_label_columns = validate_panel_schema(
        schema,
        factor_columns=factor_columns,
        label_columns=label_columns,
        group_keys=group_keys,
    )
    normalized_min_cross_section = validate_min_cross_section(min_cross_section)
    base = working.select(list(group_keys)).unique().sort(list(group_keys))

    for factor_column in selected_factor_columns:
        for label_column in selected_label_columns:
            pair_frame = _rank_ic_frame_for_pair(
                working,
                factor_column=factor_column,
                label_column=label_column,
                group_keys=group_keys,
                min_cross_section=normalized_min_cross_section,
            )
            base = base.join(pair_frame, on=list(group_keys), how="left")
            nobs_column = _nobs_column(factor_column, label_column)
            reason_column = _reason_column(factor_column, label_column)
            valid_column = _valid_column(factor_column, label_column)
            base = base.with_columns(
                [
                    pl.when(pl.col(nobs_column).is_null())
                    .then(pl.lit(_missing_pair_reason()))
                    .otherwise(pl.col(reason_column))
                    .alias(reason_column),
                    pl.col(valid_column).fill_null(False).alias(valid_column),
                ]
            )

    return base.sort(list(group_keys))


def _summary_frame_for_pair(
    timeseries: pl.LazyFrame,
    *,
    factor_column: str,
    label_column: str,
) -> pl.LazyFrame:
    ic_column = _ic_column(factor_column, label_column)
    nobs_column = _nobs_column(factor_column, label_column)
    valid_column = _valid_column(factor_column, label_column)
    reason_column = _reason_column(factor_column, label_column)
    return timeseries.select(
        [
            pl.lit(factor_column).alias("factor"),
            pl.lit(label_column).alias("label"),
            pl.len().alias("groups_total"),
            pl.col(valid_column).fill_null(False).sum().alias("groups_valid"),
            (pl.col(reason_column) == "below_min_cross_section")
            .sum()
            .alias("groups_below_min_cross_section"),
            (pl.col(reason_column) == "factor_constant")
            .sum()
            .alias("groups_factor_constant"),
            (pl.col(reason_column) == "label_constant")
            .sum()
            .alias("groups_label_constant"),
            (pl.col(reason_column) == "both_constant")
            .sum()
            .alias("groups_both_constant"),
            (pl.col(reason_column) == _missing_pair_reason())
            .sum()
            .alias("groups_missing_pair_values"),
            (pl.col(reason_column) == "nan_rank_ic").sum().alias("groups_nan_rank_ic"),
            pl.col(ic_column).mean().alias("rank_ic_mean"),
            pl.col(ic_column).std(ddof=1).alias("rank_ic_std"),
            pl.col(ic_column).count().alias("time_points"),
            pl.col(ic_column).gt(0).mean().alias("positive_ratio"),
            pl.col(nobs_column).filter(pl.col(valid_column)).mean().alias("avg_cross_section"),
        ]
    ).with_columns(
        [
            (pl.col("groups_valid") / pl.col("groups_total")).alias("valid_ratio"),
            (pl.col("groups_total") - pl.col("groups_valid")).alias("groups_invalid"),
        ]
    )


def _monthly_summary_frame_for_pair(
    timeseries: pl.LazyFrame,
    *,
    factor_column: str,
    label_column: str,
    group_keys: Sequence[str],
) -> pl.LazyFrame:
    ic_column = _ic_column(factor_column, label_column)
    nobs_column = _nobs_column(factor_column, label_column)
    valid_column = _valid_column(factor_column, label_column)
    reason_column = _reason_column(factor_column, label_column)
    if "trading_day" not in group_keys:
        raise SnapshotSchemaError("monthly summary requires 'trading_day' in group keys")

    return (
        timeseries.with_columns(
            pl.col("trading_day")
            .cast(pl.Utf8)
            .str.slice(0, 6)
            .alias("month")
        )
        .group_by("month")
        .agg(
            [
                pl.len().alias("groups_total"),
                pl.col(valid_column).fill_null(False).sum().alias("groups_valid"),
                (pl.col(reason_column) == "below_min_cross_section")
                .sum()
                .alias("groups_below_min_cross_section"),
                (pl.col(reason_column) == "factor_constant")
                .sum()
                .alias("groups_factor_constant"),
                (pl.col(reason_column) == "label_constant")
                .sum()
                .alias("groups_label_constant"),
                (pl.col(reason_column) == "both_constant")
                .sum()
                .alias("groups_both_constant"),
                (pl.col(reason_column) == _missing_pair_reason())
                .sum()
                .alias("groups_missing_pair_values"),
                (pl.col(reason_column) == "nan_rank_ic")
                .sum()
                .alias("groups_nan_rank_ic"),
                pl.col(ic_column).mean().alias("rank_ic_mean"),
                pl.col(ic_column).std(ddof=1).alias("rank_ic_std"),
                pl.col(ic_column).count().alias("time_points"),
                pl.col(ic_column).gt(0).mean().alias("positive_ratio"),
                pl.col(nobs_column).filter(pl.col(valid_column)).mean().alias("avg_cross_section"),
            ]
        )
        .with_columns(
            [
                pl.lit(factor_column).alias("factor"),
                pl.lit(label_column).alias("label"),
                (pl.col("groups_valid") / pl.col("groups_total")).alias("valid_ratio"),
                (pl.col("groups_total") - pl.col("groups_valid")).alias("groups_invalid"),
            ]
        )
        .select(
            [
                "month",
                "factor",
                "label",
                "groups_total",
                "groups_valid",
                "groups_invalid",
                "groups_below_min_cross_section",
                "groups_factor_constant",
                "groups_label_constant",
                "groups_both_constant",
                "groups_missing_pair_values",
                "groups_nan_rank_ic",
                "valid_ratio",
                "rank_ic_mean",
                "rank_ic_std",
                "time_points",
                "positive_ratio",
                "avg_cross_section",
            ]
        )
    )


def _with_derived_stats(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.with_columns(
        [
            pl.when(pl.col("rank_ic_std") > 0)
            .then(pl.col("rank_ic_mean") / pl.col("rank_ic_std"))
            .otherwise(None)
            .alias("icir"),
            pl.when((pl.col("rank_ic_std") > 0) & (pl.col("time_points") > 1))
            .then(
                pl.col("rank_ic_mean")
                / (pl.col("rank_ic_std") / pl.col("time_points").sqrt())
            )
            .otherwise(None)
            .alias("t_stat"),
        ]
    )


def build_rank_ic_summary(
    timeseries: pl.LazyFrame,
    *,
    factor_columns: Sequence[str],
    label_columns: Sequence[str],
) -> pl.LazyFrame:
    summary_frames: list[pl.LazyFrame] = []
    for factor_column in factor_columns:
        for label_column in label_columns:
            summary_frames.append(
                _summary_frame_for_pair(
                    timeseries,
                    factor_column=factor_column,
                    label_column=label_column,
                )
            )
    return _with_derived_stats(pl.concat(summary_frames, how="vertical"))


def build_monthly_rank_ic_summary(
    timeseries: pl.LazyFrame,
    *,
    factor_columns: Sequence[str],
    label_columns: Sequence[str],
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> pl.LazyFrame:
    monthly_frames: list[pl.LazyFrame] = []
    for factor_column in factor_columns:
        for label_column in label_columns:
            monthly_frames.append(
                _monthly_summary_frame_for_pair(
                    timeseries,
                    factor_column=factor_column,
                    label_column=label_column,
                    group_keys=group_keys,
                )
            )
    return _with_derived_stats(pl.concat(monthly_frames, how="vertical")).sort(
        ["month", "factor", "label"]
    )


def build_rank_ic_sweep_summary(
    frame: pl.LazyFrame,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    min_cross_sections: Sequence[int] = DEFAULT_MIN_CROSS_SECTION_SCAN,
    filter_auction: bool = True,
) -> pl.LazyFrame:
    normalized_min_cross_sections = validate_min_cross_section_scan(min_cross_sections)
    summary_frames: list[pl.LazyFrame] = []
    for min_cross_section in normalized_min_cross_sections:
        timeseries = build_rank_ic_timeseries(
            frame,
            factor_columns=factor_columns,
            label_columns=label_columns,
            group_keys=group_keys,
            min_cross_section=min_cross_section,
            filter_auction=filter_auction,
        )
        summary_frames.append(
            build_rank_ic_summary(
                timeseries,
                factor_columns=factor_columns
                or infer_factor_columns(frame.collect_schema()),
                label_columns=label_columns
                or infer_label_columns(frame.collect_schema()),
            ).with_columns(pl.lit(min_cross_section).alias("min_cross_section"))
        )
    return pl.concat(summary_frames, how="vertical").sort(
        ["min_cross_section", "factor", "label"]
    )


def compute_rank_ic_reports(
    input_path: Path,
    output_root: Path,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    min_cross_section_scan: Sequence[int] | None = None,
    filter_auction: bool = True,
    force: bool = False,
) -> EvalArtifacts:
    timeseries_path = output_root / "rank_ic_timeseries.parquet"
    summary_path = output_root / "rank_ic_summary.parquet"
    monthly_summary_path = output_root / "rank_ic_monthly_summary.parquet"
    sweep_summary_path = output_root / "rank_ic_min_cross_section_scan.parquet"
    if (
        timeseries_path.exists()
        and summary_path.exists()
        and monthly_summary_path.exists()
        and (
            min_cross_section_scan is None
            or sweep_summary_path.exists()
        )
        and not force
    ):
        return EvalArtifacts(
            timeseries_path=timeseries_path,
            summary_path=summary_path,
            monthly_summary_path=monthly_summary_path,
            sweep_summary_path=sweep_summary_path if min_cross_section_scan else None,
        )

    frame = pl.scan_parquet(str(input_path))
    working = filter_continuous_auction(frame) if filter_auction else frame
    working_schema = working.collect_schema()
    schema = frame.collect_schema()
    selected_factor_columns, selected_label_columns = validate_panel_schema(
        working_schema,
        factor_columns=factor_columns,
        label_columns=label_columns,
        group_keys=group_keys,
    )
    timeseries = build_rank_ic_timeseries(
        working,
        factor_columns=selected_factor_columns,
        label_columns=selected_label_columns,
        group_keys=group_keys,
        min_cross_section=min_cross_section,
        filter_auction=False,
    )
    summary = build_rank_ic_summary(
        timeseries,
        factor_columns=selected_factor_columns,
        label_columns=selected_label_columns,
    )
    monthly_summary = build_monthly_rank_ic_summary(
        timeseries,
        factor_columns=selected_factor_columns,
        label_columns=selected_label_columns,
        group_keys=group_keys,
    )
    sweep_summary: pl.LazyFrame | None = None
    normalized_scan: tuple[int, ...] | None = None
    if min_cross_section_scan:
        normalized_scan = validate_min_cross_section_scan(min_cross_section_scan)
        sweep_summary = build_rank_ic_sweep_summary(
            working,
            factor_columns=selected_factor_columns,
            label_columns=selected_label_columns,
            group_keys=group_keys,
            min_cross_sections=normalized_scan,
            filter_auction=False,
        )

    _sink_parquet_compat(timeseries, timeseries_path)
    _sink_parquet_compat(summary, summary_path)
    _sink_parquet_compat(monthly_summary, monthly_summary_path)
    if sweep_summary is not None:
        _sink_parquet_compat(sweep_summary, sweep_summary_path)
    return EvalArtifacts(
        timeseries_path=timeseries_path,
        summary_path=summary_path,
        monthly_summary_path=monthly_summary_path,
        sweep_summary_path=sweep_summary_path if normalized_scan else None,
    )
