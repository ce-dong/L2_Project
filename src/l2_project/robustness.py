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
)
from .factors import SnapshotSchemaError
from .panel import FACTOR_PREFIXES, LABEL_PREFIXES

DEFAULT_SESSION_SUMMARY_FILENAME = "rank_ic_session_summary.parquet"
DEFAULT_MONTH_SESSION_SUMMARY_FILENAME = "rank_ic_month_session_summary.parquet"
DEFAULT_STABILITY_SUMMARY_FILENAME = "rank_ic_stability_summary.parquet"

MORNING_WINDOW = CONTINUOUS_AUCTION_WINDOWS[0]
AFTERNOON_WINDOW = CONTINUOUS_AUCTION_WINDOWS[1]


@dataclass(frozen=True)
class RobustnessArtifacts:
    session_summary_path: Path
    month_session_summary_path: Path
    stability_summary_path: Path


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


def infer_pair_columns_from_timeseries_schema(
    schema: Mapping[str, pl.DataType],
) -> tuple[list[str], list[str]]:
    pairs: list[tuple[str, str]] = []
    for column_name in schema:
        if not column_name.startswith("rank_ic__"):
            continue
        pair_key = column_name.removeprefix("rank_ic__")
        try:
            factor_column, label_column = pair_key.split("__", maxsplit=1)
        except ValueError as exc:
            raise SnapshotSchemaError(
                f"invalid rank IC pair column naming: {column_name}"
            ) from exc
        if not any(factor_column.startswith(prefix) for prefix in FACTOR_PREFIXES):
            continue
        if not any(label_column.startswith(prefix) for prefix in LABEL_PREFIXES):
            continue
        pairs.append((factor_column, label_column))

    if not pairs:
        raise SnapshotSchemaError("no factor-label pairs were found in timeseries")

    factor_columns = sorted({factor_column for factor_column, _ in pairs})
    label_columns = sorted({label_column for _, label_column in pairs})
    return factor_columns, label_columns


def _validate_timeseries_schema(
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
            f"missing robustness timeseries columns: {missing_text}"
        )

    missing_pair_columns: list[str] = []
    for factor_column in factor_columns:
        for label_column in label_columns:
            for column_name in (
                _nobs_column(factor_column, label_column),
                _factor_nunique_column(factor_column, label_column),
                _label_nunique_column(factor_column, label_column),
                _valid_column(factor_column, label_column),
                _reason_column(factor_column, label_column),
                _ic_column(factor_column, label_column),
            ):
                if column_name not in schema:
                    missing_pair_columns.append(column_name)

    if missing_pair_columns:
        missing_text = ", ".join(missing_pair_columns)
        raise SnapshotSchemaError(
            f"missing robustness timeseries columns: {missing_text}"
        )


def _session_expr(event_time_column: str = "event_time") -> pl.Expr:
    event_time = pl.col(event_time_column)
    return (
        pl.when((event_time >= MORNING_WINDOW[0]) & (event_time <= MORNING_WINDOW[1]))
        .then(pl.lit("am"))
        .when(
            (event_time >= AFTERNOON_WINDOW[0]) & (event_time <= AFTERNOON_WINDOW[1])
        )
        .then(pl.lit("pm"))
        .otherwise(pl.lit("off_session"))
    )


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

    return pl.concat(frames, how="vertical").with_columns(
        [
            pl.col("trading_day").cast(pl.Utf8).str.slice(0, 6).alias("month"),
            _session_expr().alias("session"),
        ]
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


def build_session_rank_ic_summary(
    timeseries: pl.LazyFrame,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    schema = timeseries.collect_schema()
    selected_factor_columns, selected_label_columns = (
        factor_columns,
        label_columns,
    )
    if selected_factor_columns is None or selected_label_columns is None:
        inferred_factor_columns, inferred_label_columns = (
            infer_pair_columns_from_timeseries_schema(schema)
        )
        selected_factor_columns = list(
            selected_factor_columns or inferred_factor_columns
        )
        selected_label_columns = list(selected_label_columns or inferred_label_columns)

    _validate_timeseries_schema(
        schema,
        factor_columns=selected_factor_columns,
        label_columns=selected_label_columns,
    )

    long_frame = _pair_long_frame(
        timeseries,
        factor_columns=selected_factor_columns,
        label_columns=selected_label_columns,
    ).filter(pl.col("session") != "off_session")

    return _with_derived_stats(
        long_frame.group_by(["session", "factor", "label"])
        .agg(
            [
                pl.len().alias("groups_total"),
                pl.col("is_valid").sum().alias("groups_valid"),
                (~pl.col("is_valid")).sum().alias("groups_invalid"),
                pl.col("rank_ic").count().alias("time_points"),
                pl.col("rank_ic").mean().alias("rank_ic_mean"),
                pl.col("rank_ic").std(ddof=1).alias("rank_ic_std"),
                pl.col("rank_ic").gt(0).mean().alias("positive_ratio"),
                pl.col("n_obs").filter(pl.col("is_valid")).mean().alias(
                    "avg_cross_section"
                ),
            ]
        )
        .with_columns(
            (pl.col("groups_valid") / pl.col("groups_total")).alias("valid_ratio")
        )
    ).sort(["factor", "label", "session"])


def build_month_session_rank_ic_summary(
    timeseries: pl.LazyFrame,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    schema = timeseries.collect_schema()
    selected_factor_columns, selected_label_columns = (
        factor_columns,
        label_columns,
    )
    if selected_factor_columns is None or selected_label_columns is None:
        inferred_factor_columns, inferred_label_columns = (
            infer_pair_columns_from_timeseries_schema(schema)
        )
        selected_factor_columns = list(
            selected_factor_columns or inferred_factor_columns
        )
        selected_label_columns = list(selected_label_columns or inferred_label_columns)

    _validate_timeseries_schema(
        schema,
        factor_columns=selected_factor_columns,
        label_columns=selected_label_columns,
    )

    long_frame = _pair_long_frame(
        timeseries,
        factor_columns=selected_factor_columns,
        label_columns=selected_label_columns,
    ).filter(pl.col("session") != "off_session")

    return _with_derived_stats(
        long_frame.group_by(["month", "session", "factor", "label"])
        .agg(
            [
                pl.len().alias("groups_total"),
                pl.col("is_valid").sum().alias("groups_valid"),
                (~pl.col("is_valid")).sum().alias("groups_invalid"),
                pl.col("rank_ic").count().alias("time_points"),
                pl.col("rank_ic").mean().alias("rank_ic_mean"),
                pl.col("rank_ic").std(ddof=1).alias("rank_ic_std"),
                pl.col("rank_ic").gt(0).mean().alias("positive_ratio"),
                pl.col("n_obs").filter(pl.col("is_valid")).mean().alias(
                    "avg_cross_section"
                ),
            ]
        )
        .with_columns(
            (pl.col("groups_valid") / pl.col("groups_total")).alias("valid_ratio")
        )
    ).sort(["month", "factor", "label", "session"])


def build_rank_ic_stability_summary(
    monthly_session_summary: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        monthly_session_summary.group_by(["factor", "label"])
        .agg(
            [
                pl.col("month").n_unique().alias("months_observed"),
                pl.col("session").n_unique().alias("sessions_observed"),
                pl.col("rank_ic_mean").mean().alias("avg_segment_rank_ic"),
                pl.col("rank_ic_mean").min().alias("min_segment_rank_ic"),
                pl.col("rank_ic_mean").max().alias("max_segment_rank_ic"),
                pl.col("rank_ic_mean").gt(0).mean().alias("positive_segment_ratio"),
                pl.col("valid_ratio").mean().alias("avg_segment_valid_ratio"),
                pl.col("valid_ratio").min().alias("min_segment_valid_ratio"),
                pl.col("time_points").min().alias("min_segment_time_points"),
            ]
        )
        .with_columns(
            (
                pl.col("max_segment_rank_ic") - pl.col("min_segment_rank_ic")
            ).alias("segment_ic_range")
        )
        .sort(["factor", "label"])
    )


def compute_robustness_reports(
    timeseries_path: Path,
    output_root: Path,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    force: bool = False,
) -> RobustnessArtifacts:
    session_summary_path = output_root / DEFAULT_SESSION_SUMMARY_FILENAME
    month_session_summary_path = output_root / DEFAULT_MONTH_SESSION_SUMMARY_FILENAME
    stability_summary_path = output_root / DEFAULT_STABILITY_SUMMARY_FILENAME

    if (
        session_summary_path.exists()
        and month_session_summary_path.exists()
        and stability_summary_path.exists()
        and not force
    ):
        return RobustnessArtifacts(
            session_summary_path=session_summary_path,
            month_session_summary_path=month_session_summary_path,
            stability_summary_path=stability_summary_path,
        )

    timeseries = pl.scan_parquet(str(timeseries_path))
    session_summary = build_session_rank_ic_summary(
        timeseries,
        factor_columns=factor_columns,
        label_columns=label_columns,
    )
    monthly_session_summary = build_month_session_rank_ic_summary(
        timeseries,
        factor_columns=factor_columns,
        label_columns=label_columns,
    )
    stability_summary = build_rank_ic_stability_summary(monthly_session_summary)

    _sink_parquet_compat(session_summary, session_summary_path)
    _sink_parquet_compat(monthly_session_summary, month_session_summary_path)
    _sink_parquet_compat(stability_summary, stability_summary_path)

    return RobustnessArtifacts(
        session_summary_path=session_summary_path,
        month_session_summary_path=month_session_summary_path,
        stability_summary_path=stability_summary_path,
    )
