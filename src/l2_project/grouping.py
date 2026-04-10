from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

from .eval import (
    DEFAULT_GROUP_KEYS,
    DEFAULT_MIN_CROSS_SECTION,
    filter_continuous_auction,
    validate_min_cross_section,
    validate_panel_schema,
)
from .panel import infer_factor_columns, infer_label_columns

DEFAULT_NUM_GROUPS = 5


@dataclass(frozen=True)
class GroupingArtifacts:
    timeseries_path: Path
    summary_path: Path
    monotonicity_summary_path: Path


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


def validate_num_groups(num_groups: int) -> int:
    normalized = int(num_groups)
    if normalized < 2:
        raise ValueError("num_groups must be at least 2")
    return normalized


def effective_grouping_cross_section(
    *,
    min_cross_section: int,
    num_groups: int,
) -> int:
    return max(
        validate_min_cross_section(min_cross_section),
        validate_num_groups(num_groups) * 2,
    )


def _group_id_expr(
    *,
    rank_column: str,
    nobs_column: str,
    num_groups: int,
) -> pl.Expr:
    return (
        (
            ((pl.col(rank_column) - 1.0) * num_groups) / pl.col(nobs_column)
        )
        .floor()
        .cast(pl.Int32)
        + 1
    ).clip(1, num_groups)


def build_group_return_timeseries(
    frame: pl.LazyFrame,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    num_groups: int = DEFAULT_NUM_GROUPS,
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
    normalized_num_groups = validate_num_groups(num_groups)
    normalized_min_cross_section = effective_grouping_cross_section(
        min_cross_section=min_cross_section,
        num_groups=normalized_num_groups,
    )

    timeseries_frames: list[pl.LazyFrame] = []
    for factor_column in selected_factor_columns:
        for label_column in selected_label_columns:
            pair_frame = (
                working.select(
                    [
                        *group_keys,
                        pl.col(factor_column)
                        .cast(pl.Float64, strict=False)
                        .alias("__factor"),
                        pl.col(label_column)
                        .cast(pl.Float64, strict=False)
                        .alias("__label"),
                    ]
                )
                .filter(pl.col("__factor").is_not_null() & pl.col("__label").is_not_null())
                .with_columns(
                    [
                        pl.len().over(list(group_keys)).alias("__cross_section_n_obs"),
                        pl.col("__factor")
                        .n_unique()
                        .over(list(group_keys))
                        .alias("__factor_n_unique"),
                        pl.col("__factor")
                        .rank(method="ordinal")
                        .over(list(group_keys))
                        .cast(pl.Float64)
                        .alias("__factor_rank"),
                    ]
                )
                .filter(
                    (pl.col("__cross_section_n_obs") >= normalized_min_cross_section)
                    & (pl.col("__factor_n_unique") > 1)
                )
                .with_columns(
                    _group_id_expr(
                        rank_column="__factor_rank",
                        nobs_column="__cross_section_n_obs",
                        num_groups=normalized_num_groups,
                    ).alias("group_id")
                )
                .group_by([*group_keys, "group_id"])
                .agg(
                    [
                        pl.col("__label").mean().alias("group_mean_ret"),
                        pl.len().alias("group_n_obs"),
                        pl.col("__cross_section_n_obs")
                        .max()
                        .alias("cross_section_n_obs"),
                        pl.col("__factor_n_unique").max().alias("factor_n_unique"),
                    ]
                )
                .with_columns(
                    [
                        pl.lit(factor_column).alias("factor"),
                        pl.lit(label_column).alias("label"),
                        pl.lit(normalized_num_groups).alias("num_groups"),
                        pl.lit(normalized_min_cross_section).alias(
                            "min_group_cross_section"
                        ),
                    ]
                )
                .select(
                    [
                        *group_keys,
                        "factor",
                        "label",
                        "group_id",
                        "group_mean_ret",
                        "group_n_obs",
                        "cross_section_n_obs",
                        "factor_n_unique",
                        "num_groups",
                        "min_group_cross_section",
                    ]
                )
            )
            timeseries_frames.append(pair_frame)

    return pl.concat(timeseries_frames, how="vertical").sort(
        [*group_keys, "factor", "label", "group_id"]
    )


def build_group_return_summary(
    timeseries: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        timeseries.group_by(["factor", "label", "group_id"])
        .agg(
            [
                pl.col("group_mean_ret").mean().alias("mean_ret"),
                pl.col("group_mean_ret").std(ddof=1).alias("std_ret"),
                pl.col("group_mean_ret").count().alias("time_points"),
                pl.col("group_mean_ret").gt(0).mean().alias("positive_ratio"),
                pl.col("group_n_obs").mean().alias("avg_group_size"),
                pl.col("cross_section_n_obs").mean().alias("avg_cross_section"),
                pl.col("num_groups").max().alias("num_groups"),
                pl.col("min_group_cross_section")
                .max()
                .alias("min_group_cross_section"),
            ]
        )
        .with_columns(
            pl.when((pl.col("std_ret") > 0) & (pl.col("time_points") > 1))
            .then(
                pl.col("mean_ret")
                / (pl.col("std_ret") / pl.col("time_points").sqrt())
            )
            .otherwise(None)
            .alias("t_stat")
        )
        .sort(["factor", "label", "group_id"])
    )


def build_group_monotonicity_summary(
    timeseries: pl.LazyFrame,
    summary: pl.LazyFrame | None = None,
    *,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> pl.LazyFrame:
    group_summary = summary if summary is not None else build_group_return_summary(timeseries)
    join_keys = [*group_keys, "factor", "label"]

    spread_timeseries = (
        timeseries.filter(pl.col("group_id") == 1)
        .select(
            [
                *join_keys,
                pl.col("group_mean_ret").alias("__bottom_ret"),
            ]
        )
        .join(
            timeseries.filter(pl.col("group_id") == pl.col("num_groups"))
            .select(
                [
                    *join_keys,
                    pl.col("group_mean_ret").alias("__top_ret"),
                ]
            ),
            on=join_keys,
            how="inner",
        )
        .with_columns((pl.col("__top_ret") - pl.col("__bottom_ret")).alias("__spread"))
    )

    monotonic_from_groups = (
        group_summary.group_by(["factor", "label"])
        .agg(
            [
                pl.len().alias("groups_observed"),
                pl.col("num_groups").max().alias("num_groups"),
                pl.col("min_group_cross_section")
                .max()
                .alias("min_group_cross_section"),
                pl.col("mean_ret")
                .sort_by("group_id")
                .diff()
                .drop_nulls()
                .ge(0)
                .all()
                .alias("is_monotonic_non_decreasing"),
                pl.corr(
                    pl.col("group_id").rank(method="average"),
                    pl.col("mean_ret").rank(method="average"),
                    method="pearson",
                ).alias("spearman_group_ret_corr"),
            ]
        )
    )

    spread_summary = (
        spread_timeseries.group_by(["factor", "label"])
        .agg(
            [
                pl.col("__spread").mean().alias("top_minus_bottom_mean"),
                pl.col("__spread").std(ddof=1).alias("top_minus_bottom_std"),
                pl.col("__spread").count().alias("spread_time_points"),
                pl.col("__spread").gt(0).mean().alias("positive_spread_ratio"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("top_minus_bottom_mean") > 0)
                .then(pl.lit("positive"))
                .when(pl.col("top_minus_bottom_mean") < 0)
                .then(pl.lit("negative"))
                .otherwise(pl.lit("flat"))
                .alias("top_bottom_spread_sign"),
                pl.when(
                    (pl.col("top_minus_bottom_std") > 0)
                    & (pl.col("spread_time_points") > 1)
                )
                .then(
                    pl.col("top_minus_bottom_mean")
                    / (
                        pl.col("top_minus_bottom_std")
                        / pl.col("spread_time_points").sqrt()
                    )
                )
                .otherwise(None)
                .alias("top_minus_bottom_t_stat"),
            ]
        )
    )

    return (
        monotonic_from_groups.join(
            spread_summary,
            on=["factor", "label"],
            how="left",
        )
        .with_columns(
            (pl.col("groups_observed") == pl.col("num_groups")).alias(
                "has_complete_group_grid"
            )
        )
        .sort(["factor", "label"])
    )


def compute_grouping_reports(
    input_path: Path,
    output_root: Path,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    num_groups: int = DEFAULT_NUM_GROUPS,
    filter_auction: bool = True,
    force: bool = False,
) -> GroupingArtifacts:
    timeseries_path = output_root / "group_return_timeseries.parquet"
    summary_path = output_root / "group_return_summary.parquet"
    monotonicity_summary_path = output_root / "group_monotonicity_summary.parquet"
    if (
        timeseries_path.exists()
        and summary_path.exists()
        and monotonicity_summary_path.exists()
        and not force
    ):
        return GroupingArtifacts(
            timeseries_path=timeseries_path,
            summary_path=summary_path,
            monotonicity_summary_path=monotonicity_summary_path,
        )

    panel_frame = pl.scan_parquet(str(input_path))
    panel_schema = panel_frame.collect_schema()
    selected_factor_columns = list(factor_columns or infer_factor_columns(panel_schema))
    selected_label_columns = list(label_columns or infer_label_columns(panel_schema))

    timeseries = build_group_return_timeseries(
        panel_frame,
        factor_columns=selected_factor_columns,
        label_columns=selected_label_columns,
        group_keys=group_keys,
        min_cross_section=min_cross_section,
        num_groups=num_groups,
        filter_auction=filter_auction,
    )
    summary = build_group_return_summary(timeseries)
    monotonicity_summary = build_group_monotonicity_summary(
        timeseries,
        summary=summary,
        group_keys=group_keys,
    )

    _sink_parquet_compat(timeseries, timeseries_path)
    _sink_parquet_compat(summary, summary_path)
    _sink_parquet_compat(monotonicity_summary, monotonicity_summary_path)

    return GroupingArtifacts(
        timeseries_path=timeseries_path,
        summary_path=summary_path,
        monotonicity_summary_path=monotonicity_summary_path,
    )
