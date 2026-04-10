from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from .eval import (
    DEFAULT_GROUP_KEYS,
    DEFAULT_MIN_CROSS_SECTION,
    filter_continuous_auction,
    validate_min_cross_section,
)
from .factors import SnapshotSchemaError
from .panel import infer_factor_columns, infer_label_columns

STRONG_FACTOR_CORR_THRESHOLD = 0.80
STRONG_IC_CORR_THRESHOLD = 0.80
SAME_FAMILY_FACTOR_CORR_THRESHOLD = 0.65
SAME_FAMILY_IC_CORR_THRESHOLD = 0.60


@dataclass(frozen=True)
class RedundancyArtifacts:
    factor_corr_timeseries_path: Path
    factor_corr_summary_path: Path
    ic_corr_summary_path: Path
    recommendation_path: Path


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


def _missing_pair_reason() -> str:
    return "missing_pair_values"


def _factor_corr_column(factor_left: str, factor_right: str) -> str:
    return f"factor_corr__{factor_left}__{factor_right}"


def _rank_ic_column(factor_column: str, label_column: str) -> str:
    return f"rank_ic__{factor_column}__{label_column}"


def _validate_redundancy_panel_schema(
    schema: Mapping[str, pl.DataType],
    *,
    factor_columns: Sequence[str] | None = None,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> list[str]:
    missing_group_keys = [column for column in group_keys if column not in schema]
    if missing_group_keys:
        missing_text = ", ".join(missing_group_keys)
        raise SnapshotSchemaError(f"missing redundancy group keys: {missing_text}")

    selected_factor_columns = list(factor_columns or infer_factor_columns(schema))
    if len(selected_factor_columns) < 2:
        raise SnapshotSchemaError(
            "at least two factor columns are required for redundancy analysis"
        )

    missing_factor_columns = [
        column for column in selected_factor_columns if column not in schema
    ]
    if missing_factor_columns:
        missing_text = ", ".join(missing_factor_columns)
        raise SnapshotSchemaError(
            f"missing redundancy factor columns: {missing_text}"
        )

    return selected_factor_columns


def _factor_corr_frame_for_pair(
    frame: pl.LazyFrame,
    *,
    factor_left: str,
    factor_right: str,
    group_keys: Sequence[str],
    min_cross_section: int,
) -> pl.LazyFrame:
    factor_corr_column = _factor_corr_column(factor_left, factor_right)

    return (
        frame.select(
            [
                *group_keys,
                pl.col(factor_left).cast(pl.Float64, strict=False).alias("__left"),
                pl.col(factor_right).cast(pl.Float64, strict=False).alias("__right"),
            ]
        )
        .filter(pl.col("__left").is_not_null() & pl.col("__right").is_not_null())
        .group_by(list(group_keys))
        .agg(
            [
                pl.len().alias("n_obs"),
                pl.col("__left").n_unique().alias("left_factor_n_unique"),
                pl.col("__right").n_unique().alias("right_factor_n_unique"),
                pl.corr(
                    pl.col("__left").rank(method="average"),
                    pl.col("__right").rank(method="average"),
                    method="pearson",
                ).alias("__raw_factor_corr"),
            ]
        )
        .with_columns(
            [
                pl.lit(factor_left).alias("factor_left"),
                pl.lit(factor_right).alias("factor_right"),
                pl.when(pl.col("n_obs") < min_cross_section)
                .then(pl.lit("below_min_cross_section"))
                .when(
                    (pl.col("left_factor_n_unique") <= 1)
                    & (pl.col("right_factor_n_unique") <= 1)
                )
                .then(pl.lit("both_constant"))
                .when(pl.col("left_factor_n_unique") <= 1)
                .then(pl.lit("left_factor_constant"))
                .when(pl.col("right_factor_n_unique") <= 1)
                .then(pl.lit("right_factor_constant"))
                .when(pl.col("__raw_factor_corr").is_nan())
                .then(pl.lit("nan_factor_corr"))
                .otherwise(None)
                .alias("invalid_reason"),
            ]
        )
        .with_columns(
            [
                pl.col("invalid_reason").is_null().alias("is_valid"),
                pl.when(
                    pl.col("invalid_reason").is_null()
                    & pl.col("__raw_factor_corr").is_finite()
                )
                .then(pl.col("__raw_factor_corr"))
                .otherwise(None)
                .alias("factor_corr"),
            ]
        )
        .drop("__raw_factor_corr")
        .select(
            [
                *group_keys,
                "factor_left",
                "factor_right",
                "n_obs",
                "left_factor_n_unique",
                "right_factor_n_unique",
                "is_valid",
                "invalid_reason",
                "factor_corr",
            ]
        )
    )


def build_cross_section_factor_corr_timeseries(
    frame: pl.LazyFrame,
    *,
    factor_columns: Sequence[str] | None = None,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    filter_auction: bool = True,
) -> pl.LazyFrame:
    working = filter_continuous_auction(frame) if filter_auction else frame
    schema = working.collect_schema()
    selected_factor_columns = _validate_redundancy_panel_schema(
        schema,
        factor_columns=factor_columns,
        group_keys=group_keys,
    )
    normalized_min_cross_section = validate_min_cross_section(min_cross_section)
    base = working.select(list(group_keys)).unique().sort(list(group_keys))

    pair_frames: list[pl.LazyFrame] = []
    for factor_left, factor_right in combinations(selected_factor_columns, 2):
        pair_frame = _factor_corr_frame_for_pair(
            working,
            factor_left=factor_left,
            factor_right=factor_right,
            group_keys=group_keys,
            min_cross_section=normalized_min_cross_section,
        )
        pair_base = (
            base.with_columns(
                [
                    pl.lit(factor_left).alias("factor_left"),
                    pl.lit(factor_right).alias("factor_right"),
                ]
            )
            .join(
                pair_frame,
                on=[*group_keys, "factor_left", "factor_right"],
                how="left",
            )
            .with_columns(
                [
                    pl.when(pl.col("n_obs").is_null())
                    .then(pl.lit(_missing_pair_reason()))
                    .otherwise(pl.col("invalid_reason"))
                    .alias("invalid_reason"),
                    pl.col("is_valid").fill_null(False).alias("is_valid"),
                ]
            )
            .select(
                [
                    *group_keys,
                    "factor_left",
                    "factor_right",
                    "n_obs",
                    "left_factor_n_unique",
                    "right_factor_n_unique",
                    "is_valid",
                    "invalid_reason",
                    "factor_corr",
                ]
            )
        )
        pair_frames.append(pair_base)

    return pl.concat(pair_frames, how="vertical").sort(
        [*group_keys, "factor_left", "factor_right"]
    )


def build_cross_section_factor_corr_summary(
    timeseries: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        timeseries.group_by(["factor_left", "factor_right"])
        .agg(
            [
                pl.len().alias("groups_total"),
                pl.col("is_valid").fill_null(False).sum().alias("groups_valid"),
                (pl.col("invalid_reason") == "below_min_cross_section")
                .sum()
                .alias("groups_below_min_cross_section"),
                (pl.col("invalid_reason") == "left_factor_constant")
                .sum()
                .alias("groups_left_factor_constant"),
                (pl.col("invalid_reason") == "right_factor_constant")
                .sum()
                .alias("groups_right_factor_constant"),
                (pl.col("invalid_reason") == "both_constant")
                .sum()
                .alias("groups_both_constant"),
                (pl.col("invalid_reason") == _missing_pair_reason())
                .sum()
                .alias("groups_missing_pair_values"),
                (pl.col("invalid_reason") == "nan_factor_corr")
                .sum()
                .alias("groups_nan_factor_corr"),
                pl.col("factor_corr").mean().alias("factor_corr_mean"),
                pl.col("factor_corr").std(ddof=1).alias("factor_corr_std"),
                pl.col("factor_corr").count().alias("time_points"),
                pl.col("factor_corr").abs().mean().alias("factor_corr_abs_mean"),
                pl.col("n_obs").filter(pl.col("is_valid")).mean().alias("avg_cross_section"),
            ]
        )
        .with_columns(
            [
                (pl.col("groups_valid") / pl.col("groups_total")).alias("valid_ratio"),
                (pl.col("groups_total") - pl.col("groups_valid")).alias("groups_invalid"),
            ]
        )
        .sort(["factor_left", "factor_right"])
    )


def build_ic_corr_summary(
    rank_ic_timeseries: pl.LazyFrame,
    *,
    factor_columns: Sequence[str],
    label_columns: Sequence[str],
) -> pl.LazyFrame:
    schema = rank_ic_timeseries.collect_schema()
    pair_frames: list[pl.LazyFrame] = []
    for factor_left, factor_right in combinations(factor_columns, 2):
        for label_column in label_columns:
            left_ic_column = _rank_ic_column(factor_left, label_column)
            right_ic_column = _rank_ic_column(factor_right, label_column)
            missing_columns = [
                column
                for column in (left_ic_column, right_ic_column)
                if column not in schema
            ]
            if missing_columns:
                missing_text = ", ".join(missing_columns)
                raise SnapshotSchemaError(
                    f"missing rank IC columns for redundancy analysis: {missing_text}"
                )

            pair_summary = (
                rank_ic_timeseries.select(
                    [
                        pl.col(left_ic_column).cast(pl.Float64, strict=False).alias("__left_ic"),
                        pl.col(right_ic_column).cast(pl.Float64, strict=False).alias("__right_ic"),
                    ]
                )
                .filter(pl.col("__left_ic").is_not_null() & pl.col("__right_ic").is_not_null())
                .select(
                    [
                        pl.lit(factor_left).alias("factor_left"),
                        pl.lit(factor_right).alias("factor_right"),
                        pl.lit(label_column).alias("label"),
                        pl.len().alias("overlap_time_points"),
                        pl.corr(
                            pl.col("__left_ic"),
                            pl.col("__right_ic"),
                            method="pearson",
                        ).alias("__raw_ic_corr"),
                    ]
                )
                .with_columns(
                    [
                        pl.when(
                            (pl.col("overlap_time_points") > 1)
                            & pl.col("__raw_ic_corr").is_finite()
                        )
                        .then(pl.col("__raw_ic_corr"))
                        .otherwise(None)
                        .alias("ic_corr")
                    ]
                )
                .with_columns(pl.col("ic_corr").abs().alias("ic_corr_abs"))
                .drop("__raw_ic_corr")
            )
            pair_frames.append(pair_summary)

    return pl.concat(pair_frames, how="vertical").sort(
        ["factor_left", "factor_right", "label"]
    )


def build_redundancy_recommendation(
    factor_corr_summary: pl.LazyFrame,
    ic_corr_summary: pl.LazyFrame,
    rank_ic_summary: pl.LazyFrame,
) -> pl.LazyFrame:
    factor_strength = (
        rank_ic_summary.group_by("factor")
        .agg(
            [
                pl.col("rank_ic_mean").abs().mean().alias("avg_abs_rank_ic_mean"),
                pl.col("rank_ic_mean").abs().max().alias("max_abs_rank_ic_mean"),
                pl.col("valid_ratio").mean().alias("avg_valid_ratio"),
            ]
        )
        .sort("factor")
    )

    ic_pair_summary = (
        ic_corr_summary.group_by(["factor_left", "factor_right"])
        .agg(
            [
                pl.col("ic_corr").mean().alias("ic_corr_mean"),
                pl.col("ic_corr_abs").mean().alias("ic_corr_abs_mean"),
                pl.col("ic_corr_abs").max().alias("ic_corr_abs_max"),
                pl.col("overlap_time_points").min().alias("min_overlap_time_points"),
                pl.col("overlap_time_points").max().alias("max_overlap_time_points"),
            ]
        )
    )

    left_strength = factor_strength.rename(
        {
            "factor": "factor_left",
            "avg_abs_rank_ic_mean": "left_avg_abs_rank_ic_mean",
            "max_abs_rank_ic_mean": "left_max_abs_rank_ic_mean",
            "avg_valid_ratio": "left_avg_valid_ratio",
        }
    )
    right_strength = factor_strength.rename(
        {
            "factor": "factor_right",
            "avg_abs_rank_ic_mean": "right_avg_abs_rank_ic_mean",
            "max_abs_rank_ic_mean": "right_max_abs_rank_ic_mean",
            "avg_valid_ratio": "right_avg_valid_ratio",
        }
    )

    recommendation = (
        factor_corr_summary.join(
            ic_pair_summary,
            on=["factor_left", "factor_right"],
            how="left",
        )
        .join(left_strength, on="factor_left", how="left")
        .join(right_strength, on="factor_right", how="left")
        .with_columns(
            [
                pl.col("factor_left")
                .str.replace(r"_[0-9].*$", "")
                .alias("factor_family_left"),
                pl.col("factor_right")
                .str.replace(r"_[0-9].*$", "")
                .alias("factor_family_right"),
            ]
        )
        .with_columns(
            [
                (pl.col("factor_family_left") == pl.col("factor_family_right")).alias(
                    "same_family"
                ),
                (pl.col("factor_corr_abs_mean") >= STRONG_FACTOR_CORR_THRESHOLD).alias(
                    "high_factor_corr"
                ),
                (pl.col("ic_corr_abs_mean") >= STRONG_IC_CORR_THRESHOLD).alias(
                    "high_ic_corr"
                ),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("high_factor_corr") & pl.col("high_ic_corr"))
                    | (
                        pl.col("same_family")
                        & (pl.col("factor_corr_abs_mean") >= SAME_FAMILY_FACTOR_CORR_THRESHOLD)
                        & (pl.col("ic_corr_abs_mean") >= SAME_FAMILY_IC_CORR_THRESHOLD)
                    )
                ).alias("likely_redundant")
            ]
        )
        .with_columns(
            [
                pl.when(~pl.col("likely_redundant"))
                .then(pl.lit("keep_both"))
                .when(
                    pl.col("left_avg_abs_rank_ic_mean")
                    > pl.col("right_avg_abs_rank_ic_mean")
                )
                .then(pl.col("factor_left"))
                .when(
                    pl.col("left_avg_abs_rank_ic_mean")
                    < pl.col("right_avg_abs_rank_ic_mean")
                )
                .then(pl.col("factor_right"))
                .when(
                    pl.col("left_max_abs_rank_ic_mean")
                    > pl.col("right_max_abs_rank_ic_mean")
                )
                .then(pl.col("factor_left"))
                .when(
                    pl.col("left_max_abs_rank_ic_mean")
                    < pl.col("right_max_abs_rank_ic_mean")
                )
                .then(pl.col("factor_right"))
                .when(pl.col("left_avg_valid_ratio") >= pl.col("right_avg_valid_ratio"))
                .then(pl.col("factor_left"))
                .otherwise(pl.col("factor_right"))
                .alias("recommended_keep"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("recommended_keep") == "keep_both")
                .then(None)
                .when(pl.col("recommended_keep") == pl.col("factor_left"))
                .then(pl.col("factor_right"))
                .otherwise(pl.col("factor_left"))
                .alias("recommended_drop"),
                pl.when(~pl.col("likely_redundant"))
                .then(
                    pl.when(pl.col("same_family"))
                    .then(pl.lit("same_family_but_distinct_in_current_sample"))
                    .otherwise(pl.lit("retain_both"))
                )
                .when(pl.col("high_factor_corr") & pl.col("high_ic_corr"))
                .then(pl.lit("high_factor_and_ic_correlation"))
                .otherwise(pl.lit("same_family_with_material_correlation"))
                .alias("reason"),
            ]
        )
        .sort(["factor_left", "factor_right"])
    )
    return recommendation


def compute_redundancy_reports(
    panel_path: Path,
    rank_ic_timeseries_path: Path,
    rank_ic_summary_path: Path,
    output_root: Path,
    *,
    factor_columns: Sequence[str] | None = None,
    label_columns: Sequence[str] | None = None,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    filter_auction: bool = True,
    force: bool = False,
) -> RedundancyArtifacts:
    factor_corr_timeseries_path = output_root / "factor_corr_timeseries.parquet"
    factor_corr_summary_path = output_root / "factor_corr_summary.parquet"
    ic_corr_summary_path = output_root / "ic_corr_summary.parquet"
    recommendation_path = output_root / "factor_redundancy_recommendation.parquet"

    if (
        factor_corr_timeseries_path.exists()
        and factor_corr_summary_path.exists()
        and ic_corr_summary_path.exists()
        and recommendation_path.exists()
        and not force
    ):
        return RedundancyArtifacts(
            factor_corr_timeseries_path=factor_corr_timeseries_path,
            factor_corr_summary_path=factor_corr_summary_path,
            ic_corr_summary_path=ic_corr_summary_path,
            recommendation_path=recommendation_path,
        )

    panel_frame = pl.scan_parquet(str(panel_path))
    panel_schema = panel_frame.collect_schema()
    selected_factor_columns = _validate_redundancy_panel_schema(
        panel_schema,
        factor_columns=factor_columns,
        group_keys=group_keys,
    )
    selected_label_columns = list(label_columns or infer_label_columns(panel_schema))
    if not selected_label_columns:
        raise SnapshotSchemaError("no label columns were found for redundancy analysis")

    factor_corr_timeseries = build_cross_section_factor_corr_timeseries(
        panel_frame,
        factor_columns=selected_factor_columns,
        group_keys=group_keys,
        min_cross_section=min_cross_section,
        filter_auction=filter_auction,
    )
    factor_corr_summary = build_cross_section_factor_corr_summary(
        factor_corr_timeseries
    )
    rank_ic_timeseries = pl.scan_parquet(str(rank_ic_timeseries_path))
    ic_corr_summary = build_ic_corr_summary(
        rank_ic_timeseries,
        factor_columns=selected_factor_columns,
        label_columns=selected_label_columns,
    )
    recommendation = build_redundancy_recommendation(
        factor_corr_summary,
        ic_corr_summary,
        pl.scan_parquet(str(rank_ic_summary_path)),
    )

    _sink_parquet_compat(factor_corr_timeseries, factor_corr_timeseries_path)
    _sink_parquet_compat(factor_corr_summary, factor_corr_summary_path)
    _sink_parquet_compat(ic_corr_summary, ic_corr_summary_path)
    _sink_parquet_compat(recommendation, recommendation_path)

    return RedundancyArtifacts(
        factor_corr_timeseries_path=factor_corr_timeseries_path,
        factor_corr_summary_path=factor_corr_summary_path,
        ic_corr_summary_path=ic_corr_summary_path,
        recommendation_path=recommendation_path,
    )
