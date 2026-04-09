from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from .eval import CONTINUOUS_AUCTION_WINDOWS
from .factors import (
    BOOK_SORT,
    SnapshotSchemaError,
    _is_numeric_dtype,
    _safe_divide,
    validate_snapshot_parquet,
    with_snapshot_factors,
)

TRADE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "trading_day",
    "event_time",
    "price",
    "volume",
    "side",
)
VALID_TRADE_SIDES: tuple[int, int] = (0, 1)
TRADE_IMBALANCE_WINDOW_MS = 3_000


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


def _event_time_to_ms_expr(column: str = "event_time") -> pl.Expr:
    event_time = pl.col(column).cast(pl.Int64, strict=False)
    hour = event_time // 10_000_000
    minute = (event_time // 100_000) % 100
    second = (event_time // 1_000) % 100
    millisecond = event_time % 1_000
    return (
        ((hour * 3_600 + minute * 60 + second) * 1_000) + millisecond
    ).alias("__event_time_ms")


def _continuous_auction_mask(column: str = "event_time") -> pl.Expr:
    event_time = pl.col(column)
    mask: pl.Expr | None = None
    for start_time, end_time in CONTINUOUS_AUCTION_WINDOWS:
        window_mask = (event_time >= start_time) & (event_time <= end_time)
        mask = window_mask if mask is None else (mask | window_mask)
    if mask is None:
        raise ValueError("continuous auction windows cannot be empty")
    return mask


def _bucket_end_ms_expr(
    column: str = "__event_time_ms",
    *,
    window_ms: int = TRADE_IMBALANCE_WINDOW_MS,
) -> pl.Expr:
    event_time_ms = pl.col(column)
    return (
        pl.when(event_time_ms % window_ms == 0)
        .then(event_time_ms)
        .otherwise(((event_time_ms // window_ms) + 1) * window_ms)
        .alias("__bucket_end_ms")
    )


def validate_trade_schema(schema: Mapping[str, pl.DataType]) -> None:
    missing = [column for column in TRADE_REQUIRED_COLUMNS if column not in schema]
    if missing:
        missing_text = ", ".join(missing)
        raise SnapshotSchemaError(f"missing required trade columns: {missing_text}")

    invalid_numeric = [
        f"{column}<{schema[column]}>"
        for column in ("trading_day", "event_time", "price", "volume", "side")
        if not _is_numeric_dtype(schema[column])
    ]
    if invalid_numeric:
        invalid_text = ", ".join(invalid_numeric)
        raise SnapshotSchemaError(f"invalid trade column types: {invalid_text}")


def build_trade_imbalance_frame(
    trade_frame: pl.LazyFrame,
    *,
    window_ms: int = TRADE_IMBALANCE_WINDOW_MS,
) -> pl.LazyFrame:
    validate_trade_schema(trade_frame.collect_schema())

    working = (
        trade_frame.filter(_continuous_auction_mask())
        .with_columns(_event_time_to_ms_expr())
        .filter(
            pl.col("price").cast(pl.Float64, strict=False) > 0
            & pl.col("volume").cast(pl.Float64, strict=False).is_not_null()
            & (pl.col("volume").cast(pl.Float64, strict=False) > 0)
            & pl.col("side").cast(pl.Int64, strict=False).is_in(VALID_TRADE_SIDES)
        )
        .with_columns(_bucket_end_ms_expr(window_ms=window_ms))
    )

    side_one_volume = (
        pl.when(pl.col("side").cast(pl.Int64, strict=False) == 1)
        .then(pl.col("volume").cast(pl.Float64, strict=False))
        .otherwise(0.0)
    )
    side_zero_volume = (
        pl.when(pl.col("side").cast(pl.Int64, strict=False) == 0)
        .then(pl.col("volume").cast(pl.Float64, strict=False))
        .otherwise(0.0)
    )

    return (
        working.group_by(["symbol", "trading_day", "__bucket_end_ms"])
        .agg(
            [
                side_one_volume.sum().alias("__side_1_volume_3s"),
                side_zero_volume.sum().alias("__side_0_volume_3s"),
            ]
        )
        .with_columns(
            _safe_divide(
                pl.col("__side_0_volume_3s") - pl.col("__side_1_volume_3s"),
                pl.col("__side_1_volume_3s") + pl.col("__side_0_volume_3s"),
            ).alias("trade_imbalance_3s")
        )
        .select(["symbol", "trading_day", "__bucket_end_ms", "trade_imbalance_3s"])
    )


def with_trade_imbalance_factor(
    snapshot_factor_frame: pl.LazyFrame,
    trade_frame: pl.LazyFrame,
    *,
    window_ms: int = TRADE_IMBALANCE_WINDOW_MS,
) -> pl.LazyFrame:
    trade_imbalance_frame = build_trade_imbalance_frame(
        trade_frame,
        window_ms=window_ms,
    )
    return (
        snapshot_factor_frame.with_columns(_event_time_to_ms_expr())
        .join(
            trade_imbalance_frame,
            left_on=["symbol", "trading_day", "__event_time_ms"],
            right_on=["symbol", "trading_day", "__bucket_end_ms"],
            how="left",
        )
        .drop("__event_time_ms")
    )


def compute_snapshot_trade_factors_to_parquet(
    snapshot_source_path: Path,
    trade_source_path: Path,
    output_path: Path,
    *,
    micro_price_levels: Sequence[int] = (1,),
    relative_spread_levels: Sequence[int] = (1,),
    oir_depths: Sequence[int] = (1, 5),
    book_slope_depths: Sequence[int] = (5,),
    flow_levels: Sequence[int] = (1,),
    window_ms: int = TRADE_IMBALANCE_WINDOW_MS,
    force: bool = False,
) -> Path:
    if output_path.exists() and not force:
        return output_path

    validate_snapshot_parquet(
        snapshot_source_path,
        micro_price_levels=micro_price_levels,
        relative_spread_levels=relative_spread_levels,
        oir_depths=oir_depths,
        book_slope_depths=book_slope_depths,
        flow_levels=flow_levels,
    )
    trade_frame = pl.scan_parquet(str(trade_source_path))
    validate_trade_schema(trade_frame.collect_schema())

    snapshot_frame = pl.scan_parquet(str(snapshot_source_path))
    snapshot_factor_frame = with_snapshot_factors(
        snapshot_frame,
        micro_price_levels=micro_price_levels,
        relative_spread_levels=relative_spread_levels,
        oir_depths=oir_depths,
        book_slope_depths=book_slope_depths,
        flow_levels=flow_levels,
    )
    combined = with_trade_imbalance_factor(
        snapshot_factor_frame,
        trade_frame,
        window_ms=window_ms,
    ).sort(BOOK_SORT)
    _sink_parquet_compat(combined, output_path)
    return output_path
