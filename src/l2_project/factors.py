from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import polars as pl

BOOK_PARTITION = ["symbol", "trading_day"]
BOOK_SORT = ["symbol", "trading_day", "event_time"]
BOOK_NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


class SnapshotSchemaError(ValueError):
    """Raised when the snapshot parquet schema cannot support factor calculation."""


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype in BOOK_NUMERIC_DTYPES


def required_snapshot_columns(
    *,
    micro_price_levels: Sequence[int] = (1,),
    oir_depths: Sequence[int] = (1, 5),
    flow_levels: Sequence[int] = (1,),
) -> list[str]:
    required = set(BOOK_SORT)
    max_depth = max([1, *micro_price_levels, *oir_depths, *flow_levels])
    for level in range(1, max_depth + 1):
        required.update(
            {
                f"bid_price_{level}",
                f"ask_price_{level}",
                f"bid_volume_{level}",
                f"ask_volume_{level}",
            }
        )
    return sorted(required)


def validate_snapshot_schema(
    schema: Mapping[str, pl.DataType],
    *,
    micro_price_levels: Sequence[int] = (1,),
    oir_depths: Sequence[int] = (1, 5),
    flow_levels: Sequence[int] = (1,),
) -> None:
    required_columns = required_snapshot_columns(
        micro_price_levels=micro_price_levels,
        oir_depths=oir_depths,
        flow_levels=flow_levels,
    )
    missing = [column for column in required_columns if column not in schema]
    if missing:
        missing_text = ", ".join(missing)
        raise SnapshotSchemaError(f"missing required snapshot columns: {missing_text}")

    numeric_columns = [column for column in required_columns if column not in BOOK_SORT]
    invalid_numeric = [
        f"{column}<{schema[column]}>"
        for column in numeric_columns
        if not _is_numeric_dtype(schema[column])
    ]
    if invalid_numeric:
        invalid_text = ", ".join(invalid_numeric)
        raise SnapshotSchemaError(f"non-numeric snapshot columns: {invalid_text}")


def validate_snapshot_parquet(
    source_path: Path,
    *,
    micro_price_levels: Sequence[int] = (1,),
    oir_depths: Sequence[int] = (1, 5),
    flow_levels: Sequence[int] = (1,),
) -> None:
    frame = pl.scan_parquet(str(source_path))
    validate_snapshot_schema(
        frame.collect_schema(),
        micro_price_levels=micro_price_levels,
        oir_depths=oir_depths,
        flow_levels=flow_levels,
    )


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


def _book_volume_sum(side: str, depth: int) -> pl.Expr:
    volumes = [
        pl.col(f"{side}_volume_{level}").fill_null(0.0) for level in range(1, depth + 1)
    ]
    return pl.sum_horizontal(*volumes)


def _prev_col(name: str) -> str:
    return f"__prev_{name}"


def _best_quote_is_valid(level: int, *, prefix: str = "") -> pl.Expr:
    bid_price = pl.col(f"{prefix}bid_price_{level}")
    ask_price = pl.col(f"{prefix}ask_price_{level}")
    return (
        bid_price.is_not_null()
        & ask_price.is_not_null()
        & (bid_price > 0)
        & (ask_price > 0)
        & (ask_price >= bid_price)
    )


def micro_price_expr(level: int = 1) -> pl.Expr:
    bid_price = pl.col(f"bid_price_{level}")
    ask_price = pl.col(f"ask_price_{level}")
    bid_volume = pl.col(f"bid_volume_{level}")
    ask_volume = pl.col(f"ask_volume_{level}")
    return (
        pl.when(_best_quote_is_valid(level))
        .then(
            _safe_divide(
                (bid_price * ask_volume) + (ask_price * bid_volume),
                bid_volume + ask_volume,
            )
        )
        .otherwise(None)
        .alias(f"micro_price_{level}")
    )


def oir_expr(depth: int = 1) -> pl.Expr:
    bid_volume = _book_volume_sum("bid", depth)
    ask_volume = _book_volume_sum("ask", depth)
    return _safe_divide(
        bid_volume - ask_volume,
        bid_volume + ask_volume,
    ).alias(f"oir_{depth}")


def voi_expr(level: int = 1) -> pl.Expr:
    bid_price = pl.col(f"bid_price_{level}")
    ask_price = pl.col(f"ask_price_{level}")
    bid_volume = pl.col(f"bid_volume_{level}")
    ask_volume = pl.col(f"ask_volume_{level}")
    prev_bid_price = pl.col(_prev_col(f"bid_price_{level}"))
    prev_ask_price = pl.col(_prev_col(f"ask_price_{level}"))
    prev_bid_volume = pl.col(_prev_col(f"bid_volume_{level}"))
    prev_ask_volume = pl.col(_prev_col(f"ask_volume_{level}"))

    bid_component = (
        pl.when(bid_price > prev_bid_price)
        .then(bid_volume)
        .when(bid_price == prev_bid_price)
        .then(bid_volume - prev_bid_volume)
        .otherwise(-prev_bid_volume)
    )
    ask_component = (
        pl.when(ask_price < prev_ask_price)
        .then(ask_volume)
        .when(ask_price == prev_ask_price)
        .then(ask_volume - prev_ask_volume)
        .otherwise(-prev_ask_volume)
    )
    return (
        pl.when(
            prev_bid_price.is_not_null()
            & prev_ask_price.is_not_null()
            & _best_quote_is_valid(level)
            & _best_quote_is_valid(level, prefix="__prev_")
        )
        .then(bid_component - ask_component)
        .otherwise(None)
        .alias(f"voi_{level}")
    )


def ofi_expr(level: int = 1) -> pl.Expr:
    bid_price = pl.col(f"bid_price_{level}")
    ask_price = pl.col(f"ask_price_{level}")
    bid_volume = pl.col(f"bid_volume_{level}")
    ask_volume = pl.col(f"ask_volume_{level}")
    prev_bid_price = pl.col(_prev_col(f"bid_price_{level}"))
    prev_ask_price = pl.col(_prev_col(f"ask_price_{level}"))
    prev_bid_volume = pl.col(_prev_col(f"bid_volume_{level}"))
    prev_ask_volume = pl.col(_prev_col(f"ask_volume_{level}"))

    bid_flow = (
        pl.when(bid_price >= prev_bid_price).then(bid_volume).otherwise(0.0)
        - pl.when(bid_price <= prev_bid_price).then(prev_bid_volume).otherwise(0.0)
    )
    ask_flow = (
        pl.when(ask_price <= prev_ask_price).then(ask_volume).otherwise(0.0)
        - pl.when(ask_price >= prev_ask_price).then(prev_ask_volume).otherwise(0.0)
    )
    return (
        pl.when(
            prev_bid_price.is_not_null()
            & prev_ask_price.is_not_null()
            & _best_quote_is_valid(level)
            & _best_quote_is_valid(level, prefix="__prev_")
        )
        .then(bid_flow - ask_flow)
        .otherwise(None)
        .alias(f"ofi_{level}")
    )


def _lag_expressions(levels: Iterable[int]) -> list[pl.Expr]:
    lag_exprs: list[pl.Expr] = []
    for level in levels:
        for field in ("bid_price", "ask_price", "bid_volume", "ask_volume"):
            column = f"{field}_{level}"
            lag_exprs.append(
                pl.col(column).shift(1).over(BOOK_PARTITION).alias(_prev_col(column))
            )
    return lag_exprs


def with_snapshot_factors(
    frame: pl.LazyFrame,
    *,
    micro_price_levels: Sequence[int] = (1,),
    oir_depths: Sequence[int] = (1, 5),
    flow_levels: Sequence[int] = (1,),
) -> pl.LazyFrame:
    lag_levels = sorted(set(flow_levels))
    lag_columns = [_prev_col(f"{field}_{level}") for level in lag_levels for field in (
        "bid_price",
        "ask_price",
        "bid_volume",
        "ask_volume",
    )]

    factor_exprs = [micro_price_expr(level) for level in micro_price_levels]
    factor_exprs.extend(oir_expr(depth) for depth in oir_depths)
    factor_exprs.extend(voi_expr(level) for level in flow_levels)
    factor_exprs.extend(ofi_expr(level) for level in flow_levels)

    return (
        frame.sort(BOOK_SORT)
        .with_columns(_lag_expressions(lag_levels))
        .with_columns(factor_exprs)
        .drop(lag_columns)
    )


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


def compute_snapshot_factors_to_parquet(
    source_path: Path,
    output_path: Path,
    *,
    micro_price_levels: Sequence[int] = (1,),
    oir_depths: Sequence[int] = (1, 5),
    flow_levels: Sequence[int] = (1,),
    force: bool = False,
) -> Path:
    if output_path.exists() and not force:
        return output_path

    validate_snapshot_parquet(
        source_path,
        micro_price_levels=micro_price_levels,
        oir_depths=oir_depths,
        flow_levels=flow_levels,
    )
    frame = pl.scan_parquet(str(source_path))
    factor_frame = with_snapshot_factors(
        frame,
        micro_price_levels=micro_price_levels,
        oir_depths=oir_depths,
        flow_levels=flow_levels,
    )
    _sink_parquet_compat(factor_frame, output_path)
    return output_path
