from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from .factors import BOOK_SORT, SnapshotSchemaError, _is_numeric_dtype

LABEL_PARTITION = ["symbol", "trading_day"]
DEFAULT_HORIZONS_MS: tuple[int, ...] = (3_000, 30_000)


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


def _mid_price_expr(level: int = 1) -> pl.Expr:
    bid_price = pl.col(f"bid_price_{level}")
    ask_price = pl.col(f"ask_price_{level}")
    return (
        pl.when(
            bid_price.is_not_null()
            & ask_price.is_not_null()
            & (bid_price > 0)
            & (ask_price > 0)
            & (ask_price >= bid_price)
        )
        .then((bid_price + ask_price) / 2)
        .otherwise(None)
    )


def _label_base_price_expr(
    schema: Mapping[str, pl.DataType],
    *,
    price_column: str,
) -> pl.Expr:
    if price_column in schema:
        return pl.col(price_column).cast(pl.Float64, strict=False).alias("__label_base_price")
    if price_column == "mid_price_1":
        return _mid_price_expr(1).alias("__label_base_price")
    raise SnapshotSchemaError(
        f"price column '{price_column}' is missing and cannot be derived from snapshot columns"
    )


def validate_label_schema(
    schema: Mapping[str, pl.DataType],
    *,
    price_column: str = "mid_price_1",
) -> None:
    required_columns = ["symbol", "trading_day", "event_time"]
    if price_column == "mid_price_1" and price_column not in schema:
        required_columns.extend(["bid_price_1", "ask_price_1"])
    elif price_column != "mid_price_1":
        required_columns.append(price_column)

    missing = [column for column in required_columns if column not in schema]
    if missing:
        missing_text = ", ".join(missing)
        raise SnapshotSchemaError(f"missing required label columns: {missing_text}")

    invalid_numeric = [
        f"{column}<{schema[column]}>"
        for column in ("event_time", "trading_day")
        if not _is_numeric_dtype(schema[column])
    ]
    if price_column in schema and not _is_numeric_dtype(schema[price_column]):
        invalid_numeric.append(f"{price_column}<{schema[price_column]}>")
    for column in ("bid_price_1", "ask_price_1"):
        if column in schema and not _is_numeric_dtype(schema[column]):
            invalid_numeric.append(f"{column}<{schema[column]}>")
    if invalid_numeric:
        invalid_text = ", ".join(invalid_numeric)
        raise SnapshotSchemaError(f"invalid label column types: {invalid_text}")


def validate_horizons(horizons_ms: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(horizon) for horizon in horizons_ms)))
    if not normalized:
        raise ValueError("at least one horizon is required")
    invalid = [horizon for horizon in normalized if horizon <= 0]
    if invalid:
        invalid_text = ", ".join(str(horizon) for horizon in invalid)
        raise ValueError(f"horizons must be positive milliseconds: {invalid_text}")
    return normalized


def with_forward_returns(
    frame: pl.LazyFrame,
    *,
    horizons_ms: Sequence[int] = DEFAULT_HORIZONS_MS,
    price_column: str = "mid_price_1",
) -> pl.LazyFrame:
    schema = frame.collect_schema()
    validate_label_schema(schema, price_column=price_column)
    normalized_horizons = validate_horizons(horizons_ms)

    working = (
        frame.sort(BOOK_SORT)
        .with_columns(
            [
                _event_time_to_ms_expr(),
                _label_base_price_expr(schema, price_column=price_column),
            ]
        )
    )
    right = working.select(
        [
            "symbol",
            "trading_day",
            pl.col("__event_time_ms"),
            pl.col("__label_base_price"),
        ]
    )

    for horizon_ms in normalized_horizons:
        target_time_col = f"__target_time_ms_{horizon_ms}"
        future_time_col = f"label_time_ms_{horizon_ms}ms"
        future_price_col = f"label_price_{horizon_ms}ms"
        return_col = f"fwd_ret_{horizon_ms}ms"

        working = working.with_columns(
            (pl.col("__event_time_ms") + pl.lit(horizon_ms)).alias(target_time_col)
        )
        working = working.join_asof(
            right.rename(
                {
                    "__event_time_ms": future_time_col,
                    "__label_base_price": future_price_col,
                }
            ),
            left_on=target_time_col,
            right_on=future_time_col,
            by=LABEL_PARTITION,
            strategy="forward",
            check_sortedness=False,
        )
        working = working.with_columns(
            pl.when(
                pl.col("__label_base_price").is_not_null()
                & (pl.col("__label_base_price") > 0)
                & pl.col(future_price_col).is_not_null()
                & (pl.col(future_price_col) > 0)
            )
            .then(pl.col(future_price_col) / pl.col("__label_base_price") - 1.0)
            .otherwise(None)
            .alias(return_col)
        )

    helper_columns = ["__event_time_ms", "__label_base_price"]
    helper_columns.extend(f"__target_time_ms_{horizon_ms}" for horizon_ms in normalized_horizons)
    return working.drop(helper_columns)


def compute_forward_returns_to_parquet(
    source_path: Path,
    output_path: Path,
    *,
    horizons_ms: Sequence[int] = DEFAULT_HORIZONS_MS,
    price_column: str = "mid_price_1",
    force: bool = False,
) -> Path:
    if output_path.exists() and not force:
        return output_path

    frame = pl.scan_parquet(str(source_path))
    labeled_frame = with_forward_returns(
        frame,
        horizons_ms=horizons_ms,
        price_column=price_column,
    )
    _sink_parquet_compat(labeled_frame, output_path)
    return output_path
