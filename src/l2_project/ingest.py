from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import polars as pl

DatasetKind = Literal["snapshot", "order", "trade"]
DATASET_KINDS: tuple[DatasetKind, ...] = ("snapshot", "order", "trade")
BOOK_DEPTH = 10


@dataclass(frozen=True)
class CsvIngestJob:
    dataset: DatasetKind
    month: str
    source_path: Path
    output_path: Path


def _snapshot_raw_schema() -> dict[str, Any]:
    schema: dict[str, Any] = {
        "SecuCode": pl.Int64,
        "TradingDay": pl.Int32,
        "TickTime": pl.Int64,
        "TickTimeDiff": pl.Int32,
        "Price": pl.Float64,
        "DealNum": pl.Float64,
        "Volume": pl.Float64,
        "Turnover": pl.Float64,
        "TotalDealNum": pl.Float64,
        "TotalVolume": pl.Float64,
        "TotalTurnover": pl.Float64,
        "TotalBidVolume": pl.Float64,
        "TotalAskVolume": pl.Float64,
        "WeightBidPrice": pl.Float64,
        "WeightAskPrice": pl.Float64,
    }
    for level in range(1, BOOK_DEPTH + 1):
        schema[f"AskPrice{level}"] = pl.Float64
        schema[f"AskVolume{level}"] = pl.Float64
        schema[f"BidPrice{level}"] = pl.Float64
        schema[f"BidVolume{level}"] = pl.Float64
        schema[f"BidOrder{level}"] = pl.Float64
        schema[f"AskOrder{level}"] = pl.Float64
    return schema


def _snapshot_rename_map() -> dict[str, str]:
    rename_map = {
        "SecuCode": "secu_code",
        "TradingDay": "trading_day",
        "TickTime": "event_time",
        "TickTimeDiff": "tick_time_diff",
        "Price": "last_price",
        "DealNum": "deal_num",
        "Volume": "volume",
        "Turnover": "turnover",
        "TotalDealNum": "total_deal_num",
        "TotalVolume": "total_volume",
        "TotalTurnover": "total_turnover",
        "TotalBidVolume": "total_bid_volume",
        "TotalAskVolume": "total_ask_volume",
        "WeightBidPrice": "weight_bid_price",
        "WeightAskPrice": "weight_ask_price",
    }
    for level in range(1, BOOK_DEPTH + 1):
        rename_map[f"AskPrice{level}"] = f"ask_price_{level}"
        rename_map[f"AskVolume{level}"] = f"ask_volume_{level}"
        rename_map[f"BidPrice{level}"] = f"bid_price_{level}"
        rename_map[f"BidVolume{level}"] = f"bid_volume_{level}"
        rename_map[f"BidOrder{level}"] = f"bid_order_{level}"
        rename_map[f"AskOrder{level}"] = f"ask_order_{level}"
    return rename_map


RAW_SCHEMA_BY_DATASET: dict[DatasetKind, dict[str, Any]] = {
    "snapshot": _snapshot_raw_schema(),
    "order": {
        "SecuCode": pl.Int64,
        "TradingDay": pl.Int32,
        "OrderTime": pl.Int64,
        "OrderID": pl.Int64,
        "Price": pl.Float64,
        "Volume": pl.Float64,
        "OrderType": pl.Int32,
        "Channel": pl.Int32,
        "BizIndex": pl.Int64,
        "DBOrderID": pl.Int64,
        "LastPrice": pl.Float64,
    },
    "trade": {
        "SecuCode": pl.Int64,
        "TradingDay": pl.Int32,
        "DealTime": pl.Int64,
        "BuyID": pl.Int64,
        "SellID": pl.Int64,
        "DealID": pl.Int64,
        "Price": pl.Float64,
        "Volume": pl.Float64,
        "Side": pl.Int32,
        "Channel": pl.Int32,
        "BizIndex": pl.Int64,
    },
}

RENAME_MAP_BY_DATASET: dict[DatasetKind, dict[str, str]] = {
    "snapshot": _snapshot_rename_map(),
    "order": {
        "SecuCode": "secu_code",
        "TradingDay": "trading_day",
        "OrderTime": "event_time",
        "OrderID": "order_id",
        "Price": "price",
        "Volume": "volume",
        "OrderType": "order_type",
        "Channel": "channel",
        "BizIndex": "biz_index",
        "DBOrderID": "db_order_id",
        "LastPrice": "last_price",
    },
    "trade": {
        "SecuCode": "secu_code",
        "TradingDay": "trading_day",
        "DealTime": "event_time",
        "BuyID": "buy_id",
        "SellID": "sell_id",
        "DealID": "deal_id",
        "Price": "price",
        "Volume": "volume",
        "Side": "side",
        "Channel": "channel",
        "BizIndex": "biz_index",
    },
}


def discover_csv_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw directory does not exist: {raw_dir}")
    return sorted(path for path in raw_dir.glob("*.csv") if path.is_file())


def build_ingest_jobs(
    *,
    source_dirs_by_month: Mapping[str, Mapping[DatasetKind, Path]],
    output_root: Path,
    months: Sequence[str] | None = None,
    datasets: Sequence[DatasetKind] | None = None,
) -> list[CsvIngestJob]:
    selected_months = tuple(months or source_dirs_by_month.keys())
    selected_datasets = tuple(datasets or DATASET_KINDS)
    jobs: list[CsvIngestJob] = []
    for month in selected_months:
        month_sources = source_dirs_by_month[month]
        for dataset in selected_datasets:
            raw_dir = month_sources[dataset]
            for source_path in discover_csv_files(raw_dir):
                output_path = output_root / dataset / f"month={month}" / f"{source_path.stem}.parquet"
                jobs.append(
                    CsvIngestJob(
                        dataset=dataset,
                        month=month,
                        source_path=source_path,
                        output_path=output_path,
                    )
                )
    return jobs


def _scan_csv_compat(source_path: Path, schema: Mapping[str, Any]) -> pl.LazyFrame:
    common_kwargs = {
        "source": str(source_path),
        "has_header": True,
        "infer_schema_length": 0,
        "ignore_errors": False,
        "null_values": ["", "None", "null", "NULL"],
    }
    try:
        return pl.scan_csv(schema_overrides=dict(schema), **common_kwargs)
    except TypeError:
        return pl.scan_csv(dtypes=dict(schema), **common_kwargs)


def scan_normalized_csv(
    source_path: Path,
    *,
    dataset: DatasetKind,
    month: str,
) -> pl.LazyFrame:
    raw_frame = _scan_csv_compat(source_path, RAW_SCHEMA_BY_DATASET[dataset])
    normalized = (
        raw_frame.rename(RENAME_MAP_BY_DATASET[dataset])
        .with_columns(
            [
                pl.col("secu_code")
                .cast(pl.Int64, strict=False)
                .cast(pl.Utf8)
                .str.zfill(6)
                .alias("symbol"),
                pl.lit(month).alias("source_month"),
            ]
        )
        .select(["symbol", "source_month", *RENAME_MAP_BY_DATASET[dataset].values()])
    )
    return normalized


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


def ingest_csv_job(job: CsvIngestJob, *, force: bool = False) -> Path:
    if job.output_path.exists() and not force:
        return job.output_path

    frame = scan_normalized_csv(
        job.source_path,
        dataset=job.dataset,
        month=job.month,
    )
    _sink_parquet_compat(frame, job.output_path)
    return job.output_path


def ingest_csv_jobs(
    jobs: Iterable[CsvIngestJob],
    *,
    force: bool = False,
) -> list[Path]:
    outputs: list[Path] = []
    for job in jobs:
        outputs.append(ingest_csv_job(job, force=force))
    return outputs
