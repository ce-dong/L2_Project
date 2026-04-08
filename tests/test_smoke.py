from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from l2_project.eval import build_rank_ic_summary, build_rank_ic_timeseries
from l2_project.factors import SnapshotSchemaError, compute_snapshot_factors_to_parquet
from l2_project.labels import compute_forward_returns_to_parquet


class PipelineSmokeTest(unittest.TestCase):
    def _sample_snapshot_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "symbol": ["000001", "000001", "000001"],
                "source_month": ["202511", "202511", "202511"],
                "secu_code": [1, 1, 1],
                "trading_day": [20251103, 20251103, 20251103],
                "event_time": [93000000, 93003000, 93006000],
                "tick_time_diff": [0, 3000, 3000],
                "last_price": [10.01, 10.02, 10.05],
                "deal_num": [0.0, 1.0, 2.0],
                "volume": [0.0, 100.0, 200.0],
                "turnover": [0.0, 1000.0, 2000.0],
                "total_deal_num": [0.0, 1.0, 2.0],
                "total_volume": [0.0, 100.0, 300.0],
                "total_turnover": [0.0, 1000.0, 3000.0],
                "total_bid_volume": [100.0, 110.0, 130.0],
                "total_ask_volume": [120.0, 90.0, 80.0],
                "weight_bid_price": [10.00, 10.00, 10.02],
                "weight_ask_price": [10.02, 10.02, 10.04],
                "bid_price_1": [10.00, 10.00, 10.02],
                "bid_volume_1": [100.0, 110.0, 130.0],
                "ask_price_1": [10.02, 10.02, 10.04],
                "ask_volume_1": [120.0, 90.0, 80.0],
                "bid_price_2": [9.99, 9.99, 10.01],
                "bid_volume_2": [80.0, 82.0, 95.0],
                "ask_price_2": [10.03, 10.03, 10.05],
                "ask_volume_2": [110.0, 95.0, 90.0],
                "bid_price_3": [9.98, 9.98, 10.00],
                "bid_volume_3": [70.0, 72.0, 85.0],
                "ask_price_3": [10.04, 10.04, 10.06],
                "ask_volume_3": [100.0, 92.0, 88.0],
                "bid_price_4": [9.97, 9.97, 9.99],
                "bid_volume_4": [60.0, 61.0, 80.0],
                "ask_price_4": [10.05, 10.05, 10.07],
                "ask_volume_4": [90.0, 89.0, 86.0],
                "bid_price_5": [9.96, 9.96, 9.98],
                "bid_volume_5": [50.0, 58.0, 70.0],
                "ask_price_5": [10.06, 10.06, 10.08],
                "ask_volume_5": [80.0, 85.0, 84.0],
            }
        )

    def test_factor_and_label_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_path = temp_root / "snapshot.parquet"
            factor_path = temp_root / "factors.parquet"
            label_path = temp_root / "labels.parquet"

            self._sample_snapshot_frame().write_parquet(source_path)

            compute_snapshot_factors_to_parquet(source_path, factor_path, force=True)
            compute_forward_returns_to_parquet(
                factor_path,
                label_path,
                horizons_ms=(3_000,),
                force=True,
            )

            result = pl.read_parquet(label_path).sort(["symbol", "trading_day", "event_time"])

            self.assertTrue(label_path.exists())
            self.assertIn("micro_price_1", result.columns)
            self.assertIn("oir_1", result.columns)
            self.assertIn("voi_1", result.columns)
            self.assertIn("ofi_1", result.columns)
            self.assertIn("fwd_ret_3000ms", result.columns)

            first_row = result.row(0, named=True)
            second_row = result.row(1, named=True)

            self.assertIsNone(first_row["voi_1"])
            self.assertIsNone(first_row["ofi_1"])
            self.assertGreaterEqual(first_row["micro_price_1"], first_row["bid_price_1"])
            self.assertLessEqual(first_row["micro_price_1"], first_row["ask_price_1"])
            self.assertAlmostEqual(second_row["voi_1"], 40.0, places=6)
            self.assertAlmostEqual(second_row["ofi_1"], 40.0, places=6)
            self.assertIsNone(result.row(2, named=True)["voi_1"])
            self.assertAlmostEqual(result.row(2, named=True)["ofi_1"], 220.0, places=6)
            self.assertAlmostEqual(first_row["fwd_ret_3000ms"], 0.0, places=12)
            self.assertGreater(second_row["fwd_ret_3000ms"], 0.0)
            self.assertIsNone(result.row(2, named=True)["fwd_ret_3000ms"])

    def test_missing_snapshot_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_path = temp_root / "invalid_snapshot.parquet"

            self._sample_snapshot_frame().drop("ask_volume_1").write_parquet(source_path)

            with self.assertRaises(SnapshotSchemaError):
                compute_snapshot_factors_to_parquet(
                    source_path,
                    temp_root / "invalid_factor.parquet",
                    force=True,
                )

    def test_eval_summary_ignores_invalid_groups(self) -> None:
        panel = pl.DataFrame(
            {
                "trading_day": [20251103] * 9,
                "event_time": [
                    91400000,
                    91400000,
                    91400000,
                    93000000,
                    93000000,
                    93000000,
                    93003000,
                    93003000,
                    93003000,
                ],
                "symbol": [
                    "000001",
                    "000002",
                    "000003",
                    "000001",
                    "000002",
                    "000003",
                    "000001",
                    "000002",
                    "000003",
                ],
                "source_month": ["202511"] * 9,
                "micro_price_1": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0],
                "fwd_ret_3000ms": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        ).lazy()

        timeseries = build_rank_ic_timeseries(
            panel,
            factor_columns=["micro_price_1"],
            label_columns=["fwd_ret_3000ms"],
            min_cross_section=3,
        ).collect()
        summary = build_rank_ic_summary(
            timeseries.lazy(),
            factor_columns=["micro_price_1"],
            label_columns=["fwd_ret_3000ms"],
        ).collect()

        self.assertEqual(timeseries.shape[0], 2)
        self.assertEqual(timeseries["rank_ic__micro_price_1__fwd_ret_3000ms"].null_count(), 1)
        self.assertEqual(
            timeseries["invalid_reason__micro_price_1__fwd_ret_3000ms"].to_list(),
            [None, "factor_constant"],
        )

        row = summary.row(0, named=True)
        self.assertEqual(row["groups_total"], 2)
        self.assertEqual(row["groups_valid"], 1)
        self.assertEqual(row["groups_factor_constant"], 1)
        self.assertAlmostEqual(row["valid_ratio"], 0.5, places=12)
        self.assertAlmostEqual(row["rank_ic_mean"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
