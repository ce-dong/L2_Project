from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

CORE_FACTORS: tuple[str, ...] = (
    "oir_1",
    "book_slope_5",
    "trade_imbalance_3s",
    "voi_1",
    "relative_spread_1",
)
SUPPLEMENTARY_FACTORS: tuple[str, ...] = (
    "oir_5",
    "ofi_1",
    "micro_price_1",
)
ALL_FACTORS: tuple[str, ...] = (*CORE_FACTORS, *SUPPLEMENTARY_FACTORS)
LABEL_ORDER: tuple[str, ...] = ("fwd_ret_3000ms", "fwd_ret_30000ms")
LABEL_SHORT: dict[str, str] = {
    "fwd_ret_3000ms": "3s",
    "fwd_ret_30000ms": "30s",
}
FACTOR_DISPLAY_NAMES: dict[str, str] = {
    "oir_1": "OIR L1",
    "book_slope_5": "Book Slope L5",
    "trade_imbalance_3s": "Trade Imb 3s",
    "voi_1": "VOI L1",
    "relative_spread_1": "Rel Spread",
    "oir_5": "OIR L5",
    "ofi_1": "OFI L1",
    "micro_price_1": "Micro-price",
}
HORIZON_COLORS: dict[str, str] = {
    "fwd_ret_3000ms": "#0b6e4f",
    "fwd_ret_30000ms": "#c26d1a",
}
ACCENT_COLORS: dict[str, str] = {
    "coverage": "#3d7ea6",
    "validity": "#d17a22",
}


@dataclass(frozen=True)
class FigureArtifacts:
    core_rank_ic_png: Path
    core_rank_ic_svg: Path
    core_monotonicity_png: Path
    core_monotonicity_svg: Path
    redundancy_heatmap_png: Path
    redundancy_heatmap_svg: Path
    coverage_validity_png: Path
    coverage_validity_svg: Path


def _set_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.9,
        rc={
            "axes.facecolor": "#fbfaf6",
            "figure.facecolor": "#fbfaf6",
            "grid.color": "#d9d4c7",
            "axes.edgecolor": "#403d39",
            "font.family": "DejaVu Sans",
        },
    )


def _ensure_output_root(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _save_figure(fig: plt.Figure, output_root: Path, stem: str) -> tuple[Path, Path]:
    png_path = output_root / f"{stem}.png"
    svg_path = output_root / f"{stem}.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, svg_path


def _factor_label_value(
    frame: pl.DataFrame,
    *,
    factor: str,
    label: str,
    column: str,
) -> float:
    subset = frame.filter(
        (pl.col("factor") == factor) & (pl.col("label") == label)
    ).select(column)
    if subset.height != 1:
        raise ValueError(
            f"expected one row for factor={factor}, label={label}, column={column}"
        )
    return float(subset.item())


def plot_core_factor_rank_ic(
    eval_summary: pl.DataFrame,
    output_root: Path,
) -> tuple[Path, Path]:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(10.8, 5.8))

    x = np.arange(len(CORE_FACTORS))
    width = 0.34
    for idx, label in enumerate(LABEL_ORDER):
        values = [
            _factor_label_value(
                eval_summary,
                factor=factor,
                label=label,
                column="rank_ic_mean",
            )
            for factor in CORE_FACTORS
        ]
        offset = (-0.5 + idx) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=HORIZON_COLORS[label],
            label=f"{LABEL_SHORT[label]} forward return",
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + (0.004 if value >= 0 else -0.006),
                f"{value:.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
                color="#2b2a28",
            )

    ax.axhline(0.0, color="#403d39", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [FACTOR_DISPLAY_NAMES[factor] for factor in CORE_FACTORS],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("Rank IC mean")
    fig.suptitle(
        "Core Factor Rank IC",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.885,
        "Final headline factor pool across 3s and 30s horizons",
        fontsize=11,
        color="#666057",
    )
    fig.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.84),
    )
    sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0, 1, 0.72])
    return _save_figure(fig, output_root, "core_factor_rank_ic")


def plot_core_factor_monotonicity(
    group_summary: pl.DataFrame,
    monotonicity_summary: pl.DataFrame,
    output_root: Path,
) -> tuple[Path, Path]:
    _set_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.0), sharey=True)
    axes = axes.flatten()

    all_returns_bps: list[float] = []
    for factor in CORE_FACTORS:
        factor_rows = group_summary.filter(pl.col("factor") == factor)
        all_returns_bps.extend([float(value) * 1e4 for value in factor_rows["mean_ret"]])

    y_min = min(all_returns_bps) - 0.25
    y_max = max(all_returns_bps) + 0.25

    legend_handles = None
    legend_labels = None
    for ax, factor in zip(axes, CORE_FACTORS, strict=False):
        subset = group_summary.filter(pl.col("factor") == factor)
        for label in LABEL_ORDER:
            line_frame = subset.filter(pl.col("label") == label).sort("group_id")
            x = line_frame["group_id"].to_list()
            y = [float(value) * 1e4 for value in line_frame["mean_ret"].to_list()]
            (line_handle,) = ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.5,
                markersize=6,
                color=HORIZON_COLORS[label],
                label=f"{LABEL_SHORT[label]} horizon",
            )
            if legend_handles is None:
                legend_handles = [line_handle]
                legend_labels = [f"{LABEL_SHORT[label]} horizon"]
            elif f"{LABEL_SHORT[label]} horizon" not in legend_labels:
                legend_handles.append(line_handle)
                legend_labels.append(f"{LABEL_SHORT[label]} horizon")

        spread_ratio = _factor_label_value(
            monotonicity_summary,
            factor=factor,
            label="fwd_ret_3000ms",
            column="positive_spread_ratio",
        )
        ax.set_title(FACTOR_DISPLAY_NAMES[factor], fontsize=12, weight="bold")
        ax.text(
            0.03,
            0.95,
            f"Spread+ ratio: {spread_ratio:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#666057",
        )
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Factor group (low to high)")
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", alpha=0.22)
        sns.despine(ax=ax)

    for ax in axes:
        ax.set_ylabel("")
    axes[-1].axis("off")

    fig.suptitle(
        "Core Factor Quantile Monotonicity",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.90,
        "Five-group cross-sectional sorting on the final headline factor pool",
        fontsize=11,
        color="#666057",
    )
    fig.supylabel("Mean forward return (bps)", x=0.03)
    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.53, 0.86),
        )
    fig.tight_layout(rect=[0, 0, 1, 0.76])
    return _save_figure(fig, output_root, "core_factor_monotonicity")


def plot_factor_redundancy_heatmap(
    redundancy_recommendation: pl.DataFrame,
    output_root: Path,
) -> tuple[Path, Path]:
    _set_plot_style()
    matrix = np.zeros((len(ALL_FACTORS), len(ALL_FACTORS)))
    for idx_left, factor_left in enumerate(ALL_FACTORS):
        for idx_right, factor_right in enumerate(ALL_FACTORS):
            if idx_left == idx_right:
                matrix[idx_left, idx_right] = 1.0
                continue
            if idx_left > idx_right:
                matrix[idx_left, idx_right] = matrix[idx_right, idx_left]
                continue
            row = redundancy_recommendation.filter(
                (pl.col("factor_left") == factor_left)
                & (pl.col("factor_right") == factor_right)
            )
            if row.height == 0:
                row = redundancy_recommendation.filter(
                    (pl.col("factor_left") == factor_right)
                    & (pl.col("factor_right") == factor_left)
                )
            if row.height != 1:
                raise ValueError(
                    f"missing redundancy row for {factor_left} vs {factor_right}"
                )
            record = row.row(0, named=True)
            matrix[idx_left, idx_right] = max(
                float(record["factor_corr_abs_mean"] or 0.0),
                float(record["ic_corr_abs_mean"] or 0.0),
            )
            matrix[idx_right, idx_left] = matrix[idx_left, idx_right]

    fig, ax = plt.subplots(figsize=(9.2, 7.8))
    cmap = sns.blend_palette(
        ["#f6efe5", "#d7b98e", "#6b8f71", "#1f4d3d"],
        as_cmap=True,
    )
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.8,
        linecolor="#f8f6f2",
        cbar_kws={"shrink": 0.8, "label": "Overlap intensity"},
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
    )
    ax.set_xticklabels(
        [FACTOR_DISPLAY_NAMES[factor] for factor in ALL_FACTORS],
        rotation=35,
        ha="right",
    )
    ax.set_yticklabels(
        [FACTOR_DISPLAY_NAMES[factor] for factor in ALL_FACTORS],
        rotation=0,
    )
    fig.suptitle(
        "Factor Redundancy Heatmap",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.89,
        "Cell value = max(abs cross-sectional corr, abs IC-series corr)",
        fontsize=11,
        color="#666057",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.81])
    return _save_figure(fig, output_root, "factor_redundancy_heatmap")


def plot_factor_coverage_validity(
    eval_summary: pl.DataFrame,
    coverage_summary: pl.DataFrame,
    output_root: Path,
) -> tuple[Path, Path]:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    x = np.arange(len(ALL_FACTORS))
    width = 0.34
    coverage_values = [
        float(
            coverage_summary.filter(pl.col("column") == factor)["coverage_ratio"].item()
        )
        for factor in ALL_FACTORS
    ]
    valid_values = [
        _factor_label_value(
            eval_summary,
            factor=factor,
            label="fwd_ret_3000ms",
            column="valid_ratio",
        )
        for factor in ALL_FACTORS
    ]
    coverage_bars = ax.bar(
        x - width / 2.0,
        coverage_values,
        width=width,
        color=ACCENT_COLORS["coverage"],
        label="Column coverage",
    )
    valid_bars = ax.bar(
        x + width / 2.0,
        valid_values,
        width=width,
        color=ACCENT_COLORS["validity"],
        label="Valid ratio (3s horizon)",
    )
    for bars, values in ((coverage_bars, coverage_values), (valid_bars, valid_values)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(value + 0.012, 1.03),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="#2b2a28",
            )

    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [FACTOR_DISPLAY_NAMES[factor] for factor in ALL_FACTORS],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Ratio")
    fig.suptitle(
        "Factor Coverage and Evaluation Validity",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.885,
        "Coverage vs. 3s-horizon evaluability across core and supplementary factors",
        fontsize=11,
        color="#666057",
    )
    fig.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.84),
    )
    sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0, 1, 0.72])
    return _save_figure(fig, output_root, "factor_coverage_validity")


def build_figure_bundle(
    *,
    eval_summary_path: Path,
    group_summary_path: Path,
    monotonicity_summary_path: Path,
    redundancy_recommendation_path: Path,
    coverage_summary_path: Path,
    output_root: Path,
) -> FigureArtifacts:
    output_root = _ensure_output_root(output_root)
    eval_summary = pl.read_parquet(eval_summary_path)
    group_summary = pl.read_parquet(group_summary_path)
    monotonicity_summary = pl.read_parquet(monotonicity_summary_path)
    redundancy_recommendation = pl.read_parquet(redundancy_recommendation_path)
    coverage_summary = pl.read_parquet(coverage_summary_path)

    core_rank_ic_png, core_rank_ic_svg = plot_core_factor_rank_ic(
        eval_summary,
        output_root,
    )
    core_monotonicity_png, core_monotonicity_svg = plot_core_factor_monotonicity(
        group_summary,
        monotonicity_summary,
        output_root,
    )
    redundancy_heatmap_png, redundancy_heatmap_svg = plot_factor_redundancy_heatmap(
        redundancy_recommendation,
        output_root,
    )
    coverage_validity_png, coverage_validity_svg = plot_factor_coverage_validity(
        eval_summary,
        coverage_summary,
        output_root,
    )
    return FigureArtifacts(
        core_rank_ic_png=core_rank_ic_png,
        core_rank_ic_svg=core_rank_ic_svg,
        core_monotonicity_png=core_monotonicity_png,
        core_monotonicity_svg=core_monotonicity_svg,
        redundancy_heatmap_png=redundancy_heatmap_png,
        redundancy_heatmap_svg=redundancy_heatmap_svg,
        coverage_validity_png=coverage_validity_png,
        coverage_validity_svg=coverage_validity_svg,
    )
