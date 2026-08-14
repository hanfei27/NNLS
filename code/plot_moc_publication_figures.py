#!/usr/bin/env python3
"""Render publication-sized figures from the locked formal r3 CSV outputs.

This script is intentionally independent of the formal experiment and report
modules.  It reads their immutable CSV outputs and writes only to the new
``paper_publication`` directory below the selected result root.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_RESULT_ROOT = Path(
    "code/results/moc_experiment_completion_formal_20260810_r3"
)
PAPER_WIDTH_IN = 6.27
ERROR_COLUMNS = ("E0", "E1", "E2", "E3")
SOLUTIONS = ("u1", "u2", "u3", "u4", "u5")

INK = "#20252B"
BLUE = "#315E8A"
ORANGE = "#C45E24"
REFERENCE = "#67727C"
GRID = "#D8DDE3"

COLORS = {
    "u1": BLUE,
    "u2": "#C46A2A",
    "u3": "#6F7F3A",
    "u4": "#9A4E74",
    "u5": "#5C6F7B",
}
MARKERS = {"u1": "o", "u2": "s", "u3": "^", "u4": "D", "u5": "v"}
NORM_TITLES = {
    "E0": r"$L^2_\gamma$",
    "E1": r"$H^1_\gamma$",
    "E2": r"$H^2_\gamma$",
    "E3": r"$H^3_\gamma$",
}
CONVERGENCE_LEGEND_LOCATION = "upper right"
CONVERGENCE_LEGEND_ANCHOR = (0.985, 0.985)
CONVERGENCE_CORNER_RESERVE = 0.16
RANK_STYLES = {
    "lapack_style": ("o", "-"),
    "relative_1e-12": ("s", "--"),
    "relative_1e-14": ("^", ":"),
}
RANK_LABELS = {
    "lapack_style": r"$\tau=\max(N,J)\varepsilon$",
    "relative_1e-12": r"$\tau=10^{-12}$",
    "relative_1e-14": r"$\tau=10^{-14}$",
}


def _publication_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": ("STIXGeneral", "DejaVu Serif"),
        "mathtext.fontset": "stix",
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _require_finite_positive(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    source: Path,
) -> None:
    values = frame.loc[:, columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{source} contains non-finite or non-positive plotted values")


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(True, which="major", color=GRID, linewidth=0.55, alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.8, length=3.0, color=INK)


def _save_pair(figure: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path)
    figure.savefig(png_path, dpi=300)
    plt.close(figure)
    return pdf_path, png_path


def _load_convergence(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    required = {
        "solution",
        "level_index",
        "h_theta",
        *ERROR_COLUMNS,
        *(f"p_{norm}" for norm in ERROR_COLUMNS),
        *(f"log_intercept_{norm}" for norm in ERROR_COLUMNS),
        *(f"r2_{norm}" for norm in ERROR_COLUMNS),
    }
    _require_columns(frame, required, source)
    if set(frame["solution"]) != set(SOLUTIONS):
        raise ValueError(f"{source} must contain exactly {SOLUTIONS}")
    counts = frame.groupby("solution", observed=True).size()
    if not (counts == 6).all() or len(frame) != 30:
        raise ValueError(f"{source} must contain six common levels for each solution")
    if frame.duplicated(["solution", "level_index"]).any():
        raise ValueError(f"{source} contains duplicate solution/level rows")
    _require_finite_positive(frame, ("h_theta", *ERROR_COLUMNS), source)
    for norm in ERROR_COLUMNS:
        fit_values = frame[[f"p_{norm}", f"log_intercept_{norm}", f"r2_{norm}"]]
        if not np.all(np.isfinite(fit_values.to_numpy(dtype=float))):
            raise ValueError(f"{source} contains non-finite {norm} fit statistics")
        if not frame[f"r2_{norm}"].between(0.0, 1.0).all():
            raise ValueError(f"{source} contains invalid {norm} R-squared values")
        if (frame[f"p_{norm}"] <= 0.0).any():
            raise ValueError(f"{source} contains a non-positive {norm} fit exponent")
        for _, subset in frame.groupby("solution", observed=True):
            for column in (f"p_{norm}", f"log_intercept_{norm}", f"r2_{norm}"):
                if not np.allclose(subset[column], subset[column].iloc[0], rtol=0, atol=1e-12):
                    raise ValueError(f"{source} has inconsistent {column} within a solution")
    return frame


def _plot_five_solution_convergence(
    frame: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    with plt.rc_context(_publication_style()):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(PAPER_WIDTH_IN, 5.75),
            sharex=True,
            layout="constrained",
        )
        legend_layouts: list[
            tuple[plt.Axes, matplotlib.legend.Legend, str, dict[str, tuple[np.ndarray, np.ndarray]]]
        ] = []
        for norm, axis in zip(ERROR_COLUMNS, axes.flat, strict=True):
            traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for solution in SOLUTIONS:
                subset = frame[frame["solution"] == solution].sort_values("h_theta")
                h_theta = subset["h_theta"].to_numpy(dtype=float)
                errors = subset[norm].to_numpy(dtype=float)
                power = float(subset[f"p_{norm}"].iloc[0])
                intercept = float(subset[f"log_intercept_{norm}"].iloc[0])
                r_squared = float(subset[f"r2_{norm}"].iloc[0])
                fit_grid = np.geomspace(float(h_theta.min()), float(h_theta.max()), 160)
                fit_values = np.exp(intercept) * fit_grid**power
                axis.plot(
                    fit_grid,
                    fit_values,
                    color=COLORS[solution],
                    linewidth=1.2,
                    alpha=0.92,
                    zorder=2,
                )
                axis.plot(
                    h_theta,
                    errors,
                    linestyle="none",
                    marker=MARKERS[solution],
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    color=COLORS[solution],
                    label=(
                        rf"$u_{solution[1]}$: ${power:.2f},\,{r_squared:.3f}$"
                    ),
                    zorder=3,
                )
                traces[solution] = (
                    np.concatenate((h_theta, fit_grid)),
                    np.concatenate((errors, fit_values)),
                )

            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_title(NORM_TITLES[norm])
            axis.set_ylabel("relative error")
            _style_axis(axis)
            axis.grid(True, which="minor", color=GRID, linewidth=0.45, alpha=0.65)
            y_min, y_max = axis.get_ylim()
            reserve_factor = np.exp(
                CONVERGENCE_CORNER_RESERVE * np.log(y_max / y_min)
            )
            axis.set_ylim(y_min, y_max * reserve_factor)
            handles, labels = axis.get_legend_handles_labels()
            if len(handles) != 5 or len(labels) != 5:
                raise ValueError(f"expected five legend entries for {norm}")
            header = Line2D([], [], linestyle="none")
            # Matplotlib fills a two-column legend down each column.  This
            # order produces the three compact rows
            #   (p,R^2) | u1,   u2 | u3,   u4 | u5.
            ordered_handles = [
                header,
                handles[1],
                handles[3],
                handles[0],
                handles[2],
                handles[4],
            ]
            ordered_labels = [
                r"$(p,\ R^2)$",
                labels[1],
                labels[3],
                labels[0],
                labels[2],
                labels[4],
            ]
            legend = axis.legend(
                ordered_handles,
                ordered_labels,
                ncol=2,
                frameon=True,
                facecolor="white",
                edgecolor="#B8BEC5",
                framealpha=1.0,
                borderpad=0.05,
                labelspacing=0.00,
                columnspacing=0.48,
                handlelength=0.75,
                handletextpad=0.20,
                loc=CONVERGENCE_LEGEND_LOCATION,
                bbox_to_anchor=CONVERGENCE_LEGEND_ANCHOR,
                bbox_transform=axis.transAxes,
            )
            legend_layouts.append((axis, legend, norm, traces))

        axes[0, 0].invert_xaxis()
        for axis in axes[1, :]:
            axis.set_xlabel(r"fill distance $h_{\Theta_0}$")

        # The common upper-right anchor keeps the four panels visually
        # consistent.  Any overlap with the evidence remains a hard failure.
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis, legend, norm, traces in legend_layouts:
            legend_box = legend.get_window_extent(renderer)
            for solution, (trace_x, trace_y) in traces.items():
                display_points = axis.transData.transform(np.column_stack((trace_x, trace_y)))
                covered = (
                    (display_points[:, 0] >= legend_box.x0)
                    & (display_points[:, 0] <= legend_box.x1)
                    & (display_points[:, 1] >= legend_box.y0)
                    & (display_points[:, 1] <= legend_box.y1)
                )
                if np.any(covered):
                    raise ValueError(f"{norm} legend covers the {solution} trace")
        return _save_pair(figure, output_dir, "five_solution_convergence")


def _load_oversampling(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source).sort_values("rule_index")
    required = {
        "rule_index",
        "width_J",
        "N_over_J",
        "rank_lapack_style",
        "rank_relative_1e_12",
        "rank_relative_1e_14",
        "sigma_min_lapack",
        "effective_condition_lapack",
        "exact_condition_infinite",
    }
    _require_columns(frame, required, source)
    if len(frame) != 6 or frame["rule_index"].duplicated().any():
        raise ValueError(f"{source} must contain the six registered sampling rules")
    _require_finite_positive(
        frame,
        ("width_J", "N_over_J", "effective_condition_lapack"),
        source,
    )
    sigma = frame["sigma_min_lapack"].to_numpy(dtype=float)
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
        raise ValueError(f"{source} contains invalid smallest singular values")
    return frame


def _plot_oversampling(
    frame: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    with plt.rc_context(_publication_style()):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(PAPER_WIDTH_IN, 2.42),
            layout="constrained",
        )
        x = frame["N_over_J"].to_numpy(dtype=float)
        rank_columns = {
            "lapack_style": "rank_lapack_style",
            "relative_1e-12": "rank_relative_1e_12",
            "relative_1e-14": "rank_relative_1e_14",
        }
        for threshold, column in rank_columns.items():
            marker, linestyle = RANK_STYLES[threshold]
            axes[0].plot(
                x,
                frame[column],
                color=BLUE,
                marker=marker,
                markersize=4.2,
                markerfacecolor="white",
                markeredgewidth=0.9,
                linewidth=1.15,
                linestyle=linestyle,
                label=RANK_LABELS[threshold],
            )
        width = int(frame["width_J"].iloc[0])
        if not (frame["width_J"] == width).all():
            raise ValueError("oversampling comparison must keep J fixed")
        axes[0].axhline(width, color=REFERENCE, linewidth=1.0, linestyle="-.", label=r"$J$")
        axes[0].set_title("Rank at cutoff")
        axes[0].set_ylabel(r"$r_\tau$")
        axes[0].legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.96,
            fontsize=7.5,
            borderpad=0.2,
            labelspacing=0.18,
            handlelength=1.4,
            loc="lower right",
        )

        sigma = frame["sigma_min_lapack"].to_numpy(dtype=float)
        positive_sigma = sigma[sigma > 0.0]
        if positive_sigma.size == 0:
            raise ValueError("oversampling spectrum has no positive smallest singular value")
        sigma_floor = float(positive_sigma.min() / 8.0)
        sigma_plot = np.where(sigma > 0.0, sigma, sigma_floor)
        axes[1].plot(
            x,
            sigma_plot,
            color=BLUE,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.9,
            markersize=4.2,
        )
        zero = sigma == 0.0
        if np.any(zero):
            axes[1].scatter(x[zero], sigma_plot[zero], marker="v", color=ORANGE, s=30, zorder=5)
            axes[1].annotate(
                r"$\sigma_J=0$ ($N<J$)",
                xy=(float(x[zero][0]), float(sigma_plot[zero][0])),
                xytext=(4, 7),
                textcoords="offset points",
                fontsize=7.5,
                color=ORANGE,
            )
        axes[1].set_yscale("log")
        axes[1].set_title("Smallest singular value")
        axes[1].set_ylabel(r"$\sigma_J(\widetilde A)$")

        condition = frame["effective_condition_lapack"].to_numpy(dtype=float)
        axes[2].plot(
            x,
            condition,
            color=BLUE,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.9,
            markersize=4.2,
        )
        exact = frame["exact_condition_infinite"].astype(bool).to_numpy()
        if np.any(exact):
            axes[2].scatter(
                x[exact],
                condition[exact],
                marker="^",
                facecolors="none",
                edgecolors=ORANGE,
                s=44,
                zorder=5,
            )
            axes[2].annotate(
                r"exact $\kappa=+\infty$",
                xy=(float(x[exact][0]), float(condition[exact][0])),
                xytext=(4, -11),
                textcoords="offset points",
                fontsize=7.5,
                color=ORANGE,
            )
        axes[2].set_yscale("log")
        axes[2].set_title("Condition number at cutoff")
        axes[2].set_ylabel(r"$\kappa_{\tau_{\rm mach}}$")

        shown_ticks = x[[0, 1, 2, 3, 5]]
        shown_labels = [f"{value:.2g}" for value in shown_ticks]
        for axis in axes:
            _style_axis(axis)
            axis.set_xscale("log")
            axis.set_xticks(shown_ticks, shown_labels)
            axis.set_xlabel(r"residual-point ratio $N/J$")
        return _save_pair(figure, output_dir, "oversampling_rank_spectrum")


def _load_antipodal(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    required = {
        "regime",
        "delta_pm",
        "width_J",
        "rank_threshold_name",
        "numerical_rank",
        "rank_fraction",
        "effective_condition_number",
        "exact_condition_infinite",
    }
    _require_columns(frame, required, source)
    expected_regimes = {"separation_sequence", "large_exact_collision"}
    if set(frame["regime"]) != expected_regimes:
        raise ValueError(f"{source} must contain {sorted(expected_regimes)}")
    if set(frame["rank_threshold_name"]) != set(RANK_STYLES):
        raise ValueError(f"{source} has an unexpected rank-threshold set")
    _require_finite_positive(frame, ("width_J", "effective_condition_number"), source)
    if (frame["delta_pm"] < 0.0).any() or not np.all(np.isfinite(frame["delta_pm"])):
        raise ValueError(f"{source} contains invalid antipodal separations")
    return frame


def _plot_antipodal(
    frame: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    sequence = frame[frame["regime"] == "separation_sequence"].copy()
    large = frame[
        (frame["regime"] == "large_exact_collision")
        & (frame["rank_threshold_name"] == "lapack_style")
    ]
    if len(large) != 1:
        raise ValueError("antipodal core must contain one LAPACK-style large collision row")
    positive = sequence.loc[sequence["delta_pm"] > 0.0, "delta_pm"]
    if positive.empty:
        raise ValueError("antipodal sequence must contain positive separations")
    zero_position = float(positive.min() / 10.0)

    def plot_abscissa(values: pd.Series) -> np.ndarray:
        array = values.to_numpy(dtype=float)
        return np.where(array > 0.0, array, zero_position)

    with plt.rc_context(_publication_style()):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(PAPER_WIDTH_IN, 2.08),
            layout="constrained",
        )
        for threshold in RANK_STYLES:
            subset = sequence[
                sequence["rank_threshold_name"] == threshold
            ].sort_values("delta_pm")
            marker, linestyle = RANK_STYLES[threshold]
            axes[0].plot(
                plot_abscissa(subset["delta_pm"]),
                subset["rank_fraction"],
                color=BLUE,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=0.9,
                markersize=4.2,
                linewidth=1.15,
                linestyle=linestyle,
                label=RANK_LABELS[threshold],
            )
        large_row = large.iloc[0]
        large_width = int(large_row["width_J"])
        large_rank = int(large_row["numerical_rank"])
        axes[0].scatter(
            [zero_position],
            [float(large_row["rank_fraction"])],
            marker="*",
            s=58,
            color=ORANGE,
            label=rf"$J={large_width},\ r_\tau/J={large_rank}/{large_width}$",
            zorder=5,
        )
        axes[0].axhline(1.0, color=REFERENCE, linewidth=1.0, linestyle="--")
        axes[0].set_title("Rank at cutoff")
        axes[0].set_ylabel(r"$r_\tau/J$")
        axes[0].legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.96,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1.4,
            loc="lower right",
        )

        lapack = sequence[
            sequence["rank_threshold_name"] == "lapack_style"
        ].sort_values("delta_pm")
        finite = lapack[~lapack["exact_condition_infinite"].astype(bool)]
        if finite.empty:
            raise ValueError("antipodal sequence must contain finite-condition rows")
        axes[1].plot(
            plot_abscissa(finite["delta_pm"]),
            finite["effective_condition_number"],
            color=BLUE,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.9,
            markersize=4.2,
        )
        top = float(finite["effective_condition_number"].max() * 2.0)
        axes[1].scatter(
            [zero_position],
            [top],
            marker="^",
            facecolors="none",
            edgecolors=ORANGE,
            s=48,
            zorder=5,
        )
        axes[1].annotate(
            r"exact collision: $\kappa=+\infty$",
            xy=(zero_position, top),
            xytext=(6, -2),
            textcoords="offset points",
            fontsize=7.8,
            color=ORANGE,
            ha="left",
        )
        axes[1].set_yscale("log")
        axes[1].set_ylim(
            bottom=float(finite["effective_condition_number"].min() / 2.0),
            top=top * 2.2,
        )
        axes[1].set_title("Condition number at cutoff")
        axes[1].set_ylabel(r"$\kappa_{\tau_{\rm mach}}$")

        tick_values = [zero_position, 1e-6, 1e-4, 1e-2, 1e-1]
        tick_labels = ["0", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$", r"$10^{-1}$"]
        for axis in axes:
            _style_axis(axis)
            axis.set_xscale("log")
            axis.set_xlim(zero_position / 1.8, float(positive.max() * 1.5))
            axis.set_xticks(tick_values, tick_labels)
            axis.set_xlabel(r"antipodal separation $\delta_\pm$")
        return _save_pair(figure, output_dir, "antipodal_rank_condition")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="formal result root containing report_full and report_core",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    convergence_source = (
        result_root / "report_full" / "paper_candidates" / "five_solution_convergence.csv"
    )
    oversampling_source = result_root / "report_core" / "oversampling_core.csv"
    antipodal_source = result_root / "report_core" / "antipodal_core.csv"
    for source in (convergence_source, oversampling_source, antipodal_source):
        if not source.is_file():
            raise FileNotFoundError(source)

    output_dir = result_root / "paper_publication"
    convergence = _load_convergence(convergence_source)
    oversampling = _load_oversampling(oversampling_source)
    antipodal = _load_antipodal(antipodal_source)
    outputs = [
        *_plot_five_solution_convergence(convergence, output_dir),
        *_plot_oversampling(oversampling, output_dir),
        *_plot_antipodal(antipodal, output_dir),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
