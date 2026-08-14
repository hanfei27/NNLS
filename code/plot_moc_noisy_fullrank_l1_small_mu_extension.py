#!/usr/bin/env python3
"""Render the coefficient-radius figure from the compact released table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_WIDTH_IN = 6.27
EXPECTED_COLUMNS = (
    "solution",
    "solution_label",
    "mu",
    "clean_E3",
    "perturbed_mean_E3",
    "perturbed_sample_sd_E3",
    "seed_count",
)
EXPECTED_SOLUTIONS = (
    "v8_bilinear_broad_gaussian",
    "w4_rotated_root_exponential",
    "w10_tanh_mix",
    "w19_asinh_ridge",
)
SOLUTION_LABELS = dict(
    zip(EXPECTED_SOLUTIONS, (r"$u_1$", r"$u_2$", r"$u_3$", r"$u_4$"), strict=True)
)
ALL_MULTIPLIERS = (
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    3.0 / 4.0,
    1.0,
    1.5,
    2.0,
)
EXPECTED_SEED_COUNT = 20
EXPECTED_ROW_COUNT = len(EXPECTED_SOLUTIONS) * len(ALL_MULTIPLIERS)

INK = "#20252B"
BLUE = "#315E8A"
ORANGE = "#C45E24"
GRID = "#D8DDE3"
Y_AXIS_LABEL = r"direct relative error $E_3$"
PUBLICATION_FILENAMES = {
    "pdf": "l1_small_mu_direct_errors.pdf",
    "png": "l1_small_mu_direct_errors.png",
    "provenance": "publication_provenance.json",
}
UNCERTAINTY_BAND_PROVENANCE = {
    "center_column": "perturbed_mean_E3",
    "spread_column": "perturbed_sample_sd_E3",
    "spread_statistic": "sample_standard_deviation",
    "ddof": 1,
    "observations_per_point": EXPECTED_SEED_COUNT,
    "interval": "mean_plus_minus_one_sample_sd",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


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
        "legend.fontsize": 7.4,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }


def _uncertainty_bounds(summaries: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    counts = summaries["seed_count"].to_numpy(dtype=float)
    if not np.all(np.isfinite(counts)) or not np.all(counts == EXPECTED_SEED_COUNT):
        raise ValueError(f"seed_count must equal {EXPECTED_SEED_COUNT} in every row")
    mean = summaries["perturbed_mean_E3"].to_numpy(dtype=float)
    sample_sd = summaries["perturbed_sample_sd_E3"].to_numpy(dtype=float)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(sample_sd)):
        raise ValueError("perturbed means and sample standard deviations must be finite")
    if np.any(sample_sd < 0.0):
        raise ValueError("sample standard deviations must be nonnegative")
    lower = mean - sample_sd
    upper = mean + sample_sd
    if np.any(lower <= 0.0):
        raise ValueError("perturbed mean minus sample SD must be positive")
    if not np.all(np.isfinite(upper)):
        raise ValueError("perturbed mean plus sample SD must be finite")
    return lower, upper


def load_compact_table(source_csv: Path) -> pd.DataFrame:
    """Load and fail-closed validate the exact released 4-by-9 table."""

    source = Path(source_csv).expanduser()
    if source.is_symlink():
        raise ValueError("compact table must not be a symlink")
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    summaries = pd.read_csv(source)
    if tuple(summaries.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"compact table schema changed: expected={EXPECTED_COLUMNS}, "
            f"actual={tuple(summaries.columns)}"
        )
    if summaries.shape != (EXPECTED_ROW_COUNT, len(EXPECTED_COLUMNS)):
        raise ValueError(
            "compact table must have exactly "
            f"{EXPECTED_ROW_COUNT} rows and {len(EXPECTED_COLUMNS)} columns"
        )
    if summaries.isna().any(axis=None):
        raise ValueError("compact table contains missing values")
    if summaries.duplicated(["solution", "mu"]).any():
        raise ValueError("compact table contains duplicate solution/mu rows")

    numeric = summaries[
        ["mu", "clean_E3", "perturbed_mean_E3", "perturbed_sample_sd_E3"]
    ].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("compact table contains non-finite numerical values")
    if np.any(numeric[:, :3] <= 0.0) or np.any(numeric[:, 3] < 0.0):
        raise ValueError("compact table errors must be positive and SDs nonnegative")

    expected_mu = np.asarray(ALL_MULTIPLIERS)
    for solution in EXPECTED_SOLUTIONS:
        selected = summaries[summaries["solution"] == solution].sort_values("mu")
        if len(selected) != len(ALL_MULTIPLIERS):
            raise ValueError(f"compact table row count changed for {solution}")
        if set(selected["solution_label"].astype(str)) != {SOLUTION_LABELS[solution]}:
            raise ValueError(f"compact table label changed for {solution}")
        if not np.array_equal(selected["mu"].to_numpy(dtype=float), expected_mu):
            raise ValueError(f"compact table multiplier grid changed for {solution}")

    if set(summaries["solution"].astype(str)) != set(EXPECTED_SOLUTIONS):
        raise ValueError("compact table solution set changed")
    seed_counts = summaries["seed_count"].to_numpy(dtype=float)
    if not np.all(np.isfinite(seed_counts)) or not np.all(
        seed_counts == EXPECTED_SEED_COUNT
    ):
        raise ValueError(f"seed_count must equal {EXPECTED_SEED_COUNT} in every row")
    _uncertainty_bounds(summaries)
    return summaries.sort_values(["solution", "mu"]).reset_index(drop=True)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(True, which="major", color=GRID, linewidth=0.55, alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.8, length=3.0, color=INK)


def _plot_summaries(
    summaries: pd.DataFrame,
    pdf_path: Path,
    png_path: Path,
) -> None:
    uncertainty_lower, uncertainty_upper = _uncertainty_bounds(summaries)
    plotted = np.column_stack(
        (
            summaries["clean_E3"].to_numpy(dtype=float),
            uncertainty_lower,
            uncertainty_upper,
        )
    )
    if not np.all(np.isfinite(plotted)) or np.any(plotted <= 0.0):
        raise ValueError("plot values must be finite and positive")
    lower = float(np.min(plotted) / 1.6)
    upper = float(np.max(plotted) * 1.6)
    if not (0.0 < lower < upper < math.inf):
        raise ValueError("dynamic vertical limits are invalid")

    with plt.rc_context(_publication_style()):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(PAPER_WIDTH_IN, 4.35),
            sharex=True,
            sharey=True,
        )
        figure.subplots_adjust(
            left=0.115,
            right=0.985,
            bottom=0.15,
            top=0.88,
            wspace=0.12,
            hspace=0.24,
        )
        for axis, solution in zip(axes.flat, EXPECTED_SOLUTIONS, strict=True):
            summary = summaries[summaries["solution"] == solution].sort_values("mu")
            mu_values = summary["mu"].to_numpy(dtype=float)
            if not np.array_equal(mu_values, np.asarray(ALL_MULTIPLIERS)):
                raise ValueError(f"incomplete nine-point curve for {solution}")
            perturbed_mean = summary["perturbed_mean_E3"].to_numpy(dtype=float)
            perturbed_sample_sd = summary["perturbed_sample_sd_E3"].to_numpy(
                dtype=float
            )
            axis.fill_between(
                mu_values,
                perturbed_mean - perturbed_sample_sd,
                perturbed_mean + perturbed_sample_sd,
                color=ORANGE,
                alpha=0.14,
                linewidth=0.0,
                zorder=1,
            )
            axis.plot(
                mu_values,
                summary["clean_E3"].to_numpy(dtype=float),
                color=BLUE,
                linestyle="-",
                marker="o",
                markersize=4.6,
                markerfacecolor="white",
                markeredgewidth=0.9,
                label="unperturbed",
            )
            axis.plot(
                mu_values,
                perturbed_mean,
                color=ORANGE,
                linestyle="--",
                marker="x",
                markersize=4.6,
                markeredgewidth=0.9,
                label=r"perturbed mean $\pm$ 1 SD",
            )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xlim(ALL_MULTIPLIERS[0] / 1.18, ALL_MULTIPLIERS[-1] * 1.10)
            axis.set_ylim(lower, upper)
            axis.set_xticks(ALL_MULTIPLIERS)
            axis.set_xticklabels(
                ("1/32", "1/16", "1/8", "1/4", "1/2", "3/4", "1", "3/2", "2")
            )
            axis.set_title(SOLUTION_LABELS[solution])
            _style_axis(axis)

        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.55, 0.99),
            ncol=2,
            frameon=False,
            columnspacing=1.5,
            handlelength=1.8,
            handletextpad=0.45,
        )
        figure.supxlabel(r"coefficient-radius factor $\mu$", y=0.035, fontsize=8.5)
        figure.supylabel(Y_AXIS_LABEL, x=0.025, fontsize=8.5)
        figure.savefig(pdf_path)
        figure.savefig(png_path, dpi=300)
        plt.close(figure)


def _provenance_payload(
    source_csv: Path,
    outputs: dict[str, Path],
) -> dict[str, Any]:
    script = Path(__file__).resolve()
    return {
        "compact_table_path": str(source_csv.resolve()),
        "compact_table_sha256": _sha256(source_csv),
        "plot_script_path": str(script),
        "plot_script_sha256": _sha256(script),
        "uncertainty_band": dict(UNCERTAINTY_BAND_PROVENANCE),
        "output_sha256": {
            outputs[name].name: _sha256(outputs[name]) for name in ("pdf", "png")
        },
    }


def render_publication_assets(
    source_csv: Path,
    output_dir: Path,
) -> dict[str, Path]:
    source = Path(source_csv).expanduser()
    lexical_output = Path(output_dir).expanduser()
    if lexical_output.is_symlink():
        raise FileExistsError("publication output path must not be a symlink")
    output = lexical_output.resolve()
    if output.exists():
        raise FileExistsError(f"publication output directory already exists: {output}")

    summaries = load_compact_table(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or lexical_output.is_symlink():
        raise FileExistsError(f"publication output directory already exists: {output}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    staged = {name: staging / filename for name, filename in PUBLICATION_FILENAMES.items()}
    final = {name: output / filename for name, filename in PUBLICATION_FILENAMES.items()}
    try:
        _plot_summaries(summaries, staged["pdf"], staged["png"])
        payload = _provenance_payload(source.resolve(), staged)
        staged["provenance"].write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if {path.name for path in staging.iterdir()} != set(PUBLICATION_FILENAMES.values()):
            raise ValueError("publication staging file set changed")
        if _load_json_object(staged["provenance"]) != _provenance_payload(
            source.resolve(), staged
        ):
            raise ValueError("publication provenance verification failed")
        if output.exists() or lexical_output.is_symlink():
            raise FileExistsError(f"publication output directory already exists: {output}")
        staging.rename(output)
        return final
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = render_publication_assets(args.input_csv, args.output_dir)
    print("\n".join(f"{name}={path}" for name, path in outputs.items()))


if __name__ == "__main__":
    main()
