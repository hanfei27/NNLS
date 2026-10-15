#!/usr/bin/env python3
"""Reproduce the paper's three figures and three table summaries from CSV data."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#315E8A", "#CF7029", "#7B893C", "#A7527D", "#637884")
MARKERS = ("o", "s", "^", "D", "v")
STYLE = {
    "font.family":"serif", "font.serif":["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset":"stix", "font.size":10, "axes.labelsize":11,
    "legend.fontsize":8.5, "legend.fancybox":False, "legend.framealpha":1.,
    "lines.linewidth":1.2, "lines.markersize":4,
    "axes.axisbelow":True, "grid.color":"#E8EBEF", "grid.linewidth":.5,
    "pdf.fonttype":42, "ps.fonttype":42, "savefig.facecolor":"white",
}


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate":None, "ModDate":None})
    plt.close(fig)


def convergence(frame, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), sharex=True)
    for s, ax in enumerate(axes.flat):
        for i, name in enumerate(sorted(frame.solution.unique())):
            group = frame[frame.solution == name].sort_values("h_theta")
            h, error = group.h_theta.to_numpy(), group[f"E{s}"].to_numpy()
            p, intercept = np.polyfit(np.log(h), np.log(error), 1)
            ax.loglog(h, np.exp(intercept)*h**p, color=COLORS[i])
            ax.loglog(h, error, linestyle="none", marker=MARKERS[i], color=COLORS[i],
                      markerfacecolor="white", label=rf"$u_{i+1},\ p={p:.2f}$")
        ax.set_title(r"$L_\gamma^2$" if s == 0 else rf"$H_\gamma^{s}$")
        ax.set_ylabel("relative error")
        ax.set_ylim(top=frame[f"E{s}"].max()*6)
        ax.grid(axis="y", which="major")
        handles, labels = ax.get_legend_handles_labels()
        order = [0,2,4,1,3] if len(handles) == 5 else list(range(len(handles)))
        ax.legend([handles[i] for i in order], [labels[i] for i in order],
                  ncol=2, loc="upper right", columnspacing=.7, handlelength=.8,
                  borderpad=.3, labelspacing=.2)
        if s >= 2:
            ax.set_xlabel(r"fill distance $h_{\Theta_0}$")
    axes[0, 0].invert_xaxis()
    fig.tight_layout()
    save(fig, output, "five_solution_convergence")


def sampling(deterministic, random, output):
    deterministic = deterministic.sort_values("n_collocation_N")
    quantiles = random.groupby("n_collocation_N").E3.quantile([.25, .5, .75]).unstack()
    width = int(deterministic.width_J.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = deterministic.n_collocation_N / width
    full = deterministic["rank"] == width
    axes[0].plot(x, deterministic.condition, "--", color=COLORS[0], linewidth=.9)
    for mask, marker, color, label in (
        (full, "o", COLORS[0], rf"Numerical rank $={width}$"),
        (~full, "^", COLORS[1], rf"Numerical rank $<{width}$")):
        axes[0].plot(x[mask], deterministic.loc[mask, "condition"], linestyle="none",
                     marker=marker, color=color, markerfacecolor="white", label=label)
    axes[0].set_ylabel("condition number")
    axes[0].legend(loc="upper right", fontsize=7.2)
    x_random = quantiles.index.to_numpy() / width
    axes[1].plot(x_random, quantiles[.5], "o-", color=COLORS[0],
                 markerfacecolor="white", label="Random median")
    axes[1].plot(x[full], deterministic.loc[full, "E3"], "s--", color=COLORS[1],
                 markerfacecolor="white", label="Deterministic")
    axes[1].fill_between(x_random, quantiles[.25], quantiles[.75], color=COLORS[0],
                         alpha=.17, linewidth=0, label="Middle 50% of random runs")
    axes[1].set_ylabel(r"relative error $E_3$")
    axes[1].legend(loc="upper right", fontsize=7.2)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator([.5, 1, 2, 4, 8]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel(r"sampling ratio $N/J$")
        ax.grid(which="major")
    axes[1].set_yticks([3.e-4,1.e-3,3.e-3],
                      [r"$3\times 10^{-4}$", r"$10^{-3}$", r"$3\times 10^{-3}$"])
    fig.tight_layout()
    save(fig, output, "residual_sampling_comparison")


def antipodal(frame, output):
    seq = frame[(frame.width_J == 132) & (frame.delta_pm > 0)].sort_values("delta_pm", ascending=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    full = seq["rank"] == seq.width_J
    ax.loglog(seq.delta_pm, seq.condition, "--", color=COLORS[0])
    ax.loglog(seq.loc[full, "delta_pm"], seq.loc[full, "condition"], "o-",
              markerfacecolor="white", color=COLORS[0], label="full numerical rank")
    ax.loglog(seq.loc[~full, "delta_pm"], seq.loc[~full, "condition"], "^",
              markerfacecolor="white", color=COLORS[1], label="rank deficient; capped")
    exact = frame[frame.delta_pm == 0].sort_values("width_J")
    annotation = "Exact collision ($\\delta_\\pm=0$)\n" + "\n".join(
        f"$J={int(row.width_J)}$: rank {int(row.rank)}" for row in exact.itertuples())
    ax.text(.97, .06, annotation, ha="right", va="bottom", transform=ax.transAxes, fontsize=8.5)
    ax.set_xlabel(r"antipodal separation $\delta_\pm$")
    ax.set_ylabel(r"capped condition number $\kappa_1$")
    ax.invert_xaxis()
    ax.grid(which="major")
    ax.legend(loc="upper left", title=r"$J=132,\ N=4112$", fontsize=7.5)
    fig.tight_layout()
    save(fig, output, "antipodal_rank_condition")


def table_summaries(parameters, inverse, deterministic, random):
    # Average solutions within each parameter set, then summarize the seeds.
    errors = [f"E{s}" for s in range(4)]
    per_seed = parameters.groupby(["design", "seed"])[errors].mean()
    parameter_rows = []
    for design, group in per_seed.groupby(level="design", sort=False):
        parameter_rows.append({"design":design, "n_sets":len(group),
            **{f"{name}_{stat}":float(group[name].mean() if stat == "mean" else group[name].std(ddof=1))
               for name in errors for stat in ("mean", "sd")}})
    inverse_rows = [{"sigma":int(sigma), "fitted_exponent":float(np.polyfit(
        np.log(1/group.delta), np.log(group.inverse_norm_max), 1)[0])}
        for sigma, group in inverse.groupby("sigma")]
    sampling_rows = []
    for row in deterministic.sort_values("n_collocation_N").itertuples():
        group = random[random.n_collocation_N == row.n_collocation_N]
        sampling_rows.append({"N":int(row.n_collocation_N), "deterministic_rank":int(row.rank),
            "random_full_rank_count":int((group["rank"] == group.width_J).sum()),
            "random_realizations":len(group)})
    return (pd.DataFrame(parameter_rows), pd.DataFrame(inverse_rows), pd.DataFrame(sampling_rows))


def render(data_dir, output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    data = {name:pd.read_csv(Path(data_dir) / f"{name}.csv") for name in (
        "convergence", "parameters", "inverse", "sampling_deterministic", "sampling_random", "antipodal")}
    output.mkdir(parents=True)
    with plt.rc_context(STYLE):
        convergence(data["convergence"], output)
        sampling(data["sampling_deterministic"], data["sampling_random"], output)
        antipodal(data["antipodal"], output)
    tables = table_summaries(data["parameters"], data["inverse"],
                             data["sampling_deterministic"], data["sampling_random"])
    for number, table in enumerate(tables, 1):
        table.to_csv(output / f"table_{number}.csv", index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/paper")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/figures")
    args = parser.parse_args()
    print(render(args.data_dir, args.output_dir))


if __name__ == "__main__":
    main()
