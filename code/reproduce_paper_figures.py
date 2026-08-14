#!/usr/bin/env python3
"""Rebuild the four manuscript PNG figures from the released data."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile

import plot_moc_noisy_fullrank_l1_small_mu_extension as l1_figures
import plot_moc_publication_figures as main_figures


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
PAPER_DATA = PROJECT_ROOT / "data" / "paper"
L1_SMALL_MU_TABLE = PAPER_DATA / "l1_small_mu_direct_errors.csv"
FIGURE_NAMES = (
    "five_solution_convergence.png",
    "l1_small_mu_direct_errors.png",
    "oversampling_rank_spectrum.png",
    "antipodal_rank_condition.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reproduce_figures(output_dir: Path) -> dict[str, str]:
    """Render into a new directory and return the PNG SHA-256 mapping."""

    output = Path(output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gaussian-hermite-figures-") as temporary:
        staging = Path(temporary)
        main_output = staging / "main"
        main_figures._plot_five_solution_convergence(
            main_figures._load_convergence(PAPER_DATA / "five_solution_convergence.csv"),
            main_output,
        )
        main_figures._plot_oversampling(
            main_figures._load_oversampling(PAPER_DATA / "oversampling_core.csv"),
            main_output,
        )
        main_figures._plot_antipodal(
            main_figures._load_antipodal(PAPER_DATA / "antipodal_core.csv"),
            main_output,
        )
        l1_output = staging / "l1"
        l1_figures.render_publication_assets(
            L1_SMALL_MU_TABLE,
            l1_output,
        )

        output.mkdir()
        sources = {
            "five_solution_convergence.png": main_output
            / "five_solution_convergence.png",
            "l1_small_mu_direct_errors.png": l1_output
            / "l1_small_mu_direct_errors.png",
            "oversampling_rank_spectrum.png": main_output
            / "oversampling_rank_spectrum.png",
            "antipodal_rank_condition.png": main_output
            / "antipodal_rank_condition.png",
        }
        for name in FIGURE_NAMES:
            shutil.copy2(sources[name], output / name)

    hashes = {name: sha256(output / name) for name in FIGURE_NAMES}
    (output / "SHA256SUMS").write_text(
        "".join(f"{hashes[name]}  {name}\n" for name in FIGURE_NAMES),
        encoding="utf-8",
    )
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hashes = reproduce_figures(args.output_dir)
    for name in FIGURE_NAMES:
        print(f"{hashes[name]}  {args.output_dir / name}")


if __name__ == "__main__":
    main()
