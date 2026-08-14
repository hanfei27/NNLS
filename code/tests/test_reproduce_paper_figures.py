from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import reproduce_paper_figures as reproduction  # noqa: E402


EXPECTED_SHA256 = {
    "five_solution_convergence.png": (
        "9faa2cb1217406311713cb775c1d396c95b5bcc56900915fbb720da8d47877e2"
    ),
    "l1_small_mu_direct_errors.png": (
        "ce141cc5a79e3a87e0b4b477292eeadb039f188f254df69c4392234abcac079e"
    ),
    "oversampling_rank_spectrum.png": (
        "4b1fd6b9dd0093283455571547a333f947ae5590d93be9b5c156324b72fb6c55"
    ),
    "antipodal_rank_condition.png": (
        "eb8477bed51d8564462e75ba67afe9f29b54b13944e473d860310873b42e3c9f"
    ),
}


class PublicationFigureReproductionTests(unittest.TestCase):
    def test_released_data_reproduce_all_four_expected_pngs(self) -> None:
        original_cwd = Path.cwd()
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "figures"
            os.chdir(temporary)
            try:
                hashes = reproduction.reproduce_figures(output)
            finally:
                os.chdir(original_cwd)

            self.assertEqual(hashes, EXPECTED_SHA256)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                set(reproduction.FIGURE_NAMES) | {"SHA256SUMS"},
            )
            for name in reproduction.FIGURE_NAMES:
                self.assertEqual(
                    reproduction.sha256(output / name), EXPECTED_SHA256[name]
                )


if __name__ == "__main__":
    unittest.main()
