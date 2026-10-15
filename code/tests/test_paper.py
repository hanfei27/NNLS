from pathlib import Path
import tempfile
import unittest
import hashlib

import numpy as np
import pandas as pd

import inverse
import numerics as n
import plot_figures
from solutions import derivative

ROOT = Path(__file__).resolve().parents[2]


class NumericalChecks(unittest.TestCase):
    def test_manufactured_derivatives_by_finite_differences(self):
        points = np.array([[.7, -.8], [-1.3, .4], [1.8, 1.1]])
        step = 1.e-5
        for name in ("u1", "u2", "u3", "u4", "u5"):
            for total in range(1, 4):
                for x_order in range(total + 1):
                    alpha = (x_order, total-x_order)
                    axis = 0 if x_order else 1
                    lower = list(alpha)
                    lower[axis] -= 1
                    shift = np.zeros_like(points)
                    shift[:, axis] = step
                    difference = (derivative(name, points+shift, tuple(lower))
                                  - derivative(name, points-shift, tuple(lower))) / (2*step)
                    np.testing.assert_allclose(derivative(name, points, alpha), difference,
                                               rtol=3.e-7, atol=2.e-8)

    def test_residual_counts_and_normalized_ranks(self):
        params = n.parameters(16, 17)
        for factor, count, rank in zip((9,6,4,3,2.4,2),
                                      (112,276,624,1108,1708,2480),
                                      (112,266,272,272,272,272)):
            points, weights = n.level_rule(16,17,factor)
            self.assertEqual(len(points), count)
            matrix = n.ornstein_uhlenbeck_residual_matrix(points, params, 4)
            self.assertEqual(n.spectrum(matrix*np.sqrt(weights)[:,None])["rank"], rank)

    def test_constrained_solver_against_known_projection(self):
        result = n.solve_l1_least_squares_projected(
            np.eye(3), np.array([2.,-1.,.5]), np.ones(3), 1.,
            maxiter=100, check_interval=1, kkt_tol=1.e-12)
        np.testing.assert_allclose(result.coefficients, [1.,0.,0.], atol=1.e-12)

    def test_antipodal_polynomial_identity(self):
        points = np.random.default_rng(4).normal(size=(20,2))
        params = n.parameters(12,11,.5)
        z = n.ridge_inputs(points, params)
        np.testing.assert_allclose(np.maximum(z,0.)**4+np.maximum(-z,0.)**4,
                                   z**4, rtol=2.e-15, atol=1.e-13)

    def test_inverse_constants_against_original_records(self):
        records = pd.read_csv(ROOT/'data/paper/inverse.csv')
        for index, group in records.groupby("collision_index"):
            params = inverse.build_d0_parameters(int(index))
            grams = inverse.assemble_direct_grams(params)
            for row in group.itertuples():
                result = inverse.generalized_metrics(grams,int(row.sigma),parameters=params)
                self.assertAlmostEqual(float(result['inverse_norm_max'])/row.inverse_norm_max, 1., places=11)


class PaperOutputs(unittest.TestCase):
    def test_data_hashes(self):
        for line in (ROOT/'data/paper/SHA256SUMS').read_text().splitlines():
            expected, name = line.split()
            self.assertEqual(hashlib.sha256((ROOT/'data/paper'/name).read_bytes()).hexdigest(), expected)

    def test_render_and_published_table_values(self):
        with tempfile.TemporaryDirectory() as directory:
            output = plot_figures.render(ROOT/'data/paper', Path(directory)/'figures')
            self.assertEqual(len(list(output.glob('*.png'))),3)
            self.assertEqual(len(list(output.glob('*.pdf'))),3)
            for path in output.glob('*.png'):
                self.assertEqual(path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            one = pd.read_csv(output/'table_1.csv').set_index('design')
            expected = {
                'designed_quasi_uniform': ([7.774e-7,2.361e-6,8.534e-6,8.700e-5], None),
                'random_dirs_uniform_bias': ([1.869e-6,6.383e-6,2.866e-5,3.067e-4], [1.735e-6,6.159e-6,2.215e-5,1.927e-4]),
                'iid_uniform': ([2.494e-6,8.586e-6,4.402e-5,4.342e-4], [.648e-6,1.964e-6,.819e-5,.715e-4]),
            }
            for design, (means, deviations) in expected.items():
                np.testing.assert_allclose(one.loc[design,[f'E{s}_mean' for s in range(4)]].to_numpy(dtype=float),means,rtol=5.e-4)
                if deviations:
                    np.testing.assert_allclose(one.loc[design,[f'E{s}_sd' for s in range(4)]].to_numpy(dtype=float),deviations,rtol=1.e-3)
            two = pd.read_csv(output/'table_2.csv')
            np.testing.assert_allclose(two.fitted_exponent,[.9232,1.9106,2.9069,3.9056],atol=5.e-5,rtol=0)
            three = pd.read_csv(output/'table_3.csv')
            self.assertEqual(three.deterministic_rank.tolist(),[112,266,272,272,272,272])
            self.assertEqual(three.random_full_rank_count.tolist(),[0,29,30,30,30,30])
            self.assertEqual(three.random_realizations.tolist(),[30]*6)
            with self.assertRaises(FileExistsError):
                plot_figures.render(ROOT/'data/paper', output)


if __name__ == '__main__':
    unittest.main()
