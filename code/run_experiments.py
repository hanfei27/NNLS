#!/usr/bin/env python3
"""Run the five experiments in arXiv:2610.14457, Section 4.

Example: python code/run_experiments.py all --smoke --output-dir build/smoke
Omit --smoke for the paper settings. Full convergence and parameter studies
are substantially more expensive; existing output directories are rejected.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import mpmath as mp
import numpy as np

import inverse
import numerics as n

STUDIES = ("convergence", "parameters", "inverse", "sampling", "antipodal")
LEVELS = ((20, 21), (24, 25), (32, 33), (40, 41), (48, 49), (56, 57))
SAMPLE_COUNTS = (112, 276, 624, 1108, 1708, 2480)
SAMPLE_FACTORS = (9., 6., 4., 3., 2.4, 2.)
NAMES = ("u1", "u2", "u3", "u4", "u5")


def write_rows(path, rows):
    if not rows:
        raise ValueError("no experiment rows to save")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(params, coefficients, names, smoke, cell_width=.05, radius=8.):
    return n.errors(params, coefficients, names, radius=radius,
                    cell_width=1. if smoke else cell_width, order=4)


def convergence(output, smoke=False, truncation_check=False):
    levels = ((8, 9), (12, 13)) if smoke else LEVELS
    rows, audit = [], []
    for na, nb in levels:
        params = n.parameters(na, nb)
        points, weights = n.level_rule(na, nb)
        coefficients, _, diagnostics = n.solve(
            params, points, weights, NAMES, (1., 1., 1., 1., 2000.))
        values = evaluate(params, coefficients, NAMES, smoke)
        geometry = n.parameter_geometry(params, n_bias=nb, bias_radius=5.)
        np.savez_compressed(output / f"convergence_{na}x{nb}.npz",
                            directions=params.directions, biases=params.biases,
                            coefficients=coefficients, solutions=NAMES)
        for i, name in enumerate(NAMES):
            rows.append({"solution":name, "n_directions":na, "n_bias":nb,
                         "width_J":params.n_atoms, "n_collocation_N":len(points),
                         "h_theta":geometry["h_theta"],
                         **{f"E{s}":values[i, s] for s in range(4)}})
        write_rows(output / "convergence.csv", rows)
        if truncation_check:
            previous = values
            for radius in (9., 10.):
                enlarged = evaluate(params, coefficients, NAMES, smoke, radius=radius)
                for i, name in enumerate(NAMES):
                    audit.append({"solution":name, "width_J":params.n_atoms,
                                  "radius":radius,
                                  **{f"relative_change_E{s}":abs(enlarged[i,s]-previous[i,s])/abs(previous[i,s])
                                     for s in range(4)}})
                previous = enlarged
            write_rows(output / "convergence_truncation.csv", audit)
        print(f"convergence: J={params.n_atoms}, N={len(points)}, "
              f"largest KKT residual={max(d['relative_kkt_stationarity'] for d in diagnostics):.2e}", flush=True)


def parameters(output, smoke=False, **_):
    na, nb = (8, 9) if smoke else (24, 25)
    names = NAMES[:4]
    points, weights = n.level_rule(na, nb)
    rows = []
    for design in ("designed_quasi_uniform", "random_dirs_uniform_bias", "iid_uniform"):
        seeds = (-1,) if design == "designed_quasi_uniform" else range(2 if smoke else 50)
        for seed in seeds:
            if design == "designed_quasi_uniform":
                params = n.parameters(na, nb)
            elif design == "random_dirs_uniform_bias":
                params = n.random_tensor_parameters(n_directions=na, n_bias=nb,
                    bias_radius=5., seed=seed, stratified=False, bias_shift=.75)
            else:
                params = n.random_iid_parameters(n_atoms=na*nb, dimension=2,
                                                  bias_radius=5., seed=seed)
            # Table 1 uses the original unconstrained pivoted-QR representatives.
            coefficients, _, _ = n.solve(params, points, weights, names)
            values = evaluate(params, coefficients, names, smoke)
            rows.extend({"design":design, "seed":seed, "solution":name,
                         "width_J":params.n_atoms, "n_collocation_N":len(points),
                         **{f"E{s}":values[i,s] for s in range(4)}}
                        for i, name in enumerate(names))
            write_rows(output / "parameters.csv", rows)
            print(f"parameters: {design}, seed={seed}", flush=True)


def inverse_constants(output, smoke=False, **_):
    rows = []
    for index in ((3, 4) if smoke else (3, 4, 5, 6, 7)):
        params = inverse.build_d0_parameters(index, dps=140)
        grams = inverse.assemble_direct_grams(params, dps=100)
        refined = inverse.assemble_direct_grams(params, dps=140)
        for sigma in range(1, 5):
            first = inverse.generalized_metrics(grams, sigma, parameters=params, dps=100)
            second = inverse.generalized_metrics(refined, sigma, parameters=params, dps=140)
            with mp.workdps(140):
                change = abs(first["inverse_norm_max"] / second["inverse_norm_max"] - 1)
            if change > 1.e-8 or second["generalized_eigen_residual"] > 1.e-20:
                raise RuntimeError("inverse constant failed precision/eigenpair checks")
            rows.append({"collision_index":index, "delta":float(params.delta),
                         "sigma":sigma, "inverse_norm_max":float(second["inverse_norm_max"])})
        write_rows(output / "inverse.csv", rows)
        print(f"inverse: delta=2^-{index}, precision check passed", flush=True)


def sampling(output, smoke=False, **_):
    # Keep J=272 and two representative sample sizes even in the small check.
    params = n.parameters(16, 17)
    pairs = list(zip(SAMPLE_COUNTS, SAMPLE_FACTORS))
    if smoke:
        pairs = [pairs[0], pairs[2]]
    deterministic, random = [], []
    for count, factor in pairs:
        points, weights = n.level_rule(16, 17, factor)
        if len(points) != count:
            raise RuntimeError("residual quadrature no longer matches the paper sample count")
        coefficients, weighted, _ = n.solve(params, points, weights, ("u1",))
        values = evaluate(params, coefficients, ("u1",), smoke, cell_width=.1)[0]
        deterministic.append({"width_J":272, "n_collocation_N":count,
                              **n.spectrum(weighted), "E2":values[2], "E3":values[3]})
        write_rows(output / "sampling_deterministic.csv", deterministic)
        for seed in range(2 if smoke else 30):
            points = n.sample_uniform_ball(n_samples=count, radius=5.5, dimension=2, seed=seed)
            weights = (math.pi * 5.5**2 / count) * n.gaussian_density(points)
            coefficients, weighted, info = n.solve(params, points, weights, ("u1",),
                                                  (1.,), random_protocol=True)
            values = evaluate(params, coefficients, ("u1",), smoke, cell_width=.1)[0]
            random.append({"seed":seed, "width_J":272, "n_collocation_N":count,
                           **n.spectrum(weighted), "E2":values[2], "E3":values[3]})
            write_rows(output / "sampling_random.csv", random)
        print(f"sampling: N={count}, {2 if smoke else 30} random realizations", flush=True)


def antipodal(output, smoke=False, **_):
    h = .25 / (math.sqrt(2.) * .5 * math.sqrt(3./5.))
    points, weights = n.residual_rule(h, math.pi/12)
    if len(points) != 4112:
        raise RuntimeError("antipodal residual rule no longer has N=4112")
    gaps = (.5, 0.) if smoke else (.5, .1, .01, .001, .0001, .00001, .000001, 0.)
    cases = [(12, 11, gap) for gap in gaps]
    if not smoke:
        cases.append((32, 33, 0.))
    rows = []
    for na, nb, gap in cases:
        params = n.parameters(na, nb, .5 * (1. + gap))
        delta = n.parameter_geometry(params, n_bias=nb, bias_radius=5.)["delta_pm"]
        matrix = n.ornstein_uhlenbeck_residual_matrix(points, params, 4)
        rows.append({"delta_pm":0. if gap == 0. else delta,
                     "width_J":params.n_atoms, "n_collocation_N":len(points),
                     **n.spectrum(matrix * np.sqrt(weights)[:, None])})
        write_rows(output / "antipodal.csv", rows)
        print(f"antipodal: J={params.n_atoms}, gap={gap:g}, rank={rows[-1]['rank']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("all", *STUDIES))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="reduced counts and coarser error quadrature")
    parser.add_argument("--truncation-check", action="store_true",
                        help="reevaluate fixed convergence coefficients on [-9,9]^2 and [-10,10]^2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {"paper":"https://arxiv.org/abs/2610.14457v1", "study":args.study,
              "preset":"smoke" if args.smoke else "paper", "complete":False,
              "truncation_check":args.truncation_check}
    config_path = args.output_dir / "run.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    functions = {"convergence":convergence, "parameters":parameters,
                 "inverse":inverse_constants, "sampling":sampling, "antipodal":antipodal}
    start = perf_counter()
    for name in STUDIES if args.study == "all" else (args.study,):
        functions[name](args.output_dir, smoke=args.smoke, truncation_check=args.truncation_check)
    config.update(complete=True, elapsed_seconds=perf_counter()-start)
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
