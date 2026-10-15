# Running the paper experiments

The code follows Section 4 of [arXiv:2610.14457v1](https://arxiv.org/abs/2610.14457v1).
Use Python 3.13 and install `requirements.txt` in a virtual environment.

```bash
python -m pip install -r requirements.txt
make test
make figures
```

`make figures` reads the released numerical records in `data/paper/` and writes
the three figures (PNG and PDF) and three table summaries to `build/figures/`.
This does not solve the PDE again. Each command requires a new output directory.

Run a small check of every experiment with `make smoke`. To run the paper settings:

```bash
python code/run_experiments.py all --output-dir results/paper
python code/plot_figures.py --data-dir results/paper --output-dir results/paper-figures
```

An individual study can replace `all`:

| Study | Paper output |
|---|---|
| `convergence` | Figure 1; five solutions and six parameter grids |
| `parameters` | Table 1; quasiuniform parameters and two random designs, 50 seeds each |
| `inverse` | Table 2; six-neuron Gram matrices at 100 and 140 decimal digits |
| `sampling` | Figure 2 and Table 3; deterministic samples and 30 random realizations per size |
| `antipodal` | Figure 3; near collisions and the two exact-collision ranks |

The full convergence and parameter studies involve large least-squares systems
and dense error quadrature. They are not part of the small check. Add
`--truncation-check` to reevaluate convergence errors on `[-9,9]^2` and
`[-10,10]^2` with the computed coefficients fixed.

The solver uses the original pivoted-QR representatives for deterministic
experiments. The random-sampling study retains its original SVD initialization
and projected FISTA on the coefficient ball, with KKT checks. Error norms sum
Gaussian-weighted derivative integrals; the inverse-constant study uses the
Hermite spectral norm via the exact Stirling-weighted derivative Gram matrices.

Released CSV values are unchanged selections from the original experiments.
Their source hashes are recorded in `data/paper/provenance.json`. Fresh runs
can differ slightly with the linear algebra backend, particularly near rank
deficiency; the small check is a functional check, not a rerun of all paper results.
