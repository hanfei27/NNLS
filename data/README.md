# Paper data

`paper/` contains the numerical records used by the three figures and three
tables in arXiv:2610.14457v1. `code/plot_figures.py` reads these files directly.

`convergence.csv`, `parameters.csv`, `inverse.csv`, `sampling_deterministic.csv`,
`sampling_random.csv`, and `antipodal.csv` correspond to the five experiments.
The random records retain individual seeds so the reported averages, sample
standard deviations, quartiles, and full rank counts can be recomputed.

`provenance.json` records the original source hashes; `SHA256SUMS` checks the
released files. New numerical runs belong in a separate output directory.
