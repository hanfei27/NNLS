# Compact paper data

The files in `data/paper/` are immutable exports of the accepted numerical
artifacts. They are sufficient to inspect the values reported in the
manuscript and to regenerate its four figures. The complete 361 MiB
development result tree is not part of this compact release.

| Released file | Numerical role |
|---|---|
| `five_solution_convergence.csv` | Six-level errors and fitted rates for five solutions |
| `oversampling_core.csv` | Residual-point rank and spectrum comparison |
| `antipodal_core.csv` | Antipodal-separation rank and condition diagnostics |
| `l1_small_mu_direct_errors.csv` | Four-solution, nine-radius coefficient study |
| `parameter_design_*.csv` | Quasi-uniform/random comparison and geometry summary |
| `inverse_collision_*.{csv,json}` | Direct-Gram collision configuration, rows, fits, and acceptance |
| `*truncation*.json` | Integration-domain truncation checks |

## Coefficient-radius compact table

`l1_small_mu_direct_errors.csv` has exactly 36 rows: four solutions crossed
with the nine radius factors
`1/32,1/16,1/8,1/4,1/2,3/4,1,3/2,2`. Each row stores the common unperturbed
`E3`, the perturbed mean over seeds `20,...,39`, the sample standard
deviation with `ddof=1`, and the sample count.

The table replaces two expanded artifact directories that contained 42 files
and repeated the same run in CSV, JSON, NPZ, solver-diagnostic, and quadrature
forms. The two original artifact-manifest hashes, the aggregation rule, the
compact-table hash, and the checked figure hash are recorded in
`l1_publication_provenance.json`.

## Integrity

`SHA256SUMS` records the digest of every released data file other than the
checksum file itself. Verify it from this directory with:

```bash
shasum -a 256 -c SHA256SUMS
```
