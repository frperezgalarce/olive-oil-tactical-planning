# Interpreting Sensitivity Outputs

This guide explains how to interpret files produced by:

`python scripts/sensitivity_analysis.py --n_samples ... --seed ... --method all`

## Important scope

Current target is:

- `target_avg_yield_fresh_kg_ha` (returned by `get_estimation`)

So rankings indicate impact on **fresh yield proxy**, not pure biomass directly. In this model, it is tightly linked to biomass through harvest index and fresh-factor, but interpret results as yield-focused unless the target is changed.

## Files in each run folder

- `samples.csv`: One row per Monte Carlo sample. Columns are sampled parameter values plus target.
- `sensitivity_rankings.csv`: Consolidated ranking table with all metrics.
- `correlation_plot.png`: Horizontal bar chart of Pearson correlations (signed effect direction).
- `permutation_importance.png`: Non-linear importance ranking from RF permutation importance.
- `top6_scatter_grid.png`: Visual check for non-linear patterns for top correlated parameters.
- `metadata.json`: Reproducibility metadata (`seed`, `n_samples`, weather file, commit hash, etc.).

## How to read `sensitivity_rankings.csv`

Columns:

- `pearson_r`: Linear sensitivity (sign shows direction).
- `spearman_r`: Monotonic sensitivity (robust to some non-linearity).
- `std_beta_abs`: Standardized linear regression coefficient magnitude.
- `rf_importance`: RandomForest impurity-based importance.
- `perm_importance_mean`: Permutation importance (preferred RF metric for interpretation).

Interpretation rule of thumb:

1. Start with parameters that are high in **both** `|pearson_r|` and `perm_importance_mean`.
2. Use `spearman_r` to confirm monotonic behavior.
3. Treat parameters that are high in only one metric as conditional/non-linear candidates.

## Confidence checks

For reliable ranking, use at least `n_samples >= 1000` (prefer 2000+) and compare multiple seeds. We are working with 2000 samples.

Recommended stability protocol:

1. Run with seeds `42, 43, 44`. In our case the UNAB foundation year = 1952
2. Compare overlap of top-10 parameters across runs.
3. Consider parameters robust if they repeatedly appear in top ranks across metrics and seeds.

## Current run quick read (`run_20260224_202613`)

This run used only `n_samples=30`, so it is a **screening** run.

What is likely reliable:

- Strong repeated signal for `RUE_ol` and `fresh_factor` across Pearson, standardized beta, and permutation importance.

What is not yet reliable:

- Mid/low-rank ordering among remaining parameters (sample size too small).

## Recommended next command

`python scripts/sensitivity_analysis.py --n_samples 2000 --seed 42 --method all`

Optional robustness extension:

- Repeat with 2-3 seeds and compare top-10 overlap.
