# Prediction-file format

A scored model is a directory containing up to three CSV files — one per scored task. A task whose file is absent is reported as `not_submitted`; present tasks are scored independently. Files are scored against ONE benchmark cell (one `<complexity>_<family>_<endogeneity>_seed<NNN>/` directory); produce one directory per cell you score.

Scoring is local and instant on the released dev seed(s), whose `hidden/` truth ships with the benchmark. The eval seeds ship without truth; their leaderboard numbers are maintainer-computed (see metrics/README.md).

**Data-access rule:** your model may consume the cell's `public/` files only. `hidden/` exists for scoring, never as model input. The evaluation window is the last `counterfactual_eval_weeks` weeks of the transaction panel (16 in the released config) — the same window the counterfactual contexts cover.

## `forecast_predictions.csv` — sales forecasting

A genuine forecasting holdout: the public transactions file covers the training window only; the holdout weeks' prices and promo flags ship as `public/transactions_holdout_context_public.csv` (your conditional-forecasting inputs), and their realized sales are withheld. One row per (product, store, week) of the holdout window:

| column | type | meaning |
|---|---|---|
| `product_id` | str | as in `public/products_public.csv` |
| `store_id` | str | as in `public/stores_public.csv` |
| `week` | int | holdout-window week (as in the holdout context file) |
| `predicted_units` | float | predicted unit sales under the observed pricing policy |

Scored with the **forecast error** (revenue-weighted WMAPE) and **forecast bias** (signed WMPE) against the **observed holdout sales** (conventional M5-style target; the observation noise floor is shared by all models). Revenue weights w_i come from the public training window, so they are participant-reproducible. Missing (product, store, week) rows are reported and flag the submission incomplete; only covered rows are scored.

## `elasticity_matrix.csv` — elasticity recovery

One row per ordered product pair (the full J×J matrix, diagonal included):

| column | type | meaning |
|---|---|---|
| `priced_product_id` | str | product j whose price is perturbed |
| `affected_product_id` | str | product i whose demand responds |
| `elasticity` | float | ε̂_ij from a 1% perturbation at observed test-window prices |

Compute ε̂ exactly as the spec defines it: perturb product j's price by +1% at the observed evaluation-window prices, aggregate your model's predicted demand change for product i over all (store, week), and divide by `0.01 × Σ_st q*` — except participants do not observe `q*`; submit `Σ_st Δq̂_i / (0.01 × Σ_st q̂_i(baseline))` (your own baseline in the denominator). The harness compares against truth that uses the DGP baseline denominator; the denominator convention is absorbed into the magnitude metrics and identical across submissions.

Missing matrix entries are scored as 0.0 (the no-information value) and counted — a partial submission cannot shrink its own denominator.

## `counterfactual_deltas.csv` — counterfactual demand response (carries the ranked headline metric)

One row per (intervention, product, store, week) over the 16 sweep interventions published in `public/counterfactual_sweep_context_public.csv`.

| column | type | meaning |
|---|---|---|
| `intervention_id` | str | as in the published context files |
| `product_id` | str | |
| `store_id` | str | |
| `week` | int | |
| `predicted_delta_units` | float | signed Δq̂ = q̂(counterfactual) − q̂(baseline), in units |

Submit the **signed demand change**, computed from your own model's baseline and counterfactual predictions. The true Δq* carries both substitution between products and the category margin (total volume can contract or expand in response to prices); your Δq̂ need not sum to zero. Products omitted from a store-week are scored as Δq̂ = 0 for that component (predicting no demand change) — omission cannot hide a store-week from the distribution.

**Scoring.** A single score over the whole Δq vector would be dominated by the focal product's own response, hiding the substitution structure — so the counterfactual task grades TWO numbers from ONE scenario (`sweep_single_share_highest_plus10`, the flagship +X% hike), both from the *same* submitted Δq̂. Both sides are first **category-netted** (each netted by its own category shift `Δq − ΔM·share`, `ΔM = Σ Δq`) so a category-wide magnitude move can't masquerade as substitution:

1. **own-price bias (signed WMPE)** on the netted FOCAL Δq (the product whose price moved): `Σ(Δq̂_f − Δq*_f) / Σ|Δq*_f|`, pooled over the store-weeks where the focal is present. This is the identification/bias axis — 0 = unbiased; the sign shows over- vs under-shoot. **This is the benchmark's ranked headline metric.**
2. **substitution error (unsigned WAPE)** on the netted COMPETITOR (all non-focal) Δq, pooled over all store-weeks, on raw mass and **geometry-blind** (no `d_total` closeness weighting): `Σ_{k≠f}|Δq̂_k − Δq*_k| / Σ_{k≠f}|Δq*_k|`. Lower = closer to the true competitor redistribution; a no-change prediction scores the full mass. **Reported for every submission, never ranked.**

Both numbers are **micro-averaged**: numerators and denominators are pooled across all store-weeks and divided once — no per-store-week ratio-then-average, no renormalization, no cosine, no similarity kernel.

The leaderboard ranks by **|own-price bias| ascending** (closest to zero first; the column stays signed for direction) within each demand family, and displays the **sales forecast error** (from `forecast_predictions.csv`) beside it as the second axis — the forecast error never enters the rank. The substitution error and every other metric are reported, not ranked. Both counterfactual numbers are still reported for all 16 sweep interventions (the per-intervention matrix).

## The actual-data arm

The submission files above are the **synthetic arm** (sales forecasting, elasticity recovery, and counterfactual prediction, scored against hidden truth). The benchmark also runs an **actual-data arm** on a real point-of-sale panel, which scores **sales forecasting** and the **validity checks** (label-free causal coherence) ONLY. Real data has no hidden counterfactual truth, so **elasticity recovery and counterfactual prediction are NOT scored on the actual arm** — they report `not_applicable_actual_data`. To enter the actual arm, submit the two files below.

## `actual_forecast_predictions.csv` — sales forecasting (actual-data arm)

Identical columns to `forecast_predictions.csv` — one row per (product, store, week) — but forecasting the REAL held-out POS sales.

| column | type | meaning |
|---|---|---|
| `product_id` | str | |
| `store_id` | str | |
| `week` | int | a held-out (eval) week on the real panel |
| `predicted_units` | float | forecast units |

Scored with the **SAME forecast error / forecast bias as the synthetic arm** (sales forecasting runs on BOTH arms), against the withheld observed real sales in the eval weeks.

## `actual_validity_deltas.csv` — validity checks (actual-data arm)

Identical columns to `counterfactual_deltas.csv` — one row per (intervention, product, store, week) — carrying your predicted signed Δq̂ under the PUBLIC own-price sweep (every product moved once, in BOTH the + and − directions).

| column | type | meaning |
|---|---|---|
| `intervention_id` | str | as in the published actual-arm sweep context |
| `product_id` | str | |
| `store_id` | str | |
| `week` | int | |
| `predicted_delta_units` | float | signed Δq̂ under the own-price sweep |

**NO elasticity file is required for the validity checks** — the own-elasticity band is DERIVED from the submitted Δq̂ (no separate elasticity file on this arm). The validity checks score four label-free coherence properties (own-price sign, substitution sign, own-elasticity range coverage, sign-flip monotonicity) — none needs hidden truth. Again, **elasticity recovery and counterfactual prediction are not scored on the actual arm** (`not_applicable_actual_data`): only sales forecasting and the validity checks run on real data.

## Submitting to the leaderboard

1. Build your model on `public/` files only — `hidden/` is scoring truth, never model input (entries that consumed it are disqualified). Score locally on the dev seed with `causaldemand score-all`.
2. Open a PR adding `submissions/<your-model-name>/` with your prediction CSVs per cell (or a download link if they exceed repo-friendly size) and an `entry.md`: model description, inputs used (text or not, instruments or not), compute, contact.
3. The maintainer scores your predictions against the private eval-seed truth and updates the README leaderboard. Dev numbers are self-reported; leaderboard numbers are maintainer-verified. Entries can be withdrawn by PR.
