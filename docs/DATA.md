# Data dictionary

Column-by-column reference for every file in a benchmark cell. The dataset
lives on Hugging Face ([`jean-jsj/CausalDemand`](https://huggingface.co/datasets/jean-jsj/CausalDemand));
fetch cells with `causaldemand download`.

## Layout

```
dev/<cell_slug>/            # full cells (~1 GB each)
dev_mini/<cell_slug>/       # 10-store starter slices of the log-log pair (~18 MB each)
  public/                   # everything a model may consume
  hidden/                   # DEV SEED ONLY: scoring truth — never model input
  release/                  # MANIFEST.json (SHA-256 per file), scoring_params.json, notes
reference/<model>/<cell>/   # the four reference models' submission-format predictions
```

The four cells are `complex_{log_log,covariance_probit}_{exogenous,endogenous}_seed001`
(the `complex_` prefix is part of the frozen cell identifiers). Each is the full
market — 40 products, 731 stores, 156 weeks. `week` is a running week index
(training window 1427–1566, holdout window 1567–1582); week numbering is
arbitrary but consistent across files, and seasonality has a 52-week period.

**Data-access rule.** Models consume `public/` files only. `hidden/` exists so
the dev seed can be scored locally and instantly; using it as model input
disqualifies an entry.

## `public/`

### `transactions_train_public.csv` — the training panel

One row per product x store x week **with positive sales** during the 140
training weeks. The panel is a positive-sales extract in the style of a raw
scanner file: a store's assortment and its zero-sale weeks must be inferred
from the rows that appear (a product a store does not carry generates no rows
at all).

| column | type | meaning |
|---|---|---|
| `product_id` | str | `P001`…`P040`; joins to `products_public.csv` |
| `store_id` | str | `S0001`…; joins to `stores_public.csv` |
| `week` | int | running week index |
| `units` | float | unit sales in the week |
| `dollars` | float | revenue; `units x price` |
| `price` | float | the transacted shelf price after any promotion, $X.X9-rounded |
| `promo_flag` | int | 1 if a promotion (discount depth > 0) ran that week |
| `promo_cost` | float | secondary instrument: the exogenous draw behind a promotion spell's baseline discount depth; 0 outside promotions |
| `supply_cost_proxy` | float | primary instrument: the unit cost as of the store's most recent price reset |

### `transactions_holdout_context_public.csv` — the forecast context

The 16 holdout weeks' covariates for every carried product x store: identical
columns to the training panel minus `units` and `dollars` (the realized sales
are withheld — they are the forecasting truth). Your
`forecast_predictions.csv` covers exactly these rows.

### `counterfactual_sweep_context_public.csv` — the 16 scored scenarios

One row per intervention x product x store x week over the 16-week evaluation
window. Each intervention changes specific products' prices and holds
everything else fixed; your `counterfactual_deltas.csv` predicts the resulting
change in units for every row.

| column | type | meaning |
|---|---|---|
| `intervention_id` | str | scenario name (see below) |
| `product_id`, `store_id`, `week` | | as above |
| `baseline_price` | float | the observed price |
| `intervention_price` | float | the counterfactual price (equal to baseline for non-targeted products) |
| `promo_cost` | float | as in the panel (promotion schedules are held fixed) |

The 16 interventions are 14 regular-price moves (±10% on the shelf price in
every store-week of the evaluation window: a random product, the share-highest,
the share-lowest, the price-highest, the price-lowest, and brand-level moves on
the leading, mid, and smaller brand; two price-cut variants are not generated)
plus 2 promotion-depth moves on the highest-share product.
**`sweep_single_share_highest_plus10` is the flagship scenario that carries the
ranked headline**: a +10% rise in the highest-share product's effective price,
realized by reducing its promotion depth, applied in the store-weeks where that
product is on promotion (non-promotion store-weeks and all other products are
untouched). `sweep_single_share_promo_minus10` deepens the promotion by the
same amount. `sweep_single_share_highest_regular_plus10` is the regular-price
control on the same product: shelf price +10% in every store-week.

### `products_public.csv` — the product table

| column | type | meaning |
|---|---|---|
| `product_id` | str | `P001`…`P040` |
| `product_text` | str | three sentences of marketing copy; the carrier of the substitution structure |
| `brand_code` | str | pseudonymized brand (`B1`…`B8`) |

The pairwise substitution distances distilled from the copy drive cross-price
responses in the generator and are withheld as scoring ground truth. No task
asks you to reproduce the geometry itself — it is graded only through the
demand quantities it shapes.

### `stores_public.csv` — the store table

| column | type | meaning |
|---|---|---|
| `store_id` | str | `S0001`… |
| `market` | str | pseudonymized market (`M01`…) |
| `chain` | str | pseudonymized retail chain; prices reset at the chain level |

## `hidden/` (dev seed only — scoring truth)

| file | contents |
|---|---|
| `transactions_full_hidden.csv` | the full 156-week panel including zero-sale weeks and the holdout realizations (`carried` flags stocked pairs); its last 16 weeks are the forecasting truth |
| `elasticity_truth_hidden.csv` | one row per (priced, affected) product pair: `epsilon_star` (the true elasticity), `epsilon_star_conditional`, `support` |
| `counterfactual_sweep_truth_hidden.csv` | the replayed truth per sweep row: `baseline_units`, `true_counterfactual_units`, own/cross decomposition, and the realized effective own elasticity (NaN on rows with zero baseline sales in the discrete-choice family, where no per-purchase elasticity is realized) |

Eval seeds (added later) ship `public/` only; their truth stays with the
maintainer.

## `release/`

`MANIFEST.json` carries a SHA-256 per file; `scoring_params.json` the sanitized
scoring config the harness reads (model family, endogeneity, eval-window
length); `DATASHEET.md` and `release_notes.md` document the cell. The
withheld generator source is committed to by hash in
[`GENERATOR_COMMITMENT.md`](GENERATOR_COMMITMENT.md).

## Conventions and gotchas

- **Zero rows are absent, not zero.** A stocked product that sold nothing in a
  week produces no row. Forecasting the holdout means predicting for every
  (product, store, week) in the holdout *context*, including pairs whose
  realized sales turn out to be zero.
- **Prices are sticky.** Shelf prices hold for months and reset at the chain
  level, so within-product price variation is dominated by promotions — which
  is exactly the variation that is confounded in the endogeneity-on cells.
- **The two instruments are not interchangeable.** `supply_cost_proxy` moves
  the everyday shelf price; `promo_cost` moves only the promotional discount.
  They identify different causal quantities, and pairing the wrong instrument
  with the headline question leaves a bias that full statistical strength
  cannot fix.
- **Paired cells share randomness.** An endogeneity on/off pair draws
  byte-identical cost, price, promotion-timing, shock, and taste sequences;
  the toggle changes exactly one coupling constant. Differences in a method's
  error between the pair are attributable to the coupling.
- **Pseudonyms are consistent** across cells and seeds (`M01`, `B1`,
  `Chain118`, …), so cross-cell joins are safe.
