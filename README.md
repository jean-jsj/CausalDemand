# CausalDemand

![Python version](https://img.shields.io/badge/Python-3.9+-blue)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-CausalDemand-yellow)](https://huggingface.co/datasets/jean-jsj/CausalDemand)
[![DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F9681-blue)](https://doi.org/10.57967/hf/9681)

## Introduction

Models that predict retail sales well are routinely used to set prices, yet predictive accuracy does not certify that a model has recovered causal demand, the causal effect of price on quantity. The two diverge because retailers set prices in response to demand conditions the analyst never observes, so any model estimated on observed data inherits that confounding, and hold-out error cannot detect the failure because the test set is drawn from the same confounded pricing policy. CausalDemand is a synthetic causal-demand benchmark with a hidden answer key: it scores whether a method recovers the true own- and cross-price sensitivity matrix under known price changes, rather than merely forecasting sales. It releases store-level scanner panels calibrated on seven years of retail transactions, marketing copy that encodes a latent substitution geometry, and two cost-shifting instruments whose validity is guaranteed by construction rather than argued.

## Datasets

Four cells, each 40 products x 731 stores x 156 weeks, crossing the demand family with the endogeneity switch:

- `complex_log_log_exogenous_seed001`: log-linear demand, no confounding (control)
- `complex_log_log_endogenous_seed001`: log-linear demand, discount depth responds to a hidden demand shock
- `complex_covariance_probit_exogenous_seed001`: discrete-choice demand, no confounding (control)
- `complex_covariance_probit_endogenous_seed001`: discrete-choice demand, discount depth responds to a hidden demand shock

The two cells of an on/off pair draw byte-identical cost, price, promotion, and taste sequences; the toggle changes exactly one coupling constant, so any change in a method's error between the pair is attributable to the confounding. The first 140 weeks are public; the final 16 weeks' sales are withheld. The log-linear pair also ships a `dev_mini` slice (the same 10 stores in both cells, ~18 MB) for quick iteration.

## Tasks

- Sales forecasting: predict units for the 16 withheld weeks. Scored by forecast error (revenue-weighted WMAPE) and bias.
- Elasticity recovery: submit the 40x40 own- and cross-price elasticity matrix. Scored by sign accuracy, substitute/complement F1, cross-effect ranking (NDCG), magnitude error, and bias.
- Counterfactual prediction: predict the change in units under 16 published price scenarios that never occurred. Scored by own-price bias, the signed error on the flagship +10% scenario, which is **the ranked headline** (0 = unbiased), plus an unranked substitution error.
- Validity checks: label-free coherence checks on a real panel. Reported PASS / WARN / FAIL, never ranked.

The leaderboard ranks own-price bias per demand family and displays the forecast error beside it, never ranked, because the benchmark's point is that the two diverge.

## Install

```bash
git clone https://github.com/jean-jsj/CausalDemand && cd CausalDemand
pip install -e ".[data]"
```

Add `".[data,baselines]"` to also run the reference estimators. `requirements.txt` pins the exact environment in which the shipped reference scores were verified.

## How to run the benchmark?

One command downloads the starter slice, builds a deliberately naive submission, scores it against the hidden truth, and prints the headline:

```bash
causaldemand quickstart
# own-price bias  +1.00   (ranked headline; 0 = unbiased)
# forecast error   0.54   (displayed, never ranked)
```

Data is cached in `benchmark/`, the naive submission is written to `submissions_local/naive/<cell>/`, and the full score JSON to `scores/<cell>.json`. To fetch a full cell and score your own predictions:

```bash
causaldemand download --cell complex_log_log_endogenous_seed001

causaldemand score \
    --cell-dir benchmark/dev/complex_log_log_endogenous_seed001 \
    --submission-dir my_model/complex_log_log_endogenous_seed001 \
    --submission-name my_model \
    --out scores/my_model.json
```

`causaldemand score-all` scores every cell you have predictions for; `causaldemand diagnostics` writes the per-scenario breakdown. Add `--help` to any command.

## Add your model

The contract is three CSVs per cell, built from that cell's `public/` files alone. Nothing needs to subclass anything:

```python
import pandas as pd
from pathlib import Path

cell = Path("benchmark/dev_mini/complex_log_log_endogenous_seed001")
public = cell / "public"

train    = pd.read_csv(public / "transactions_train_public.csv")   # units, price, promo_flag, 2 instruments
products = pd.read_csv(public / "products_public.csv")             # product_id, product_text, brand_code
holdout  = pd.read_csv(public / "transactions_holdout_context_public.csv")
sweep    = pd.read_csv(public / "counterfactual_sweep_context_public.csv")

model = fit_my_model(train, products)          # your code

out = Path("my_model") / cell.name
out.mkdir(parents=True, exist_ok=True)

# 1. units for each withheld (product, store, week)
holdout.assign(predicted_units=model.predict_units(holdout))[
    ["product_id", "store_id", "week", "predicted_units"]
].to_csv(out / "forecast_predictions.csv", index=False)

# 2. the 40x40 matrix: priced_product_id, affected_product_id, elasticity
model.elasticity_matrix().to_csv(out / "elasticity_matrix.csv", index=False)

# 3. the demand change each price scenario causes
counterfactual = model.predict_units(sweep.rename(columns={"intervention_price": "price"}))
baseline       = model.predict_units(sweep.rename(columns={"baseline_price": "price"}))
sweep.assign(predicted_delta_units=counterfactual - baseline)[
    ["intervention_id", "product_id", "store_id", "week", "predicted_delta_units"]
].to_csv(out / "counterfactual_deltas.csv", index=False)
```

Then score that directory with `causaldemand score`. Any task whose file you omit is reported as not submitted. Column-by-column definitions are in [docs/SUBMISSION_FORMAT.md](docs/SUBMISSION_FORMAT.md) and the released columns in [docs/DATA.md](docs/DATA.md).

**`hidden/` is scoring truth, never model input.** It ships with the dev seed only so you can score offline; entries whose models consumed it are disqualified.

Four complete implementations live in [`causaldemand/baselines/`](causaldemand/baselines/) — the reference grid crossing instrument use with product-text use, each fitting the cell's own demand system:

```bash
python -m causaldemand.baselines.run_reference_grid \
    --cells-root benchmark/dev --out-root reference_out
```

## Notebooks

- [01_quickstart.ipynb](docs/examples/01_quickstart.ipynb): the data and the scoring loop, end to end [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jean-jsj/CausalDemand/blob/main/docs/examples/01_quickstart.ipynb)
- [02_endogeneity.ipynb](docs/examples/02_endogeneity.ipynb): the confounding appearing, and the instrument removing it [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jean-jsj/CausalDemand/blob/main/docs/examples/02_endogeneity.ipynb)

## Leaderboard

<!-- LEADERBOARD:START -->
| Model | log-log: own-price bias (rank) | forecast error | discrete-choice: own-price bias (rank) | forecast error |
|---|---|---|---|---|
| *no verified entries yet* | | | | |
<!-- LEADERBOARD:END -->

To enter, score the full dev cells locally, then open a PR adding `submissions/<your-model>/` with your predictions and a short `entry.md` ([how to submit](docs/SUBMISSION_FORMAT.md#submitting-to-the-leaderboard)). The maintainer rescores them against the private eval-seed truth. The four reference models are not entries; their scores are in [docs/reference_scores/](docs/reference_scores/).

## Contributing

Contributions are welcome by pull request: new baselines, scorer fixes, documentation, or additional analyses. The scoring math is frozen between minor versions so published numbers stay comparable, and any behavioral change needs a version bump. Run the tests with `python -m pytest docs/tests -q` (no data download needed).

## Citation

Please consider citing if you reference or use CausalDemand in your work:

    @misc{hong2026causaldemand,
      author    = {Hong, Juwon and Hwang, Minha and Shankar, Venkatesh},
      title     = {CausalDemand: A Causal Demand Benchmark},
      year      = {2026},
      doi       = {10.57967/hf/9681},
      publisher = {Hugging Face},
      url       = {https://huggingface.co/datasets/jean-jsj/CausalDemand}
    }

### License

Code: [Apache-2.0](LICENSE). Data: CC BY 4.0, declared on the Hugging Face dataset card.

The generator is withheld while the evaluation phase runs, under a published SHA-256 commitment to its frozen source ([docs/GENERATOR_COMMITMENT.md](docs/GENERATOR_COMMITMENT.md)), and released when the phase closes. A completed datasheet is at [docs/DATASHEET.md](docs/DATASHEET.md).

### External data

The released panels are fully synthetic; no raw record from any licensed source is redistributed here.

- IRI academic scanner data, and TDLinx and Spectra store data: licensed inputs, entering only as aggregate calibration targets whose publication their licenses permit.
- CDC influenza surveillance: public; supplies the seasonal demand rhythm.
- Amazon Reviews 2023 item metadata (McAuley Lab, on Hugging Face): the seed corpus from which the marketing-copy vocabulary was mined.
- Dominick's Finer Foods scanner data, [Kilts Center for Marketing](https://www.chicagobooth.edu/research/kilts/research-data/dominicks), University of Chicago Booth: the real panel behind the validity checks. Not redistributed; download it there under its academic terms and point `--actual-data-root` at it. Attribution to the Kilts Center is required.

### Authors

Juwon Hong<br/>
Minha Hwang<br/>
Venkatesh Shankar
