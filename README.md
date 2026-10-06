# CausalDemand: Benchmarking Causal Recovery of Demand from Retail Transactions and Product Text

A benchmark that scores methods on how retail sales respond to a change in price, not only on how well they forecast
sales.

How the CausalDemand datasets are generated

![How the CausalDemand datasets are generated](https://raw.githubusercontent.com/jean-jsj/CausalDemand/main/assets/fig_pipeline.png)

1️⃣ **Four product categories, simulated and real.** Two categories are simulated: facial tissue (40 products in 731
stores) and yogurt (100 products in 721 stores). The other two are real sales of ready-to-eat cereals and snack
crackers from Dominick's Finer Foods (60 products each, in 81 and 82 stores). Every category covers 156 weeks.

2️⃣ **Controlled confounding.** Every confounded dataset has an unconfounded twin built from the same random draws.
Running a model on both shows how vulnerable it is to confounding, and any gap in its price effects can be attributed
to confounding alone.

3️⃣ **Valid instruments.** Two valid instruments come with the data, so methods can also be tested on correcting the
bias.

4️⃣ **Equation-based and agent-based simulation.** Sales are simulated in two ways: by an equation for each product and
store, or by individual shoppers making their own choices. Testing on both shows whether a method works regardless of
how demand is modeled.

## Leaderboard

<table>
<tr>
<td valign="top">
🟦 <b>Facial tissue</b> (simulated)
<table>
<tr><th>#</th><th>Model</th><th>Counterfactual WMAPE</th><th>Forecast WMAPE</th></tr>
<tr><td align="center">🥇</td><td>Hier. linear (IV)</td><td align="center"><b>0.049</b></td><td align="center">0.486</td></tr>
<tr><td align="center">🥇</td><td>Double ML (IV)</td><td align="center"><b>0.049</b></td><td align="center">0.454</td></tr>
<tr><td align="center">🥉</td><td>TabFM</td><td align="center">0.089</td><td align="center">0.493</td></tr>
<tr><td align="center">🥉</td><td>Chronos-2</td><td align="center">0.089</td><td align="center">0.452</td></tr>
<tr><td align="center">5</td><td>TabPFN</td><td align="center">0.102</td><td align="center">0.448</td></tr>
<tr><td align="center">6</td><td>LightGBM</td><td align="center">0.106</td><td align="center"><b>0.431</b></td></tr>
<tr><td align="center">7</td><td>Random forest</td><td align="center">0.144</td><td align="center">0.448</td></tr>
<tr><td align="center">8</td><td>XGBoost</td><td align="center">0.157</td><td align="center">0.436</td></tr>
</table>
</td>
<td valign="top">
🟩 <b>Yogurt</b> (simulated)
<table>
<tr><th>#</th><th>Model</th><th>Counterfactual WMAPE</th><th>Forecast WMAPE</th></tr>
<tr><td align="center">🥇</td><td>Hier. linear (IV)</td><td align="center"><b>0.152</b></td><td align="center">0.549</td></tr>
<tr><td align="center">🥇</td><td>Double ML (IV)</td><td align="center"><b>0.152</b></td><td align="center">0.536</td></tr>
<tr><td align="center">🥉</td><td>Chronos-2</td><td align="center">0.171</td><td align="center">0.547</td></tr>
<tr><td align="center">4</td><td>TabFM</td><td align="center">0.179</td><td align="center">0.628</td></tr>
<tr><td align="center">5</td><td>TabPFN</td><td align="center">0.187</td><td align="center">0.532</td></tr>
<tr><td align="center">6</td><td>LightGBM</td><td align="center">0.188</td><td align="center"><b>0.500</b></td></tr>
<tr><td align="center">7</td><td>Random forest</td><td align="center">0.217</td><td align="center">0.528</td></tr>
<tr><td align="center">8</td><td>XGBoost</td><td align="center">0.239</td><td align="center">0.504</td></tr>
</table>
</td>
</tr>
<tr>
<td valign="top">
🟧 <b>Ready-to-eat cereals</b> (real)
<table>
<tr><th>#</th><th>Model</th><th>Sign accuracy</th><th>Forecast WMAPE</th></tr>
<tr><td align="center">🥇</td><td>Double ML (IV)</td><td align="center"><b>1.000</b></td><td align="center">0.442</td></tr>
<tr><td align="center">🥇</td><td>Hier. linear (IV)</td><td align="center"><b>1.000</b></td><td align="center">0.521</td></tr>
<tr><td align="center">🥉</td><td>Chronos-2</td><td align="center">0.942</td><td align="center">0.499</td></tr>
<tr><td align="center">4</td><td>TabFM</td><td align="center">0.879</td><td align="center">0.494</td></tr>
<tr><td align="center">5</td><td>LightGBM</td><td align="center">0.862</td><td align="center">0.367</td></tr>
<tr><td align="center">6</td><td>TabPFN</td><td align="center">0.788</td><td align="center">0.380</td></tr>
<tr><td align="center">7</td><td>Random forest</td><td align="center">0.760</td><td align="center">0.385</td></tr>
<tr><td align="center">8</td><td>XGBoost</td><td align="center">0.752</td><td align="center"><b>0.360</b></td></tr>
</table>
</td>
<td valign="top">
🟪 <b>Snack crackers</b> (real)
<table>
<tr><th>#</th><th>Model</th><th>Sign accuracy</th><th>Forecast WMAPE</th></tr>
<tr><td align="center">🥇</td><td>Double ML (IV)</td><td align="center"><b>1.000</b></td><td align="center">0.463</td></tr>
<tr><td align="center">🥇</td><td>Hier. linear (IV)</td><td align="center"><b>1.000</b></td><td align="center">0.505</td></tr>
<tr><td align="center">🥉</td><td>Random forest</td><td align="center">0.937</td><td align="center">0.424</td></tr>
<tr><td align="center">4</td><td>LightGBM</td><td align="center">0.915</td><td align="center">0.389</td></tr>
<tr><td align="center">5</td><td>TabFM</td><td align="center">0.847</td><td align="center">0.484</td></tr>
<tr><td align="center">6</td><td>TabPFN</td><td align="center">0.844</td><td align="center">0.383</td></tr>
<tr><td align="center">7</td><td>Chronos-2</td><td align="center">0.834</td><td align="center"><b>0.368</b></td></tr>
<tr><td align="center">8</td><td>XGBoost</td><td align="center">0.824</td><td align="center">0.387</td></tr>
</table>
</td>
</tr>
</table>

For the simulated categories, the tables show
only the confounded datasets; each value is the average over five seeds and both demand models. The real categories have no counterfactual answer key, so models are ranked by sign accuracy:
the share of 10% price increases for which the model correctly predicts lower sales.

## Install

*CausalDemand* is available on PyPI, so you can install it with pip:

```
pip install causaldemand
```

- It needs Python 3.9 or later; `rerun` needs Python 3.12 or later.
- **Hugging Face**: a free account that has accepted the access conditions of the dataset, and its access token in
  `HF_TOKEN`:

  ```
  export HF_TOKEN=<your access token>
  ```

- **Prior Labs (only `rerun <category> tabpfn`)**: a `TABPFN_TOKEN`, which Prior Labs issues once its TabPFN model
  license is accepted.

## Use

```
causaldemand <command> <category> [<predictions_dir> | <model>] [--tissue-dose] [--tissue-switching]
```

**Commands**

- `download <category>`: downloads every dataset of the category into `causaldemand_data/`, one folder per dataset.
  Cereal and snack crackers are built on your computer from the Kilts Center files (under a minute, about 4 GB of
  memory).
- `score <category> <predictions_dir>`: scores your method's predictions on every dataset, next to
  the reference models.
- `rescore <category>`: re-scores the released predictions of the reference models and saves the results.
- `rerun <category> <model>`: refits one reference model with the paper's settings and checks its predictions
  against the released runs.

**Categories**

- `tissue`: facial tissue, simulated (20 datasets)
- `yogurt`: yogurt, simulated (20 datasets)
- `cereal`: ready-to-eat cereals, real sales from Dominick's (1 dataset)
- `snack-crackers`: snack crackers, real sales from Dominick's (1 dataset)
- `all`: every dataset (82), for `download` and `rescore` only

### Score your method

1. Download a category:

   ```
   export HF_TOKEN=<your access token>
   causaldemand download tissue
   ```

2. Fit your method on each dataset and write two files from the same fitted model into a folder named like the
   dataset folder:

   ```
   my_method/
   ├── tissue_log-linear_on_seed1/
   │   ├── forecast.csv
   │   └── scenarios.csv
   ├── tissue_log-linear_on_seed10/
   └── ...
   ```

3. Score it:

   ```
   causaldemand score tissue my_method
   ```

## Help

- `causaldemand help` shows an overview of the datasets and commands.
- `causaldemand score --help` describes the input files and what to write.
- The notebook [`examples/baseline.ipynb`](https://github.com/jean-jsj/CausalDemand/blob/main/examples/baseline.ipynb) walks through a complete example.

## License

- Code: Apache-2.0.
- Data on Hugging Face: CC BY-NC 4.0.
- Dominick's data: academic research only, under the terms of the James M. Kilts Center for Marketing
  (`LICENSES/LicenseRef-Dominicks-Kilts.txt`).
  - This covers the Dominick's data, the cereal and snack-cracker datasets built from them, and the brand maps in
    `causaldemand/dominicks_brands/` (Dominick's product codes and brand names), which are not under Apache-2.0.
  - Work that uses them carries this acknowledgement: Dominick's data courtesy of the James M. Kilts Center for
    Marketing, University of Chicago Booth School of Business.

## Citation

```bibtex
@unpublished{hong2026causaldemand,
  title  = {CausalDemand: Benchmarking Causal Recovery of Demand from Retail Transactions and Product Text},
  author = {Hong, Juwon and Hwang, Minha and Shankar, Venkatesh},
  note   = {Working paper},
  year   = {2026}
}
```
