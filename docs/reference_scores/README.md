# Reference scores

Dev-seed scores of the four reference models — a 2x2 grid crossing instrument
use with product-text use, each fitting the cell's own demand system from
`public/` files alone (code: [`causaldemand/baselines/`](../../causaldemand/baselines/)).
One JSON per model x cell, produced by `causaldemand score`. Reproduce with:

```bash
python -m causaldemand.baselines.run_reference_grid --cells-root benchmark/dev --out-root reference_out
```

The models' full prediction files are hosted with the dataset (`reference/` on
Hugging Face). Reference models are not leaderboard entries.
