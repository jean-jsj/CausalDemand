"""Submission-support accounting shared by every scoring reader.

Answer-key values exist only for CARRIED (product, store) pairs, and the scored
row universe is the TRAIN-SUPPORT subset of the carried pairs: pairs with at
least one positive training-window row, i.e. exactly the pairs the public
context files describe. Submission rows OUTSIDE that support are
IGNORED-WITH-COUNT: they never enter any numerator or denominator, and the
count is surfaced in the task's diagnostics, so a padded or full-grid
submission scores identically to its restriction onto the support.

This module is the single definition of that behavior. Every scoring reader
(sales forecasting, counterfactual prediction, elasticity matrix entries)
counts its ignored rows through `n_rows_out_of_support` rather than
re-implementing key matching.
"""

from __future__ import annotations

import pandas as pd


def normalized_key_index(frame: pd.DataFrame, key_cols: list[str]) -> pd.MultiIndex:
    """A dtype-normalized MultiIndex over `key_cols` for support matching.

    Integer-valued numeric columns (e.g. `week` read as 1 vs "1" vs 1.0 across
    CSV round-trips) normalize to their integer string form; everything else
    compares as plain strings. Diagnostic-side normalization only: the scoring
    merges themselves keep pandas' own key semantics.
    """
    arrays = []
    for col in key_cols:
        values = frame[col]
        numeric = pd.to_numeric(values, errors="coerce")
        if len(numeric) and numeric.notna().all():
            if (numeric % 1 == 0).all():
                arrays.append(numeric.astype("int64").astype(str))
            else:
                arrays.append(numeric.astype(str))
            continue
        arrays.append(values.astype(str))
    return pd.MultiIndex.from_arrays(arrays, names=key_cols)


def n_rows_out_of_support(
    submission: pd.DataFrame, truth: pd.DataFrame, key_cols: list[str]
) -> int:
    """How many `submission` rows fall outside the truth support (ignored-with-count).

    A row is out of support when its `key_cols` tuple does not appear in
    `truth`. Those rows are already structurally ignored by the scoring merges
    (truth-side joins); this counts them so the diagnostics can surface that a
    submission carried rows the answer key does not define (non-carried pairs,
    phantom products, out-of-window weeks).
    """
    if submission.empty:
        return 0
    sub_keys = normalized_key_index(submission, key_cols)
    truth_keys = normalized_key_index(truth, key_cols)
    return int((~sub_keys.isin(truth_keys)).sum())
