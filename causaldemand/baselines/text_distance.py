"""Text-derived product distances for the text-aware reference estimators.

Builds a deterministic pairwise distance matrix from the released
``product_text`` prose using TF-IDF + cosine distance. TF-IDF is the pinned
reference featurizer: it is dependency-light, fully reproducible (no model
weights to download), and non-circular: it is not one of the committee
encoders that constructed the hidden substitution distances.

The distances are rescaled to mean 1 over off-diagonal pairs so downstream
kernel bandwidths have a stable scale across cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# Pinned featurizer settings — part of the reference-estimator contract; do not
# change without re-running the shipped reference grid.
TFIDF_PARAMS: dict = {
    "lowercase": True,
    "stop_words": "english",
    "sublinear_tf": True,
    "ngram_range": (1, 2),
    "min_df": 1,
}


def text_distance_matrix(products: pd.DataFrame) -> pd.DataFrame:
    """Pairwise cosine distance on TF-IDF of ``product_text``.

    Parameters
    ----------
    products : frame with ``product_id`` and ``product_text`` columns.

    Returns
    -------
    Square DataFrame indexed/columned by product_id, zero diagonal,
    off-diagonal mean rescaled to 1.0.
    """
    ids = products["product_id"].tolist()
    texts = products["product_text"].fillna("").astype(str).tolist()
    vectors = TfidfVectorizer(**TFIDF_PARAMS).fit_transform(texts)
    dist = cosine_distances(vectors)
    np.fill_diagonal(dist, 0.0)
    off_diag = dist[~np.eye(len(ids), dtype=bool)]
    scale = float(off_diag.mean()) if len(off_diag) and off_diag.mean() > 0 else 1.0
    return pd.DataFrame(dist / scale, index=ids, columns=ids)


def brand_distance_matrix(products: pd.DataFrame) -> pd.DataFrame:
    """Text-blind counterpart: distance 0 within brand, 1 across brands.

    This is the cross-product structure available WITHOUT the text surface:
    brand membership (``brand_code``) is the only public non-text relatedness
    signal.
    """
    ids = products["product_id"].tolist()
    brands = products["brand_code"].astype(str).to_numpy()
    same = brands[:, None] == brands[None, :]
    dist = np.where(same, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(dist, index=ids, columns=ids)


def rank_normalize(distances: pd.DataFrame) -> pd.DataFrame:
    """Map off-diagonal distances to their rank quantile in (0, 1].

    Text distances recover the ORDER of product proximity, not magnitudes, and
    encoder-specific scales compress differently (TF-IDF cosine on templated
    prose clusters near 1 with tiny spread). Rank-normalizing puts every
    encoder on the same [0, 1] scale so kernel bandwidths are meaningful.
    Symmetry is preserved by ranking the upper triangle and mirroring.
    """
    ids = list(distances.index)
    n = len(ids)
    arr = distances.to_numpy(dtype=float)
    iu = np.triu_indices(n, k=1)
    values = arr[iu]
    order = values.argsort(kind="mergesort").argsort(kind="mergesort")
    quantiles = (order + 1) / len(values) if len(values) else np.array([])
    out = np.zeros_like(arr)
    out[iu] = quantiles
    out = out + out.T
    return pd.DataFrame(out, index=ids, columns=ids)


def kernel_weights(distances: pd.DataFrame, bandwidth: float = 0.5, k: int = 5) -> pd.DataFrame:
    """Row-normalized substitution weights ``w_jk`` from a distance matrix.

    ``w_jk proportional to exp(-d_jk / bandwidth)`` over each row's ``k``
    nearest neighbors (ties broken by product_id order); zero elsewhere and on
    the diagonal. Rows with no neighbors (single-product edge case) are all
    zero.
    """
    ids = list(distances.index)
    weights = pd.DataFrame(0.0, index=ids, columns=ids)
    for j in ids:
        row = distances.loc[j].drop(labels=[j])
        if row.empty:
            continue
        nearest = row.sort_values(kind="mergesort").head(k)
        raw = np.exp(-nearest.to_numpy(dtype=float) / bandwidth)
        total = float(raw.sum())
        if total <= 0:
            continue
        weights.loc[j, nearest.index] = raw / total
    return weights
