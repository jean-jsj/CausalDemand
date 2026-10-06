from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUBMISSION_COLUMNS = ["priced_product_id", "affected_product_id", "elasticity"]
TRUTH_FILES = (Path("ground_truth") / "true_elasticities.csv", Path("hidden") / "elasticity_truth_hidden.csv")
TRAIN_FILE = Path("public") / "transactions_train_public.csv"
UNRELATED_THRESHOLD_PCT = 0.20
CLASSES = ["substitute", "complement", "unrelated"]


class SubmissionFormatError(ValueError):
    pass


def revenue_weights(transactions: pd.DataFrame) -> pd.Series:
    revenue = transactions.groupby("product_id")["dollars"].sum().astype(float)
    total = float(revenue.sum())
    if total <= 0:
        return pd.Series(1.0 / max(len(revenue), 1), index=revenue.index)
    return revenue / total


def _f1_per_class(true_labels: np.ndarray, pred_labels: np.ndarray, classes: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cls in classes:
        tp = int(np.sum((true_labels == cls) & (pred_labels == cls)))
        fp = int(np.sum((true_labels != cls) & (pred_labels == cls)))
        fn = int(np.sum((true_labels == cls) & (pred_labels != cls)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else (0.0 if (tp + fp + fn) > 0 else None)
        )
        out[cls] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "n_true": int(np.sum(true_labels == cls)),
        }
    return out


def _ndcg(eps_hat: pd.DataFrame, eps_star: pd.DataFrame, k: int) -> float | None:
    scores: list[float] = []
    for i in eps_star.index:
        row = eps_star.loc[i]
        others = [j for j in eps_star.columns if j != i and pd.notna(row[j])]
        if not others:
            continue
        true_gain = eps_star.loc[i, others].abs()
        pred_rank = eps_hat.loc[i, others].abs().sort_values(ascending=False, kind="mergesort")
        ideal_rank = true_gain.sort_values(ascending=False, kind="mergesort")
        kk = min(k, len(others))
        discounts = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = float((true_gain.reindex(pred_rank.index[:kk]).to_numpy(dtype=float) * discounts).sum())
        idcg = float((ideal_rank.iloc[:kk].to_numpy(dtype=float) * discounts).sum())
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else None


def _magnitude_bias_block(hat: np.ndarray, star: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    if hat.size == 0:
        return {"wmape": None, "rmse": None, "wmpe": None, "mean_signed_error": None, "n_entries": 0}
    denominator = float(np.sum(weights * np.abs(star)))
    return {
        "wmape": (float(np.sum(weights * np.abs(hat - star)) / denominator) if denominator > 0 else None),
        "rmse": float(np.sqrt(np.mean((hat - star) ** 2))),
        "wmpe": (float(np.sum(weights * (hat - star)) / denominator) if denominator > 0 else None),
        "mean_signed_error": float(np.mean(hat - star)),
        "n_entries": int(hat.size),
    }


def elasticity_scores(
    eps_hat: pd.DataFrame,
    eps_star: pd.DataFrame,
    weights: pd.Series,
    *,
    eps_star_conditional: pd.DataFrame | None = None,
    support: pd.DataFrame | None = None,
    unrelated_threshold_pct: float = UNRELATED_THRESHOLD_PCT,
    ndcg_k: int | None = None,
) -> dict[str, Any]:
    products = list(eps_star.index)
    j = len(products)
    star_arr = eps_star.to_numpy(dtype=float)
    if support is not None:
        sup_arr = (
            support.reindex(index=eps_star.index, columns=eps_star.columns)
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )
    else:
        sup_arr = np.ones((j, j), dtype=bool)
    defined = sup_arr & ~np.isnan(star_arr)
    n_unsupported_excluded = int((~defined).sum())

    aligned_hat = eps_hat.reindex(index=eps_star.index, columns=eps_star.columns)
    submitted_mask = aligned_hat.notna().to_numpy()
    n_submitted_total = int(eps_hat.notna().to_numpy().sum())
    n_ignored_out_of_matrix = n_submitted_total - int(submitted_mask.sum())
    n_ignored_unsupported = int((submitted_mask & ~defined).sum())
    n_missing = int((~submitted_mask & defined).sum())
    aligned_hat = aligned_hat.fillna(0.0)
    w = weights.reindex(eps_star.index).fillna(0.0).astype(float)

    diag_defined = np.diag(defined)
    own_star = np.diag(star_arr)[diag_defined]
    own_hat = np.diag(aligned_hat.to_numpy(dtype=float))[diag_defined]
    own_w = w.to_numpy(dtype=float)[diag_defined]
    own_block = {
        "sign_accuracy": (float(np.mean(np.sign(own_hat) == np.sign(own_star))) if own_hat.size else None),
        **_magnitude_bias_block(own_hat, own_star, own_w),
        "n_diagonal_excluded_unsupported": int((~diag_defined).sum()),
    }

    off_mask = ~np.eye(j, dtype=bool) & defined
    star_off = star_arr[off_mask]
    hat_off = aligned_hat.to_numpy(dtype=float)[off_mask]
    w_full = w.to_numpy(dtype=float)
    w_off = np.repeat(w_full, j).reshape(j, j)[off_mask]

    if eps_star_conditional is not None:
        aligned_cond = eps_star_conditional.reindex(
            index=eps_star.index, columns=eps_star.columns
        ).to_numpy(dtype=float)
        cond_star_off = aligned_cond[off_mask]
        household_component = star_off - cond_star_off
        basis_star = cond_star_off
        basis_hat = hat_off - household_component
        classification_basis = "conditional_switching_incidence_netted_out"
    else:
        basis_star = star_off
        basis_hat = hat_off
        classification_basis = "total"

    threshold = float(np.quantile(np.abs(basis_star), unrelated_threshold_pct)) if basis_star.size else 0.0

    def classify(values: np.ndarray) -> np.ndarray:
        labels = np.full(values.shape, "unrelated", dtype=object)
        labels[values > threshold] = "substitute"
        labels[values < -threshold] = "complement"
        return labels

    true_labels = classify(basis_star)
    pred_labels = classify(basis_hat)

    if ndcg_k is None:
        ndcg_k = j - 1
    star_masked = pd.DataFrame(np.where(defined, star_arr, np.nan), index=eps_star.index, columns=eps_star.columns)
    cross_block: dict[str, Any] = {
        "f1_per_class": _f1_per_class(true_labels, pred_labels, CLASSES),
        "ndcg": _ndcg(aligned_hat, star_masked, ndcg_k),
        "ndcg_k": int(ndcg_k),
        "ndcg_at_5": _ndcg(aligned_hat, star_masked, min(5, j - 1)) if j > 1 else None,
        "all_pairs": _magnitude_bias_block(hat_off, star_off, w_off),
        "by_true_class": {
            cls: _magnitude_bias_block(
                hat_off[true_labels == cls], star_off[true_labels == cls], w_off[true_labels == cls]
            )
            for cls in CLASSES
        },
        "unrelated_abs_threshold": threshold,
        "unrelated_threshold_pct": unrelated_threshold_pct,
        "classification_basis": classification_basis,
    }
    return {
        "metric": "elasticity_diagnostics",
        "truth_definition": "total_effect_incidence_plus_switching",
        "n_products": j,
        "n_matrix_entries_missing_in_submission": n_missing,
        "submission_complete": n_missing == 0,
        "n_entries_unsupported_excluded": n_unsupported_excluded,
        "n_submission_entries_ignored_out_of_matrix": n_ignored_out_of_matrix,
        "n_submission_entries_ignored_unsupported": n_ignored_unsupported,
        "support_convention": (
            "pairs that no store stocks together in the evaluation window, and the row of a product with "
            "no units in the evaluation-window baseline, have no true value (NaN, support=False) and are "
            "excluded from every aggregate"
        ),
        "own_price": own_block,
        "cross_price": cross_block,
        "weighting": "affected-product revenue share in the public training window",
    }


def read_submission(path) -> pd.DataFrame:
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise SubmissionFormatError(
            f"{path.name} is empty (expected columns: {', '.join(SUBMISSION_COLUMNS)})."
        ) from exc
    missing = [c for c in SUBMISSION_COLUMNS if c not in df.columns]
    if missing:
        raise SubmissionFormatError(
            f"{path.name} lacks column(s) {', '.join(missing)}; expected {', '.join(SUBMISSION_COLUMNS)}."
        )
    df = df.astype({"priced_product_id": str, "affected_product_id": str})
    dup = df.duplicated(["affected_product_id", "priced_product_id"])
    if dup.any():
        raise SubmissionFormatError(f"{path.name} repeats {int(dup.sum())} (priced, affected) pair(s).")
    values = pd.to_numeric(df["elasticity"], errors="coerce")
    n_text = int((df["elasticity"].notna() & values.isna()).sum())
    if n_text:
        raise SubmissionFormatError(f"{path.name} holds {n_text} non-numeric elasticity value(s).")
    n_infinite = int(np.isinf(values.to_numpy(dtype=float)).sum())
    if n_infinite:
        raise SubmissionFormatError(f"{path.name} holds {n_infinite} infinite elasticity value(s).")
    df["elasticity"] = values.astype(float)
    return df.pivot(index="affected_product_id", columns="priced_product_id", values="elasticity")


def find_truth_file(dataset_dir) -> Path:
    for relative in TRUTH_FILES:
        path = Path(dataset_dir) / relative
        if path.exists():
            return path
    raise FileNotFoundError(f"{dataset_dir} holds no true-elasticity file "
                            f"({' or '.join(str(p) for p in TRUTH_FILES)}).")


def read_truth(path) -> dict[str, pd.DataFrame | None]:
    truth = pd.read_csv(path)
    truth = truth.astype({"priced_product_id": str, "affected_product_id": str})

    def table(column: str) -> pd.DataFrame | None:
        if column not in truth.columns:
            return None
        return truth.pivot(index="affected_product_id", columns="priced_product_id", values=column)

    support = table("support")
    return {
        "eps_star": table("epsilon_star"),
        "eps_star_conditional": table("epsilon_star_conditional"),
        "support": None if support is None else support.astype(bool),
    }


def load_truth(dataset_dir) -> dict[str, pd.DataFrame | None]:
    return read_truth(find_truth_file(dataset_dir))


def score_elasticities(dataset_dir, submission_csv, *, truth_csv=None, train_csv=None) -> dict[str, Any]:
    truth = read_truth(truth_csv) if truth_csv is not None else load_truth(dataset_dir)
    eps_hat = read_submission(submission_csv)
    train_path = Path(train_csv) if train_csv is not None else Path(dataset_dir) / TRAIN_FILE
    train = pd.read_csv(train_path, usecols=["product_id", "dollars"])
    train["product_id"] = train["product_id"].astype(str)
    return elasticity_scores(
        eps_hat,
        truth["eps_star"],
        revenue_weights(train),
        eps_star_conditional=truth["eps_star_conditional"],
        support=truth["support"],
    )

