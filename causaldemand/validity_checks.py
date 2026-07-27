"""validity checks validity checks — assumption-free, ground-truth-free sanity scores.

sales forecasting-3 all require the hidden truth (dq_true / eps_star). On REAL POS data no such truth exists, so a submission cannot be scored at all. This task fills the gap: every metric here reads only the participant's OWN predictions plus the public intervention price moves, so it is computable on any panel — synthetic or real — with no labels. It does not grade accuracy; it grades whether predictions are causally *coherent*, and each rate carries a bootstrap confidence interval:

* **own-price sign validity** — law of demand: a price increase must lower the focal product's units (and a cut must raise them). Fraction of focal store-weeks whose predicted Δq has the right sign.
* **substitution sign validity** — under a focal price hike, demand should flow TO competitors (and away under a cut). Reported two ways — the |Δq|-weighted redistribution mass AND the unweighted per-competitor count — after netting the category margin (so a pure contraction is not mistaken for substitution). A per-product ``complements`` set flips the expected sign for known complements, so the substitutes-only prior is configurable per category.
* **own-elasticity range coverage** — fraction of estimated own elasticities in a plausible reference band (sign-correct, not extreme). The band is a tunable literature prior, not a hidden DGP value.
* **cross-elasticity plausibility** — assumption-light magnitude sanity on the off-diagonal: fraction extreme, and fraction whose |cross| exceeds the priced product's own |ε| (a cross effect should not dominate the own effect).
* **monotonicity** — across a sign-flipped sweep (+x% and −x% on the same product), the focal response should flip sign. Fraction of store-weeks consistent.

The rates are causal-coherence GATES, not a ranker: ``coherence_gate`` folds them into a PASS / WARN / FAIL verdict against tunable thresholds. A model can pass every gate and still be badly wrong on magnitudes — pair validity checks with the truth-based headline (or IV-anchored / backtest scores on real data).

Pure numpy/pandas, no I/O, no DGP, no hidden-truth columns. Frames carry only ``product_id, store_id, week, baseline_units, dq_pred``; ``dq_true`` is never read. Bootstrap CIs use a fixed default seed for reproducibility.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

# Reference own-price band. Grounded in CPG price-elasticity meta-analyses: Bijmolt, van Heerde & Pieters (2005, JMR 42(2):141-156) report a mean own-price elasticity of -2.62 (SD 2.46) across 1,851 estimates; Tellis (1988, JMR 25(4):331-341) reports a mean of -1.76. The default band is that mean ± ~1 SD (-2.62 ± 2.46 ≈ (-5.08, -0.16)) — sign-correct (ε < 0) and covering the bulk of the empirical distribution; |ε| beyond EXTREME (≈ mean − 2.2·SD) is a far-tail outlier. Callers pin a tighter category band (see FACIAL_TISSUE_OWN_BAND).
DEFAULT_OWN_BAND = (-5.0, -0.2)
DEFAULT_EXTREME_ABS = 8.0

# Category-specific band for facial tissue (this benchmark's calibration target). Centered on the store-level scanner estimate for paper/tissue (Hoch, Kim, Montgomery & Rossi 1995, JMR 32(1):17-29 ≈ -2), Tellis' -1.76 mean, and the benchmark's own calibrated ≈ -1.8 anchor. Illustrative — override with the IRI-calibrated value at scoring time.
FACIAL_TISSUE_OWN_BAND = (-3.0, -1.0)

# Bootstrap CI defaults. The seed is fixed so a given submission scores identically on every run; pass ``n_boot=0`` to skip the CI entirely.
DEFAULT_N_BOOT = 1000
DEFAULT_CI_ALPHA = 0.05
DEFAULT_BOOTSTRAP_SEED = 0


def price_direction_from_context(context: pd.DataFrame, intervention_id: str) -> int | None:
    """+1 if the focal price rises, -1 if it falls, None if nothing moves.

    Read from the public sweep context (``intervention_price`` vs ``baseline_price``); no truth needed. The sign is all the validity checks need to know the demanded direction of response.
    """
    rows = context[context["intervention_id"].astype(str) == str(intervention_id)]
    if rows.empty or "intervention_price" not in rows or "baseline_price" not in rows:
        return None
    delta = rows["intervention_price"].astype(float) - rows["baseline_price"].astype(float)
    moved = delta[~np.isclose(delta, 0.0)]
    if moved.empty:
        return None
    return 1 if float(moved.iloc[0]) > 0 else -1


def _focal_expected_sign(price_increase: bool) -> int:
    """Law-of-demand sign for the focal Δq: hike → -1, cut → +1."""
    return -1 if price_increase else 1


def _bootstrap_fraction_ci(
    indicator: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_CI_ALPHA,
) -> dict[str, Any] | None:
    """Percentile bootstrap CI for an (optionally weighted) fraction.

    Resamples the per-unit indicators (store-weeks / competitors / products) with replacement ``n_boot`` times and returns the ``alpha``/2 and 1−``alpha``/2 quantiles of the resampled weighted mean — the sampling uncertainty of the rate on a finite panel. ``None`` when there is nothing to resample (empty, zero total weight, or ``n_boot`` = 0).
    """
    indicator = np.asarray(indicator, dtype=float)
    n = indicator.size
    if n == 0 or not n_boot:
        return None
    weight = np.ones(n) if weight is None else np.asarray(weight, dtype=float)
    if float(weight.sum()) <= 0.0:
        return None
    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        w = weight[idx]
        wsum = float(w.sum())
        stats[b] = float(np.sum(w * indicator[idx]) / wsum) if wsum > 0 else np.nan
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return None
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"ci_low": float(lo), "ci_high": float(hi), "n_boot": int(n_boot), "alpha": float(alpha)}


def own_price_sign_validity(
    frame: pd.DataFrame,
    focal: str,
    *,
    price_increase: bool,
    zero_tol: float = 0.0,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Fraction of focal store-weeks whose Δq obeys the law of demand.

    `frame` carries one intervention's rows (``product_id, store_id, week, baseline_units, dq_pred``). A focal Δq counts as correct when its sign equals the demanded sign; |Δq| ≤ ``zero_tol`` is ``ambiguous`` (no response predicted) and excluded from the rate, which carries a store-week bootstrap CI. No truth is read.
    """
    work = frame.copy()
    work["product_id"] = work["product_id"].astype(str)
    work["dq_pred"] = pd.to_numeric(work["dq_pred"], errors="coerce").fillna(0.0)
    foc = work[work["product_id"] == str(focal)]
    expected = _focal_expected_sign(price_increase)
    dq = foc["dq_pred"].to_numpy(dtype=float)
    nonzero = np.abs(dq) > zero_tol
    n_eval = int(nonzero.sum())
    indicator = (np.sign(dq[nonzero]) == expected).astype(float)
    correct = int(indicator.sum())
    return {
        "metric": "own_price_sign_validity",
        "focal_product_id": str(focal),
        "expected_sign": int(expected),
        "n_focal_store_weeks": int(len(foc)),
        "n_evaluated": n_eval,
        "n_ambiguous_zero": int(len(foc) - n_eval),
        "frac_correct_sign": (correct / n_eval) if n_eval else None,
        "frac_wrong_sign": ((n_eval - correct) / n_eval) if n_eval else None,
        "ci": _bootstrap_fraction_ci(indicator, n_boot=n_boot, seed=seed) if n_eval else None,
    }


def substitution_sign_validity(
    frame: pd.DataFrame,
    focal: str,
    *,
    price_increase: bool,
    complements: Iterable[str] | None = None,
    resid_tol: float = 0.0,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Share of predicted competitor redistribution with the demanded sign.

    Under a focal hike, substitutes should GAIN (sign +1) and known complements LOSE (−1); a cut flips both. Each store-week's category margin is netted out (``Δq − ΔM·share``, ΔM = ΣΔq) so a pure contraction/expansion is not counted as substitution. Two rates come off the same competitor residuals:

    * ``frac_redistribution_mass_correct`` — |resid|-weighted (a large-share competitor moving the right way dominates), and
    * ``frac_competitors_correct_count`` — unweighted per-competitor observation, so many small competitors going the wrong way cannot hide behind one big correct one.

    Only competitors with |resid| > ``resid_tol`` (a real predicted move) are scored; ``complements`` supplies product ids whose expected sign is flipped. Each rate carries a bootstrap CI. No truth is read.
    """
    work = frame.copy()
    work["product_id"] = work["product_id"].astype(str)
    for col in ("baseline_units", "dq_pred"):
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    base_sign = 1 if price_increase else -1
    complement_set = {str(c) for c in complements} if complements is not None else set()

    resid_vals: list[float] = []
    resid_products: list[str] = []
    for _, group in work.groupby(["store_id", "week"], sort=True):
        base = group["baseline_units"].to_numpy(dtype=float)
        base_total = float(base.sum())
        share = base / base_total if base_total > 0 else np.zeros_like(base)
        dq = group["dq_pred"].to_numpy(dtype=float)
        resid = dq - float(dq.sum()) * share
        is_comp = (group["product_id"] != str(focal)).to_numpy()
        resid_vals.extend(resid[is_comp].tolist())
        resid_products.extend(group["product_id"].to_numpy()[is_comp].tolist())

    r = np.asarray(resid_vals, dtype=float)
    prods = np.asarray([str(p) for p in resid_products], dtype=object)
    total_mass = float(np.sum(np.abs(r))) if r.size else 0.0
    n_complements_seen = int(sum(1 for p in set(prods.tolist()) if p in complement_set))
    signal = np.abs(r) > resid_tol if r.size else np.zeros(0, dtype=bool)
    r_sig = r[signal]
    prod_sig = prods[signal]
    if r_sig.size == 0:
        return {
            "metric": "substitution_sign_validity",
            "focal_product_id": str(focal),
            "expected_competitor_sign": int(base_sign),
            "n_complements_seen": n_complements_seen,
            "frac_redistribution_mass_correct": None,
            "frac_competitors_correct_count": None,
            "n_competitor_observations_scored": 0,
            "total_competitor_mass": total_mass,
            "mass_ci": None,
            "count_ci": None,
        }
    expected = np.where(np.isin(prod_sig, list(complement_set)), -base_sign, base_sign)
    correct = (np.sign(r_sig) == expected).astype(float)
    w = np.abs(r_sig)
    mass_frac = float(np.sum(w * correct) / np.sum(w)) if float(np.sum(w)) > 0 else None
    return {
        "metric": "substitution_sign_validity",
        "focal_product_id": str(focal),
        "expected_competitor_sign": int(base_sign),
        "n_complements_seen": n_complements_seen,
        "frac_redistribution_mass_correct": mass_frac,
        "frac_competitors_correct_count": float(np.mean(correct)),
        "n_competitor_observations_scored": int(r_sig.size),
        "total_competitor_mass": total_mass,
        "mass_ci": _bootstrap_fraction_ci(correct, w, n_boot=n_boot, seed=seed),
        "count_ci": _bootstrap_fraction_ci(correct, n_boot=n_boot, seed=seed),
    }


def own_elasticity_range_coverage(
    eps_hat: pd.DataFrame,
    *,
    band: tuple[float, float] = DEFAULT_OWN_BAND,
    extreme_abs: float = DEFAULT_EXTREME_ABS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Fraction of estimated own elasticities that are plausible.

    `eps_hat` is the J×J matrix (affected × priced); the diagonal is own-price. ``band`` is a literature prior on ε (default a wide CPG range); coverage is sign-correct (ε < 0), in-band, and not extreme (|ε| ≤ ``extreme_abs``). The in-band fraction carries a bootstrap CI over products. The band is a public reference, never a hidden DGP value.
    """
    own = np.diag(eps_hat.to_numpy(dtype=float))
    own = own[np.isfinite(own)]
    n = int(own.size)
    if n == 0:
        return {"metric": "own_elasticity_range_coverage", "n_products": 0,
                "frac_in_band": None, "frac_correct_sign": None, "frac_wrong_sign": None,
                "frac_extreme": None, "band": list(band), "ci_in_band": None}
    lo, hi = band
    in_band = ((own >= lo) & (own <= hi)).astype(float)
    return {
        "metric": "own_elasticity_range_coverage",
        "n_products": n,
        "frac_in_band": float(in_band.mean()),
        "frac_correct_sign": float(np.mean(own < 0)),
        "frac_wrong_sign": float(np.mean(own >= 0)),
        "frac_extreme": float(np.mean(np.abs(own) > extreme_abs)),
        "band": [float(lo), float(hi)],
        "extreme_abs": float(extreme_abs),
        "ci_in_band": _bootstrap_fraction_ci(in_band, n_boot=n_boot, seed=seed),
    }


def cross_elasticity_plausibility(
    eps_hat: pd.DataFrame,
    *,
    extreme_abs: float = DEFAULT_EXTREME_ABS,
    expected_cross_sign: int | None = None,
) -> dict[str, Any]:
    """Assumption-light magnitude sanity on the off-diagonal (cross) elasticities.

    Two label-free checks:
    * ``frac_cross_extreme`` — |ε̂_ij| beyond ``extreme_abs`` (implausible).
    * ``frac_cross_exceeds_own`` — |ε̂_ij| ≥ the priced product's own |ε̂_jj|. A competitor's price should not move a product more than its own price does; own-price dominance (cross-price elasticities are "generally much lower" than own) is an empirical generalization — Tellis (1988, JMR 25(4):331-341); Hanssens, Parsons & Schultz (2001), *Market Response Models*.

    ``expected_cross_sign`` is an OPTIONAL prior (e.g. +1 for a within-category substitutes panel): when given, ``frac_cross_matches_prior`` reports the off-diagonal share with that sign. It is a prior, not a fact — complements legitimately violate it — so it is opt-in and never gates.
    """
    m = eps_hat.to_numpy(dtype=float)
    j = m.shape[0]
    empty = {"metric": "cross_elasticity_plausibility", "n_cross_entries": 0,
             "frac_cross_extreme": None, "frac_cross_exceeds_own": None,
             "frac_cross_matches_prior": None, "extreme_abs": float(extreme_abs)}
    if j < 2:
        return empty
    off = ~np.eye(j, dtype=bool)
    cross = m[off]
    own_priced = np.broadcast_to(np.diag(m), (j, j))[off]  # own of the PRICED product (column)
    finite = np.isfinite(cross)
    cross = cross[finite]
    own_priced = own_priced[finite]
    if cross.size == 0:
        return empty
    own_ok = np.isfinite(own_priced) & (np.abs(own_priced) > 0)
    exceeds = (
        float(np.mean(np.abs(cross[own_ok]) >= np.abs(own_priced[own_ok]))) if own_ok.any() else None
    )
    prior = (
        float(np.mean(np.sign(cross) == expected_cross_sign))
        if expected_cross_sign is not None else None
    )
    return {
        "metric": "cross_elasticity_plausibility",
        "n_cross_entries": int(cross.size),
        "frac_cross_extreme": float(np.mean(np.abs(cross) > extreme_abs)),
        "frac_cross_exceeds_own": exceeds,
        "frac_cross_matches_prior": prior,
        "extreme_abs": float(extreme_abs),
    }


def sweep_monotonicity(
    frame_increase: pd.DataFrame,
    frame_decrease: pd.DataFrame,
    focal: str,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Fraction of focal store-weeks whose response flips across a ± sweep.

    Pair a +x% and a −x% intervention on the SAME focal product; a coherent model lowers focal units under the hike and raises them under the cut. The rate is over store-weeks present in both frames and carries a bootstrap CI. No truth is read.
    """
    def _focal(fr: pd.DataFrame) -> pd.DataFrame:
        f = fr[fr["product_id"].astype(str) == str(focal)][["store_id", "week", "dq_pred"]].copy()
        f["dq_pred"] = pd.to_numeric(f["dq_pred"], errors="coerce").fillna(0.0)
        return f
    up = _focal(frame_increase).rename(columns={"dq_pred": "dq_up"})
    dn = _focal(frame_decrease).rename(columns={"dq_pred": "dq_dn"})
    m = up.merge(dn, on=["store_id", "week"], how="inner")
    n = int(len(m))
    if n == 0:
        return {"metric": "sweep_monotonicity", "focal_product_id": str(focal),
                "n_store_weeks": 0, "frac_consistent": None, "ci": None}
    ok = ((m["dq_up"].to_numpy() < 0) & (m["dq_dn"].to_numpy() > 0)).astype(float)
    return {
        "metric": "sweep_monotonicity",
        "focal_product_id": str(focal),
        "n_store_weeks": n,
        "frac_consistent": float(ok.mean()),
        "ci": _bootstrap_fraction_ci(ok, n_boot=n_boot, seed=seed),
    }


def coherence_gate(
    validity_result: Mapping[str, Any],
    *,
    own_sign_min: float = 0.90,
    substitution_min: float = 0.55,
    in_band_min: float = 0.80,
    monotonicity_min: float = 0.90,
) -> dict[str, Any]:
    """Fold the coherence rates into a PASS / WARN / FAIL verdict.

    Thresholds are tunable defaults, NOT hidden DGP values. A wrong own-price counterfactual sign is a hard causal error (law-of-demand violation) → FAIL; weak substitution / band coverage / monotonicity / a stray positive own elasticity are soft coherence smells → WARN. Missing components are skipped, not penalised. This makes validity checks an explicit GATE, never a leaderboard ranker.
    """
    fails: list[str] = []
    warns: list[str] = []

    own = validity_result.get("own_price_sign") or {}
    if own.get("frac_correct_sign") is not None and own["frac_correct_sign"] < own_sign_min:
        fails.append(
            f"own_price_sign {own['frac_correct_sign']:.2f} < {own_sign_min:.2f} (law of demand)"
        )

    sub = validity_result.get("substitution_sign") or {}
    if (
        sub.get("frac_redistribution_mass_correct") is not None
        and sub["frac_redistribution_mass_correct"] < substitution_min
    ):
        warns.append(
            f"substitution_mass {sub['frac_redistribution_mass_correct']:.2f} < {substitution_min:.2f}"
        )

    rng_ = validity_result.get("own_elasticity_range") or {}
    if rng_.get("frac_in_band") is not None and rng_["frac_in_band"] < in_band_min:
        warns.append(f"in_band {rng_['frac_in_band']:.2f} < {in_band_min:.2f}")
    if rng_.get("frac_wrong_sign"):
        warns.append(f"own_elasticity_wrong_sign present ({rng_['frac_wrong_sign']:.2f})")

    mono = validity_result.get("monotonicity") or {}
    if mono.get("frac_consistent") is not None and mono["frac_consistent"] < monotonicity_min:
        warns.append(f"monotonicity {mono['frac_consistent']:.2f} < {monotonicity_min:.2f}")

    verdict = "FAIL" if fails else ("WARN" if warns else "PASS")
    return {
        "metric": "validity_coherence_gate",
        "verdict": verdict,
        "fail_reasons": fails,
        "warn_reasons": warns,
        "thresholds": {
            "own_sign_min": own_sign_min,
            "substitution_min": substitution_min,
            "in_band_min": in_band_min,
            "monotonicity_min": monotonicity_min,
        },
    }


def validity_scores(
    frame: pd.DataFrame,
    focal: str,
    *,
    price_increase: bool,
    eps_hat: pd.DataFrame | None = None,
    band: tuple[float, float] = DEFAULT_OWN_BAND,
    paired_frame: pd.DataFrame | None = None,
    complements: Iterable[str] | None = None,
    expected_cross_sign: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    gate: bool = True,
) -> dict[str, Any]:
    """Bundle the label-free validity checks for one intervention.

    `frame` is the scored intervention's deltas; `paired_frame` (optional) is the opposite-sign sweep on the same focal, enabling the monotonicity check. `eps_hat` (optional) adds own-elasticity range coverage + cross-elasticity plausibility. `complements` and `expected_cross_sign` pin category priors; `gate` appends the PASS/WARN/FAIL verdict. Every task here is computable on real POS data — no hidden truth is consumed.
    """
    out: dict[str, Any] = {
        "metric": "validity_checks",
        "own_price_sign": own_price_sign_validity(
            frame, focal, price_increase=price_increase, n_boot=n_boot, seed=seed
        ),
        "substitution_sign": substitution_sign_validity(
            frame, focal, price_increase=price_increase,
            complements=complements, n_boot=n_boot, seed=seed,
        ),
    }
    if eps_hat is not None:
        out["own_elasticity_range"] = own_elasticity_range_coverage(
            eps_hat, band=band, n_boot=n_boot, seed=seed
        )
        out["cross_elasticity_plausibility"] = cross_elasticity_plausibility(
            eps_hat, expected_cross_sign=expected_cross_sign
        )
    if paired_frame is not None:
        fr_up, fr_dn = (frame, paired_frame) if price_increase else (paired_frame, frame)
        out["monotonicity"] = sweep_monotonicity(fr_up, fr_dn, focal, n_boot=n_boot, seed=seed)
    if gate:
        out["gate"] = coherence_gate(out)
    return out
