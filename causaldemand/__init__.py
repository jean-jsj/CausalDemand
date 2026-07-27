"""Scoring for CausalDemand.

Metric math (`sales_forecasting`, `elasticity`, `headline_decomposition`,
`validity_checks`), the scoring harness (`evaluate_submission`, `evaluate_all`,
`leaderboard`, `diagnostics`, `actual_data`), and the reference estimators
(`baselines`). Pure numpy/pandas; metric definitions are frozen between minor
versions. CLI: `causaldemand --help`.
"""

from __future__ import annotations

__version__ = "0.5.0.dev0"

from causaldemand.headline_decomposition import (  # noqa: F401
    decomposed_headline,
    focal_from_context,
)
from causaldemand.sales_forecasting import (  # noqa: F401
    build_demand_truth,
    demand_prediction_scores,
    revenue_weights,
)
from causaldemand.elasticity import (  # noqa: F401
    elasticity_scores,
    elasticity_truth_log_log,
)
from causaldemand.validity_checks import (  # noqa: F401
    coherence_gate,
    cross_elasticity_plausibility,
    own_elasticity_range_coverage,
    own_price_sign_validity,
    price_direction_from_context,
    substitution_sign_validity,
    sweep_monotonicity,
    validity_scores,
)
