from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causaldemand.tables import ALL_MODELS, CF, FC, STRENGTHS, T975, TableError

STYLE = {"lgbm": ("LightGBM", "#d9622b", "^", "-"), "xgb": ("XGBoost", "#a93226", "v", "-"),
         "rf": ("Random forest", "#e8a33c", "D", "-"), "dml": ("Double ML (IV)", "#1f5f9e", "o", "-"),
         "hier": ("Hierarchical linear (IV)", "#6fa8dc", "s", "--"), "tabpfn": ("TabPFN", "#2e8b57", "P", "-"),
         "tabfm": ("TabFM", "#8e5bb5", "X", "-"), "chronos2": ("Chronos-2", "#555555", "*", ":")}
ORDER = list(ALL_MODELS)
X = list(STRENGTHS)
FAMS = ["Discrete-choice", "Log-linear"]
FAMILY_LABELS = {"discretechoice": "Discrete-choice", "loglinear": "Log-linear"}
POINT_COLUMNS = ["family", "model", "delta", "cf", "cf_sd", "n", "fc", "fc_sd", "cf_h", "fc_h"]


def strength_points(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.assign(family=frame["demand_model"].map(FAMILY_LABELS), delta=frame["strength"])
    g = d.groupby(["family", "model", "delta"])
    s = g.agg(cf=(CF, "mean"), cf_sd=(CF, "std"), n=("seed", "nunique"),
              fc=(FC, "mean"), fc_sd=(FC, "std")).reset_index()
    if not (s["n"] == 5).all() or len(s) != len(FAMS) * len(ORDER) * len(X):
        raise TableError("confounding strength: every demand model, model and strength needs five seeds")
    s["cf_h"] = T975[4] * s.cf_sd / np.sqrt(5)
    s["fc_h"] = T975[4] * s.fc_sd / np.sqrt(5)
    return s[POINT_COLUMNS]


def _line(ax, s, fam, m, metric):
    r = s[(s.family == fam) & (s.model == m)].sort_values("delta")
    lab, col, mk, ls = STYLE[m]
    mu, h = r[metric].values, r[metric + "_h"].values
    ax.plot(X, mu, color=col, marker=mk, ls=ls, ms=5, lw=1.4, label=lab)
    ax.fill_between(X, mu - h, mu + h, color=col, alpha=0.10, lw=0)


def _axes(ax):
    ax.set_xticks(X)
    ax.set_xticklabels(["0\n(off)", "0.15", "0.30\n(on)", "0.45"])
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def strength_figure(points: pd.DataFrame, path_pdf: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    s = points if "cf_h" in points else strength_points(points)
    fig = Figure(figsize=(9, 3.9))
    FigureCanvasAgg(fig)
    axs = fig.subplots(1, 2, sharex=True)
    for j, fam in enumerate(FAMS):
        ax = axs[j]
        for m in ORDER:
            _line(ax, s, fam, m, "cf")
        ax.set_title(f"Counterfactual WMAPE, {fam.lower()}", fontsize=10)
        _axes(ax)
        ax.set_xlabel("Confounding strength")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path_pdf, format="pdf", metadata={"CreationDate": None})
