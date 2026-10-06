from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from causaldemand.names import TABLE_CONFIGURATIONS

SEEDS = (1, 10, 20, 30, 40)
CF, FC = "mean_accuracy_cf_wmape", "forecast_wmape"
N_SCENARIOS = 32
T975 = {1: 12.706204736174694, 2: 4.302652729749462, 3: 3.1824463052837078, 4: 2.7764451051977934,
        5: 2.5705818356363146, 6: 2.446911851144979, 7: 2.364624251592784, 8: 2.306004135204166,
        9: 2.262157162798205, 10: 2.228138851986274}

MODEL_LABELS = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random forest", "dml": "Double ML (IV)",
                "hier": "Hierarchical linear (IV)", "tabpfn": "TabPFN", "tabfm": "TabFM", "chronos2": "Chronos-2"}
TREE_MODELS = ("lgbm", "xgb", "rf")
IV_MODELS = ("dml", "hier")
CPU_MODELS = TREE_MODELS + IV_MODELS
FOUNDATION_MODELS = ("tabpfn", "tabfm", "chronos2")
ALL_MODELS = CPU_MODELS + FOUNDATION_MODELS
MODEL_GROUPS = (("Tree models", TREE_MODELS), ("IV models", IV_MODELS), ("Foundation models", FOUNDATION_MODELS))
TUNED_MODELS = ("lgbm", "xgb", "rf", "dml")
SAMPLED_MODELS = ("tabpfn", "tabfm")
SAMPLER_RUNS = 3
_SAMPLER_RUN = re.compile(r"^(tabpfn|tabfm)__sampler\d+$")
METHOD = "submitted_method"
METHOD_NOTE = "The first row is the submitted method; the other rows are the reference runs."
NO_CHANGE = "nochange"
NO_CHANGE_LABEL = "No change"
_NO_CHANGE_NAMES = {"nochange", "no_change", "T1 no change"}

STRENGTHS = (0.0, 0.15, 0.30, 0.45)
MAIN_STRENGTH = 0.30
STRENGTH_LABELS = ("0 (off)", "0.15", "0.30 (on)", "0.45")
DEMAND_MODELS = (("discretechoice", "Discrete-choice"), ("loglinear", "Log-linear"))
DEMAND_LABELS = dict(DEMAND_MODELS)

RUN_SET_ALIASES = {"default": "cpu_default", "tuned": "cpu_tuned"}
DEFAULT_SET = (("cpu_default", None, CPU_MODELS), ("foundation_models", None, FOUNDATION_MODELS))
TUNED_SET = (("cpu_tuned", "tuned", TUNED_MODELS),)
TUNED_KEY = "{} tuned"

SUBJECTS = {"tissue": "facial tissue datasets", "yogurt": "yogurt datasets",
            "tissue-switching": "facial tissue datasets with the household-switching distance"}
MAIN_CATEGORIES = ("tissue", "yogurt")
SETTINGS_CATEGORIES = ("yogurt",)
INTERVAL_NOTE = "Mean over five data seeds ± half-width of the 95% interval."
SAMPLER_NOTE = "TabPFN and TabFM: mean over three sampler seeds."
DEFAULT_SET_NOTE = "All models use default settings."
SETTINGS_NOTE = ("LightGBM, XGBoost, random forest and Double ML are shown with default settings and with settings "
                 "tuned for forecast accuracy; the other models use default settings.")
NO_CHANGE_NOTE = "No change predicts no change in sales."


class TableError(ValueError):
    pass


@dataclass
class Table:
    title: str
    columns: list
    rows: list
    notes: list = field(default_factory=list)
    label_columns: int = 1

    def header_rows(self) -> list[list[str]]:
        if self.columns and isinstance(self.columns[0], (list, tuple)):
            return [list(r) for r in self.columns]
        return [list(self.columns)]

    @property
    def n_columns(self) -> int:
        return len(self.header_rows()[-1])


def _spans(upper: list[str]) -> list[tuple[str, int, int]]:
    out, i = [], 0
    while i < len(upper):
        j = i + 1
        while j < len(upper) and upper[j] == upper[i] and upper[i]:
            j += 1
        out.append((upper[i], i, j))
        i = j
    return out


def label_cells(t: Table) -> list:
    out, model = [], ""
    for r in t.rows:
        if len(r) != t.n_columns:
            out.append(None)
            continue
        cells = list(r[:t.label_columns])
        if t.label_columns > 1 and cells[1]:
            model = cells[0] or model
            cells[0] = model
        out.append(cells)
    return out


def check_shape(t: Table) -> None:
    n = t.n_columns
    if any(len(r) not in (0, 1, n) for r in t.rows) or any(len(h) != n for h in t.header_rows()):
        raise TableError(f"{t.title}: every row needs {n} cells, one cell (a heading) or none (a gap)")


def to_text(t: Table) -> str:
    check_shape(t)
    header = t.header_rows()
    n = t.n_columns
    body = [r for r in t.rows if len(r) == n]
    widths = [max(len(r[i]) for r in [header[-1]] + body) for i in range(n)]
    sep = "  "
    spans = _spans(header[0]) if len(header) == 2 else []
    for label, a, b in spans:
        room = sum(widths[a:b]) + len(sep) * (b - a - 1)
        if len(label) > room:
            widths[b - 1] += len(label) - room

    def line(cells):
        parts = [c.ljust(w) if i < t.label_columns else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths))]
        return sep.join(parts).rstrip()

    total = sum(widths) + len(sep) * (n - 1)
    out = [t.title, ""]
    if spans:
        parts = []
        for label, a, b in spans:
            room = sum(widths[a:b]) + len(sep) * (b - a - 1)
            parts.append(label.center(room) if b - a > 1 else
                         (label.ljust(room) if a < t.label_columns else label.rjust(room)))
        out.append(sep.join(parts).rstrip())
    out.append(line(header[-1]))
    out.append("-" * total)
    for r in t.rows:
        if not r:
            out.append("")
        elif len(r) == 1:
            out.append(r[0])
        else:
            out.append(line(r))
    out.append("-" * total)
    for note in t.notes:
        out.extend(textwrap.wrap(note, width=max(72, total)))
    return "\n".join(out) + "\n"


def _md(text: str) -> str:
    return text.replace("|", "\\|")


def to_markdown(t: Table, heading: str = "###") -> str:
    check_shape(t)
    header = t.header_rows()
    if len(header) == 2:
        cols = [f"{u}, {lo}" if u and lo else (u or lo) for u, lo in zip(*header)]
    else:
        cols = header[0]
    n = len(cols)
    out = [f"{heading} {t.title}".strip(), "",
           "| " + " | ".join(_md(c) for c in cols) + " |",
           "|" + "|".join(":---" if i < t.label_columns else "---:" for i in range(n)) + "|"]
    for r in t.rows:
        if not r:
            continue
        cells = [f"*{r[0]}*"] + [""] * (n - 1) if len(r) == 1 else r
        out.append("| " + " | ".join(_md(c) for c in cells) + " |")
    for note in t.notes:
        out += ["", note]
    return "\n".join(out) + "\n"


def interval(values) -> tuple[float, float]:
    x = np.asarray(values, float)
    return x.mean(), T975[len(x) - 1] * x.std(ddof=1) / np.sqrt(len(x))


def number(v: float, signed: bool = False) -> str:
    s = f"{v:.3f}"
    if s in ("-0.000", "0.000"):
        return "0.000"
    if s.startswith("-"):
        return s
    return ("+" + s) if signed else s


def cell(values, signed: bool = False) -> str:
    m, h = interval(values)
    return f"{number(m, signed)} ± {h:.3f}"


def configuration_columns(first=("Model",)) -> list[list[str]]:
    upper = [""] * len(first) + [DEMAND_LABELS[c.split("_", 1)[0]] for c in TABLE_CONFIGURATIONS]
    lower = list(first) + [c.rsplit("_", 1)[1] for c in TABLE_CONFIGURATIONS]
    return [upper, lower]


_GENERATOR = re.compile(r"(?:^|__)complex_(log_log|covariance_probit)_(endogenous|exogenous)_seed0*(\d+)$")
_DOSE_GENERATOR = re.compile(r"^delta0p(\d\d)__complex_(log_log|covariance_probit)_seed0*(\d+)$")
_RELEASE = re.compile(r"^(?:delta0p(\d\d)/)?(?:(yogurt|switching)_)?(loglinear|discretechoice)"
                      r"_confounding_(on|off)_seed0*(\d+)$")
_CLI_ID = re.compile(r"^(tissue|yogurt|tissue-switching)_(log-linear|discrete-choice)_(on|off|on-0\.15|on-0\.45)"
                     r"_seed(\d+)$")
_ARM = re.compile(r"^([a-z]+)__complex_")
_FAMILY = {"log_log": "loglinear", "covariance_probit": "discretechoice",
           "log-linear": "loglinear", "discrete-choice": "discretechoice"}
_DOSE_LEVELS = {"delta0p15": 0.15, "delta0p45": 0.45}
_RELEASE_PREFIX = {"yogurt": "yogurt", "switching": "tissue-switching"}


def _parse(name: str) -> tuple[str, int, float | None, str, str | None]:
    name = str(name)
    m = _DOSE_GENERATOR.match(name)
    if m:
        strength = int(m.group(1)) / 100
        setting = f"{_FAMILY[m.group(2)]}_confounding_{'off' if strength == 0 else 'on'}"
        return setting, int(m.group(3)), strength, "", "tissue"
    m = _GENERATOR.search(name)
    if m:
        arm = _ARM.match(name)
        conf = "on" if m.group(2) == "endogenous" else "off"
        return f"{_FAMILY[m.group(1)]}_confounding_{conf}", int(m.group(3)), None, arm.group(1) if arm else "", None
    m = _RELEASE.match(name)
    if m:
        strength = int(m.group(1)) / 100 if m.group(1) else None
        return (f"{m.group(3)}_confounding_{m.group(4)}", int(m.group(5)), strength, "",
                _RELEASE_PREFIX.get(m.group(2), "tissue"))
    m = _CLI_ID.match(name)
    if m:
        conf, _, level = m.group(3).partition("-")
        return (f"{_FAMILY[m.group(2)]}_confounding_{conf}", int(m.group(4)), float(level) if level else None, "",
                m.group(1))
    raise TableError(f"{name!r} is not a dataset name")


def is_sampler_run(model) -> bool:
    return bool(_SAMPLER_RUN.match(str(model)))


def base_model(model) -> str:
    model = str(model)
    return model.split("__", 1)[0] if is_sampler_run(model) else model


def _agree(named: pd.Series, given: pd.Series, what: str) -> pd.Series:
    both = named.notna() & given.notna()
    if both.any():
        differ = (~np.isclose(named[both].astype(float), given[both].astype(float))
                  if named.dtype.kind == "f" else named[both] != given[both])
        if differ.any():
            raise TableError(f"the {what} column differs from the dataset name ({int(differ.sum())} rows)")
    return named.where(named.notna(), given)


def prepare(runs: pd.DataFrame) -> pd.DataFrame:
    df = runs.copy()
    parsed = {n: _parse(n) for n in df["dataset"].unique()}
    named = lambda i: df["dataset"].map(lambda n: parsed[n][i])
    setting = named(0)
    df["setting"] = _agree(setting, df["setting"], "setting") if "setting" in df else setting
    seed = named(1).astype(int)
    df["seed"] = _agree(seed, df["seed"].astype(int), "seed") if "seed" in df else seed
    strength = named(2).astype(float)
    if "dose_level" in df:
        strength = _agree(strength, df["dose_level"].map(_DOSE_LEVELS).astype(float), "dose_level")
    for col in ("delta", "strength"):
        if col in df:
            strength = _agree(strength, df[col].astype(float), col)
    df["strength"] = strength
    df["arm"] = named(3)
    df["named_category"] = named(4)
    df["base"] = df["model"].map(base_model)
    return df


def check_category(df: pd.DataFrame, category: str, what: str) -> None:
    other = df["named_category"].notna() & (df["named_category"] != category)
    if other.any():
        found = ", ".join(sorted(set(df.loc[other, "named_category"])))
        raise TableError(f"{what}: the table is for the category {category}, the scores name the category {found}")


def _true(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def check_runs(runs: pd.DataFrame, what: str = "scores") -> None:
    if "status" in runs:
        bad = runs[runs["status"].notna() & (runs["status"] != "valid")]
        if len(bad):
            raise TableError(f"{what}: runs with an invalid score: "
                             + ", ".join(f"{d}/{m}" for d, m in zip(bad["dataset"], bad["model"])))
    if "n_scenarios" in runs and "forecast_rows_missing_prediction" in runs:
        missing = [c for c in ("complete_panel", "rows_missing_prediction") if c not in runs]
        if missing:
            raise TableError(f"{what}: the scores have no column {', '.join(missing)}")
        bad = runs[(runs["n_scenarios"] != N_SCENARIOS) | ~_true(runs["complete_panel"])
                   | (runs["rows_missing_prediction"] != 0) | (runs["forecast_rows_missing_prediction"] != 0)]
        if len(bad):
            raise TableError(f"{what}: incomplete runs: "
                             + ", ".join(f"{d}/{m}" for d, m in zip(bad["dataset"], bad["model"])))


def select_arm(df: pd.DataFrame, arm: str | None) -> pd.DataFrame:
    arms = set(df["arm"]) - {""}
    if not arms:
        return df
    if arm is None or arm not in arms:
        raise TableError(f"the scores are labelled {', '.join(sorted(arms))}, not {arm or 'without a label'}")
    return df[df["arm"] == arm]


def run_set_name(key: str) -> str:
    last = str(key).rsplit("/", 1)[-1]
    return RUN_SET_ALIASES.get(last, last)


def run_set(frames: Mapping[str, pd.DataFrame], name: str) -> pd.DataFrame:
    found = [k for k in frames if run_set_name(k) == name]
    if len(found) > 1:
        raise TableError(f"the run sets {', '.join(found)} are all {name}; a table takes the scores of one category")
    if not found:
        raise TableError(f"no scores of the run set {name} (given: {', '.join(map(str, frames))})")
    return frames[found[0]]


def select_runs(frames: Mapping[str, pd.DataFrame], model_set=DEFAULT_SET, check: bool = True,
                category: str | None = None) -> pd.DataFrame:
    parts = []
    for name, arm, models in model_set:
        df = prepare(run_set(frames, name))
        if category is not None:
            check_category(df, category, name)
        df = select_arm(df, arm)
        df = df[df["base"].isin(models)]
        if check:
            check_runs(df, name)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def average_samplers(df: pd.DataFrame, keys: Sequence[str] = ("setting", "seed"),
                     values: Sequence[str] = (CF, FC)) -> pd.DataFrame:
    keys = list(keys)
    repeated = df[df.duplicated(keys + ["model"], keep=False)]
    if len(repeated):
        names = sorted(set(repeated["model"].astype(str)))
        text = f"more than one run per dataset of {', '.join(names)}"
        if any(base_model(m) in ALL_MODELS for m in names):
            text += ("; the scores of two models share a name (a submitted method cannot take the name of a "
                     f"reference model: {', '.join(ALL_MODELS)})")
        raise TableError(text)
    keys = keys + ["base"]
    marked = df.assign(_sampler=df["model"].map(is_sampler_run))
    g = marked.groupby(keys, dropna=False, sort=True)
    n = g.agg(n=("model", "size"), sampler=("_sampler", "sum"))
    sampled = n.index.get_level_values("base").isin(SAMPLED_MODELS)
    ok = np.where(sampled, ((n["n"] == SAMPLER_RUNS) & (n["sampler"] == SAMPLER_RUNS))
                  | ((n["n"] == 1) & (n["sampler"] == 0)), n["n"] == 1)
    if not ok.all():
        bad = n[~ok].reset_index()
        raise TableError("runs per dataset and model are incomplete or repeated (TabPFN and TabFM need "
                         f"{SAMPLER_RUNS} sampler runs, the other models one run):\n{bad.to_string(index=False)}")
    out = g[list(values)].mean().reset_index()
    return out.rename(columns={"base": "model"})


def seed_values(avg: pd.DataFrame, model: str, setting: str, col: str, name: str | None = None) -> np.ndarray:
    name = model if name is None else name
    s = avg[(avg["model"] == model) & (avg["setting"] == setting)]
    if s.empty:
        raise TableError(f"no scores of {name} in {setting}")
    if sorted(s["seed"]) != list(SEEDS):
        raise TableError(f"{name}: {setting}: seeds {sorted(s['seed'])}, expected {', '.join(map(str, SEEDS))} "
                         f"({col})")
    return s.set_index("seed")[col].loc[list(SEEDS)].to_numpy(float)


def no_change_frame(no_change: pd.DataFrame, category: str | None = None) -> pd.DataFrame:
    df = no_change
    if "model" in df:
        df = df[df["model"].isin(_NO_CHANGE_NAMES)]
    if "reference" in df:
        df = df[df["reference"].isin(_NO_CHANGE_NAMES)]
    df = prepare(df.assign(model=NO_CHANGE))
    if category is not None:
        check_category(df, category, "No change")
    df = df[~df["strength"].isin(list(_DOSE_LEVELS.values()))]
    g = df.groupby(["setting", "seed"])[CF]
    spread = (g.max() - g.min()) / g.max().abs().clip(lower=1e-300)
    if (spread > 1e-12).any():
        bad = spread[spread > 1e-12].index[0]
        raise TableError(f"the No-change scores give different values for {bad[0]} seed {bad[1]}")
    df = df.drop_duplicates(["setting", "seed"])
    return df.assign(**{FC: np.nan})[["setting", "seed", "model", CF, FC]]


def _check_default_arm(frames: Mapping[str, pd.DataFrame], what: str) -> None:
    tuned = prepare(run_set(frames, "cpu_tuned"))
    stored = tuned[(tuned["arm"] == "default") & tuned["base"].isin(TUNED_MODELS)]
    if stored.empty:
        return
    base = prepare(run_set(frames, "cpu_default"))
    base = base[base["base"].isin(TUNED_MODELS)]
    k = ["setting", "seed", "base"]
    m = stored.merge(base, on=k, suffixes=("", "_b"))
    if len(m) != len(stored) or not np.allclose(m[CF], m[CF + "_b"], atol=1e-12) \
            or not np.allclose(m[FC], m[FC + "_b"], atol=1e-12):
        raise TableError(f"{what}: the default-settings scores stored with the tuned runs differ from the default "
                         "runs")


def merge_notes(tables: list[Table]) -> list[Table]:
    notes = list(dict.fromkeys(n for t in tables for n in t.notes))
    for t in tables:
        t.notes = []
    if tables and notes:
        tables[-1].notes = [" ".join(notes)]
    return tables


def results_tables(runs: pd.DataFrame, rows: Sequence, no_change: pd.DataFrame | None, subject: str,
                   notes: Sequence[str] = (), category: str | None = None, first=("Model",)) -> list[Table]:
    check_runs(runs, subject)
    runs = prepare(runs)
    if category is not None:
        check_category(runs, category, subject)
    avg = average_samplers(runs)
    tables = []
    for col, what in ((CF, "Counterfactual WMAPE"), (FC, "Forecast WMAPE")):
        body = []
        for entry in rows:
            if entry is None:
                body.append([])
                continue
            label, model = entry
            cells = list(label) if isinstance(label, tuple) else [label]
            body.append(cells + [cell(seed_values(avg, model, c, col, " ".join(x for x in cells if x)))
                                 for c in TABLE_CONFIGURATIONS])
        table_notes = [INTERVAL_NOTE] + list(notes)
        if col == CF and no_change is not None:
            nc = no_change_frame(no_change, category)
            body += [[], [NO_CHANGE_LABEL] + [""] * (len(first) - 1)
                     + [cell(seed_values(nc, NO_CHANGE, c, CF)) for c in TABLE_CONFIGURATIONS]]
            table_notes.append(NO_CHANGE_NOTE)
        tables.append(Table(f"{what}, {subject}", configuration_columns(first), body, table_notes,
                            label_columns=len(first)))
    return tables


def _grouped_rows(labels: Mapping[str, str]) -> list:
    rows = []
    for gi, (_, models) in enumerate(MODEL_GROUPS):
        if gi:
            rows.append(None)
        rows += [(labels[m], m) for m in models]
    return rows


def method_runs(method: tuple[str, pd.DataFrame], category: str) -> tuple[str, pd.DataFrame]:
    label, runs = method
    label = str(label)
    if not label.strip():
        raise TableError("the submitted method needs a name")
    if "model" in runs and runs["model"].nunique() > 1:
        raise TableError(f"the scores of the submitted method {label} name several models: "
                         + ", ".join(sorted(map(str, runs["model"].unique()))))
    runs = runs.assign(model=METHOD)
    check_runs(runs, label)
    check_category(prepare(runs), category, label)
    return label, runs


def _method_rows(runs: pd.DataFrame, rows: list, notes: list, method, category: str, width: int = 1):
    if method is None:
        return runs, rows, notes
    label, mine = method_runs(method, category)
    cells = (label,) + ("",) * (width - 1) if width > 1 else label
    return (pd.concat([runs, mine], ignore_index=True), [(cells, METHOD), None] + rows, notes + [METHOD_NOTE])


def _check_category_word(category: str, allowed: Sequence[str], table: str) -> None:
    if category not in allowed:
        raise TableError(f"{category!r} is not a category of the {table}; categories: {', '.join(allowed)}")


def main_table(per_run: Mapping[str, pd.DataFrame], no_change: pd.DataFrame | None,
               method: tuple[str, pd.DataFrame] | None = None) -> list[Table]:
    category = "tissue"
    runs = select_runs(per_run, DEFAULT_SET, category=category)
    runs, rows, notes = _method_rows(runs, _grouped_rows(MODEL_LABELS), [DEFAULT_SET_NOTE, SAMPLER_NOTE], method,
                                     category)
    return results_tables(runs, rows, no_change, SUBJECTS[category], notes, category)


def _settings_rows() -> list:
    rows = []
    for gi, (_, models) in enumerate(MODEL_GROUPS):
        if gi:
            rows.append(None)
        for m in models:
            rows.append(((MODEL_LABELS[m], "default"), m))
            if m in TUNED_MODELS:
                rows.append((("", "tuned"), TUNED_KEY.format(m)))
    return rows


def settings_table(per_run: Mapping[str, pd.DataFrame], no_change: pd.DataFrame | None, category: str,
                   method: tuple[str, pd.DataFrame] | None = None) -> list[Table]:
    _check_category_word(category, SETTINGS_CATEGORIES, "table of default and forecast-tuned settings")
    _check_default_arm(per_run, category)
    default = select_runs(per_run, DEFAULT_SET, category=category)
    tuned = select_runs(per_run, TUNED_SET, category=category)
    tuned = tuned.assign(model=tuned["base"].map(TUNED_KEY.format), base=tuned["base"].map(TUNED_KEY.format))
    runs = pd.concat([default, tuned], ignore_index=True)
    runs, rows, notes = _method_rows(runs, _settings_rows(), [SETTINGS_NOTE, SAMPLER_NOTE], method, category, 2)
    return results_tables(runs, rows, no_change, SUBJECTS[category], notes, category, ("Model", "Settings"))


def switching_table(per_run: Mapping[str, pd.DataFrame], no_change: pd.DataFrame | None,
                    method: tuple[str, pd.DataFrame] | None = None) -> list[Table]:
    category = "tissue-switching"
    runs = select_runs(per_run, DEFAULT_SET, category=category)
    runs, rows, notes = _method_rows(runs, _grouped_rows(MODEL_LABELS), [DEFAULT_SET_NOTE, SAMPLER_NOTE], method,
                                     category)
    return results_tables(runs, rows, no_change, SUBJECTS[category], notes, category)


def yogurt_table(per_run: Mapping[str, pd.DataFrame], no_change: pd.DataFrame | None,
                 method: tuple[str, pd.DataFrame] | None = None) -> list[Table]:
    return settings_table(per_run, no_change, "yogurt", method)


def strength_frame(*per_run: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for f in per_run:
        df = prepare(f)
        check_runs(df, "confounding strength")
        check_category(df, "tissue", "confounding strength")
        if (df["arm"] != "").any():
            raise TableError("the confounding-strength table takes runs at default settings only")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    on = df["setting"].str.endswith("_on")
    df["strength"] = df["strength"].fillna(pd.Series(np.where(on, MAIN_STRENGTH, 0.0), index=df.index))
    snapped = pd.Series(np.nan, index=df.index)
    for s in STRENGTHS:
        snapped[np.isclose(df["strength"], s)] = s
    if snapped.isna().any():
        raise TableError(f"confounding strengths outside {STRENGTHS}: {sorted(set(df['strength'][snapped.isna()]))}")
    df["strength"] = snapped
    df["demand_model"] = df["setting"].str.split("_").str[0]
    key = ["demand_model", "strength", "seed", "model"]
    dup = df[df.duplicated(key, keep=False)]
    if len(dup):
        spread = dup.groupby(key)[[CF, FC]].agg(lambda v: (v.max() - v.min()) / max(abs(v).max(), 1e-300))
        if (spread.to_numpy() > 1e-9).any():
            raise TableError("a run listed twice has different scores")
        df = df.drop_duplicates(key)
    return average_samplers(df, ["demand_model", "strength", "seed"])


def _strength_grid(frame: pd.DataFrame, demand_model: str, model: str, col: str,
                   name: str | None = None) -> pd.DataFrame:
    g = frame[(frame["demand_model"] == demand_model) & (frame["model"] == model)].pivot(
        index="seed", columns="strength", values=col)
    if sorted(g.index) != list(SEEDS) or not np.allclose(sorted(g.columns), STRENGTHS):
        raise TableError(f"confounding strength: {demand_model} {model if name is None else name} is incomplete")
    return g.loc[list(SEEDS)].sort_index(axis=1)


STRENGTH_METHOD_NOTE = ("The first row of each panel is the submitted method; the other rows are the reference "
                        "runs.")


def strength_table(frame: pd.DataFrame, metric: str = FC, method: tuple[str, pd.DataFrame] | None = None) -> Table:
    mine = None
    if method is not None:
        label, runs = method_runs(method, "tissue")
        mine = (label, strength_frame(runs))
    body = []
    for i, (dm, dlabel) in enumerate(DEMAND_MODELS):
        if i:
            body.append([])
        body.append([f"{dlabel} demand"])
        if mine is not None:
            label, mframe = mine
            g = _strength_grid(mframe, dm, METHOD, metric, label)
            body.append([label] + [cell(g[c].to_numpy(float)) for c in g.columns])
            body.append([])
        for gi, (_, models) in enumerate(MODEL_GROUPS):
            if gi:
                body.append([])
            for m in models:
                g = _strength_grid(frame, dm, m, metric)
                body.append([MODEL_LABELS[m]] + [cell(g[c].to_numpy(float)) for c in g.columns])
    what = "Counterfactual WMAPE" if metric == CF else "Forecast WMAPE" if metric == FC else metric
    columns = ["Model"] + list(STRENGTH_LABELS)
    settings = DEFAULT_SET_NOTE if mine is None else "Reference models at default settings."
    notes = [INTERVAL_NOTE, settings + " " + SAMPLER_NOTE]
    if mine is not None:
        notes.append(STRENGTH_METHOD_NOTE)
    return Table(f"{what} by confounding strength, facial tissue datasets", columns, body, notes)


REAL_COLUMNS = ["Model", "Forecast WMAPE [95% CI]", "Sign correct, price up [95% CI]"]


def real_table(rows: Sequence[Sequence[str]], title: str, notes: Sequence[str] = ()) -> Table:
    body = []
    for r in rows:
        if len(r) != len(REAL_COLUMNS):
            raise TableError(f"a row of the observed-data table needs {len(REAL_COLUMNS)} cells: {r!r}")
        body.append([str(c) for c in r])
    return Table(title, list(REAL_COLUMNS), body, list(notes))


def tuned_table(per_run: Mapping[str, pd.DataFrame], category: str = "tissue") -> list[Table]:
    _check_category_word(category, MAIN_CATEGORIES, "table of default and forecast-tuned settings")
    _check_default_arm(per_run, category)
    default = select_runs(per_run, (("cpu_default", None, TUNED_MODELS),), category=category)
    tuned = select_runs(per_run, (("cpu_tuned", "tuned", TUNED_MODELS),), category=category)
    avg = {"default": average_samplers(default), "tuned": average_samplers(tuned)}
    tables = []
    for col, what in ((CF, "Counterfactual WMAPE"), (FC, "Forecast WMAPE")):
        body = []
        for i, m in enumerate(TUNED_MODELS):
            if i:
                body.append([])
            for j, arm in enumerate(("default", "tuned")):
                body.append([MODEL_LABELS[m] if j == 0 else "", arm]
                            + [cell(seed_values(avg[arm], m, c, col)) for c in TABLE_CONFIGURATIONS])
        tables.append(Table(f"{what} with default and forecast-tuned settings, {SUBJECTS[category]}",
                            configuration_columns(("Model", "Settings")), body, [INTERVAL_NOTE], label_columns=2))
    return tables


SCENARIO_GROUPS = (("product", "plus10", "Products, price +10%"), ("product", "minus10", "Products, price -10%"),
                   ("brand", "plus10", "Brands, price +10%"), ("brand", "minus10", "Brands, price -10%"))
SCENARIO_KEYS = ["setting", "seed", "intervention_id", "scope", "direction"]
SETTING_TITLES = {"discretechoice_confounding_on": "discrete-choice model with confounding on",
                  "discretechoice_confounding_off": "discrete-choice model with confounding off",
                  "loglinear_confounding_on": "log-linear model with confounding on",
                  "loglinear_confounding_off": "log-linear model with confounding off"}


def scenario_id(scope: str, k: int, direction: str) -> str:
    return (f"sweep_panel_product{k:02d}" if scope == "product" else f"sweep_panel_brand{k}") + f"_promo_{direction}"


def scenario_scores(per_scenario: Mapping[str, pd.DataFrame], model_set=DEFAULT_SET,
                    per_run: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    ps = select_runs(per_scenario, model_set, check=False, category="tissue")
    if (ps["n_missing_prediction"] != 0).any() or (ps["n_dropped_qhat_nonpositive"] != 0).any():
        raise TableError("the per-scenario scores have rows without a prediction or with a dropped prediction")
    ps = ps.assign(abs_bias=ps["bias"].abs())
    avg = average_samplers(ps, SCENARIO_KEYS, ("accuracy_cf_wmape", "bias", "abs_bias"))
    if per_run is not None:
        runs = average_samplers(select_runs(per_run, model_set, category="tissue"))
        agg = avg.groupby(["setting", "seed", "model"], as_index=False)["accuracy_cf_wmape"].agg(["mean", "size"])
        m = agg.merge(runs, on=["setting", "seed", "model"])
        if len(m) != len(agg) or len(m) != len(runs) or (m["size"] != N_SCENARIOS).any() \
                or not np.allclose(m["mean"], m[CF], atol=1e-9):
            raise TableError("the per-scenario scores do not reproduce the per-run counterfactual WMAPE")
    return avg


def scenario_tables(per_scenario: Mapping[str, pd.DataFrame],
                    per_run: Mapping[str, pd.DataFrame] | None = None) -> dict[str, Table]:
    d = scenario_scores(per_scenario, DEFAULT_SET, per_run)
    columns = [[""] + [g for g, ms in MODEL_GROUPS for _ in ms], ["Scenario"] + [MODEL_LABELS[m] for m in ALL_MODELS]]
    short = {"discretechoice_confounding_on": "dc-on", "discretechoice_confounding_off": "dc-off",
             "loglinear_confounding_on": "ll-on", "loglinear_confounding_off": "ll-off"}
    out = {}
    for col, key, signed, what in (("accuracy_cf_wmape", "cf", False, "Counterfactual WMAPE"),
                                   ("bias", "bias", True, "Counterfactual bias")):
        for setting in TABLE_CONFIGURATIONS:
            sub = d[d["setting"] == setting]
            body = []
            for g, (scope, direction, glabel) in enumerate(SCENARIO_GROUPS):
                if g:
                    body.append([])
                body.append([glabel])
                for k in range(1, (12 if scope == "product" else 4) + 1):
                    sid = scenario_id(scope, k, direction)
                    cells = []
                    for m in ALL_MODELS:
                        x = sub[(sub["model"] == m) & (sub["intervention_id"] == sid)]
                        if sorted(x["seed"]) != list(SEEDS):
                            raise TableError(f"{setting} {m} {sid}: seeds {sorted(x['seed'])}")
                        cells.append(cell(x.sort_values("seed")[col].to_numpy(float), signed=signed))
                    body.append([f"{'Product' if scope == 'product' else 'Brand'} {k}"] + cells)
            notes = [INTERVAL_NOTE, DEFAULT_SET_NOTE, SAMPLER_NOTE,
                     "Products and brands are numbered by sales rank; each scenario changes the price of one product "
                     "or of every product of one brand."]
            out[f"scen-{key}-{short[setting]}"] = Table(f"{what} by scenario in the {SETTING_TITLES[setting]}",
                                                        columns, body, notes)
    return out
