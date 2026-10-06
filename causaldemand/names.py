from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

CATEGORY_WORDS = ("tissue", "yogurt", "cereal", "snack-crackers", "all")
CATEGORIES = ("tissue", "yogurt", "cereal", "snack-crackers")
CATEGORY_TITLES = {
    "tissue": "facial tissue, synthetic sales",
    "yogurt": "yogurt, synthetic sales",
    "cereal": "ready-to-eat cereals, observed sales (Dominick's)",
    "snack-crackers": "snack crackers, observed sales (Dominick's)",
    "all": "every dataset of the four categories",
}
VARIANTS = ("main", "dose", "switching", "yogurt", "real")

DEMAND_MODELS = {"loglinear": ("log-linear", "log_log"), "discretechoice": ("discrete-choice", "covariance_probit")}
CONFOUNDING = {"on": "endogenous", "off": "exogenous"}
SEEDS = (1, 10, 20, 30, 40)
TABLE_CONFIGURATIONS = ("discretechoice_confounding_on", "discretechoice_confounding_off",
                        "loglinear_confounding_on", "loglinear_confounding_off")

MAIN_STRENGTH = 0.30
DOSE_LEVELS = {"delta0p15": 0.15, "delta0p45": 0.45}
YOGURT_STRENGTH = {"loglinear": 0.42, "discretechoice": 0.36}

YOGURT = "yogurt"
DOSE_FOLDER = "dose_response"
SWITCHING_FOLDER = "switching"
REAL_FOLDER = "dominicks"
REAL_DATASETS = {
    "cereal": ("cereal", "ready-to-eat cereals, observed sales (Dominick's Finer Foods)"),
    "snack-crackers": ("snack_crackers", "snack crackers, observed sales (Dominick's Finer Foods)"),
}


class UnknownNameError(ValueError):
    pass


@dataclass(frozen=True)
class Dataset:
    category: str
    variant: str
    model: str | None
    confounding: str | None
    seed: int | None
    dose_level: str | None = None

    @property
    def is_real(self) -> bool:
        return self.variant == "real"

    @property
    def demand_model(self) -> str | None:
        return None if self.model is None else DEMAND_MODELS[self.model][0]

    @property
    def id(self) -> str:
        if self.is_real:
            return self.category
        conf = self.confounding if self.dose_level is None else f"on-{DOSE_LEVELS[self.dose_level]:.2f}"
        prefix = "tissue-switching" if self.variant == "switching" else self.category
        return f"{prefix}_{self.demand_model}_{conf}_seed{self.seed}"

    @property
    def setting(self) -> str | None:
        return None if self.is_real else f"{self.model}_confounding_{self.confounding}"

    @property
    def release_name(self) -> str:
        if self.is_real:
            return REAL_DATASETS[self.category][0]
        prefix = {"switching": "switching_", "yogurt": "yogurt_"}.get(self.variant, "")
        return f"{prefix}{self.setting}_seed{self.seed:03d}"

    @property
    def release_path(self) -> str:
        if self.variant == "main":
            return f"data/{self.release_name}"
        if self.variant == "dose":
            return f"{DOSE_FOLDER}/{self.dose_level}/{self.release_name}"
        if self.variant == "switching":
            return f"{SWITCHING_FOLDER}/{self.release_name}"
        if self.variant == "yogurt":
            return f"{YOGURT}/{self.release_name}"
        return f"{REAL_FOLDER}/{self.release_name}"

    @property
    def generator_name(self) -> str | None:
        if self.is_real:
            return None
        return f"complex_{self.family}_{CONFOUNDING[self.confounding]}_seed{self.seed:03d}"

    @property
    def family(self) -> str | None:
        return None if self.model is None else DEMAND_MODELS[self.model][1]

    @property
    def strength(self) -> float | None:
        if self.is_real:
            return None
        if self.confounding == "off":
            return 0.0
        if self.dose_level is not None:
            return DOSE_LEVELS[self.dose_level]
        if self.variant == "yogurt":
            return YOGURT_STRENGTH[self.model]
        return MAIN_STRENGTH

    @property
    def title(self) -> str:
        if self.is_real:
            return REAL_DATASETS[self.category][1]
        subject = {"switching": "facial tissue with household-switching product distances",
                   "yogurt": "yogurt"}.get(self.variant, "facial tissue")
        conf = "confounding off" if self.confounding == "off" else f"confounding on (strength {self.strength:.2f})"
        return f"{subject}, {self.demand_model} demand, {conf}, seed {self.seed}"

    @property
    def name(self) -> str:
        return self.release_name

    @property
    def configuration(self) -> str:
        return self.release_name if self.is_real else self.release_name.rsplit("_seed", 1)[0]

    @property
    def label(self) -> str:
        return self.release_name if self.dose_level is None else f"{self.dose_level}/{self.release_name}"


def _synthetic(category: str, variant: str, confounding_values=("on", "off"), dose_level: str | None = None):
    for setting in TABLE_CONFIGURATIONS:
        model, conf = setting.split("_confounding_")
        if conf not in confounding_values:
            continue
        for seed in SEEDS:
            yield Dataset(category, variant, model, conf, seed, dose_level)


def _build() -> tuple[Dataset, ...]:
    out = list(_synthetic("tissue", "main"))
    for level in DOSE_LEVELS:
        out += _synthetic("tissue", "dose", ("on",), level)
    out += _synthetic("tissue", "switching")
    out += _synthetic(YOGURT, "yogurt")
    out += [Dataset(cat, "real", None, None, None) for cat in REAL_DATASETS]
    return tuple(out)


_CATALOG = _build()
_BY_ID = {ds.id: ds for ds in _CATALOG}
_BY_PATH = {ds.release_path: ds for ds in _CATALOG}
_BY_RELEASE_NAME = {(ds.release_name, ds.dose_level): ds for ds in _CATALOG}
_BY_GENERATOR_NAME = {(ds.generator_name, ds.variant, ds.dose_level): ds for ds in _CATALOG if not ds.is_real}
_RELEASE_NAME = re.compile(r"^(?:(yogurt|switching)_)?(loglinear|discretechoice)_confounding_(on|off)"
                           r"(?:_seed(\d{3}))?$")


def all_datasets() -> list[Dataset]:
    return list(_CATALOG)


def _did_you_mean(word: str, choices) -> str:
    close = difflib.get_close_matches(word, list(choices), n=1, cutoff=0.6)
    return f" Did you mean {close[0]}?" if close else ""


def release_name_hint(word: str) -> str:
    m = _RELEASE_NAME.match(word)
    if not m:
        return ""
    prefix, model, conf, seed = m.groups()
    category = "yogurt" if prefix == "yogurt" else "tissue"
    id_prefix = {"yogurt": "yogurt", "switching": "tissue-switching"}.get(prefix, "tissue")
    option = " --tissue-switching" if prefix == "switching" else ""
    seed_text = str(int(seed)) if seed else "<N>"
    return (f"Commands take a category and act on all its datasets: {category}{option} (the dataset ID of this "
            f"name is {id_prefix}_{DEMAND_MODELS[model][0]}_{conf}_seed{seed_text}).")


_TISSUE_PART_WORDS = {
    "dose": "--tissue-dose", "tissue-dose": "--tissue-dose", "tissue_dose": "--tissue-dose",
    "dose-response": "--tissue-dose", "dose_response": "--tissue-dose",
    "switching": "--tissue-switching", "tissue-switching": "--tissue-switching",
    "tissue_switching": "--tissue-switching",
}
_TISSUE_PARTS = {"--tissue-dose": "The datasets with confounding strengths 0.15 and 0.45",
                 "--tissue-switching": "The datasets with household-switching product distances"}
_GROUP_WORDS = {"real": ("cereal", "snack-crackers"), "dominicks": ("cereal", "snack-crackers"),
                "synthetic": ("tissue", "yogurt")}
_ID_PREFIX = re.compile(r"^(tissue-switching|tissue|yogurt)_")


def _dataset_id_hint(word: str) -> str:
    ds = _BY_ID.get(word)
    if ds is not None:
        option = {"dose": " --tissue-dose", "switching": " --tissue-switching"}.get(ds.variant, "")
        return (f"\"{word}\" is a dataset ID, not a category. Commands act on all datasets of a category; this "
                f"dataset is part of {ds.category}{option}.")
    m = _ID_PREFIX.match(word)
    if not m:
        return ""
    prefix = m.group(1)
    category = "yogurt" if prefix == "yogurt" else "tissue"
    option = (" --tissue-switching" if prefix == "tissue-switching"
              else " --tissue-dose" if prefix == "tissue" and "_on-0." in word else "")
    return (f"\"{word}\" looks like a dataset ID, not a category. Commands act on all datasets of a category, "
            f"here {category}{option}.")


def check_category(word: str, allowed=CATEGORY_WORDS) -> str:
    if word in allowed:
        return word
    listed = ", ".join(allowed)
    if word in CATEGORY_WORDS:
        raise UnknownNameError(f"{word} is not accepted here. Categories: {listed}.")
    if "," in word:
        raise UnknownNameError(f"\"{word}\" is not a category. This command takes one category; run it once for "
                               f"each category. Categories: {listed}.")
    if word == "crackers":
        raise UnknownNameError("\"crackers\" is not a category. The Dominick's category of the benchmark is snack "
                               f"crackers: snack-crackers. Categories: {listed}.")
    if word.lower() in _GROUP_WORDS:
        first, second = _GROUP_WORDS[word.lower()]
        raise UnknownNameError(f"\"{word}\" is not a category. This command takes one category; run it once for "
                               f"{first} and once for {second}. Categories: {listed}.")
    if word.lower() in _TISSUE_PART_WORDS:
        option = _TISSUE_PART_WORDS[word.lower()]
        raise UnknownNameError(f"\"{word}\" is not a category. {_TISSUE_PARTS[option]} are part of tissue: give "
                               f"tissue {option}. Categories: {listed}.")
    hint = release_name_hint(word)
    if hint:
        raise UnknownNameError(f"\"{word}\" is a name in the release, not a category. {hint} Categories: {listed}.")
    hint = _dataset_id_hint(word)
    if hint:
        raise UnknownNameError(f"{hint} Categories: {listed}.")
    raise UnknownNameError(f"\"{word}\" is not a category.{_did_you_mean(word, allowed)} Categories: {listed}.")


def select(category: str, tissue_dose: bool = False, tissue_switching: bool = False) -> list[Dataset]:
    check_category(category)
    if category == "all":
        return all_datasets()
    if category != "tissue" and (tissue_dose or tissue_switching):
        options = [o for o, on in (("--tissue-dose", tissue_dose), ("--tissue-switching", tissue_switching)) if on]
        verb = "apply" if len(options) == 2 else "applies"
        raise UnknownNameError(f"{' and '.join(options)} {verb} to tissue only, not to {category}.")
    if category == "tissue":
        variants = {"main"} | ({"dose"} if tissue_dose else set()) | ({"switching"} if tissue_switching else set())
    else:
        variants = {"yogurt" if category == YOGURT else "real"}
    return [ds for ds in _CATALOG if ds.category == category and ds.variant in variants]


def by_id(dataset_id: str) -> Dataset:
    try:
        return _BY_ID[dataset_id]
    except KeyError:
        raise UnknownNameError(
            f"\"{dataset_id}\" is not a dataset ID.{_did_you_mean(dataset_id, _BY_ID)} Dataset IDs look like "
            "tissue_log-linear_on_seed1, tissue_discrete-choice_on-0.15_seed10, "
            "tissue-switching_log-linear_off_seed40, yogurt_discrete-choice_on_seed20, cereal or snack-crackers.") \
            from None


def by_release_name(name: str, dose_level: str | None = None) -> Dataset:
    if dose_level is not None and dose_level not in DOSE_LEVELS:
        raise UnknownNameError(f"{dose_level!r} is not a level of confounding strength; levels: "
                               f"{', '.join(DOSE_LEVELS)}")
    try:
        return _BY_RELEASE_NAME[(name, dose_level)]
    except KeyError:
        where = f" at level {dose_level}" if dose_level else ""
        raise UnknownNameError(f"{name!r} is not a released dataset{where}") from None


def by_release_path(path: str) -> Dataset:
    try:
        return _BY_PATH[path.strip("/")]
    except KeyError:
        raise UnknownNameError(f"{path!r} is not a dataset folder of the release") from None


def by_generator_name(name: str, variant: str = "main", dose_level: str | None = None) -> Dataset:
    try:
        return _BY_GENERATOR_NAME[(name, variant, dose_level)]
    except KeyError:
        raise UnknownNameError(f"{name!r} is not a generator name of the {variant} datasets") from None


def count_text(n: int, word: str = "dataset") -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")
