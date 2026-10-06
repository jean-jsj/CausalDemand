from __future__ import annotations

import logging
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("dff_loglog")

_HERE = Path(__file__).resolve().parent

CATEGORIES = {
    "ana": "Analgesics",
    "bat": "Bath Soap",
    "ber": "Beer",
    "bjc": "Bottled Juices",
    "cer": "Cereals",
    "che": "Cheeses",
    "cig": "Cigarettes",
    "coo": "Cookies",
    "cra": "Crackers",
    "cso": "Canned Soup",
    "did": "Dish Detergent",
    "fec": "Front-end-candies",
    "frd": "Frozen Dinners",
    "fre": "Frozen Entrees",
    "frj": "Frozen Juices",
    "fsf": "Fabric Softeners",
    "gro": "Grooming Products",
    "lnd": "Laundry Detergents",
    "oat": "Oatmeal",
    "ptw": "Paper Towels",
    "sdr": "Soft Drinks",
    "sha": "Shampoos",
    "sna": "Snack Crackers",
    "soa": "Soaps",
    "tbr": "Toothbrushes",
    "tna": "Canned Tuna",
    "tpa": "Toothpastes",
    "tti": "Bathroom Tissues",
}
SALE_CODES = ("B", "C", "S")

_DISCONTINUED = "~"
_TRIAL = "<"
_COMBO = "#"
_MARKER = re.compile(r"[~$*#<]")

MOVE_COLS = {
    "STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "PROFIT",
    "OK", "PRICE_HEX", "PROFIT_HEX",
}

DESC_COLS = [
    "upc", "manufacturer_code", "product_code", "commodity_code", "nitem",
    "distribution", "case_pack", "descrip", "size", "ounces", "vol_eq",
    "size_parsed", "discontinued", "trial_size", "combo_store_only",
    "brand", "description",
]


@dataclass
class Config:
    category: str = "cra"
    data_dir: Path = field(default_factory=lambda: _HERE / "data")
    out_dir: Path = field(default_factory=lambda: _HERE / "out")
    upc_file: Path | None = None
    move_file: Path | None = None

    oz_per_eq_unit: float = 16.0
    filter_ok: bool = True
    use_hex_price: bool = True
    use_hex_profit: bool = True

    product_level: str = "upc"
    min_rev_share: float = 0.02
    top_n: int | None = None
    small_products: str = "pool"
    xprice_exclude_pooled: bool = True

    feature_codes: tuple[str, ...] = ("S",)
    display_codes: tuple[str, ...] = ("B",)
    min_cost_coverage: float = 0.5
    expand_fe: bool = True
    product_fe: bool = True
    chunksize: int = 2_000_000

    @property
    def coupon_codes(self) -> tuple[str, ...]:
        used = {c.upper() for c in self.feature_codes + self.display_codes}
        return tuple(c for c in SALE_CODES if c not in used)

    @property
    def category_name(self) -> str:
        return CATEGORIES.get(self.category, self.category)


_WEIGHT = {"OZ": 1.0, "O": 1.0, "Z": 1.0, "LB": 16.0, "#": 16.0}
_SIZE_TOKEN = re.compile(
    r"(\d*\.\d+|\d+)\s*(OZ|O|Z|LB|#|CT|PK|PKS|PACK|CNT|COUNT|EA)\b", re.IGNORECASE)
_SIZE_MULTI = re.compile(
    r"^\s*(\d*\.\d+|\d+)\s*/\s*(\d*\.\d+|\d+)\s*(OZ|O|Z|LB|#)\b", re.IGNORECASE)


def parse_size_oz(size: str) -> float:
    if not isinstance(size, str) or not size.strip():
        return np.nan
    multi = _SIZE_MULTI.match(size)
    if multi:
        unit = multi.group(3).upper()
        return float(multi.group(1)) * float(multi.group(2)) * _WEIGHT[unit]
    total = 0.0
    found = False
    for num, unit in _SIZE_TOKEN.findall(size):
        scale = _WEIGHT.get(unit.upper())
        if scale is None:
            continue
        total += float(num) * scale
        found = True
    return total if found else np.nan


def _text_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[name].fillna("").astype(str).str.strip()


def _status_suffix(discontinued: pd.Series, trial: pd.Series, combo: pd.Series) -> pd.Series:
    labels = ("discontinued", "trial size", "combo stores only")
    flags = (discontinued.astype(bool), trial.astype(bool), combo.astype(bool))
    parts = [
        "; ".join(label for label, on in zip(labels, row) if on)
        for row in zip(*flags)
    ]
    return pd.Series([f" ({p})" if p else "" for p in parts], index=discontinued.index)


def build_upc_descriptions(upc: pd.DataFrame, oz_per_eq_unit: float) -> pd.DataFrame:
    raw = _text_col(upc, "descrip").str.upper()
    size = _text_col(upc, "size").str.upper()
    discontinued = raw.str.contains(_DISCONTINUED, regex=False).astype("int8")
    trial = raw.str.contains(_TRIAL, regex=False).astype("int8")
    combo = raw.str.contains(_COMBO, regex=False).astype("int8")
    name = (raw.str.replace(_MARKER, "", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip())
    name = name.mask(name.eq(""), "UPC " + upc["upc"].astype(str))
    size_bit = pd.Series(np.where(size.ne(""), ", " + size, ""), index=upc.index)
    description = name + size_bit + _status_suffix(discontinued, trial, combo)

    manufacturer = (upc["upc"] // 100_000).astype(int)
    ounces = size.map(parse_size_oz)
    missing_size = ounces.isna()
    if missing_size.any():
        log.warning(
            "SIZE is not a weight for %d of %d UPCs; those UPCs are left out of the model",
            int(missing_size.sum()), len(upc),
        )

    nitem = pd.to_numeric(_text_col(upc, "nitem").replace("", np.nan), errors="coerce")
    last_digit = nitem % 10
    distribution = pd.Series(pd.NA, index=upc.index, dtype="string")
    distribution = distribution.mask(last_digit.eq(0), "drop-shipped").mask(last_digit.eq(1), "warehoused")

    token = name.str.split().str[0].replace("", np.nan)
    modal = (pd.DataFrame({"manufacturer_code": manufacturer, "token": token})
               .dropna()
               .groupby("manufacturer_code")["token"]
               .agg(lambda s: s.value_counts().index[0]))
    brand = manufacturer.map(modal).fillna(manufacturer.astype(str))
    brand = brand.astype(str) + " (" + manufacturer.astype(str) + ")"

    out = pd.DataFrame({
        "upc": upc["upc"].astype("int64"),
        "manufacturer_code": manufacturer,
        "product_code": (upc["upc"] % 100_000).astype(int),
        "commodity_code": pd.to_numeric(_text_col(upc, "com_code").replace("", np.nan), errors="coerce"),
        "nitem": nitem,
        "distribution": distribution,
        "case_pack": pd.to_numeric(_text_col(upc, "case").replace("", np.nan), errors="coerce"),
        "descrip": raw,
        "size": size,
        "ounces": ounces,
        "vol_eq": ounces / oz_per_eq_unit,
        "size_parsed": (~missing_size).astype("int8"),
        "discontinued": discontinued,
        "trial_size": trial,
        "combo_store_only": combo,
        "brand": brand,
        "description": description,
    })
    return out.sort_values("upc", ignore_index=True)[DESC_COLS]


def read_upc_file(path: Path) -> pd.DataFrame:
    upc = pd.read_csv(path, encoding="latin-1", dtype=str)
    upc.columns = [c.strip().lower() for c in upc.columns]
    upc["upc"] = pd.to_numeric(upc["upc"], errors="coerce").astype("Int64")
    upc = upc.dropna(subset=["upc"]).copy()
    upc["upc"] = upc["upc"].astype("int64")
    n_dup = int(upc["upc"].duplicated().sum())
    if n_dup:
        log.warning("UPC file has %d duplicate UPC rows; kept the first of each", n_dup)
    return upc.drop_duplicates("upc")


def _hex_to_float(s) -> float:
    if not isinstance(s, str) or not s.strip():
        return np.nan
    s = s.strip()
    try:
        if s.lower().startswith(("0x", "-0x")) or "p" in s.lower():
            return float.fromhex(s)
        if len(s) == 16:
            return struct.unpack(">d", bytes.fromhex(s))[0]
    except (ValueError, struct.error):
        pass
    return np.nan


def _apply_hex(mv: pd.DataFrame, hex_col: str, dest_col: str, tolerance: float) -> None:
    if hex_col not in mv:
        return
    uniq = mv[hex_col].dropna().unique()
    lut = {h: _hex_to_float(h) for h in uniq}
    parsed = mv[hex_col].map(lut)
    orig = pd.to_numeric(mv[dest_col], errors="coerce")
    both = parsed.notna() & orig.notna()
    close = both & ((parsed - orig).abs() <= tolerance)
    disagree = int((both & ~close).sum())
    if disagree:
        log.warning(
            "%s disagrees with %s by more than %s on %s rows; those rows keep %s",
            hex_col, dest_col, tolerance, f"{disagree:,}", dest_col,
        )
    use = close | (parsed.notna() & orig.isna())
    mv.loc[use, dest_col] = parsed[use]
    log.info("Full-precision %s used for %.1f%% of rows", hex_col, 100 * float(use.mean()))


def read_movement(path: Path, cfg: Config, upcs: set[int]) -> pd.DataFrame:
    comp = "zip" if path.suffix.lower() == ".zip" else "infer"
    reader = pd.read_csv(
        path, compression=comp, chunksize=cfg.chunksize,
        usecols=lambda c: c.strip().upper() in MOVE_COLS,
        dtype={"SALE": str, "PRICE_HEX": str, "PROFIT_HEX": str,
               "sale": str, "price_hex": str, "profit_hex": str},
    )
    parts, n_raw = [], 0
    for ch in reader:
        ch.columns = [c.strip().lower() for c in ch.columns]
        n_raw += len(ch)
        parts.append(ch[ch["upc"].isin(upcs)])
    if not parts:
        raise SystemExit("Movement file contained no rows.")
    mv = pd.concat(parts, ignore_index=True)
    log.info("Movement rows: %s read, %s kept (all stores, category UPCs)", f"{n_raw:,}", f"{len(mv):,}")

    if cfg.use_hex_price:
        _apply_hex(mv, "price_hex", "price", tolerance=0.02)
    if cfg.use_hex_profit and "profit" in mv.columns:
        _apply_hex(mv, "profit_hex", "profit", tolerance=0.1)
    if "profit" not in mv.columns:
        log.warning("Movement file has no PROFIT column; wholesale instruments will be missing")
        mv["profit"] = np.nan

    if cfg.filter_ok and "ok" in mv:
        n0 = len(mv)
        mv = mv[mv["ok"] == 1]
        log.info("OK filter dropped %s rows", f"{n0 - len(mv):,}")
    mv = mv[(mv["qty"] > 0) & (mv["move"] >= 0) & (mv["price"] > 0)].copy()
    n_zero = int((mv["move"] == 0).sum())
    log.info("Zero-sale rows with a positive price kept for rival prices: %s", f"{n_zero:,}")
    return mv


def profit_to_margin(profit: pd.Series) -> pd.Series:
    p = pd.to_numeric(profit, errors="coerce")
    positive = p[(p > 0) & np.isfinite(p)]
    med = float(positive.median()) if len(positive) else np.nan
    if pd.notna(med) and med > 1.5:
        p = p / 100.0
        scale = "percent (divided by 100)"
    else:
        scale = "fraction"
    log.info("PROFIT interpreted as a %s; median of positive raw values is %s",
             scale, "nan" if pd.isna(med) else f"{med:.3f}")
    return p.where(np.isfinite(p) & (p < 1))
