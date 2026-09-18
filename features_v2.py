"""Признаки v2: к 9 признакам похожести (experiments.Featurizer) добавляем признаки РАЗЛИЧИЙ,
нормализацию единиц, поиск кода в тексте другого оффера и явный признак «один магазин»."""
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from baseline import CODE
from experiments import Featurizer

# --------------------------------------------------------------------------- нормализация единиц
UNIT_RULES = [
    (r"(\d+(?:\.\d+)?)\s*(?:gb|go|gig|gigabytes?)\b", r"\1gb"),
    (r"(\d+(?:\.\d+)?)\s*(?:tb|to|terabytes?)\b", r"\1tb"),
    (r"(\d+(?:\.\d+)?)\s*(?:mb|mo|megabytes?)\b", r"\1mb"),
    (r"(\d+(?:\.\d+)?)\s*ghz\b", r"\1ghz"),
    (r"(\d+(?:\.\d+)?)\s*mhz\b", r"\1mhz"),
    (r"(\d+(?:\.\d+)?)\s*(?:inch(?:es)?|in\b|''|\")", r"\1in"),
    (r"(\d+(?:\.\d+)?)\s*rpm\b", r"\1rpm"),
    (r"(\d+(?:\.\d+)?)\s*w\b", r"\1w"),
]
UNIT_RULES = [(re.compile(p), r) for p, r in UNIT_RULES]
MEASURE = re.compile(r"(\d+(?:\.\d+)?)(gb|tb|mb|ghz|mhz|in|rpm|w)\b")
NUM = re.compile(r"\d+(?:\.\d+)?")


def norm_units(s):
    for rx, rep in UNIT_RULES:
        s = rx.sub(rep, s)
    return s


def measures(s):
    """{единица: множество значений}; tb переводим в gb, чтобы 1tb == 1000gb."""
    out = {}
    for val, unit in MEASURE.findall(s):
        v = float(val)
        if unit == "tb":
            unit, v = "gb", v * 1000
        out.setdefault(unit, set()).add(round(v, 2))
    return out


# --------------------------------------------------------------------------- магазин
DOMAIN = re.compile(r"\b([a-z0-9-]+\.(?:com|co\.uk|de|es|nl|fr|net|it|pl))\b")
KNOWN_DASH_SHOPS = re.compile(r"\s-\s(tweakers|prijzen tweakers|specificaties tweakers|ncix|cnet|pcdiga|micro center|"
                              r"black friday 2017|hp store [a-z]+|server nexus llc|powerplanetonline)\s*$")


def shop(title):
    m = re.search(r"\|\s*([^|]+)$", title)
    if m:
        return re.sub(r"\s*-us$", "", m.group(1)).strip()
    m = DOMAIN.search(title)
    if m:
        return m.group(1)
    m = KNOWN_DASH_SHOPS.search(title)
    if m:
        return "tweakers" if "tweakers" in m.group(1) else m.group(1)
    if "wholesale" in title:
        return "wholesale"
    return ""


# --------------------------------------------------------------------------- коды
def codes(s):
    return {c.replace("-", "").replace("/", "") for c in CODE.findall(s)} - {"ddr3", "ddr4", "gddr5", "gddr6", "1080p"}


def code_pair_feats(cl, cr):
    """max сходство разных кодов, «почти совпадение» (hx426c15fb vs hx424c15fb), «один — префикс другого»."""
    best, near, prefix = 0.0, 0.0, 0.0
    for a in cl:
        for b in cr:
            if a == b:
                continue
            r = SequenceMatcher(None, a, b).ratio()
            best = max(best, r)
            if min(len(a), len(b)) >= 5 and r >= 0.8:
                near = 1.0
            if min(len(a), len(b)) >= 6 and (a.startswith(b) or b.startswith(a)):
                prefix = 1.0
    return best, near, prefix


def code_in_text(cl, text):
    """Доля кодов (длиной ≥5) одного оффера, найденных подстрокой в тексте другого (без дефисов/пробелов)."""
    cl = [c for c in cl if len(c) >= 5]
    if not cl:
        return 0.0
    return sum(c in text for c in cl) / len(cl)


GROUPS_V2 = {
    "measures": ["measure_conflict", "measure_match", "capacity_conflict"],
    "num_unmatched": ["num_unmatched_max", "num_unmatched_min", "digit_token_diff"],
    "code_diff": ["code_max_sim", "code_near_miss", "code_prefix", "code_exact"],
    "code_in_text": ["code_in_other_text"],
    "shop": ["same_shop", "both_shop_known"],
    "length": ["len_ratio"],
}


class FeaturizerV2(Featurizer):
    def transform(self, df):
        base = super().transform(df)
        tl, tr = df.title_left.map(norm_units), df.title_right.map(norm_units)
        rows = []
        for a, b, dl, dr in zip(tl, tr, df.description_left, df.description_right):
            ma, mb = measures(a), measures(b)
            common_units = ma.keys() & mb.keys()
            na, nb = set(NUM.findall(a)), set(NUM.findall(b))
            da = {t for t in a.split() if any(ch.isdigit() for ch in t)}
            db = {t for t in b.split() if any(ch.isdigit() for ch in t)}
            ca, cb = codes(a), codes(b)
            best, near, prefix = code_pair_feats(ca, cb)
            text_a = re.sub(r"[\s/-]", "", a + " " + dl)
            text_b = re.sub(r"[\s/-]", "", b + " " + dr)
            sa, sb = shop(a), shop(b)
            la, lb = len(a.split()), len(b.split())
            rows.append({
                "measure_conflict": sum(not (ma[u] & mb[u]) for u in common_units),
                "measure_match": sum(bool(ma[u] & mb[u]) for u in common_units),
                "capacity_conflict": float("gb" in common_units and not (ma["gb"] & mb["gb"])),
                "num_unmatched_max": max(len(na - nb), len(nb - na)),
                "num_unmatched_min": min(len(na - nb), len(nb - na)),
                "digit_token_diff": len(da ^ db),
                "code_max_sim": best,
                "code_near_miss": near * float(not (ca & cb)),
                "code_prefix": prefix,
                "code_exact": float(bool(ca & cb)),
                "code_in_other_text": max(code_in_text(ca, text_b), code_in_text(cb, text_a)),
                "same_shop": float(sa != "" and sa == sb),
                "both_shop_known": float(sa != "" and sb != ""),
                "len_ratio": min(la, lb) / max(la, lb, 1),
            })
        return pd.concat([base, pd.DataFrame(rows, index=base.index)], axis=1)


V2_COLS = [c for cols in GROUPS_V2.values() for c in cols]
