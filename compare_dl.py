"""Сравнение трансформеров (predictions/dl__*) с лучшей классикой — бустингом на признаках v2.

Предсказания бустинга считаются один раз и кэшируются в predictions/gb_v2__{size}__{val,test}.csv.
Порог для каждой модели подбирается по её val-предсказаниям.
Запуск: python3 compare_dl.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import PRED_DIR, best_threshold, bootstrap_f1, load_split, metrics, paired_compare

SIZES = ["small", "medium", "large", "xlarge"]


def gb_v2_preds(size):
    """Бустинг v2 с параметрами по умолчанию (как 'grad_boosting | v2' в experiments_v2)."""
    paths = {part: PRED_DIR / f"gb_v2__{size}__{part}.csv" for part in ("val", "test")}
    if not all(p.exists() for p in paths.values()):
        from experiments_v2 import FULL, gb
        from features_v2 import FeaturizerV2

        train, val, test = load_split(size)
        feat = FeaturizerV2().fit(train)
        clf = gb().fit(feat.transform(train)[FULL], train.label)
        for part, df in [("val", val), ("test", test)]:
            s = clf.predict_proba(feat.transform(df)[FULL])[:, 1]
            pd.DataFrame({"pair_id": df.pair_id, "label": df.label, "score": s}).to_csv(paths[part], index=False)
    return {part: pd.read_csv(p) for part, p in paths.items()}


def load_preds(prefix):
    return {part: pd.read_csv(f"{prefix}__{part}.csv") for part in ("val", "test")}


def score(p):
    t = best_threshold(p["val"].label.values, p["val"].score.values)
    y, s = p["test"].label.values, p["test"].score.values
    return t, (s >= t).astype(int), metrics(y, s, t)


def main():
    rows = []
    for size in SIZES:
        files = sorted(Path(PRED_DIR).glob(f"dl__*__{size}__*__test.csv"))
        if not files:
            continue
        ref = gb_v2_preds(size)
        _, ref_pred, ref_m = score(ref)
        rows.append({"train": size, "model": "grad_boosting v2 (классика)", "test_F1": ref_m["F1"],
                     "test_P": ref_m["P"], "test_R": ref_m["R"]})
        for f in files:
            prefix = str(f)[: -len("__test.csv")]
            p = load_preds(prefix)
            assert (p["test"].pair_id.values == ref["test"].pair_id.values).all()
            _, pred, m = score(p)
            lo, hi = bootstrap_f1(p["test"].label.values, pred)
            cmp = paired_compare(p["test"].label.values, ref_pred, pred)
            rows.append({"train": size, "model": re.sub(r"^dl__", "", Path(prefix).name), "test_F1": m["F1"],
                         "CI": f"[{lo:.3f}, {hi:.3f}]", "test_P": m["P"], "test_R": m["R"],
                         "dF1_vs_gb": cmp["dF1"], "dF1_CI": f"[{cmp['dF1_lo']:+.3f}, {cmp['dF1_hi']:+.3f}]",
                         "mcnemar_p": cmp["mcnemar_p"]})
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(df.round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
