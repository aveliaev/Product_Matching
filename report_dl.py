"""Сводка этапа 3: трансформеры vs бустинг v2. Таблица mean ± std по seeds, кривая обучения,
разбор ошибок, простые ансамбли. Всё — по сохранённым предсказаниям в predictions/ (без переобучения).

Запуск: python3 report_dl.py   (сначала python3 compare_dl.py, чтобы были предсказания бустинга)
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import PRED_DIR, best_threshold, bootstrap_f1, load_split, metrics, paired_compare
from compare_dl import gb_v2_preds, load_preds
from experiments import GRID, INK, MUTED, SERIES, style

SIZES = ["small", "medium", "large", "xlarge"]
PATTERN = re.compile(r"dl__(?P<model>[^_]+(?:-[^_]+)*)__(?P<fields>[^_]+)__(?P<size>\w+?)__seed(?P<seed>\d+)(?:__(?P<tag>.+))?$")


def collect():
    rows = []
    for f in sorted(PRED_DIR.glob("dl__*__test.csv")):
        prefix = str(f)[: -len("__test.csv")]
        m = PATTERN.match(Path(prefix).name)
        p = load_preds(prefix)
        t = best_threshold(p["val"].label.values, p["val"].score.values)
        met = metrics(p["test"].label.values, p["test"].score.values, t)
        rows.append({**m.groupdict(), "seed": int(m["seed"]), "tag": m["tag"] or "", "t": t, "prefix": prefix,
                     **{f"test_{k}": v for k, v in met.items()}})
    return pd.DataFrame(rows)


def scores_with_threshold(p):
    t = best_threshold(p["val"].label.values, p["val"].score.values)
    return (p["test"].score.values >= t).astype(int)


def ensemble(parts, weights=None):
    """Среднее скоров val и test по нескольким моделям (веса — опционально)."""
    w = np.ones(len(parts)) / len(parts) if weights is None else np.asarray(weights)
    out = {}
    for part in ("val", "test"):
        s = sum(wi * p[part].score.values for wi, p in zip(w, parts))
        out[part] = parts[0][part].assign(score=s)
    return out


def main():
    df = collect()
    main_runs = df[(df.fields == "title") & (df.tag == "")]
    gb = {s: gb_v2_preds(s) for s in SIZES}
    gb_f1 = {s: metrics(gb[s]["test"].label.values, gb[s]["test"].score.values,
                        best_threshold(gb[s]["val"].label.values, gb[s]["val"].score.values))["F1"] for s in SIZES}

    # ---------- сводная таблица
    print("=== F1 на test: mean ± std по seeds (только заголовки, 5 эпох) ===")
    tab = main_runs.groupby(["model", "size"]).test_F1.agg(["mean", "std", "count"]).reset_index()
    for s in SIZES:
        print(f"{s:7} бустинг v2: {gb_f1[s]:.3f}")
        for _, r in tab[tab["size"] == s].iterrows():
            print(f"        {r['model']:24} {r['mean']:.3f} ± {r['std']:.3f}  (n={int(r['count'])})  Δ={r['mean'] - gb_f1[s]:+.3f}")

    # ---------- кривая обучения
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SIZES))
    lo_hi = np.array([bootstrap_f1(gb[s]["test"].label.values, scores_with_threshold(gb[s])) for s in SIZES])
    ax.fill_between(x, lo_hi[:, 0], lo_hi[:, 1], color=SERIES[0], alpha=0.12, lw=0)
    ax.plot(x, [gb_f1[s] for s in SIZES], color=SERIES[0], lw=2, marker="o", ms=8, label="градиентный бустинг v2 (95% ДИ)")
    for color, model in [(SERIES[2], "multilingual-e5-small"), (SERIES[1], "multilingual-e5-base")]:
        d = tab[tab.model == model].set_index("size").reindex(SIZES)
        ok = d["mean"].notna().values
        runs = main_runs[main_runs.model == model]
        for i, s in enumerate(SIZES):
            v = runs[runs["size"] == s].test_F1
            ax.scatter([i] * len(v), v, color=color, s=18, alpha=0.5, zorder=3)
        ax.plot(x[ok], d["mean"].values[ok], color=color, lw=2, marker="o", ms=8, label=f"{model} (среднее, точки — seeds)")
        last = np.where(ok)[0][-1]
        ax.text(x[last] + 0.08, d["mean"].values[last], f"{d['mean'].values[last]:.3f}", color=INK, va="center", fontsize=9)
    ax.text(x[-1] + 0.08, gb_f1["xlarge"], f"{gb_f1['xlarge']:.3f}", color=INK, va="center", fontsize=9)
    ax.set_xticks(x, SIZES)
    ax.set_xlim(-0.2, len(SIZES) - 0.5)
    ax.set_ylim(0.8, 0.98)
    ax.set_ylabel("F1 на test", color=MUTED)
    ax.set_title("Трансформер vs бустинг: F1 на test по размеру train", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig("report/dl_learning_curve.png", dpi=120)

    # ---------- ансамбли (xlarge и small)
    print("\n=== Ансамбли (среднее вероятностей, порог по val) ===")
    for s in SIZES:
        runs = main_runs[(main_runs.model == "multilingual-e5-base") & (main_runs["size"] == s)].sort_values("seed")
        dl = [load_preds(p) for p in runs.prefix]
        if not dl:
            continue
        variants = {"e5-base, 3 seeds": ensemble(dl), "e5-base (3 seeds) + бустинг": ensemble([ensemble(dl), gb[s]])}
        for name, p in variants.items():
            t = best_threshold(p["val"].label.values, p["val"].score.values)
            m = metrics(p["test"].label.values, p["test"].score.values, t)
            c = paired_compare(p["test"].label.values, scores_with_threshold(gb[s]), (p["test"].score.values >= t).astype(int))
            print(f"{s:7} {name:30} F1={m['F1']:.3f} P={m['P']:.3f} R={m['R']:.3f}  "
                  f"vs бустинг ΔF1={c['dF1']:+.3f} [{c['dF1_lo']:+.3f}, {c['dF1_hi']:+.3f}]")

    # ---------- разбор ошибок: лучший по val seed e5-base на xlarge vs бустинг
    _, _, test = load_split("xlarge")
    runs = main_runs[(main_runs.model == "multilingual-e5-base") & (main_runs["size"] == "xlarge")]
    best = runs.loc[runs.prefix.map(lambda p: load_preds(p)["val"]).map(
        lambda v: metrics(v.label.values, v.score.values, best_threshold(v.label.values, v.score.values))["F1"]).idxmax()]
    p = load_preds(best.prefix)
    dl_pred, gb_pred = scores_with_threshold(p), scores_with_threshold(gb["xlarge"])
    y = test.label.values
    t = test.assign(dl=dl_pred, gb=gb_pred, dl_score=p["test"].score.values)
    print(f"\n=== Ошибки на test, xlarge: e5-base seed {best.seed} (лучший по val) vs бустинг v2 ===")
    for name, pr in [("e5-base", dl_pred), ("бустинг", gb_pred)]:
        print(f"{name:8} FP={int(((pr == 1) & (y == 0)).sum())}  FN={int(((pr == 0) & (y == 1)).sum())}")
    both = ((dl_pred != y) & (gb_pred != y)).sum()
    print(f"ошибаются обе: {both}; только бустинг: {((gb_pred != y) & (dl_pred == y)).sum()}; "
          f"только e5-base: {((dl_pred != y) & (gb_pred == y)).sum()}")
    cols = ["label", "dl_score", "title_left", "title_right"]
    with pd.option_context("display.max_colwidth", 75, "display.width", 250):
        print("\n--- бустинг ошибался, e5-base исправил (примеры):")
        print(t[(t.gb != t.label) & (t.dl == t.label)].sample(12, random_state=0)[cols].to_string(index=False))
        print("\n--- e5-base ошибается (все FP и FN):")
        print(t[t.dl != t.label].sort_values("label")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
