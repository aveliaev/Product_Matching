"""Серия 2 классического ML: признаки различий (features_v2), абляция новых групп,
подбор гиперпараметров бустинга на val, несколько seeds для RandomForest.

Результаты: results_v2.csv, графики report/v2_*.png, разбор ошибок печатается.
Запуск: python3 experiments_v2.py [small medium large xlarge]   |   --plots — только графики
"""
import itertools
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

from common import evaluate, load_split, paired_compare
from experiments import ALL, CLASSIFIERS, GRID, INK, MUTED, REPORT, SERIES, SIZES, ci, error_analysis, style
from features_v2 import GROUPS_V2, V2_COLS, FeaturizerV2

FULL = ALL + V2_COLS
REFERENCE = "grad_boosting | v1"  # лучшая конфигурация серии 1 — с ней сравниваем
RF_SEEDS = [0, 1, 2]
GB_GRID = {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31, 63],
           "min_samples_leaf": [20, 100], "l2_regularization": [0.0, 1.0]}


def gb(**kw):
    return HistGradientBoostingClassifier(max_iter=kw.pop("max_iter", 300), learning_rate=kw.pop("learning_rate", 0.05),
                                          random_state=0, **kw)


def rf(seed):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed)


def run_size(size):
    train, val, test = load_split(size)
    feat = FeaturizerV2().fit(train)
    X_tr, X_va, X_te = feat.transform(train), feat.transform(val), feat.transform(test)
    rows, preds = [], {}

    def record(section, name, clf, cols, extra=None):
        clf.fit(X_tr[cols], train.label)
        va_s, te_s = clf.predict_proba(X_va[cols])[:, 1], clf.predict_proba(X_te[cols])[:, 1]
        r = evaluate(name, size, val, va_s, test, te_s, save=False)
        preds[name] = (r.pop("test_pred"), te_s)
        r.update(section=section, **(extra or {}))
        rows.append(r)
        print(f"  {size:7} {name:40} val F1={r['val_F1']:.3f}  test F1={r['test_F1']:.3f} {r['test_F1_CI']}  "
              f"P={r['test_P']:.3f} R={r['test_R']:.3f}", flush=True)
        return r

    # A. v1 vs v2 для трёх классификаторов
    for fs, cols in [("v1", ALL), ("v2", FULL)]:
        record("A v1 vs v2", f"logreg | {fs}", CLASSIFIERS["logreg"](), cols)
        record("A v1 vs v2", f"grad_boosting | {fs}", gb(), cols)
        for seed in RF_SEEDS:
            record("A v1 vs v2", f"random_forest | {fs} | seed {seed}", rf(seed), cols, {"seed": seed})

    # B. абляция новых групп (бустинг, v2 минус группа)
    for g, cols in GROUPS_V2.items():
        record("B ablation v2", f"grad_boosting | v2 − {g}", gb(), [c for c in FULL if c not in cols])

    # C. подбор гиперпараметров бустинга по val F1 (test при выборе не используется)
    grid = [dict(zip(GB_GRID, v)) for v in itertools.product(*GB_GRID.values())]
    tuned = []
    for params in grid:
        clf = gb(max_iter=500, **params).fit(X_tr[FULL], train.label)
        va_s = clf.predict_proba(X_va[FULL])[:, 1]
        t = np.quantile(va_s, np.linspace(0, 1, 201))
        tuned.append((max(f1_score(val.label, va_s >= x, zero_division=0) for x in t), params))
    best_val, best_params = max(tuned, key=lambda x: x[0])
    print(f"  лучшие параметры по val (F1={best_val:.3f}): {best_params}")
    record("C tuned", "grad_boosting | v2 | tuned", gb(max_iter=500, **best_params), FULL,
           {"params": str(best_params)})

    y = test.label
    for r in rows:
        r.update(paired_compare(y, preds[REFERENCE][0], preds[r["model"]][0]))
        # второе сравнение — с полной моделью v2 (для абляции и тюнинга)
        v2 = paired_compare(y, preds["grad_boosting | v2"][0], preds[r["model"]][0])
        r.update({f"{k}_vs_v2": v for k, v in v2.items()})
    return rows, preds, test


def plot_v1_v2(df):
    """F1 на test: v1 vs v2 для трёх классификаторов (RF — среднее по seeds), xlarge."""
    d = df.copy()
    d["base"] = d.model.str.replace(r" \| seed \d+", "", regex=True)
    agg = d[d.section == "A v1 vs v2"].groupby(["train", "base"]).test_F1.mean().unstack()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    x = np.arange(len(SIZES))
    for ax, clf in zip(axes, ["logreg", "random_forest", "grad_boosting"]):
        for i, (fs, color) in enumerate([("v1", SERIES[0]), ("v2", SERIES[1])]):
            vals = agg.loc[SIZES, f"{clf} | {fs}"]
            ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=color, label=f"признаки {fs}")
            for xi, v in zip(x, vals):
                ax.text(xi + (i - 0.5) * 0.38, v + 0.005, f"{v:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, SIZES)
        ax.set_title(clf, color=INK, loc="left")
        ax.set_ylim(0.6, 0.95)
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].set_ylabel("F1 на test", color=MUTED)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Признаки v1 (только сходство) vs v2 (+ различия, коды, магазин) — F1 на test "
                 "(RF — среднее по 3 seeds)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(REPORT / "v2_v1_vs_v2.png", dpi=120)


def plot_ablation(df):
    """ΔF1 на test от удаления каждой новой группы (относительно полной v2), paired bootstrap 95% ДИ."""
    groups = list(GROUPS_V2)
    fig, axes = plt.subplots(1, len(SIZES), figsize=(14, 3.8), sharey=True, sharex=True)
    for ax, size in zip(axes, SIZES):
        d = df[(df.train == size) & (df.section == "B ablation v2")].set_index("model")
        d = d.loc[[f"grad_boosting | v2 − {g}" for g in groups]]
        y = np.arange(len(groups))
        lo, hi = d.dF1_lo_vs_v2.values, d.dF1_hi_vs_v2.values
        significant = hi < 0
        ax.hlines(y, lo, hi, color=MUTED, lw=1.5)
        ax.scatter(d.dF1_vs_v2, y, s=60, zorder=3, color=[SERIES[1] if sg else SERIES[0] for sg in significant])
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(y, [f"без {g}" for g in groups])
        ax.set_title(size, color=INK, loc="left")
        style(ax)
    axes[0].invert_yaxis()
    fig.supxlabel("ΔF1 на test относительно полной модели v2 (отрицательно — группа полезна)", color=MUTED, fontsize=10)
    fig.suptitle("Абляция новых групп признаков (бустинг): оранжевые — вклад значим (95% ДИ целиком < 0)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(REPORT / "v2_ablation.png", dpi=120)


def plots(df):
    plot_v1_v2(df)
    plot_ablation(df)


def main(sizes):
    REPORT.mkdir(exist_ok=True)
    all_rows = []
    for size in sizes:
        print(f"\n=== train = {size} ===")
        rows, preds, test = run_size(size)
        all_rows += rows
        if size == sizes[-1]:
            name = "grad_boosting | v2 | tuned"
            print(f"\n=== Разбор ошибок: {name} (train={size}) ===")
            error_analysis(test, *preds[name])
    df = pd.DataFrame(all_rows)
    df.to_csv("results_v2.csv", index=False)
    if sizes == SIZES:
        plots(df)

    cols = ["val_F1", "test_F1", "test_F1_CI", "test_P", "test_R", "test_PR_AUC", "dF1", "dF1_lo", "dF1_hi", "mcnemar_p"]
    with pd.option_context("display.width", 250, "display.max_rows", 300, "display.max_colwidth", 60):
        for size in sizes:
            print(f"\n=== {size}: сравнение с «{REFERENCE}» ===")
            print(df[df.train == size].set_index(["section", "model"])[cols].round(3).to_string())
            print(f"--- {size}: сравнение с «grad_boosting | v2» ---")
            v2cols = ["test_F1", "dF1_vs_v2", "dF1_lo_vs_v2", "dF1_hi_vs_v2", "mcnemar_p_vs_v2"]
            print(df[(df.train == size) & df.section.isin(["B ablation v2", "C tuned"])].set_index("model")[v2cols]
                  .round(3).to_string())
        d = df.assign(base=df.model.str.replace(r" \| seed \d+", "", regex=True))
        print("\n=== F1 на test (RF: mean ± std по seeds) ===")
        g = d.groupby(["base", "train"]).test_F1.agg(["mean", "std"])
        g = g.apply(lambda r: f"{r['mean']:.3f}" + (f" ± {r['std']:.3f}" if r["std"] == r["std"] else ""), axis=1)
        print(g.unstack()[sizes].to_string())
        print("\n=== подобранные параметры ===")
        print(df[df.section == "C tuned"][["train", "params"]].to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["--plots"]:
        plots(pd.read_csv("results_v2.csv"))
    else:
        main(sys.argv[1:] or SIZES)
