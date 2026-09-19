"""Этап 4: сводка всех этапов — одна таблица и одна кривая обучения (report/final_learning_curve.png).

Берёт готовые результаты: results_experiments.csv (серия 1), results_v2.csv (серия 2), predictions/ (DL).
Запуск: python3 final_report.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import best_threshold, bootstrap_f1, metrics
from compare_dl import gb_v2_preds, load_preds
from experiments import GRID, INK, MUTED, style
from report_dl import collect, ensemble

SIZES = ["small", "medium", "large", "xlarge"]
COLORS = ["#52514e", "#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#4a3aa7"]


def fmt(v, std=None):
    return f"{v:.3f}" + (f" ± {std:.3f}" if std is not None else "")


def main():
    s1 = pd.read_csv("results_experiments.csv")
    s2 = pd.read_csv("results_v2.csv")
    rows = {}  # название -> {size: (F1, std, CI)}

    for label, df, model in [("cosine заголовков, без обучения", s1, "title cosine (no training)"),
                             ("9 признаков сходства + логрег", s1, "logreg | all"),
                             ("9 признаков сходства + бустинг", s1, "grad_boosting | all"),
                             ("23 признака (+ различия) + бустинг", s2, "grad_boosting | v2")]:
        d = df[df.model == model].set_index("train")
        rows[label] = {s: (d.loc[s, "test_F1"], None, d.loc[s, "test_F1_CI"]) for s in SIZES}

    dl = collect()
    base = dl[(dl.model == "multilingual-e5-base") & (dl.fields == "title") & (dl.tag == "")]
    rows["трансформер e5-base (3 seeds)"] = {}
    rows["ансамбль e5-base + бустинг"] = {}
    for s in SIZES:
        runs = base[base["size"] == s].sort_values("seed")
        rows["трансформер e5-base (3 seeds)"][s] = (runs.test_F1.mean(), runs.test_F1.std(), None)
        p = ensemble([ensemble([load_preds(x) for x in runs.prefix]), gb_v2_preds(s)])
        t = best_threshold(p["val"].label.values, p["val"].score.values)
        y, sc = p["test"].label.values, p["test"].score.values
        lo, hi = bootstrap_f1(y, (sc >= t).astype(int))
        rows["ансамбль e5-base + бустинг"][s] = (metrics(y, sc, t)["F1"], None, f"[{lo:.3f}, {hi:.3f}]")

    print("| модель | " + " | ".join(SIZES) + " |")
    print("|---|" + "---|" * len(SIZES))
    for label, vals in rows.items():
        print(f"| {label} | " + " | ".join(fmt(v[0], v[1]) for v in vals.values()) + " |")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(SIZES))
    ends = {label: vals["xlarge"][0] for label, vals in rows.items()}
    label_y, prev = {}, None
    for label in sorted(ends, key=ends.get):  # раздвигаем подписи на конце линий
        label_y[label] = ends[label] if prev is None else max(ends[label], prev + 0.013)
        prev = label_y[label]
    for color, (label, vals) in zip(COLORS, rows.items()):
        y = np.array([vals[s][0] for s in SIZES])
        ax.plot(x, y, color=color, lw=2, marker="o", ms=7, label=label)
        ax.text(x[-1] + 0.08, label_y[label], f"{y[-1]:.3f}", color=INK, va="center", fontsize=9)
    ax.set_xticks(x, [f"{s}\n({n} пар)" for s, n in zip(SIZES, ["2.8 тыс.", "8.1 тыс.", "33 тыс.", "68 тыс."])])
    ax.set_xlim(-0.2, len(SIZES) - 0.45)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("F1 на test", color=MUTED)
    ax.set_title("WDC Computers: F1 на test по этапам проекта и размеру train", color=INK, loc="left")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.17), fontsize=9)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig("report/final_learning_curve.png", dpi=120)


if __name__ == "__main__":
    main()
