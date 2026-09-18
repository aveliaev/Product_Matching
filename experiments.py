"""Серия экспериментов с классическим ML (этап 2): абляции признаков, классификаторы, чистка текста,
представление пары, выбор порога. Все конфигурации × 4 размера train.

Результаты: results_experiments.csv, графики в report/, разбор ошибок печатается.
Запуск: python3 experiments.py [small medium large xlarge]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import NUM, codes, jaccard, rowwise_cos
from common import best_threshold, evaluate, load_split, paired_compare

SIZES = ["small", "medium", "large", "xlarge"]
REPORT = Path("report")
REFERENCE = "logreg | all"  # с ней сравниваем все остальные конфигурации

GROUPS = {
    "title_word": ["title_word_cos"],
    "title_char": ["title_char_cos"],
    "title_jaccard": ["title_jaccard"],
    "description": ["desc_cos", "desc_missing"],
    "codes": ["code_jaccard", "codes_disjoint"],
    "numbers": ["num_jaccard"],
    "brand": ["brand_eq"],
}
ALL = [c for cols in GROUPS.values() for c in cols]

CLASSIFIERS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0),
    "grad_boosting": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
}


class Featurizer:
    """Все признаки похожести разом; конфигурации выбирают подмножество колонок."""

    def fit(self, train):
        titles = pd.concat([train.title_left, train.title_right])
        self.word = TfidfVectorizer(token_pattern=r"[a-z0-9]+", ngram_range=(1, 2), sublinear_tf=True).fit(titles)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True).fit(titles)
        self.desc = TfidfVectorizer(token_pattern=r"[a-z0-9]+", sublinear_tf=True).fit(
            pd.concat([train.description_left, train.description_right]))
        return self

    def transform(self, df):
        cl, cr = df.title_left.map(codes), df.title_right.map(codes)
        nl, nr = df.title_left.map(lambda s: set(NUM.findall(s))), df.title_right.map(lambda s: set(NUM.findall(s)))
        has_codes = (cl.str.len() > 0) & (cr.str.len() > 0)
        return pd.DataFrame({
            "title_word_cos": rowwise_cos(self.word, df.title_left, df.title_right),
            "title_char_cos": rowwise_cos(self.char, df.title_left, df.title_right),
            "title_jaccard": [jaccard(set(a.split()), set(b.split())) for a, b in zip(df.title_left, df.title_right)],
            "desc_cos": rowwise_cos(self.desc, df.description_left, df.description_right),
            "desc_missing": ((df.description_left == "") | (df.description_right == "")).astype(float),
            "code_jaccard": [jaccard(a, b) for a, b in zip(cl, cr)],
            "codes_disjoint": (has_codes & np.array([not (a & b) for a, b in zip(cl, cr)])).astype(float),
            "num_jaccard": [jaccard(a, b) for a, b in zip(nl, nr)],
            "brand_eq": ((df.brand_left == df.brand_right) & (df.brand_left != "")).astype(float),
        })


class PairVectors:
    """Альтернативное представление: TF-IDF векторы заголовков, пара = [|a-b|, a*b]."""

    def fit(self, train):
        self.vec = TfidfVectorizer(token_pattern=r"[a-z0-9]+", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(
            pd.concat([train.title_left, train.title_right]))
        return self

    def transform(self, df):
        a, b = self.vec.transform(df.title_left), self.vec.transform(df.title_right)
        return sp.hstack([abs(a - b), a.multiply(b)]).tocsr()


def feature_configs():
    """(название, колонки) для абляций на логреге."""
    yield "all", ALL
    for g, cols in GROUPS.items():
        yield f"only {g}", cols
    for g in GROUPS:
        yield f"all − {g}", [c for c in ALL if c not in GROUPS[g]]


def run_size(size):
    rows, preds = [], {}

    def record(section, name, val, val_s, test, test_s):
        r = evaluate(name, size, val, val_s, test, test_s, save=False)
        pred = r.pop("test_pred")
        # чувствительность к порогу: стандартный 0.5 и «оракул» (лучший порог на test — только для анализа!)
        r["test_F1_t05"] = f1_score(test.label, test_s >= 0.5) if test_s.max() <= 1 else np.nan
        r["test_F1_oracle"] = f1_score(test.label, test_s >= best_threshold(test.label, test_s))
        r["section"] = section
        rows.append(r)
        preds[name] = (pred, test_s)
        print(f"  {size:7} {name:32} test F1={r['test_F1']:.3f} {r['test_F1_CI']}  P={r['test_P']:.3f} "
              f"R={r['test_R']:.3f}  PR-AUC={r['test_PR_AUC']:.3f}", flush=True)

    for clean_mode in ["light", "none", "full"]:
        train, val, test = load_split(size, clean_mode)
        feat = Featurizer().fit(train)
        X_tr, X_va, X_te = feat.transform(train), feat.transform(val), feat.transform(test)

        if clean_mode == "light":
            # 0. тривиальный бейзлайн
            record("0 trivial", "title cosine (no training)", val, X_va.title_word_cos.values, test, X_te.title_word_cos.values)
            # A. абляции признаков
            for name, cols in feature_configs():
                clf = CLASSIFIERS["logreg"]().fit(X_tr[cols], train.label)
                record("A features", f"logreg | {name}", val, clf.predict_proba(X_va[cols])[:, 1],
                       test, clf.predict_proba(X_te[cols])[:, 1])
            # B. классификаторы
            for cname in ["random_forest", "grad_boosting"]:
                clf = CLASSIFIERS[cname]().fit(X_tr, train.label)
                record("B classifier", f"{cname} | all", val, clf.predict_proba(X_va)[:, 1], test, clf.predict_proba(X_te)[:, 1])
            # D. представление пары векторами
            pv = PairVectors().fit(train)
            clf = LogisticRegression(max_iter=2000, C=4).fit(pv.transform(train), train.label)
            record("D pair vectors", "logreg | tfidf pair vectors", val, clf.predict_proba(pv.transform(val))[:, 1],
                   test, clf.predict_proba(pv.transform(test))[:, 1])
            best_clean = (train, val, test, X_tr, X_va, X_te)
        else:
            # C. другие режимы чистки текста (по умолчанию "light")
            for cname in ["logreg", "grad_boosting"]:
                clf = CLASSIFIERS[cname]().fit(X_tr, train.label)
                record("C cleaning", f"{cname} | all | clean={clean_mode}", val, clf.predict_proba(X_va)[:, 1],
                       test, clf.predict_proba(X_te)[:, 1])

    y = best_clean[2].label
    for r in rows:
        r.update(paired_compare(y, preds[REFERENCE][0], preds[r["model"]][0]))
    return rows, preds, best_clean


def error_analysis(test, pred, scores):
    """Разбор ошибок: к каким типам относятся FP и FN."""
    cl, cr = test.title_left.map(codes), test.title_right.map(codes)
    nl, nr = test.title_left.map(lambda s: set(NUM.findall(s))), test.title_right.map(lambda s: set(NUM.findall(s)))
    t = test.assign(pred=pred, score=scores, shared_code=[bool(a & b) for a, b in zip(cl, cr)],
                    nums_differ=[bool(a ^ b) for a, b in zip(nl, nr)])
    fp, fn = t[(t.pred == 1) & (t.label == 0)], t[(t.pred == 0) & (t.label == 1)]
    print(f"\nFP={len(fp)}  FN={len(fn)}")
    print(f"FP: есть общий код модели {fp.shared_code.mean():.0%}, числа в заголовках различаются {fp.nums_differ.mean():.0%}")
    print(f"FN: нет общего кода {(~fn.shared_code).mean():.0%}, числа различаются {fn.nums_differ.mean():.0%}")
    cols = ["score", "title_left", "title_right"]
    with pd.option_context("display.max_colwidth", 80, "display.width", 250):
        print("\nсамые уверенные FP:\n", fp.nlargest(10, "score")[cols].to_string(index=False))
        print("\nсамые уверенные FN:\n", fn.nsmallest(10, "score")[cols].to_string(index=False))


# --------------------------------------------------------------------------- графики
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def style(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ci(s):
    return np.array([float(x) for x in s.strip("[]").split(",")])


def plot_ablation(df, size):
    d = df[(df.train == size) & (df.section == "A features")].set_index("model")
    ref = d.loc[REFERENCE, "test_F1"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for ax, prefix, title in [(axes[0], "logreg | only ", "Только одна группа признаков"),
                              (axes[1], "logreg | all − ", "Все, кроме одной группы")]:
        sub = d[d.index.str.startswith(prefix)].sort_values("test_F1")
        labels = sub.index.str.replace(prefix, "", regex=False)
        err = np.abs(np.stack(sub.test_F1_CI.map(ci)).T - sub.test_F1.values)
        ax.barh(labels, sub.test_F1, color=SERIES[0], height=0.6, xerr=err, error_kw={"ecolor": MUTED, "lw": 1})
        ax.axvline(ref, color=SERIES[1], lw=2, ls="--")
        ax.text(ref, len(sub) - 0.4, f" все признаки {ref:.3f}", color=INK, fontsize=9, va="bottom")
        for i, v in enumerate(sub.test_F1):
            ax.text(0.01, i, f"{v:.3f}", color="white", va="center", fontsize=9, fontweight="bold")
        ax.set_title(title, color=INK, loc="left")
        style(ax)
    axes[0].set_xlim(0, 0.85)
    fig.suptitle(f"Абляция признаков, логрег, train={size} — F1 на test (95% ДИ)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(REPORT / f"ablation_{size}.png", dpi=120)


def plot_learning_curve(df):
    names = ["title cosine (no training)", "logreg | all", "random_forest | all", "grad_boosting | all",
             "logreg | tfidf pair vectors"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SIZES))
    ends = {n: df[(df.model == n) & (df.train == SIZES[-1])].test_F1.iloc[0] for n in names}
    label_y, prev = {}, None
    for n in sorted(ends, key=ends.get):  # раздвигаем подписи на конце линий, чтобы не наезжали
        label_y[n] = ends[n] if prev is None else max(ends[n], prev + 0.014)
        prev = label_y[n]
    for color, name in zip(SERIES, names):
        d = df[df.model == name].set_index("train").reindex(SIZES)
        lo_hi = np.stack(d.test_F1_CI.map(ci))
        ax.fill_between(x, lo_hi[:, 0], lo_hi[:, 1], color=color, alpha=0.12, lw=0)
        ax.plot(x, d.test_F1, color=color, lw=2, marker="o", ms=8, label=name)
        ax.text(x[-1] + 0.08, label_y[name], f"{d.test_F1.iloc[-1]:.3f}", color=INK, va="center", fontsize=9)
    ax.set_xticks(x, [f"{s}" for s in SIZES])
    ax.set_xlim(-0.2, len(SIZES) - 0.5)
    ax.set_ylabel("F1 на test", color=MUTED)
    ax.set_title("Кривая обучения: F1 на test vs размер train (заливка — 95% ДИ)", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=2)
    ax.set_ylim(0.5, 0.95)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(REPORT / "learning_curve.png", dpi=120)


def plots_only():
    """Перерисовать графики из results_experiments.csv без пересчёта."""
    df = pd.read_csv("results_experiments.csv")
    for size in SIZES:
        plot_ablation(df, size)
    plot_learning_curve(df)


def main(sizes):
    REPORT.mkdir(exist_ok=True)
    all_rows = []
    for size in sizes:
        print(f"\n=== train = {size} ===")
        rows, preds, (train, val, test, *_) = run_size(size)
        all_rows += rows
        if size == sizes[-1]:
            best = max(rows, key=lambda r: r["val_F1"])  # лучшую выбираем по val, не по test
            print(f"\n=== Разбор ошибок: {best['model']} (train={size}, выбрана по val F1) ===")
            error_analysis(test, *preds[best["model"]])

    df = pd.DataFrame(all_rows)
    df.to_csv("results_experiments.csv", index=False)
    for size in sizes:
        plot_ablation(df, size)
    if sizes == SIZES:
        plot_learning_curve(df)

    cols = ["test_F1", "test_P", "test_R", "test_PR_AUC", "dF1", "dF1_lo", "dF1_hi", "mcnemar_p"]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        for size in sizes:
            print(f"\n=== {size}: сравнение с «{REFERENCE}» ===")
            print(df[df.train == size].set_index(["section", "model"])[cols].round(3).to_string())
        print("\n=== F1 на test: конфигурация × размер ===")
        print(df.pivot_table(index=["section", "model"], columns="train", values="test_F1")[sizes].round(3).to_string())
        print("\n=== порог: val-подобранный vs 0.5 vs оракул (test) ===")
        print(df[df.section.isin(["B classifier"]) | (df.model == REFERENCE)]
              .set_index(["train", "model"])[["threshold", "test_F1", "test_F1_t05", "test_F1_oracle"]].round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["--plots"]:
        plots_only()
    else:
        main(sys.argv[1:] or SIZES)
