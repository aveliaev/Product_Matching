"""Этап 1: простые бейзлайны.

  1) cosine TF-IDF заголовков + порог (без обучения)
  2) несколько признаков похожести -> LogisticRegression

Запуск: python3 baseline.py [small medium large xlarge]
"""
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import evaluate, load_split

# токен с буквами и цифрами длиной 4+ (коды моделей: hx426c15fb, 70dg007qux, ...)
CODE = re.compile(r"\b(?=[a-z0-9/-]*\d)(?=[a-z0-9/-]*[a-z])[a-z0-9/-]{4,}\b")
NUM = re.compile(r"\d+(?:\.\d+)?")


def codes(s):
    return {c.replace("-", "") for c in CODE.findall(s)}


def rowwise_cos(vec, a, b):
    return np.asarray(vec.transform(a).multiply(vec.transform(b)).sum(1)).ravel()


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


class TitleCosine:
    """Бейзлайн 1: только cosine TF-IDF заголовков, порог подбирается на val."""

    def fit(self, train):
        self.vec = TfidfVectorizer(token_pattern=r"[a-z0-9]+", sublinear_tf=True).fit(
            pd.concat([train.title_left, train.title_right]))
        return self

    def predict_proba(self, df):
        return rowwise_cos(self.vec, df.title_left, df.title_right)


class SimilarityLogReg:
    """Бейзлайн 2: признаки похожести пары -> логистическая регрессия."""

    def features(self, df):
        cl, cr = df.title_left.map(codes), df.title_right.map(codes)
        nl, nr = df.title_left.map(lambda s: set(NUM.findall(s))), df.title_right.map(lambda s: set(NUM.findall(s)))
        has_codes = (cl.str.len() > 0) & (cr.str.len() > 0)
        return np.column_stack([
            rowwise_cos(self.word, df.title_left, df.title_right),
            rowwise_cos(self.char, df.title_left, df.title_right),
            rowwise_cos(self.desc, df.description_left, df.description_right),
            [jaccard(set(a.split()), set(b.split())) for a, b in zip(df.title_left, df.title_right)],
            [jaccard(a, b) for a, b in zip(cl, cr)],                        # пересечение кодов моделей
            has_codes & np.array([not (a & b) for a, b in zip(cl, cr)]),    # коды есть, но ни один не совпал
            [jaccard(a, b) for a, b in zip(nl, nr)],                        # пересечение чисел (объём, частота...)
            (df.brand_left == df.brand_right) & (df.brand_left != ""),
            (df.description_left == "") | (df.description_right == ""),
        ]).astype(float)

    def fit(self, train):
        titles = pd.concat([train.title_left, train.title_right])
        self.word = TfidfVectorizer(token_pattern=r"[a-z0-9]+", ngram_range=(1, 2), sublinear_tf=True).fit(titles)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True).fit(titles)
        self.desc = TfidfVectorizer(token_pattern=r"[a-z0-9]+", sublinear_tf=True).fit(
            pd.concat([train.description_left, train.description_right]))
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        self.clf.fit(self.features(train), train.label)
        return self

    def predict_proba(self, df):
        return self.clf.predict_proba(self.features(df))[:, 1]


MODELS = {"title_cosine": TitleCosine, "similarity_logreg": SimilarityLogReg}


def main(sizes):
    results = []
    for size in sizes:
        train, val, test = load_split(size)
        for name, Model in MODELS.items():
            m = Model().fit(train)
            r = evaluate(name, size, val, m.predict_proba(val), test, m.predict_proba(test))
            results.append(r)
            print(f"{size:7} {name:18} val F1={r['val_F1']:.3f} (hard {r['val_hard_F1']:.3f})  |  "
                  f"test F1={r['test_F1']:.3f} {r['test_F1_CI']}  P={r['test_P']:.3f} R={r['test_R']:.3f} "
                  f"PR-AUC={r['test_PR_AUC']:.3f}", flush=True)

    df = pd.DataFrame(results).drop(columns="test_pred")
    df.to_csv("results.csv", mode="a", header=not pd.io.common.file_exists("results.csv"), index=False)
    print("\nF1 на test:")
    print(df.pivot(index="train", columns="model", values="test_F1").reindex(sizes).round(3))


if __name__ == "__main__":
    main(sys.argv[1:] or ["small", "medium", "large", "xlarge"])
