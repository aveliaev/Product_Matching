"""Общее для всех экспериментов: загрузка, чистка, val-split, оценка."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

TEST_PATH = "computers_gs.json"
TRAIN_PATH = "computers_train/computers_train_{}.json.gz"
SPLITS_DIR = Path("splits")
PRED_DIR = Path("predictions")
SEED = 42

# хвосты с названием магазина/страницы, найденные в EDA. По умолчанию НЕ удаляем: в экспериментах
# (REPORT_classic.md) их удаление ухудшило F1 на ~3 п. — название магазина оказалось полезным сигналом.
SHOP_TAIL = re.compile(r"(\s*[|▷»].*$)|(\s*-\s*black friday 2017)|(\b\S+\.com\b)|(\s-us\b)")


def clean(s, mode="light"):
    """mode: "none" — только нижний регистр; "light" — + убираем "..."@en и 'null';
    "full" — + удаляем хвосты с названием магазина."""
    if not isinstance(s, str):
        return ""
    if mode == "none":
        return s.lower()
    s = re.sub(r'"@\w+', " ", s).replace('"', " ").lower()
    s = re.sub(r"\bnull\b", " ", s)
    if mode == "full":
        s = SHOP_TAIL.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def load(path, clean_mode="light"):
    fn = lambda s: clean(s, clean_mode)
    df = pd.read_json(path, lines=True, dtype={"label": int})
    for col in ["title", "description", "brand"]:
        for side in ["left", "right"]:
            df[f"{col}_{side}"] = df[f"{col}_{side}"].map(fn)
    return df


def load_split(size, clean_mode="light"):
    """train/val из train_{size} (фиксированный стратифицированный split 80/20, id сохраняются в splits/) + test."""
    full = load(TRAIN_PATH.format(size), clean_mode)
    split_file = SPLITS_DIR / f"val_ids_{size}.txt"
    if split_file.exists():
        val_ids = set(split_file.read_text().split())
    else:
        _, val = train_test_split(full, test_size=0.2, stratify=full.label, random_state=SEED)
        SPLITS_DIR.mkdir(exist_ok=True)
        split_file.write_text("\n".join(val.pair_id))
        val_ids = set(val.pair_id)
    is_val = full.pair_id.isin(val_ids)
    return full[~is_val].reset_index(drop=True), full[is_val].reset_index(drop=True), load(TEST_PATH, clean_mode)


def hard_mask(df):
    """«Трудные» пары: негативы с похожими заголовками и позитивы с непохожими (как в test)."""
    vec = TfidfVectorizer(token_pattern=r"[a-z0-9]+", sublinear_tf=True).fit(pd.concat([df.title_left, df.title_right]))
    cos = np.asarray(vec.transform(df.title_left).multiply(vec.transform(df.title_right)).sum(1)).ravel()
    return ((df.label == 0) & (cos >= 0.5)) | ((df.label == 1) & (cos < 0.3))


def best_threshold(y, scores):
    grid = np.unique(np.quantile(scores, np.linspace(0, 1, 201)))
    f1s = [f1_score(y, scores >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def metrics(y, scores, t):
    pred = scores >= t
    return {"F1": f1_score(y, pred, zero_division=0), "P": precision_score(y, pred, zero_division=0),
            "R": recall_score(y, pred, zero_division=0), "PR_AUC": average_precision_score(y, scores)}


def bootstrap_f1(y, pred, n=1000, seed=SEED):
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    tp = ((pred[idx] == 1) & (y[idx] == 1)).sum(1)
    f1 = 2 * tp / (pred[idx].sum(1) + y[idx].sum(1))
    return np.percentile(f1, [2.5, 97.5])


def paired_compare(y, pred_a, pred_b, n=1000, seed=SEED):
    """Сравнение моделей A и B на одном test: ΔF1 = F1(B) − F1(A) с 95% ДИ (paired bootstrap)
    и p-value теста Макнемара по парам, где модели ошиблись по-разному."""
    from scipy.stats import binomtest

    y, a, b = (np.asarray(v).astype(int) for v in (y, pred_a, pred_b))
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n, len(y)))

    def f1(p):
        tp = ((p[idx] == 1) & (y[idx] == 1)).sum(1)
        return 2 * tp / (p[idx].sum(1) + y[idx].sum(1))

    delta = f1(b) - f1(a)
    only_a = int(((a == y) & (b != y)).sum())
    only_b = int(((b == y) & (a != y)).sum())
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return {"dF1": float(f1_score(y, b) - f1_score(y, a)), "dF1_lo": float(np.percentile(delta, 2.5)),
            "dF1_hi": float(np.percentile(delta, 97.5)), "mcnemar_p": float(p)}


def evaluate(name, size, val, val_scores, test, test_scores, save=True):
    """Порог выбираем по F1 на val, применяем к test. Сохраняем предсказания для будущих парных сравнений."""
    t = best_threshold(val.label, val_scores)
    hard = hard_mask(val)
    res = {"model": name, "train": size, "threshold": t,
           **{f"val_{k}": v for k, v in metrics(val.label, val_scores, t).items()},
           "val_hard_F1": metrics(val.label[hard], val_scores[hard], t)["F1"],
           **{f"test_{k}": v for k, v in metrics(test.label, test_scores, t).items()}}
    lo, hi = bootstrap_f1(test.label, test_scores >= t)
    res["test_F1_CI"] = f"[{lo:.3f}, {hi:.3f}]"

    res["test_pred"] = (test_scores >= t).astype(int)
    if not save:
        return res
    PRED_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"\W+", "_", name).strip("_").lower()
    pd.DataFrame({"pair_id": test.pair_id, "label": test.label, "score": test_scores, "pred": (test_scores >= t).astype(int)}
                 ).to_csv(PRED_DIR / f"{slug}__{size}__test.csv", index=False)
    return res
