"""Этап 0: EDA датасета WDC Computers. Печатает статистику, графики сохраняет в eda/."""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

OUT = Path("eda")
OUT.mkdir(exist_ok=True)
SIZES = ["small", "medium", "large", "xlarge"]
FIELDS = ["title", "description", "brand", "price", "specTableContent", "keyValuePairs"]
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 90)


def header(s):
    print(f"\n{'=' * 80}\n{s}\n{'=' * 80}")


def clean(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r'"@\w+', " ", s).replace('"', " ").lower()
    return re.sub(r"\s+", " ", s).strip()


def filled(col):
    return col.notna() & (col.astype(str).str.strip() != "")


test = pd.read_json("computers_gs.json", lines=True, dtype={"label": int})
train = {s: pd.read_json(f"computers_train/computers_train_{s}.json.gz", lines=True, dtype={"label": int}) for s in SIZES}
tr = train["xlarge"]

# ---------------------------------------------------------------------------
header("1. Вложенность выборок и пересечения")
ids = {s: set(train[s].pair_id) for s in SIZES}
for a, b in zip(SIZES, SIZES[1:]):
    print(f"{a} ⊂ {b}: {len(ids[a] & ids[b]) / len(ids[a]):.1%} пар {a} есть в {b}")
print(f"test пар в xlarge: {test.pair_id.isin(tr.pair_id).mean():.1%}")
tr_offers = set(tr.id_left) | set(tr.id_right)
te_offers = set(test.id_left) | set(test.id_right)
print(f"офферов в test: {len(te_offers)}, из них есть в xlarge: {len(te_offers & tr_offers) / len(te_offers):.1%}")
for s in SIZES:
    offers = set(train[s].id_left) | set(train[s].id_right)
    clusters = set(train[s].cluster_id_left) | set(train[s].cluster_id_right)
    print(f"  {s:7} офферов={len(offers):6}  кластеров={len(clusters):5}  "
          f"test-кластеров покрыто={test.cluster_id_left.isin(clusters).mean():.1%}  "
          f"test-офферов покрыто={len(te_offers & offers) / len(te_offers):.1%}")

# ---------------------------------------------------------------------------
header("2. Качество разметки: дубли, перевёрнутые пары, противоречия")
for name, df in [("xlarge", tr), ("test", test)]:
    key = df.apply(lambda r: tuple(sorted((r.id_left, r.id_right))), axis=1)
    dup = key.duplicated(keep=False)
    conflict = df[dup].groupby(key[dup]).label.nunique().gt(1).sum()
    same_offer = (df.id_left == df.id_right).sum()
    label_vs_cluster = ((df.cluster_id_left == df.cluster_id_right).astype(int) != df.label).sum()
    print(f"{name:7} пар={len(df)}  дублей (с учётом A↔B)={dup.sum()}  противоречий={conflict}  "
          f"пар оффера с самим собой={same_offer}  label≠(cluster_left==cluster_right): {label_vs_cluster}")

# ---------------------------------------------------------------------------
header("3. Заполненность полей по классам")
rows = []
for f in FIELDS:
    for name, df in [("train_xl", tr), ("test", test)]:
        both = filled(df[f + "_left"]) & filled(df[f + "_right"])
        rows.append({"field": f, "set": name, "any_side": (filled(df[f + "_left"]).mean() + filled(df[f + "_right"]).mean()) / 2,
                     "both_pos": both[df.label == 1].mean(), "both_neg": both[df.label == 0].mean()})
print(pd.DataFrame(rows).pivot(index="field", columns="set").round(2))

# ---------------------------------------------------------------------------
header("4. Сырые заголовки: мусор от schema.org и названия сайтов")
print(tr.title_left.head(8).to_string())
suffix = tr.title_left.str.extract(r'"@\w+\s+"(.+?)"@\w+\s*$')[0]
print(f"\nзаголовков с «хвостом» (второй строкой, обычно название страницы/магазина): {suffix.notna().mean():.1%}")
print("частые хвосты:\n", suffix.str.replace(r".*[|\-–»]\s*", "", regex=True).value_counts().head(10).to_string())

for f in ["title", "description", "brand"]:
    for side in ["left", "right"]:
        tr[f"{f}_{side}"] = tr[f"{f}_{side}"].map(clean)
        test[f"{f}_{side}"] = test[f"{f}_{side}"].map(clean)

# ---------------------------------------------------------------------------
header("5. Длины текстов (слова) — важно для max_len трансформера")
lens = {}
for f in ["title", "description"]:
    for name, df in [("train_xl", tr), ("test", test)]:
        l = pd.concat([df[f + "_left"], df[f + "_right"]]).str.split().str.len()
        l = l[l > 0]
        lens[(f, name)] = l
        print(f"{f:12} {name:8} median={l.median():5.0f}  p90={l.quantile(.9):6.0f}  p99={l.quantile(.99):6.0f}  max={l.max()}")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, f, clip in [(axes[0], "title", 60), (axes[1], "description", 400)]:
    for name in ["train_xl", "test"]:
        ax.hist(lens[(f, name)].clip(upper=clip), bins=40, alpha=.5, density=True, label=name)
    ax.set_title(f"{f}: длина в словах (обрезано на {clip})")
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / "lengths.png", dpi=110)

# ---------------------------------------------------------------------------
header("6. Бренды")
brands = pd.concat([tr.brand_left, tr.brand_right])
print("топ брендов (train_xl):\n", brands[brands != ""].value_counts().head(15).to_string())
for name, df in [("train_xl", tr), ("test", test)]:
    known = (df.brand_left != "") & (df.brand_right != "")
    eq = df.brand_left == df.brand_right
    print(f"{name}: оба бренда есть в {known.mean():.1%} пар; P(label=1 | бренды равны)={df[known & eq].label.mean():.2f}, "
          f"P(label=1 | бренды разные)={df[known & ~eq].label.mean():.2f}")

# ---------------------------------------------------------------------------
header("7. Насколько похожи заголовки у дублей и не-дублей (TF-IDF cosine)")
vec = TfidfVectorizer(token_pattern=r"[a-z0-9]+", sublinear_tf=True).fit(pd.concat([tr.title_left, tr.title_right]))


def cos(df):
    a, b = vec.transform(df.title_left), vec.transform(df.title_right)
    return np.asarray(a.multiply(b).sum(1)).ravel()


tr["title_cos"], test["title_cos"] = cos(tr), cos(test)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, df) in zip(axes, [("train_xl", tr), ("test", test)]):
    for lab, c in [(0, "tab:red"), (1, "tab:green")]:
        ax.hist(df.title_cos[df.label == lab], bins=40, alpha=.5, density=True, color=c, label=f"label={lab}")
    ax.set_title(f"{name}: cosine TF-IDF заголовков")
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / "title_cosine.png", dpi=110)
for name, df in [("train_xl", tr), ("test", test)]:
    print(f"{name}: median cos  pos={df.title_cos[df.label == 1].median():.2f}  neg={df.title_cos[df.label == 0].median():.2f}; "
          f"негативов с cos>0.7: {(df.title_cos[df.label == 0] > .7).mean():.1%}; позитивов с cos<0.3: {(df.title_cos[df.label == 1] < .3).mean():.1%}")

# лучший порог по F1 без обучения — нижняя граница для всех моделей (подбор на train, применяем к test)
def f1_at(df, t):
    p = df.title_cos >= t
    tp = (p & (df.label == 1)).sum()
    return 2 * tp / (p.sum() + (df.label == 1).sum())


grid = np.linspace(0, 1, 101)
best_t = grid[np.argmax([f1_at(tr, t) for t in grid])]
print(f"\nбейзлайн «cosine ≥ порог»: порог, выбранный на train_xl = {best_t:.2f} → F1 на test = {f1_at(test, best_t):.3f}")

header("   трудные негативы в test (разные товары, почти одинаковые заголовки)")
print(test[test.label == 0].nlargest(8, "title_cos")[["title_cos", "title_left", "title_right"]].to_string(index=False))
header("   трудные позитивы в test (один товар, непохожие заголовки)")
print(test[test.label == 1].nsmallest(8, "title_cos")[["title_cos", "title_left", "title_right"]].to_string(index=False))

# ---------------------------------------------------------------------------
header("8. Коды моделей (буквы+цифры) — сильный сигнал?")
CODE = re.compile(r"\b(?=[a-z0-9-]*\d)(?=[a-z0-9-]*[a-z])[a-z0-9-]{4,}\b")


def codes(s):
    return {c.replace("-", "") for c in CODE.findall(s)} - {"ddr3", "ddr4", "ddr5", "gddr5", "gddr6", "usb3", "1080p", "4k"}


for name, df in [("train_xl", tr), ("test", test)]:
    cl, cr = df.title_left.map(codes), df.title_right.map(codes)
    has = (cl.str.len() > 0) & (cr.str.len() > 0)
    share = pd.Series([bool(a & b) for a, b in zip(cl, cr)], index=df.index)
    print(f"{name}: коды есть в обоих заголовках у {has.mean():.1%} пар; "
          f"P(1 | общий код)={df.label[has & share].mean():.2f} (n={int((has & share).sum())}), "
          f"P(1 | коды есть, но не пересекаются)={df.label[has & ~share].mean():.2f} (n={int((has & ~share).sum())})")

# ---------------------------------------------------------------------------
header("9. Структура: кластеры и офферы")
per_cluster = pd.concat([tr[["id_left", "cluster_id_left"]].set_axis(["id", "c"], axis=1),
                         tr[["id_right", "cluster_id_right"]].set_axis(["id", "c"], axis=1)]).drop_duplicates().groupby("c").size()
print(f"офферов на кластер (train_xl): median={per_cluster.median():.0f}, mean={per_cluster.mean():.1f}, max={per_cluster.max()}")
per_offer = pd.concat([tr.id_left, tr.id_right]).value_counts()
print(f"пар на оффер (train_xl): median={per_offer.median():.0f}, mean={per_offer.mean():.1f}, max={per_offer.max()}")
print(f"\nграфики сохранены в {OUT}/")
