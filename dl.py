"""Этап 3: трансформер-кросс-энкодер для пар офферов.

  [CLS] оффер A [SEP] оффер B  ->  P(дубль)

Тот же val-split и та же оценка, что у классики (common.evaluate): порог по val F1, bootstrap-ДИ на test.
Лучшая эпоха выбирается по val F1. Предсказания val/test сохраняются в predictions/ для парных сравнений.

Пример:
  python3 dl.py --size small --model intfloat/multilingual-e5-small --epochs 5
"""
import argparse
import copy
import os
import random
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # модели берём из локального кэша

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import PRED_DIR, best_threshold, evaluate, load_split

RESULTS = "results_dl.csv"


def serialize(df, side, fields, desc_words):
    """Оффер -> текст. fields='title' или 'title+desc' (описание обрезаем до desc_words слов)."""
    title = df[f"title_{side}"]
    if fields == "title":
        return title.tolist()
    desc = df[f"description_{side}"].str.split().str[:desc_words].str.join(" ")
    return [f"{t} [SEP] {d}" if d else t for t, d in zip(title, desc)]


class PairData(torch.utils.data.Dataset):
    def __init__(self, df, args):
        self.a = serialize(df, "left", args.fields, args.desc_words)
        self.b = serialize(df, "right", args.fields, args.desc_words)
        self.y = df.label.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i]


def make_collate(tok, max_len, swap_prob=0.0):
    def collate(batch):
        a, b, y = zip(*batch)
        if swap_prob:  # аугментация: случайно меняем офферы местами
            a, b = zip(*[(y_, x_) if random.random() < swap_prob else (x_, y_) for x_, y_ in zip(a, b)])
        enc = tok(list(a), list(b), truncation="longest_first", max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y)
        return enc
    return collate


@torch.no_grad()
def predict(model, df, tok, args, device):
    """P(дубль); при --tta усредняем по порядкам (A,B) и (B,A)."""
    model.eval()
    orders = [False, True] if args.tta else [False]
    out = []
    for swapped in orders:
        data = PairData(df, args)
        if swapped:
            data.a, data.b = data.b, data.a
        loader = torch.utils.data.DataLoader(data, batch_size=args.bs * 2, collate_fn=make_collate(tok, args.max_len))
        probs = []
        for enc in loader:
            enc = {k: v.to(device) for k, v in enc.items() if k != "labels"}
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**enc).logits
            probs.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
        out.append(torch.cat(probs).numpy())
    return np.mean(out, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--fields", default="title", choices=["title", "title+desc"])
    p.add_argument("--desc-words", type=int, default=40)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--swap-aug", type=float, default=0.5, help="вероятность поменять A и B местами при обучении")
    p.add_argument("--tta", type=int, default=1, help="усреднять предсказания по двум порядкам пары")
    p.add_argument("--freeze-emb", type=int, default=1,
                   help="заморозить эмбеддинги слов (у multilingual-моделей это ~80%% параметров и памяти Adam)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="взять первые N пар train (для быстрой отладки)")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = device.type == "cuda"  # fp16 на GPU (Kaggle/Colab); на MPS не используем
    train, val, test = load_split(args.size)
    if args.limit:
        train = train.sample(args.limit, random_state=args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    if args.freeze_emb:
        model.get_input_embeddings().weight.requires_grad_(False)
    loader = torch.utils.data.DataLoader(PairData(train, args), batch_size=args.bs, shuffle=True,
                                         collate_fn=make_collate(tok, args.max_len, args.swap_aug))
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)

    short = args.model.split("/")[-1]
    name = f"{short} | {args.fields} | len{args.max_len} lr{args.lr:g} ep{args.epochs}" + (f" | {args.tag}" if args.tag else "")
    print(f"{name} | train={args.size} ({len(train)} пар) | device={device}", flush=True)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = (-1.0, None, -1)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for i, enc in enumerate(loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**enc).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt), scaler.update(), sched.step(), opt.zero_grad()
            losses.append(loss.item())
            if i % 50 == 0:
                print(f"  ep{epoch} step {i}/{len(loader)} loss={np.mean(losses[-100:]):.4f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
        if device.type == "mps":
            torch.mps.empty_cache()
        va_s = predict(model, val, tok, args, device)
        va_f1 = f1_score(val.label, va_s >= best_threshold(val.label, va_s))
        print(f"  ep{epoch}: train loss={np.mean(losses):.4f}  val F1={va_f1:.4f}  [{time.time() - t0:.0f}s]", flush=True)
        if va_f1 > best[0]:
            best = (va_f1, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}), epoch)

    model.load_state_dict(best[1])
    va_s, te_s = predict(model, val, tok, args, device), predict(model, test, tok, args, device)
    r = evaluate(name, args.size, val, va_s, test, te_s, save=False)
    r.pop("test_pred")
    r.update(best_epoch=best[2], seed=args.seed, train_minutes=round((time.time() - t0) / 60, 1),
             hf_model=args.model, **{k: v for k, v in vars(args).items() if k not in ("size", "seed", "model")})
    print(f"\nИТОГ ({name}, лучшая эпоха {best[2]}): val F1={r['val_F1']:.3f} | test F1={r['test_F1']:.3f} "
          f"{r['test_F1_CI']} P={r['test_P']:.3f} R={r['test_R']:.3f} PR-AUC={r['test_PR_AUC']:.3f}", flush=True)

    pd.DataFrame([r]).to_csv(RESULTS, mode="a", header=not os.path.exists(RESULTS), index=False)
    PRED_DIR.mkdir(exist_ok=True)
    slug = f"dl__{short}__{args.fields}__{args.size}__seed{args.seed}" + (f"__{args.tag}" if args.tag else "")
    for part, df, s in [("val", val, va_s), ("test", test, te_s)]:
        pd.DataFrame({"pair_id": df.pair_id, "label": df.label, "score": s}).to_csv(
            PRED_DIR / f"{slug}__{part}.csv", index=False)


if __name__ == "__main__":
    main()
