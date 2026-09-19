"""Запуск обучения трансформеров на Kaggle GPU через официальный CLI.

  python3 kaggle_job.py setup-data                      # один раз: данные + val-сплиты -> приватный датасет
  python3 kaggle_job.py push NAME "--size medium" "--size medium --seed 1" ...   # каждая строка — аргументы dl.py
  python3 kaggle_job.py status NAME
  python3 kaggle_job.py fetch NAME                      # results_dl.csv и predictions/ -> в проект

Kernel получает текущие common.py и dl.py (встраиваются в run.py), так что код на Kaggle всегда совпадает с локальным.
Требуется API-токен в ~/.kaggle/access_token и CLI в ~/.venvs/kaggle. На kernel два T4 — прогоны идут параллельно.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

KAGGLE = str(Path.home() / ".venvs/kaggle/bin/kaggle")
DATASET_SLUG = "wdc-computers-product-matching"
BUILD = Path("kaggle_build")
OUT = Path("kaggle_out")
ACCELERATOR = "NvidiaTeslaT4"


def username():
    out = subprocess.run([str(Path(KAGGLE).parent / "python"), "-c",
                          "from kaggle.api.kaggle_api_extended import KaggleApi as A; a=A(); a.authenticate(); "
                          "print(a.get_config_value('username'))"], capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def kaggle(*args):
    print("$ kaggle", " ".join(args), flush=True)
    return subprocess.run([KAGGLE, *args], check=True)


def setup_data():
    d = BUILD / "data"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy("computers_gs.json", d)
    for f in Path("computers_train").glob("*.json.gz"):
        shutil.copy(f, d)
    for f in Path("splits").glob("val_ids_*.txt"):
        shutil.copy(f, d)
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "WDC Computers product matching", "id": f"{username()}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}]}))
    kaggle("datasets", "create", "-p", str(d))  # по умолчанию приватный


RUNNER = r'''
import glob, os, shlex, shutil, subprocess, sys, threading
from pathlib import Path

os.chdir("/kaggle/working")
src = Path(glob.glob("/kaggle/input/**/computers_gs.json", recursive=True)[0]).parent
print("input:", sorted(p.name for p in src.iterdir()), flush=True)
for d in ["computers_train", "splits", "logs"]:
    Path(d).mkdir(exist_ok=True)
os.symlink(src / "computers_gs.json", "computers_gs.json")
# Kaggle распаковывает .gz при загрузке датасета — поддерживаем оба варианта
gz = list(src.glob("computers_train_*.json.gz"))
for f in gz or src.glob("computers_train_*.json"):
    os.symlink(f, Path("computers_train") / f.name)
if not gz:
    COMMON_SRC = COMMON_SRC.replace("computers_train_{}.json.gz", "computers_train_{}.json")
for f in src.glob("val_ids_*.txt"):
    shutil.copy(f, "splits")
Path("common.py").write_text(COMMON_SRC)
Path("dl.py").write_text(DL_SRC)
os.environ["HF_HUB_OFFLINE"] = "0"
subprocess.run(["nvidia-smi", "-L"])

# прогоны раскладываем по GPU: каждая карта берёт свою очередь
n_gpu = max(1, int(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU ")))
lock = threading.Lock()

def worker(gpu, jobs):
    for i, args in jobs:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        with open(f"logs/run{i}.txt", "w") as log:
            p = subprocess.run([sys.executable, "dl.py", *shlex.split(args)], stdout=log, stderr=subprocess.STDOUT, env=env)
        with lock:
            print(f"\n######## run {i} (gpu {gpu}): dl.py {args} -> exit {p.returncode}", flush=True)
            print(open(f"logs/run{i}.txt").read()[-1200:], flush=True)

queues = [[(i, a) for i, a in enumerate(RUNS) if i % n_gpu == g] for g in range(n_gpu)]
threads = [threading.Thread(target=worker, args=(g, q)) for g, q in enumerate(queues)]
[t.start() for t in threads]
[t.join() for t in threads]
for p in ["computers_gs.json", *glob.glob("computers_train/*")]:
    os.remove(p)
'''


def push(name, runs):
    d = BUILD / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    code = (f"COMMON_SRC = {Path('common.py').read_text()!r}\n"
            f"DL_SRC = {Path('dl.py').read_text()!r}\n"
            f"RUNS = {runs!r}\n" + RUNNER)
    (d / "run.py").write_text(code)
    user = username()
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/wdc-{name}", "title": f"wdc-{name}", "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"], "competition_sources": [], "kernel_sources": []}))
    kaggle("kernels", "push", "-p", str(d), "--accelerator", ACCELERATOR)


def status(name):
    kaggle("kernels", "status", f"{username()}/wdc-{name}")


def fetch(name):
    d = OUT / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    kaggle("kernels", "output", f"{username()}/wdc-{name}", "-p", str(d), "-o")
    res = d / "results_dl.csv"
    if res.exists():
        new = pd.read_csv(res).assign(where=f"kaggle:{name}")
        local = pd.read_csv("results_dl.csv") if os.path.exists("results_dl.csv") else pd.DataFrame()
        pd.concat([local, new], ignore_index=True).to_csv("results_dl.csv", index=False)
        print(new[["model", "train", "seed", "val_F1", "test_F1", "test_F1_CI", "train_minutes"]].round(3).to_string())
    Path("predictions").mkdir(exist_ok=True)
    for f in (d / "predictions").glob("dl__*.csv"):
        shutil.copy(f, "predictions")
    Path("logs").mkdir(exist_ok=True)
    for f in (d / "logs").glob("*.txt"):
        shutil.copy(f, Path("logs") / f"kaggle_{name}_{f.name}")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    {"setup-data": lambda: setup_data(), "push": lambda: push(rest[0], rest[1:]),
     "status": lambda: status(rest[0]), "fetch": lambda: fetch(rest[0])}[cmd]()
