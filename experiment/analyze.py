"""Auswertung: Tabellen, Break-even, Energie pro korrekter Klassifikation, Abbildung.

Lebenszyklus-Energie einer Variante v nach N Inferenzen:  E_v(N) = E_train,v + N * e_v
Break-even CNN c gegen CLIP-Variante z:  N* = (E_train,c - E_train,z) / (e_z - e_c),  nur wenn e_z > e_c.
Zero-shot hat E_train = 0 (CLIP-Pretraining bewusst ausgeschlossen, siehe Limitationen).
Unsicherheit: Bootstrap über Trainings-Seeds und Inferenz-Wiederholungen.

Beispiel:  python -I experiment/analyze.py --mode net --bs 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import RESULTS, ROOT

LABEL = {"mobilenetv3": "MobileNetV3-L (FT)", "resnet50": "ResNet-50 (FT)",
         "clip_zs": "CLIP ViT-B/32 (zero-shot)", "clip_probe": "CLIP ViT-B/32 (linear probe)"}
# Kategoriale Farben in fester Reihenfolge (validiert, CVD-sicher) + Linienstil als zweite Kodierung.
COLOR = {"mobilenetv3": "#2a78d6", "resnet50": "#eb6834", "clip_zs": "#1baf7a", "clip_probe": "#eda100"}
STYLE = {"mobilenetv3": "-", "resnet50": "--", "clip_zs": "-.", "clip_probe": ":"}
CNNS, CLIPS = ("mobilenetv3", "resnet50"), ("clip_zs", "clip_probe")


def load_all(mode: str, bs: int, device: str, prec: str):
    key_e = "energy_net" if mode == "net" else "energy"
    train, acc = {}, {}
    for p in sorted((RESULTS / "train").glob("*.json")):
        r = json.loads(p.read_text())
        train.setdefault(r["model"], []).append(r[key_e]["total_j"])
        acc.setdefault(r["model"], []).append(r["test"])
    c = json.loads((RESULTS / "clip" / "clip_eval.json").read_text())
    train["clip_zs"] = [0.0]
    acc["clip_zs"] = [c["zero_shot"][c["chosen_prompt_set"]]["test"]]
    train["clip_probe"] = [p[key_e]["total_j"] for p in c["linear_probe"]]
    acc["clip_probe"] = [p["test"] for p in c["linear_probe"]]
    infer = {}
    k = "j_per_img_net" if mode == "net" else "j_per_img"
    for v in LABEL:
        p = RESULTS / "infer" / f"{v}_bs{bs}_{device}_{prec}.json"
        if p.exists():
            infer[v] = [rep[k] for rep in json.loads(p.read_text())["repeats"]]
    return train, acc, infer, c


def breakeven(e_train_c, e_train_z, e_c, e_z):
    """N* >= 0. inf: CNN amortisiert sich nie (e_c >= e_z). 0: CNN ist von Anfang an sparsamer
    (auch weniger Trainingsenergie als die CLIP-Variante)."""
    d = e_z - e_c
    n = np.where(d > 0, (e_train_c - e_train_z) / np.where(d > 0, d, 1), np.inf)
    return np.maximum(n, 0.0)


def boot(x, n, rng):
    x = np.asarray(x, dtype=float)
    return x[rng.integers(0, len(x), n)] if len(x) > 1 else np.full(n, x[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("net", "gross"), default="net")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prec", choices=("fp16", "fp32"), default="fp16",
                    help="Inferenzpräzision (Training/Accuracy stammen immer aus dem Hauptlauf)")
    ap.add_argument("--rate", type=float, default=1.0, help="Teile pro Sekunde an der Linie (für Zeitangabe)")
    a = ap.parse_args()
    out = RESULTS / "summary"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    train, acc, infer, clip = load_all(a.mode, a.bs, a.device, a.prec)
    rng = np.random.default_rng(0)
    B = 10000

    rows = []
    for v in LABEL:
        if v not in infer or v not in train:
            continue
        accs = np.array([t["accuracy"] for t in acc[v]])
        f1s = np.array([t["f1_defect"] for t in acc[v]])
        e = np.array(infer[v])
        rows.append({
            "variant": v, "n_runs": len(acc[v]),
            "acc_mean": accs.mean(), "acc_std": accs.std(ddof=1) if len(accs) > 1 else 0.0,
            "f1_mean": f1s.mean(), "f1_std": f1s.std(ddof=1) if len(f1s) > 1 else 0.0,
            "train_kj_mean": np.mean(train[v]) / 1e3, "train_kj_std": np.std(train[v], ddof=1) / 1e3 if len(train[v]) > 1 else 0.0,
            "mj_per_img_mean": e.mean() * 1e3, "mj_per_img_std": e.std(ddof=1) * 1e3 if len(e) > 1 else 0.0,
            "mj_per_correct": e.mean() / accs.mean() * 1e3,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / f"variants_{a.mode}_bs{a.bs}_{a.device}_{a.prec}.csv", index=False)
    print(df.round(4).to_string(index=False))

    be = []
    for c in CNNS:
        for z in CLIPS:
            if not all(k in infer for k in (c, z)):
                continue
            ec, ez = boot(infer[c], B, rng), boot(infer[z], B, rng)
            tc, tz = boot(train[c], B, rng), boot(train[z], B, rng)
            n = breakeven(tc, tz, ec, ez)
            acc_c = boot([t["accuracy"] for t in acc[c]], B, rng)
            acc_z = boot([t["accuracy"] for t in acc[z]], B, rng)
            n_corr = breakeven(tc, tz, ec / acc_c, ez / acc_z)  # in korrekten Klassifikationen
            fin = np.isfinite(n)
            med = float(np.median(n)) if fin.mean() > 0.5 else np.inf
            be.append({
                "cnn": c, "clip": z, "p_breakeven_exists": float(fin.mean()),
                "n_star_median": med,
                "n_star_ci95": [float(np.percentile(n[fin], 2.5)), float(np.percentile(n[fin], 97.5))] if fin.any() else None,
                "n_star_correct_median": float(np.median(n_corr)) if np.isfinite(n_corr).mean() > 0.5 else np.inf,
                "hours_at_rate": med / a.rate / 3600 if np.isfinite(med) else np.inf,
                "p_cnn_dominates": float((n == 0).mean()),
            })
            print(f"[{a.prec}] Break-even {c} vs {z}: N* ~ {med:,.0f} Inferenzen "
                  f"(~{be[-1]['hours_at_rate']:.2f} h bei {a.rate:g} Teil/s), P(existiert) {fin.mean():.2f}")
    (out / f"breakeven_{a.mode}_bs{a.bs}_{a.device}_{a.prec}.json").write_text(json.dumps(be, indent=2))

    # Prompt-Sets (zero-shot) als eigene Tabelle
    pd.DataFrame([{"prompt_set": k, "val_f1": v["val"]["f1_defect"], "test_acc": v["test"]["accuracy"],
                   "test_f1": v["test"]["f1_defect"]} for k, v in clip["zero_shot"].items()]
                 ).to_csv(out / "prompt_sets.csv", index=False)

    plot(df, train, infer, be, a, fig_dir)


def plot(df, train, infer, be, a, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                         "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    n = np.logspace(0, 9, 400)
    for v in df["variant"]:
        e_tr, e_inf = np.mean(train[v]), np.mean(infer[v])
        y = (e_tr + n * e_inf) / 3.6e6  # kWh
        ax.plot(n, y, STYLE[v], color=COLOR[v], lw=1.5, label=LABEL[v])
    for b in be:
        if np.isfinite(b["n_star_median"]):
            x = b["n_star_median"]
            y = (np.mean(train[b["cnn"]]) + x * np.mean(infer[b["cnn"]])) / 3.6e6
            ax.plot(x, y, "o", ms=4, mfc="white", mec="#0b0b0b", mew=0.8, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of inferences $N$")
    ax.set_ylabel("Cumulative energy [kWh]")
    ax.grid(True, which="major", color="#e4e3df", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(pad=0.3)
    p = fig_dir / f"breakeven_{a.mode}_bs{a.bs}_{a.device}_{a.prec}.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"), dpi=200)
    print(f"-> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
