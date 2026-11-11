"""Erzeugt die Zahlen für das Paper direkt aus den Ergebnisdateien (keine Handübertragung).

Ausgaben:
  paper/tables/variants.tex   Tabelle: Accuracy, F1, Trainingsenergie, Energie/Bild (bs 1, 32; brutto/netto)
  paper/tables/breakeven.tex  Tabelle: N* (Median + 95%-KI) für alle CNN/CLIP-Paare, bs 1/32, brutto/netto
  results/summary/paper_numbers.json  alle Kennzahlen, die im Text vorkommen (inkl. fp32, Early-Stopping)

Beispiel:  python -I experiment/paper_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from analyze import CLIPS, CNNS, LABEL, boot, breakeven, load_all
from common import RESULTS, ROOT

SHORT = {"mobilenetv3": "MobileNetV3-L", "resnet50": "ResNet-50",
         "clip_zs": "CLIP zero-shot", "clip_probe": "CLIP lin.\\ probe"}
B = 10000


def fmt_n(x: float) -> str:
    if not np.isfinite(x):
        return "$\\infty$"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    return f"\\num{{{m:.1f}e{e}}}" if e >= 4 else f"{x:,.0f}".replace(",", "{,}")


def stats_all(prec="fp16"):
    out = {}
    for mode in ("gross", "net"):
        for bs in (1, 32):
            train, acc, infer, _ = load_all(mode, bs, "cuda", prec)
            out[(mode, bs)] = (train, acc, infer)
    return out


def infer_meta(v, bs, prec="fp16"):
    r = json.loads((RESULTS / "infer" / f"{v}_bs{bs}_cuda_{prec}.json").read_text())
    reps = r["repeats"]
    return {"img_per_s": float(np.mean([x["img_per_s"] for x in reps])),
            "gpu_w": float(np.mean([x["energy"]["gpu_mean_w"] for x in reps])),
            "cpu_w": float(np.mean([x["energy"]["cpu_j"] / x["energy"]["duration_s"] for x in reps])),
            "idle_gpu_w": float(np.mean([x["idle"]["gpu_w"] for x in reps])),
            "idle_cpu_w": float(np.mean([x["idle"]["cpu_w"] for x in reps])),
            "n_images": int(np.mean([x["n_images"] for x in reps])),
            "cv_gross": float(np.std([x["j_per_img"] for x in reps], ddof=1) / np.mean([x["j_per_img"] for x in reps]))}


def be_ci(train, acc, infer, c, z, rng, per_correct=False):
    ec, ez = boot(infer[c], B, rng), boot(infer[z], B, rng)
    tc, tz = boot(train[c], B, rng), boot(train[z], B, rng)
    if per_correct:
        ec = ec / boot([t["accuracy"] for t in acc[c]], B, rng)
        ez = ez / boot([t["accuracy"] for t in acc[z]], B, rng)
    n = breakeven(tc, tz, ec, ez)
    fin = np.isfinite(n)
    med = float(np.median(n)) if fin.mean() > 0.5 else np.inf
    lo, hi = (np.percentile(n[fin], [2.5, 97.5]) if fin.any() else (np.inf, np.inf))
    return {"median": med, "lo": float(lo), "hi": float(hi), "p_exists": float(fin.mean())}


def main():
    S = stats_all("fp16")
    rng = np.random.default_rng(0)
    nums = {"variants": {}, "breakeven": {}, "fp32": {}, "early_stop": {}}

    # ---- Variantentabelle ----
    rows = []
    train_g, acc, _ = S[("gross", 1)]
    train_n = S[("net", 1)][0]
    for v in LABEL:
        a = np.array([t["accuracy"] for t in acc[v]]) * 100
        f = np.array([t["f1_defect"] for t in acc[v]])
        d = {"acc_mean": a.mean(), "acc_std": a.std(ddof=1) if len(a) > 1 else 0.0,
             "f1_mean": f.mean(), "f1_std": f.std(ddof=1) if len(f) > 1 else 0.0,
             "train_kj_gross": np.mean(train_g[v]) / 1e3, "train_kj_net": np.mean(train_n[v]) / 1e3,
             "train_kj_gross_std": np.std(train_g[v], ddof=1) / 1e3 if len(train_g[v]) > 1 else 0.0}
        for mode in ("gross", "net"):
            for bs in (1, 32):
                e = np.array(S[(mode, bs)][2][v]) * 1e3
                d[f"mj_{mode}_bs{bs}"] = e.mean()
                d[f"mj_{mode}_bs{bs}_std"] = e.std(ddof=1)
                d[f"mj_per_correct_{mode}_bs{bs}"] = e.mean() / (a.mean() / 100)
        for bs in (1, 32):
            d[f"meta_bs{bs}"] = infer_meta(v, bs)
        nums["variants"][v] = d
        tr = "0" if v == "clip_zs" else f"{d['train_kj_gross']:.1f} / {d['train_kj_net']:.1f}"
        acc_s = f"{d['acc_mean']:.1f}" + (f"$\\pm${d['acc_std']:.1f}" if d["acc_std"] > 0 else "")
        rows.append(f"{SHORT[v]} & {acc_s} & {d['f1_mean']:.3f} & {tr} & "
                    f"{d['mj_gross_bs1']:.0f} / {d['mj_net_bs1']:.0f} & {d['mj_gross_bs32']:.1f} / {d['mj_net_bs32']:.1f} \\\\")
    tdir = ROOT / "paper" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "variants.tex").write_text("\n".join(rows) + "\n")

    # ---- Break-even-Tabelle ----
    rows = []
    hw = []
    for c in CNNS:
        for z in CLIPS:
            cells = []
            for bs in (1, 32):
                for mode in ("gross", "net"):
                    train, acc_m, infer = S[(mode, bs)]
                    r = be_ci(train, acc_m, infer, c, z, rng)
                    rc = be_ci(train, acc_m, infer, c, z, rng, per_correct=True)
                    nums["breakeven"][f"{c}|{z}|{mode}|bs{bs}"] = {"all": r, "per_correct": rc}
                    cells.append(fmt_n(r["median"]))  # KI-Breite steht in der Tabellenunterschrift
                    if np.isfinite(r["median"]) and r["median"] > 0:
                        hw.append(max(r["hi"] / r["median"] - 1, 1 - r["lo"] / r["median"]))
            rows.append(f"{SHORT[c]} vs.\\ {SHORT[z]} & " + " & ".join(cells) + " \\\\")
    (tdir / "breakeven.tex").write_text("\n".join(rows) + "\n")
    nums["breakeven_max_rel_ci_halfwidth"] = float(max(hw))

    # ---- fp32-Sensitivität (nur Inferenz) ----
    for bs in (1, 32):
        for v in LABEL:
            p = RESULTS / "infer" / f"{v}_bs{bs}_cuda_fp32.json"
            if p.exists():
                reps = json.loads(p.read_text())["repeats"]
                g = np.mean([x["j_per_img"] for x in reps]) * 1e3
                nums["fp32"][f"{v}|bs{bs}"] = {"mj_gross": float(g), "img_per_s": float(np.mean([x["img_per_s"] for x in reps])),
                                               "ratio_to_fp16": float(g / nums["variants"][v][f"mj_gross_bs{bs}"])}
    if nums["fp32"]:
        for bs in (1, 32):
            tr, ac, _ = S[("gross", bs)]
            inf32 = {}
            for v in LABEL:
                p = RESULTS / "infer" / f"{v}_bs{bs}_cuda_fp32.json"
                if p.exists():
                    inf32[v] = [x["j_per_img"] for x in json.loads(p.read_text())["repeats"]]
            for c in CNNS:
                for z in CLIPS:
                    if c in inf32 and z in inf32:
                        nums["fp32"][f"breakeven|{c}|{z}|gross|bs{bs}"] = be_ci(tr, ac, inf32, c, z, rng)

    # ---- Early stopping: Energie bis zur besten Epoche (linear über Epochen anteilig) ----
    for c in CNNS:
        fr = []
        for p in sorted((RESULTS / "train").glob(f"{c}_s*.json")):
            r = json.loads(p.read_text())
            fr.append((r["best_epoch"] + 1) / r["epochs"])
        nums["early_stop"][c] = {"best_epochs": [int(round(f * 10)) for f in fr], "mean_fraction": float(np.mean(fr))}
        for bs in (1, 32):
            train, acc_m, infer = S[("gross", bs)]
            t_es = {**train, c: [e * f for e, f in zip(train[c], fr)]}
            for z in CLIPS:
                nums["early_stop"][f"{c}|{z}|gross|bs{bs}"] = be_ci(t_es, acc_m, infer, c, z, rng)

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (np.floating, float)):
            return float(o) if np.isfinite(o) else "inf"
        return o
    (RESULTS / "summary" / "paper_numbers.json").write_text(json.dumps(conv(nums), indent=1))
    print((tdir / "variants.tex").read_text())
    print((tdir / "breakeven.tex").read_text())
    print(json.dumps(conv({k: nums[k] for k in ("fp32", "early_stop")}), indent=1))


if __name__ == "__main__":
    main()
