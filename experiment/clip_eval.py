"""CLIP ViT-B/32: Zero-shot mit allen Prompt-Sets + Linear Probe (logistische Regression auf Bildfeatures).

Zero-shot: Prompt-Set wird über das Validierungsset gewählt, berichtet werden alle Sets auf Test.
Linear Probe: Zwischenvariante (wenig Labels/Training, aber CLIP-Inferenzkosten). Die Energie für
Feature-Extraktion + Fit wird als deren "Trainingsenergie" gemessen.

Beispiel:  python -I experiment/clip_eval.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

import data
from common import PRECISION, RESULTS, env_info, metrics, predict, save_json, seed_all
from energy import PowerMonitor, measure_idle
from models import PROMPT_SETS, CLIPClassifier, count_params, load_clip, text_classifier

PROBE_C = (0.1, 1.0, 10.0, 100.0, 1000.0)


@torch.no_grad()
def extract(model: CLIPClassifier, x: torch.Tensor, bs=128, dev="cuda") -> np.ndarray:
    return torch.cat([model.features(x[i:i + bs].to(dev)).cpu() for i in range(0, len(x), bs)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle-s", type=float, default=30.0)
    ap.add_argument("--settle-s", type=float, default=15.0)
    ap.add_argument("--probe-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    d = data.load()
    x_tr, y_tr = d["train"]
    x_va, y_va = d["val"]
    x_te, y_te = d["test"]
    if a.smoke:
        (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = (data.subset(x_tr, y_tr, 256), data.subset(x_va, y_va, 64),
                                                    data.subset(x_te, y_te, 64))
        a.idle_s, a.settle_s, a.probe_seeds = 2.0, 0.0, [0]

    clip_model, tok = load_clip(dev)

    # ---- Zero-shot ----
    zs = {}
    for name in PROMPT_SETS:
        clf = CLIPClassifier(clip_model, zs_weights=text_classifier(clip_model, tok, name, dev)).to(dev)
        zs[name] = {"val": metrics(y_va.numpy(), predict(clf, x_va)), "test": metrics(y_te.numpy(), predict(clf, x_te))}
        print(f"zero-shot {name:12s} val F1 {zs[name]['val']['f1_defect']:.4f}  test acc "
              f"{zs[name]['test']['accuracy']:.4f} F1 {zs[name]['test']['f1_defect']:.4f}")
    chosen = max(zs, key=lambda k: zs[k]["val"]["f1_defect"])
    print(f"-> gewählt über Val: {chosen}")

    # ---- Linear Probe ----
    clf = CLIPClassifier(clip_model, zs_weights=text_classifier(clip_model, tok, chosen, dev)).to(dev)
    extract(clf, x_tr)  # Warmup über alle Batch-Formen (auch Rest-Batch), nicht gemessen
    extract(clf, x_va)
    torch.cuda.synchronize()
    time.sleep(a.settle_s)  # GPU nach Last abklingen lassen
    idle = measure_idle(a.idle_s)
    print(f"Idle: GPU {idle.gpu_w:.1f} W, CPU {'-' if idle.cpu_w is None else f'{idle.cpu_w:.1f}'} W", flush=True)
    probes = []
    for s in a.probe_seeds:
        seed_all(s)
        with PowerMonitor() as pm:
            # Wie bei den CNNs (Checkpoint-Wahl über Val): Regularisierung C über das Val-Set wählen.
            f_tr, f_va = extract(clf, x_tr), extract(clf, x_va)
            fits = {c: LogisticRegression(C=c, max_iter=5000, random_state=s).fit(f_tr, y_tr.numpy())
                    for c in PROBE_C}
            val_f1 = {c: metrics(y_va.numpy(), f.predict(f_va))["f1_defect"] for c, f in fits.items()}
            best_c = max(val_f1, key=val_f1.get)
            lr = fits[best_c]
            torch.cuda.synchronize()
        e = pm.result
        head = nn.Linear(f_tr.shape[1], 2)
        # sklearn binär: ein Gewichtsvektor für Klasse 1 -> als 2-Klassen-Logits [0, w·x+b]
        with torch.no_grad():
            head.weight.zero_(); head.bias.zero_()
            head.weight[1] = torch.from_numpy(lr.coef_[0]).float()
            head.bias[1] = float(lr.intercept_[0])
        probe = CLIPClassifier(clip_model, head=head).to(dev)
        m = metrics(y_te.numpy(), predict(probe, x_te))
        print(f"linear probe seed {s}: C={best_c:g}, {e.total_j / 1000:.2f} kJ in {e.duration_s:.1f} s, test acc {m['accuracy']:.4f} F1 {m['f1_defect']:.4f}")
        probes.append({"seed": s, "C": best_c, "val_f1_by_C": {str(k): v for k, v in val_f1.items()}, "energy": e.to_dict(), "energy_net": e.net(idle), "test": m})
        if not a.smoke:
            (RESULTS / "ckpt").mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), RESULTS / "ckpt" / f"clip_probe_s{s}.pt")

    if a.smoke:
        return
    save_json({
        "kind": "clip", "precision": PRECISION, "arch": "ViT-B-32/openai", "params_visual": count_params(clip_model.visual),
        "params_total": count_params(clip_model), "prompt_sets": PROMPT_SETS, "zero_shot": zs,
        "chosen_prompt_set": chosen, "linear_probe": probes, "idle": idle.to_dict(), "env": env_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, RESULTS / "clip" / "clip_eval.json")


if __name__ == "__main__":
    main()
