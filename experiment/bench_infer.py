"""Inferenzenergie pro Bild.

Jede Messung: Idle-Phase (Grundlast), dann für mindestens --min-s Sekunden Inferenz über das
Testset (zyklisch), mit Synchronisation nach jedem Batch (Ergebnis liegt vor, bevor das nächste
Teil kommt, wie an einer Prüfstation). Daten liegen vorab auf dem Gerät; JPEG-Dekodierung und
Kamera sind nicht enthalten.

Beispiel:  python -I experiment/bench_infer.py --variants mobilenetv3 resnet50 clip_zs --bs 1 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

import data
from common import PRECISION, RESULTS, env_info, save_json
from energy import PowerMonitor, measure_idle
from models import CNNClassifier, CLIPClassifier, count_params, load_clip, text_classifier

VARIANTS = ("mobilenetv3", "resnet50", "clip_zs", "clip_probe")


def build(variant: str, dev: str, seed: int) -> nn.Module:
    if variant in ("mobilenetv3", "resnet50"):
        m = CNNClassifier(variant, pretrained=False)
        m.load_state_dict(torch.load(RESULTS / "ckpt" / f"{variant}_s{seed}.pt", map_location="cpu"))
        return m.to(dev).eval()
    clip_model, tok = load_clip(dev)
    if variant == "clip_zs":
        chosen = json.loads((RESULTS / "clip" / "clip_eval.json").read_text())["chosen_prompt_set"]
        return CLIPClassifier(clip_model, zs_weights=text_classifier(clip_model, tok, chosen, dev)).to(dev).eval()
    head = nn.Linear(clip_model.visual.output_dim, 2)
    head.load_state_dict(torch.load(RESULTS / "ckpt" / f"clip_probe_s{seed}.pt", map_location="cpu"))
    return CLIPClassifier(clip_model, head=head).to(dev).eval()


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run_for(model, x, bs, min_s, dev) -> tuple[int, int]:
    """Läuft bis min_s überschritten; gibt (Bilder, Batches) zurück."""
    n_img = n_batch = 0
    i = 0
    t_end = time.perf_counter() + min_s
    while time.perf_counter() < t_end:
        if i + bs > len(x):
            i = 0
        model(x[i:i + bs]).argmax(1)
        sync(dev)
        i += bs
        n_img += bs
        n_batch += 1
    return n_img, n_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 32])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--min-s", type=float, default=60.0)
    ap.add_argument("--idle-s", type=float, default=20.0)
    ap.add_argument("--settle-s", type=float, default=15.0)
    ap.add_argument("--warmup-s", type=float, default=10.0)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="welcher Checkpoint")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.repeats, a.min_s, a.idle_s, a.warmup_s, a.settle_s = 1, 2.0, 1.0, 1.0, 0.0
    dev = a.device
    x_te = data.load()["test"][0].to(dev)

    for v in a.variants:
        model = build(v, dev, a.seed)
        for bs in a.bs:
            run_for(model, x_te, bs, a.warmup_s, dev)
            reps = []
            for r in range(a.repeats):
                time.sleep(a.settle_s)  # GPU nach Last abklingen lassen
                idle = measure_idle(a.idle_s)
                with PowerMonitor() as pm:
                    n_img, n_batch = run_for(model, x_te, bs, a.min_s, dev)
                e = pm.result
                net = e.net(idle)
                rep = {
                    "repeat": r, "n_images": n_img, "n_batches": n_batch, "energy": e.to_dict(),
                    "energy_net": net, "idle": idle.to_dict(),
                    "j_per_img": e.total_j / n_img, "j_per_img_net": net["total_j"] / n_img,
                    "gpu_j_per_img": e.gpu_j / n_img, "gpu_j_per_img_net": net["gpu_j"] / n_img,
                    "ms_per_batch": 1000 * e.duration_s / n_batch, "img_per_s": n_img / e.duration_s,
                }
                reps.append(rep)
                print(f"{v:12s} bs {bs:3d} rep {r}: {1000 * rep['j_per_img']:.2f} mJ/img brutto, "
                      f"{1000 * rep['j_per_img_net']:.2f} netto, {rep['img_per_s']:.0f} img/s, "
                      f"GPU {e.gpu_mean_w:.0f} W (idle {idle.gpu_w:.1f} W), CPU {(e.cpu_j or 0) / e.duration_s:.0f} W "
                      f"(idle {idle.cpu_w or 0:.1f} W) [{PRECISION}]", flush=True)
            if not a.smoke:
                save_json({
                    "kind": "infer", "precision": PRECISION, "variant": v, "bs": bs, "device": dev, "min_s": a.min_s,
                    "params": count_params(model), "repeats": reps, "env": env_info(),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, RESULTS / "infer" / f"{v}_bs{bs}_{dev}_{PRECISION}.json")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
