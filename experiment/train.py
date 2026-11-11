"""Fine-Tuning einer CNN mit Energiemessung über den gesamten Trainingslauf (inkl. Validierung).

Beispiel:  python -I experiment/train.py --model mobilenetv3 --seed 0
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

import data
from common import PRECISION, RESULTS, env_info, metrics, predict, save_json, seed_all
from energy import PowerMonitor, measure_idle
from models import CNN_NAMES, CNNClassifier, count_params

DEFAULT_LR = {"mobilenetv3": 1e-3, "resnet50": 3e-4}


def augment(x: torch.Tensor) -> torch.Tensor:
    """Zufällige Flips und 90°-Drehungen (Draufsicht, rotationsinvariant), komplett auf der GPU."""
    b = x.shape[0]
    hf = torch.rand(b, device=x.device) < 0.5
    vf = torch.rand(b, device=x.device) < 0.5
    x = torch.where(hf.view(-1, 1, 1, 1), x.flip(3), x)
    x = torch.where(vf.view(-1, 1, 1, 1), x.flip(2), x)
    if torch.rand(()) < 0.5:
        x = x.transpose(2, 3)
    return x


def run_epochs(model, opt, sched, scaler, x_tr, y_tr, x_va, y_va, epochs, bs, gen, log):
    best = {"f1": -1.0, "state": None, "epoch": -1}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y_tr), device=x_tr.device, generator=gen)
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(augment(x_tr[idx])), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.detach() * len(idx)
        tot = float(tot)
        m = metrics(y_va.cpu().numpy(), predict(model, x_va))
        log.append({"epoch": ep, "loss": tot / len(perm), "val_f1": m["f1_defect"], "val_acc": m["accuracy"]})
        print(f"  ep {ep:2d} loss {tot / len(perm):.4f} val_acc {m['accuracy']:.4f} val_f1 {m['f1_defect']:.4f}")
        if m["f1_defect"] > best["f1"]:
            best = {"f1": m["f1_defect"], "state": copy.deepcopy(model.state_dict()), "epoch": ep}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=CNN_NAMES, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--idle-s", type=float, default=30.0)
    ap.add_argument("--settle-s", type=float, default=15.0)
    ap.add_argument("--smoke", action="store_true", help="Mini-Lauf zum Testen, Ergebnis wird nicht gespeichert")
    a = ap.parse_args()
    lr = a.lr or DEFAULT_LR[a.model]
    dev = "cuda"

    d = data.load()
    x_tr, y_tr = (t.to(dev) for t in d["train"])
    x_va, y_va = (t.to(dev) for t in d["val"])
    x_te, y_te = d["test"]
    if a.smoke:
        (x_tr, y_tr), (x_va, y_va) = data.subset(x_tr, y_tr, 256), data.subset(x_va, y_va, 64)
        a.epochs, a.idle_s, a.settle_s = 1, 2.0, 0.0

    seed_all(a.seed)
    model = CNNClassifier(a.model).to(dev)

    # Warmup auf einer Kopie: Kernel-Kompilierung/MIOpen-Suche soll nicht in die Messung fallen.
    w = copy.deepcopy(model).train()
    wopt = torch.optim.AdamW(w.parameters(), lr=1e-6)
    wscaler = torch.amp.GradScaler(enabled=PRECISION == "fp16")
    # Auch die Form des letzten, kleineren Batches aufwärmen (sonst MIOpen-Suche in der Messung).
    tail = len(y_tr) % a.bs
    for n in [a.bs] * 3 + ([tail] if tail else []):
        wopt.zero_grad(set_to_none=True)
        wscaler.scale(F.cross_entropy(w(augment(x_tr[:n])), y_tr[:n])).backward()
        wscaler.step(wopt)
        wscaler.update()
    predict(w, x_va)
    predict(w, x_te)
    del w, wopt, wscaler
    torch.cuda.synchronize()

    # GPU taktet nach Last nur langsam herunter: erst abklingen lassen, dann Idle messen.
    time.sleep(a.settle_s)
    print(f"[{a.model} seed {a.seed}] Idle-Messung {a.idle_s:.0f} s ...", flush=True)
    idle = measure_idle(a.idle_s)
    print(f"  Idle: GPU {idle.gpu_w:.1f} W, CPU {'-' if idle.cpu_w is None else f'{idle.cpu_w:.1f}'} W", flush=True)

    seed_all(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = a.epochs * math.ceil(len(y_tr) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=PRECISION == "fp16")
    log = []
    with PowerMonitor() as pm:
        best = run_epochs(model, opt, sched, scaler, x_tr, y_tr, x_va, y_va, a.epochs, a.bs, gen, log)
        torch.cuda.synchronize()
    e = pm.result

    model.load_state_dict(best["state"])
    test = metrics(y_te.numpy(), predict(model, x_te))
    print(f"  Energie {e.total_j / 1000:.1f} kJ (GPU {e.gpu_j / 1000:.1f}, CPU "
          f"{'-' if e.cpu_j is None else f'{e.cpu_j / 1000:.1f}'}) in {e.duration_s:.0f} s | "
          f"Test acc {test['accuracy']:.4f} F1 {test['f1_defect']:.4f}")
    if a.smoke:
        return

    tag = f"{a.model}_s{a.seed}"
    ckpt = RESULTS / "ckpt" / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], ckpt)
    save_json({
        "kind": "train", "precision": PRECISION, "model": a.model, "seed": a.seed, "epochs": a.epochs, "bs": a.bs, "lr": lr,
        "params": count_params(model), "best_epoch": best["epoch"], "n_train": len(y_tr),
        "energy": e.to_dict(), "energy_net": e.net(idle), "idle": idle.to_dict(),
        "test": test, "log": log, "ckpt": str(ckpt.relative_to(RESULTS.parent)), "env": env_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, RESULTS / "train" / f"{tag}.json")


if __name__ == "__main__":
    main()
