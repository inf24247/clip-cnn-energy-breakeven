from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(os.environ.get("NKI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = ROOT / "results"

# Rechenpräzision für alle Modelle (Training + Inferenz). fp16 = Mixed Precision (autocast), Standard
# beim Deployment. fp32 nur als Sensitivitätsanalyse: auf gfx1201/ROCm sind fp32-GEMMs ungetunt.
PRECISION = os.environ.get("NKI_PRECISION", "fp16")
assert PRECISION in ("fp16", "fp32"), PRECISION


def autocast():
    return torch.autocast("cuda", dtype=torch.float16, enabled=PRECISION == "fp16")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, x: torch.Tensor, bs: int = 128, device="cuda") -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(model(x[i:i + bs].to(device, non_blocking=True)).argmax(1).cpu())
    return torch.cat(out).numpy()


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_defect": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_correct": int(tp + tn), "n": int(len(y_true)),
    }


def env_info() -> dict:
    import torchvision
    d = {
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
        "precision": PRECISION,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d["gpu"] = f"{torch.cuda.get_device_name(0)} ({p.gcnArchName})"
        d["hip"] = torch.version.hip
    return d


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"-> {path.relative_to(ROOT)}")
