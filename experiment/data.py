"""Casting-Defect-Datensatz (Kaggle: ravirajsinh45/real-life-industrial-dataset-of-casting-product).

Nutzt den mitgelieferten Train/Test-Split (300x300, Graustufen). Aus Train wird ein fester,
stratifizierter Validierungsanteil abgetrennt. Alle Bilder werden einmalig auf 224x224 skaliert
und als uint8-Tensor gecacht, damit JPEG-Dekodierung nicht in die Energiemessung fällt.
Label 1 = defekt (positive Klasse für F1), 0 = ok.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(os.environ.get("NKI_ROOT", Path(__file__).resolve().parents[1]))
RAW = ROOT / "data" / "raw" / "casting_data" / "casting_data"
CACHE = ROOT / "data" / "cache_224.pt"
SIZE = 224
VAL_FRACTION = 0.1
SPLIT_SEED = 0


def _load_dir(split: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    imgs, labels, names = [], [], []
    for cls, lab in (("ok_front", 0), ("def_front", 1)):
        for p in sorted((RAW / split / cls).glob("*.jpeg")):
            im = Image.open(p).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
            imgs.append(torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()))
            labels.append(lab)
            names.append(f"{split}/{cls}/{p.name}")
    return torch.stack(imgs).unsqueeze(1), torch.tensor(labels), names


def build_cache() -> dict:
    x_tr, y_tr, n_tr = _load_dir("train")
    x_te, y_te, n_te = _load_dir("test")
    g = np.random.default_rng(SPLIT_SEED)
    val_idx = []
    for c in (0, 1):
        idx = np.flatnonzero(y_tr.numpy() == c)
        val_idx.extend(g.choice(idx, size=round(len(idx) * VAL_FRACTION), replace=False))
    val_mask = np.zeros(len(y_tr), dtype=bool)
    val_mask[val_idx] = True
    tr, va = torch.from_numpy(~val_mask), torch.from_numpy(val_mask)
    d = {
        "train": (x_tr[tr], y_tr[tr]),
        "val": (x_tr[va], y_tr[va]),
        "test": (x_te, y_te),
        "names": {"train": [n for n, m in zip(n_tr, val_mask) if not m],
                  "val": [n for n, m in zip(n_tr, val_mask) if m], "test": n_te},
    }
    torch.save(d, CACHE)
    return d


def load() -> dict:
    if CACHE.exists():
        return torch.load(CACHE, weights_only=False)
    return build_cache()


def subset(x: torch.Tensor, y: torch.Tensor, n: int, seed: int = 0):
    """Zufällige Teilmenge (für --smoke), damit beide Klassen vorkommen."""
    idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))[:n].to(y.device)
    return x[idx], y[idx]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize(x_uint8: torch.Tensor, mean, std) -> torch.Tensor:
    """uint8 [B,1,H,W] -> float [B,3,H,W], normalisiert. Läuft auf dem Gerät des Tensors."""
    x = x_uint8.float().div_(255.0).expand(-1, 3, -1, -1)
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


if __name__ == "__main__":
    d = build_cache()
    for k in ("train", "val", "test"):
        x, y = d[k]
        print(f"{k:5s} n={len(y):5d}  defekt={int(y.sum()):5d}  ok={int((y == 0).sum()):5d}  shape={tuple(x.shape)}")
