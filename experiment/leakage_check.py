"""Leakage-Check: Wie ähnlich sind Testbilder ihrem nächsten Trainingsbild?

Laut Kaggle sind alle 300x300-Bilder bereits augmentiert (aus ~1300 Originalen). Liegen Varianten
desselben Teils in Train und Test, sind die Testwerte der überwachten Modelle optimistisch.
Maß: Kosinus-Ähnlichkeit der CLIP-Bildfeatures (L2-normiert). Verglichen wird
  test -> nächstes Train-Bild   mit   val -> nächstes Train-Bild   und   train -> nächstes anderes Train-Bild.
Zusätzlich pixelbasiert (normierte Graustufen, 56x56) als modellunabhängige Kontrolle.

Beispiel:  python -I experiment/leakage_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

import data
from common import RESULTS, save_json
from models import CLIPClassifier, load_clip, text_classifier


@torch.no_grad()
def clip_feats(clf, x, bs=256):
    return torch.cat([clf.features(x[i:i + bs].cuda()) for i in range(0, len(x), bs)])


def pixel_feats(x):
    z = F.interpolate(x.float().cuda(), size=56, mode="area").flatten(1)
    z = z - z.mean(1, keepdim=True)
    return F.normalize(z, dim=1)


@torch.no_grad()
def nn_sim(q, ref, exclude_self=False, bs=1024):
    out = []
    for i in range(0, len(q), bs):
        s = q[i:i + bs] @ ref.T
        if exclude_self:
            idx = torch.arange(i, min(i + bs, len(q)), device=s.device)
            s[torch.arange(len(idx)), idx] = -1
        out.append(s.max(1).values)
    return torch.cat(out).cpu().numpy()


def stats(v):
    return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "frac_ge_0.95": float((v >= 0.95).mean()), "frac_ge_0.98": float((v >= 0.98).mean())}


def main():
    d = data.load()
    x_tr, x_va, x_te = d["train"][0], d["val"][0], d["test"][0]
    clip_model, tok = load_clip("cuda")
    clf = CLIPClassifier(clip_model, zs_weights=text_classifier(clip_model, tok, "basic", "cuda")).cuda()
    res = {}
    for name, fn in (("clip", lambda x: clip_feats(clf, x)), ("pixel", pixel_feats)):
        f_tr, f_va, f_te = fn(x_tr), fn(x_va), fn(x_te)
        res[name] = {
            "test_to_train": stats(nn_sim(f_te, f_tr)),
            "val_to_train": stats(nn_sim(f_va, f_tr)),
            "train_to_train": stats(nn_sim(f_tr, f_tr, exclude_self=True)),
        }
        for k, v in res[name].items():
            print(f"{name:5s} {k:15s} median {v['median']:.4f}  p10 {v['p10']:.4f}  >=0.95 {v['frac_ge_0.95']:.3f}")
    save_json(res, RESULTS / "summary" / "leakage_check.json")


if __name__ == "__main__":
    main()
