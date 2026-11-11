"""Modelle: fine-tuned CNNs (MobileNetV3-Large, ResNet-50) und CLIP ViT-B/32 (zero-shot / linear probe).

Alle Modelle bekommen denselben Eingang: uint8-Tensor [B,1,224,224] auf dem Zielgerät.
Die modellspezifische Normalisierung passiert im forward und zählt damit zur Inferenzenergie.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from common import autocast
from data import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD, normalize

CNN_NAMES = ("mobilenetv3", "resnet50")
CLIP_ARCH, CLIP_PRETRAINED = "ViT-B-32", "openai"

# Prompt-Sets werden vorab festgelegt und NICHT auf dem Testset optimiert.
# Auswahl eines Sets nur über das Validierungsset; berichtet werden alle.
PROMPT_SETS: dict[str, dict[str, list[str]]] = {
    "basic": {
        "ok": ["a photo of a good metal casting."],
        "def": ["a photo of a defective metal casting."],
    },
    "descriptive": {
        "ok": ["a top-view photo of a flawless cast metal impeller with a smooth, intact surface."],
        "def": ["a top-view photo of a cast metal impeller with defects such as blowholes, "
                "pinholes, cracks, burrs or a damaged edge."],
    },
    "ensemble": {
        "ok": [t.format(c) for t in (
            "a photo of a {}.", "a close-up photo of a {}.", "a grayscale photo of a {}.",
            "an industrial inspection image of a {}.", "a photo of the {} on a production line.")
            for c in ("good metal casting", "flawless cast impeller", "non-defective casting part")],
        "def": [t.format(c) for t in (
            "a photo of a {}.", "a close-up photo of a {}.", "a grayscale photo of a {}.",
            "an industrial inspection image of a {}.", "a photo of the {} on a production line.")
            for c in ("defective metal casting", "cracked cast impeller", "casting part with blowholes")],
    },
}


class CNNClassifier(nn.Module):
    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        self.name = name
        if name == "mobilenetv3":
            w = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            m = torchvision.models.mobilenet_v3_large(weights=w)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
        elif name == "resnet50":
            w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            m = torchvision.models.resnet50(weights=w)
            m.fc = nn.Linear(m.fc.in_features, 2)
        else:
            raise ValueError(name)
        self.net = m

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        with autocast():
            return self.net(normalize(x_uint8, IMAGENET_MEAN, IMAGENET_STD)).float()


def load_clip(device="cpu"):
    import open_clip
    # Die OpenAI-Gewichte wurden mit QuickGELU trainiert; ohne force_quick_gelu nutzt open_clip GELU
    # und das Modell wird messbar schlechter (Warnung "QuickGELU mismatch").
    model, _, _ = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_PRETRAINED,
                                                        force_quick_gelu=True)
    tok = open_clip.get_tokenizer(CLIP_ARCH)
    return model.to(device).eval(), tok


@torch.no_grad()
def text_classifier(clip_model, tokenizer, prompt_set: str, device) -> torch.Tensor:
    """Gemittelte, normierte Text-Embeddings je Klasse -> Matrix [D, 2] (Spalte 0 = ok, 1 = defekt)."""
    cols = []
    for cls in ("ok", "def"):
        t = clip_model.encode_text(tokenizer(PROMPT_SETS[prompt_set][cls]).to(device))
        t = F.normalize(t, dim=-1).mean(0)
        cols.append(F.normalize(t, dim=0))
    return torch.stack(cols, dim=1)


class CLIPClassifier(nn.Module):
    """CLIP-Bildencoder + feste Klassifikationsmatrix (zero-shot) oder linearer Kopf (linear probe).

    Der Textencoder wird nur einmal vorab benutzt und gehört nicht zur Inferenz pro Bild.
    """

    def __init__(self, clip_model, zs_weights: torch.Tensor | None = None, head: nn.Linear | None = None):
        super().__init__()
        self.visual = clip_model.visual
        self.logit_scale = float(clip_model.logit_scale.detach().exp())
        if (zs_weights is None) == (head is None):
            raise ValueError("genau eins von zs_weights / head angeben")
        self.register_buffer("zs_weights", zs_weights if zs_weights is not None else torch.empty(0))
        self.head = head

    def features(self, x_uint8: torch.Tensor) -> torch.Tensor:
        with autocast():
            f = self.visual(normalize(x_uint8, CLIP_MEAN, CLIP_STD))
        return F.normalize(f.float(), dim=-1)

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        f = self.features(x_uint8)
        if self.head is not None:
            return self.head(f)
        return self.logit_scale * f @ self.zs_weights


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
