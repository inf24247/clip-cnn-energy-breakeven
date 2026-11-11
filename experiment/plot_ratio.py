"""Abbildung fürs Paper: Lebenszyklus-Energie der CNNs relativ zu einer CLIP-Variante.

R_c(N) = (E_train,c + N e_c) / (E_train,z + N e_z).  R < 1: das CNN hat insgesamt weniger verbraucht.
Der Schnitt mit 1 ist N*. Liegt e_c über e_z, nähert sich R von oben einem Wert > 1 (kein N*).
Zwei Panels: Batch 1 und Batch 32, brutto, fp16. Werte direkt aus den Ergebnisdateien.
Referenz z ist standardmäßig die Linear Probe (Hauptvergleich im Paper), alternativ clip_zs.

Beispiel:  python -I experiment/plot_ratio.py                      -> ratio_gross_fp16.pdf
           python -I experiment/plot_ratio.py gross fp16 clip_zs   -> ratio_gross_fp16_clip_zs.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import COLOR, STYLE, load_all
from common import ROOT

NAME = {"mobilenetv3": "MobileNetV3-L", "resnet50": "ResNet-50"}
REF_NAME = {"clip_probe": "CLIP linear probe", "clip_zs": "zero-shot CLIP"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def paper_font() -> dict:
    """Times-Ersatz des Papers (TeX Gyre Termes X aus dem tectonic-Cache), sonst generische Serifenschrift."""
    from matplotlib import font_manager as fm
    files = sorted((ROOT / ".cache" / "tectonic").rglob("TeXGyreTermesX-*.otf"))
    if not files:
        return {"font.family": "serif", "mathtext.fontset": "dejavuserif"}
    for f in files:
        fm.fontManager.addfont(str(f))
    name = fm.FontProperties(fname=str(files[0])).get_name()
    return {"font.family": name, "mathtext.fontset": "custom", "mathtext.rm": name,
            "mathtext.it": f"{name}:italic", "mathtext.bf": f"{name}:bold"}


def main(mode="gross", prec="fp16", ref="clip_probe"):
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK, **paper_font()})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.4), sharey=True)
    n = np.logspace(3, 8, 500)
    for ax, bs in zip(axes, (1, 32)):
        train, _, infer, _ = load_all(mode, bs, "cuda", prec)
        t_ref, e_ref = np.mean(train[ref]), np.mean(infer[ref])
        ax.axhline(1.0, color=MUTED, lw=0.8, zorder=1)
        for c in ("mobilenetv3", "resnet50"):
            et, ec = np.mean(train[c]), np.mean(infer[c])
            r = (et + n * ec) / (t_ref + n * e_ref)
            ax.plot(n, r, STYLE[c], color=COLOR[c], lw=1.5, label=NAME[c], zorder=3)
            if ec < e_ref:
                ns = (et - t_ref) / (e_ref - ec)
                ax.plot(ns, 1.0, "o", ms=4.5, mfc="white", mec=INK, mew=0.8, zorder=5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e3, 1e8); ax.set_ylim(0.4, 20)
        ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
        ax.set_title(f"batch size {bs}", fontsize=7.5, color=INK, pad=3)
        ax.set_xlabel("inferences $N$")
        ax.grid(True, which="major", color=GRID, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"$E_\mathrm{CNN}(N)\,/\,E_\mathrm{CLIP}(N)$")
    axes[0].set_yticks([0.5, 1, 2, 5, 10, 20])
    axes[0].set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    axes[0].text(9e7, 1.08, REF_NAME[ref], ha="right", color=MUTED, fontsize=6.5, va="bottom")
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=2.2)
    fig.tight_layout(pad=0.25, w_pad=0.6)
    suffix = "" if ref == "clip_probe" else f"_{ref}"
    out = ROOT / "paper" / "figures" / f"ratio_{mode}_{prec}{suffix}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=250)
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
