# Zero-Shot or Fine-Tuned? The Energy Break-Even Between CLIP and Compact CNNs in Industrial Visual Inspection

Code, prompts, raw measurements and LaTeX sources of the short paper by Albert Helbing, Ari Said and
Robin Mevißen (DHBW Stuttgart, 2026). The paper is available as [`paper/main.pdf`](paper/main.pdf).

CLIP ViT-B/32 is evaluated zero-shot and with a linear probe and compared with fine-tuned
MobileNetV3-Large and ResNet-50 on a public casting-defect dataset. For every variant, accuracy,
training energy and inference energy per image are measured on one workstation. From these values,
the number of inferences is derived after which a fine-tuned CNN has used less energy in total than CLIP.

## Results in brief

Zero-shot CLIP reaches 67.1 % test accuracy, the linear probe 98.7 % and the CNNs 99.6 to 99.7 %.
Against the linear probe, the break-even at batch size 1 lies at 6.9×10⁴ inferences for MobileNetV3
and at 1.6×10⁵ for ResNet-50 (gross energy, fp16). At batch size 32, ResNet-50 needs more energy per
image than CLIP and never breaks even. Every number of the paper is stored in
`results/summary/paper_numbers.json`.

## Contents

| Path | Content |
|---|---|
| `experiment/` | Code. The prompt sets are defined in `experiment/models.py`. |
| `results/` | One JSON file per run, the log of the main run and the summaries |
| `paper/` | LaTeX sources, generated tables, figures and the PDF |
| `env.sh`, `requirements.txt` | Environment variables and pinned Python packages |

The dataset and the model checkpoints are not included. The dataset is licensed under CC BY-NC-ND 4.0.
It has to be downloaded from
[Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
and unpacked to `data/raw/`, so that the images lie in `data/raw/casting_data/casting_data/`. The
checkpoints (about 530 MB) are written to `results/ckpt/` during a run.

Comments and console output of the code are in German.

## Reproduction

The measurements were taken on an AMD Radeon RX 9070 XT and an AMD Ryzen 7 7800X3D under Linux with
ROCm 7.2. Python 3.12 and the packages in `requirements.txt` are required. PyTorch 2.14.1 and
torchvision 0.29.1 are ROCm 7.2 builds.

```bash
source env.sh             # bash or zsh
sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj   # CPU energy counter readable until reboot
experiment/run_all.sh     # full run, about 1.5 to 2 hours
```

`env.sh` expects the virtual environment in `.venv/` and keeps all caches inside the repository
folder. It also sets `HF_HUB_OFFLINE=1`, which means the CLIP weights have to be in the cache before
the run. The paper is built with `cd paper && tectonic main.tex`.

| Script | Purpose |
|---|---|
| `experiment/data.py` | Builds the cache of 224×224 images (5,969 training, 664 validation, 715 test) |
| `experiment/energy.py` | Power measurement: GPU `power1_average` sampled at 50 Hz and the RAPL package counter |
| `experiment/train.py` | Fine-tuning of MobileNetV3 or ResNet-50 with energy measurement |
| `experiment/clip_eval.py` | Zero-shot CLIP with three prompt sets and the linear probe |
| `experiment/bench_infer.py` | Inference energy per image at batch size 1 and 32 |
| `experiment/analyze.py` | Break-even with bootstrap intervals for one setting (gross or net, batch size, precision) |
| `experiment/leakage_check.py` | Similarity of each test image to the closest training image |
| `experiment/paper_tables.py` | Tables of the paper and `paper_numbers.json` |
| `experiment/plot_ratio.py` | Fig. 1 of the paper |

`data.py`, `train.py`, `clip_eval.py` and `bench_infer.py` accept `--smoke` for a run of a few seconds.

## Notes on the method

- All models run in fp16 mixed precision. Inference in fp32 is measured as a sensitivity analysis
  with `NKI_PRECISION=fp32`, and the results are stored in the `*_fp32.json` files.
- One MIOpen Winograd solver returns wrong values for fp16 depthwise convolutions on this GPU, which
  produced NaN in MobileNetV3. It is switched off in `env.sh`.
- CLIP is loaded with `force_quick_gelu=True`, because the OpenAI weights were trained with QuickGELU.
- Warm-up runs before every measurement and is not counted. Idle power is measured before every run,
  and both gross and net energy are reported.
- RAM, mainboard, power supply losses, labelling effort and the pre-training of CLIP are not covered.
