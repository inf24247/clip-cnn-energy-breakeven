#!/usr/bin/env bash
# Kompletter Experimentlauf (~1,5–2 h, GPU unter Volllast). Vorher:
#   sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj
# Während des Laufs: keine anderen GPU-/CPU-lastigen Programme, Browser/Spiele zu.
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
PY="python -I -u"   # -u: ungepuffert, damit das Log live mitläuft
SEEDS="${SEEDS:-0 1 2 3 4}"
COOL="${COOL:-60}"   # Abkühlpause zwischen Läufen in s

$PY -c "import sys; sys.path.insert(0,'experiment'); from energy import rapl_available as r; assert r(), 'RAPL nicht lesbar (sudo chmod o+r ...)'"
mkdir -p results/logs
exec > >(tee -a "results/logs/run_$(date +%Y%m%d_%H%M%S).log") 2>&1

[ -f data/cache_224.pt ] || $PY experiment/data.py
for m in mobilenetv3 resnet50; do
  for s in $SEEDS; do
    $PY experiment/train.py --model "$m" --seed "$s"
    sleep "$COOL"
  done
done
$PY experiment/clip_eval.py
sleep "$COOL"
$PY experiment/bench_infer.py --bs 1 32 --repeats 5 --min-s 60
for mode in net gross; do for bs in 1 32; do
  $PY experiment/analyze.py --mode $mode --bs $bs --prec fp16
done; done
echo HAUPTLAUF_FERTIG
# Sensitivität: Inferenz in reinem fp32 (gleiche Checkpoints), kürzer.
sleep "$COOL"
NKI_PRECISION=fp32 $PY experiment/bench_infer.py --bs 1 32 --repeats 3 --min-s 30
for mode in net gross; do for bs in 1 32; do
  $PY experiment/analyze.py --mode $mode --bs $bs --prec fp32
done; done
$PY experiment/leakage_check.py
echo FERTIG
