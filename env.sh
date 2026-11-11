# Projekt-Umgebung: alle Caches und Tools bleiben in diesem Ordner.
# Benutzung (bash/zsh): source env.sh
export NKI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PATH="$NKI_ROOT/tools:$NKI_ROOT/.venv/bin:$PATH"
export UV_CACHE_DIR="$NKI_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$NKI_ROOT/.cache/python"
export UV_PROJECT_ENVIRONMENT="$NKI_ROOT/.venv"
export PIP_CACHE_DIR="$NKI_ROOT/.cache/pip"
export TORCH_HOME="$NKI_ROOT/.cache/torch"
export HF_HOME="$NKI_ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$NKI_ROOT/.cache/xdg"
export TECTONIC_CACHE_DIR="$NKI_ROOT/.cache/tectonic"
export MIOPEN_USER_DB_PATH="$NKI_ROOT/.cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$NKI_ROOT/.cache/miopen"
export HIP_VISIBLE_DEVICES=0   # nur RX 9070 XT, nicht die iGPU
export HF_HUB_OFFLINE=1      # CLIP-Gewichte liegen im Cache; keine Netzanfragen während der Messung
# MIOpen-Bug auf gfx1201: Winograd-Fury-Solver liefert bei fp16-Depthwise-Faltungen falsche Werte
# (MobileNetV3 -> NaN). Nur diesen Solver abschalten; MIOpen wählt dann einen korrekten.
export MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3=0
export MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2=0
