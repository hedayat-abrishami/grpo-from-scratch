#!/usr/bin/env bash
# Usage: source setup.sh
# Creates .venv if needed, installs deps, leaves you inside the venv.
# Works on the Mac (MPS) and on Lambda (CUDA).

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Run this with: source setup.sh"
    exit 1
fi

VENV=".venv"
REQ='requirements.txt'
MIN_PY="3.10"

# pyenv global is pinned to 3.6 on the Mac, so bare `python3` is too old for
# torch/transformers. Pick the newest usable interpreter explicitly.
find_python() {
    for c in python3.13 python3.12 python3.11 python3.10 python3; do
        command -v "$c" >/dev/null 2>&1 || continue
        "$c" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null && {
            command -v "$c"; return 0
        }
    done
    return 1
}

if [ ! -f "$REQ" ]; then
    echo "No $REQ found in $PWD"
    return 1
fi

if [ ! -d "$VENV" ]; then
    PY="$(find_python)" || { echo "No python >= $MIN_PY found"; return 1; }
    echo "Creating $VENV with $PY ($("$PY" -V 2>&1)) ..."
    "$PY" -m venv "$VENV"
fi

source "$VENV/bin/activate"

# Reinstall only if requirements.txt is newer than the last successful sync
STAMP="$VENV/req-stamp"
if [ ! -f "$STAMP" ] || [ "$REQ" -nt "$STAMP" ]; then
    echo "Syncing dependencies ..."
    python -m pip install --upgrade pip -q
    python -m pip install -r "$REQ" && touch "$STAMP"
    python -m pip freeze > "$VENV/freeze.txt"
    python -m ipykernel install --user --name="$(basename "$PWD")" >/dev/null 2>&1
fi

echo "Activate: $(which python)"
python -c "
import torch, transformers, datasets
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    d = f'cuda ({p.name}, {p.total_memory / 2**30:.0f} GB, bf16={torch.cuda.is_bf16_supported()})'
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    d = 'mps'
else:
    d = 'cpu'
print('torch', torch.__version__, '| transformers', transformers.__version__,
      '| datasets', datasets.__version__, '| device', d)
" 2>/dev/null
