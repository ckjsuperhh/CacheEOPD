#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)

env_name=${ENV_NAME:-cacheeopd}
python_version=${PYTHON_VERSION:-3.10}
torch_spec=${TORCH_SPEC:-torch}
torch_index_url=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}
install_vllm=${INSTALL_VLLM:-1}
install_requirements=${INSTALL_REQUIREMENTS:-0}
install_flash_attn=${INSTALL_FLASH_ATTN:-0}

conda_bin=${CONDA_EXE:-}
if [[ -z "$conda_bin" ]]; then
    conda_bin=$(command -v conda || true)
fi
if [[ -z "$conda_bin" ]]; then
    echo "Conda was not found. Install Miniconda or Anaconda, then run this script again." >&2
    exit 1
fi

conda_base=$("$conda_bin" info --base)
conda_sh="$conda_base/etc/profile.d/conda.sh"
if [[ ! -f "$conda_sh" ]]; then
    echo "Could not locate Conda activation script: $conda_sh" >&2
    exit 1
fi
source "$conda_sh"

if "$conda_bin" env list | awk -v requested_name="$env_name" '$1 == requested_name {found=1} END {exit found ? 0 : 1}'; then
    echo "Using existing Conda environment: $env_name"
else
    echo "Creating Conda environment: $env_name (Python $python_version)"
    "$conda_bin" create -y -n "$env_name" "python=$python_version" pip
fi

conda activate "$env_name"
python -m pip install --upgrade pip setuptools wheel

if [[ -n "$torch_index_url" ]]; then
    python -m pip install "$torch_spec" --index-url "$torch_index_url"
else
    python -m pip install "$torch_spec"
fi

python -m pip install "transformers>=4.51.0" accelerate safetensors sentencepiece
if [[ "$install_vllm" == "1" ]]; then
    python -m pip install -e "$repo_root[vllm,math]"
else
    python -m pip install -e "$repo_root[math]"
fi

if [[ "$install_requirements" == "1" ]]; then
    python -m pip install -r "$repo_root/requirements.txt"
fi

if [[ "$install_vllm" == "1" ]]; then
    python -m pip install "vllm>=0.13.0"
fi

if [[ "$install_flash_attn" == "1" ]]; then
    python -m pip install -r "$repo_root/requirements-cuda.txt" --no-build-isolation
fi

python - <<'PY'
import torch
import transformers

print(f"Python environment ready: torch={torch.__version__}, transformers={transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
PY

echo
echo "Environment setup complete. Activate it with: conda activate $env_name"
echo "Next: cp $script_dir/env.example $script_dir/.env and fill in the model/data paths."
