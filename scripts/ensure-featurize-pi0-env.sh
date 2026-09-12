#!/usr/bin/env bash
set -Eeuo pipefail

# Check and repair the Python environment used for LeRobot 0.6.1 Pi0 training
# on Featurize. Existing compatible packages are kept; pip only resolves and
# installs missing/incompatible dependencies.

LEROBOT_VERSION="${LEROBOT_VERSION:-0.6.1}"
VENV_DIR="${VENV_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
CLOUD_WORK_ROOT="${CLOUD_WORK_ROOT:-}"
INSTALL=1

usage() {
    cat <<'EOF'
Usage: ensure-featurize-pi0-env.sh [--check-only] [--venv PATH] [--python PATH]

Environment overrides:
  LEROBOT_VERSION  Required LeRobot version (default: 0.6.1)
  VENV_DIR         Virtualenv directory
  PYTHON_BIN       Python executable; takes precedence over VENV_DIR
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --check-only) INSTALL=0 ;;
        --venv)
            [ "$#" -ge 2 ] || { echo "--venv requires a path" >&2; exit 2; }
            VENV_DIR="$2"; shift
            ;;
        --python)
            [ "$#" -ge 2 ] || { echo "--python requires a path" >&2; exit 2; }
            PYTHON_BIN="$2"; shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [ -z "$PYTHON_BIN" ]; then
    if [ -n "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ]; then
        PYTHON_BIN="$VENV_DIR/bin/python"
    else
        if [ -n "$CLOUD_WORK_ROOT" ]; then
            for candidate in \
                "$CLOUD_WORK_ROOT/lr312/bin/python" \
                "$CLOUD_WORK_ROOT/lrv/bin/python" \
                "$CLOUD_WORK_ROOT/.venv/bin/python"; do
                if [ -x "$candidate" ]; then
                    PYTHON_BIN="$candidate"
                    break
                fi
            done
        fi
    fi
fi

if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
    echo "No usable Featurize virtualenv found." >&2
    echo "Set CLOUD_WORK_ROOT, or pass --venv PATH or --python PATH." >&2
    exit 1
fi

VENV_DIR="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
PIP=("$PYTHON_BIN" -m pip)

echo "[INFO] virtualenv: $VENV_DIR"
echo "[INFO] python: $($PYTHON_BIN -V 2>&1)"
echo "[INFO] GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo unavailable)"

check_python() {
    "$PYTHON_BIN" - "$LEROBOT_VERSION" <<'PY'
import importlib
import sys

expected = sys.argv[1]
required = {
    "lerobot": expected,
    "torch": None,
    "torchvision": None,
    "transformers": None,
    "accelerate": None,
    "datasets": None,
    "draccus": None,
    "safetensors": None,
    "sentencepiece": None,
    "wandb": None,
    "huggingface_hub": None,
    "av": None,
}
failures = []
for name, wanted in required.items():
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {name} {version}")
        if wanted is not None and version != wanted:
            failures.append(f"{name}=={wanted} required, found {version}")
    except Exception as exc:
        failures.append(f"{name}: {exc}")

try:
    import torch
    if not torch.cuda.is_available():
        failures.append("torch.cuda.is_available() is False")
    else:
        props = torch.cuda.get_device_properties(0)
        print(f"[OK] CUDA {torch.version.cuda}; {props.name}; {props.total_memory / 2**30:.1f} GiB")
        print(f"[OK] bfloat16_supported={torch.cuda.is_bf16_supported()}")
except Exception as exc:
    failures.append(f"CUDA check: {exc}")

try:
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.scripts.lerobot_train import train
    print("[OK] Pi0 policy and training entrypoint import")
except Exception as exc:
    failures.append(f"Pi0 training import: {exc}")

if failures:
    print("[FAIL] " + " | ".join(failures), file=sys.stderr)
    raise SystemExit(1)
PY
}

if check_python; then
    echo "[OK] Pi0 training environment is already complete; nothing installed."
else
    if [ "$INSTALL" -eq 0 ]; then
        echo "[FAIL] Environment is incomplete and --check-only was requested." >&2
        exit 1
    fi

    echo "[INFO] Installing only missing/incompatible Pi0 training dependencies..."
    # LeRobot 0.6.1 and the bundled Torch build require setuptools < 82.
    "${PIP[@]}" install --upgrade pip "setuptools>=80,<82" wheel
    "${PIP[@]}" install --upgrade-strategy only-if-needed \
        "lerobot[pi]==${LEROBOT_VERSION}" \
        "wandb>=0.19,<0.23" \
        "huggingface-hub>=1.6,<2.0" \
        "sentencepiece>=0.2,<0.3" \
        "safetensors>=0.4" \
        "av>=12"

    check_python
fi

echo "[INFO] CLI tools"
for command in lerobot-train hf wandb; do
    if [ -x "$VENV_DIR/bin/$command" ]; then
        echo "[OK] $VENV_DIR/bin/$command"
    else
        echo "[FAIL] missing $VENV_DIR/bin/$command" >&2
        exit 1
    fi
done

if command -v git-lfs >/dev/null 2>&1 || git lfs version >/dev/null 2>&1; then
    echo "[OK] git-lfs available"
else
    echo "[WARN] git-lfs is missing. HF CLI uploads still work, but Git-based large-file operations will not."
fi

echo "[PASS] Featurize LeRobot Pi0 environment is ready."
