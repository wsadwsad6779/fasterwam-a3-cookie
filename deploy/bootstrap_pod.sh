#!/usr/bin/env bash
# Rebuild the FasterWAM -> A3 cookie environment on a fresh GPU pod.
#
# Code comes from GitHub (pinned commits), weights and data from Hugging Face.
# Nothing secret lives in this file: HF_TOKEN is read from the caller's
# environment and handed to the Hugging Face CLI only, never echoed or written.
#
#   HF_TOKEN=... bash deploy/bootstrap_pod.sh --smoke
#
# Optional environment:
#   WORKSPACE         default /workspace
#   HF_ENDPOINT       default https://hf-mirror.com   (empty string = huggingface.co)
#   PIP_INDEX         default https://mirrors.aliyun.com/pypi/simple
#   GH_MIRROR         URL prefix for GitHub clones when github.com is blocked,
#                     e.g. GH_MIRROR=https://ghfast.top  (empty = direct)
#   A3_COMMIT         a3_dual_arm_sim commit (default 08ebe8cb370856051d0b32b07e691cc9b056716b)
#   FASTERWAM_COMMIT  FasterWAM commit       (default 83667817df0d4f823f39d90700e61ea2f432ac45)
#   BRIDGE_REPO       default https://github.com/wsadwsad6779/fasterwam-a3-cookie.git
#   BRIDGE_BRANCH     default a3-cookie-r27
#   BRIDGE_SRC        if set, copy the bridge tree from this local path instead
#                     of cloning it (the instance had no GitHub access on
#                     2026-10-27, so an uploaded copy is a valid fallback)
#
# Flags:
#   --skip-pip        reuse an existing pip environment
#   --skip-weights    code + pip only (no HF downloads)
#   --with-training   also fetch the 42.6k-frame dataset, frames.dat, trainer
#                     checkpoint and optimizer state (~37 GB extra)
#   --smoke           run the 2-step run_matrix smoke for both checkpoints
#
# Disk: inference only needs ~37 GB of weights; --with-training brings it to
# ~75 GB. Run under `tmux`/`nohup`: the big downloads take 20-40 minutes.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
HF_ENDPOINT="${HF_ENDPOINT-https://hf-mirror.com}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
GH_MIRROR="${GH_MIRROR:-}"
A3_COMMIT="${A3_COMMIT:-08ebe8cb370856051d0b32b07e691cc9b056716b}"
FASTERWAM_COMMIT="${FASTERWAM_COMMIT:-83667817df0d4f823f39d90700e61ea2f432ac45}"
BRIDGE_REPO="${BRIDGE_REPO:-https://github.com/wsadwsad6779/fasterwam-a3-cookie.git}"
BRIDGE_BRANCH="${BRIDGE_BRANCH:-a3-cookie-r27}"
BRIDGE_SRC="${BRIDGE_SRC:-}"

SKIP_PIP=0
SKIP_WEIGHTS=0
WITH_TRAINING=0
SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --skip-pip) SKIP_PIP=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;;
    --with-training) WITH_TRAINING=1 ;;
    --smoke) SMOKE=1 ;;
    -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

BRIDGE_DIR="$WORKSPACE/fasterwam_a3_bridge"
A3_DIR="$WORKSPACE/a3_dual_arm_sim"
FW_DIR="$WORKSPACE/FasterWAM"
CKPT_DIR="$FW_DIR/checkpoints"

log() { printf '\n[bootstrap] %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

if [ -n "$HF_ENDPOINT" ]; then
  export HF_ENDPOINT
fi
if python3 -c 'import hf_transfer' 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi

# ---------------------------------------------------------------- 0. preflight
log "workspace=$WORKSPACE  hf_endpoint=${HF_ENDPOINT:-https://huggingface.co}  gh_mirror=${GH_MIRROR:-direct}"
have python3 || { echo "python3 missing" >&2; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "python3 >= 3.10 required" >&2; exit 1; }
mkdir -p "$WORKSPACE"
df -h "$WORKSPACE" | tail -1

gh_url() {  # gh_url https://github.com/org/repo.git
  if [ -n "$GH_MIRROR" ]; then printf '%s/%s' "${GH_MIRROR%/}" "$1"; else printf '%s' "$1"; fi
}

clone_at() {  # clone_at url commit dir
  local url="$1" commit="$2" dir="$3"
  if [ -d "$dir/.git" ]; then
    log "reusing $dir ($(git -C "$dir" rev-parse --short HEAD))"
  else
    log "cloning $(gh_url "$url") -> $dir"
    git clone "$(gh_url "$url")" "$dir"
  fi
  local want have_commit
  want="$(git -C "$dir" rev-parse --verify "$commit^{commit}" 2>/dev/null || true)"
  have_commit="$(git -C "$dir" rev-parse HEAD)"
  if [ -z "$want" ]; then
    git -C "$dir" fetch --all --tags --quiet
    want="$(git -C "$dir" rev-parse --verify "$commit^{commit}")"
  fi
  if [ "$have_commit" != "$want" ]; then
    log "checking out $commit in $dir"
    git -C "$dir" checkout --quiet --detach "$want"
  fi
  git -C "$dir" rev-parse --short HEAD
}

# ---------------------------------------------------------------- 1. pip
if [ "$SKIP_PIP" -eq 0 ]; then
  log "installing python deps from $PIP_INDEX"
  pip3 install -q -i "$PIP_INDEX" --upgrade pip
  pip3 install -q -i "$PIP_INDEX" \
    mujoco==3.3.2 numpy==2.1.2 robosuite==1.4.0 gymnasium
  pip3 install -q -i "$PIP_INDEX" --no-deps libero bddl robomimic
  pip3 install -q -i "$PIP_INDEX" \
    easydict av thop wandb jsonlines GitPython einops omegaconf hydra-core \
    imageio imageio-ffmpeg scipy h5py matplotlib opencv-python numba pandas \
    transformers==4.57.6 diffusers accelerate==1.13.0 "huggingface_hub[cli]" \
    hf_transfer termcolor pynvml rich
  log "pip done: $(python3 -c 'import mujoco, numpy, robosuite; print(mujoco.__version__, numpy.__version__, robosuite.__version__)')"
else
  log "skipping pip"
fi
have huggingface-cli || pip3 install -q -i "$PIP_INDEX" "huggingface_hub[cli]"

# ---------------------------------------------------------------- 2. code
if [ -n "$BRIDGE_SRC" ]; then
  log "copying bridge tree from $BRIDGE_SRC -> $BRIDGE_DIR"
  mkdir -p "$BRIDGE_DIR"
  cp -a "$BRIDGE_SRC"/. "$BRIDGE_DIR"/
else
  log "cloning bridge repo ($BRIDGE_BRANCH)"
  if [ -d "$BRIDGE_DIR/.git" ]; then
    git -C "$BRIDGE_DIR" fetch --quiet origin "$BRIDGE_BRANCH" || true
    git -C "$BRIDGE_DIR" checkout --quiet "$BRIDGE_BRANCH"
    git -C "$BRIDGE_DIR" pull --quiet --ff-only origin "$BRIDGE_BRANCH" || \
      echo "[bootstrap] WARNING: $BRIDGE_DIR has local changes; left as is" >&2
  else
    git clone --quiet --branch "$BRIDGE_BRANCH" "$(gh_url "$BRIDGE_REPO")" "$BRIDGE_DIR"
  fi
fi
A3_SHORT="$(clone_at https://github.com/Eter0109/a3_dual_arm_sim.git "$A3_COMMIT" "$A3_DIR")"
FW_SHORT="$(clone_at https://github.com/hustvl/FasterWAM.git "$FASTERWAM_COMMIT" "$FW_DIR")"
log "code ready: a3_dual_arm_sim@$A3_SHORT  FasterWAM@$FW_SHORT"

NEXT_DIR="$BRIDGE_DIR/_next"
[ -d "$NEXT_DIR" ] || { echo "missing $NEXT_DIR" >&2; exit 1; }

# The _next scripts are stdlib+numpy at import time, so this catches syntax and
# import-path mistakes before any 11 GB download starts.
python3 -m py_compile "$NEXT_DIR"/*.py
PYTHONPATH="$NEXT_DIR:$A3_DIR/src:$FW_DIR/src:$FW_DIR:$BRIDGE_DIR" \
  python3 "$NEXT_DIR/selftest.py" >/dev/null
log "py_compile + selftest ok"

# ---------------------------------------------------------------- 3. weights
hf_download() {  # hf_download repo localdir [include...]
  local repo="$1" dest="$2"; shift 2
  local args=(download "$repo" --local-dir "$dest")
  if [ -n "${HF_TOKEN:-}" ]; then
    args+=(--token "$HF_TOKEN")
  fi
  for inc in "$@"; do args+=(--include "$inc"); done
  huggingface-cli "${args[@]}"
}

verify_size() {  # verify_size path expected_bytes
  local path="$1" expect="$2"
  [ -f "$path" ] || { echo "[bootstrap] MISSING $path" >&2; return 1; }
  local got; got="$(stat -c %s "$path")"
  if [ "$got" != "$expect" ]; then
    echo "[bootstrap] WARNING size mismatch $path: $got != $expect" >&2
  else
    echo "[bootstrap] ok $(basename "$path") $got B"
  fi
}

verify_min_size() {  # verify_min_size path min_bytes
  local path="$1" min="$2"
  [ -f "$path" ] || { echo "[bootstrap] MISSING $path" >&2; return 1; }
  local got; got="$(stat -c %s "$path")"
  if [ "$got" -lt "$min" ]; then
    echo "[bootstrap] WARNING $path is only $got B (< $min); rerun to resume" >&2
  else
    echo "[bootstrap] ok $(basename "$path") $got B"
  fi
}

if [ "$SKIP_WEIGHTS" -eq 0 ]; then
  # Base release: raw checkpoint + LIBERO normalizer (must stay paired).
  log "downloading released base checkpoint"
  hf_download hustvl/FasterWAM "$CKPT_DIR/fasterwam_release" \
    "libero/step_021700.pt" "libero/dataset_stats.json"

  # Merged LoRA: the checkpoint r26/r27 actually ran, with its A3 normalizer.
  log "downloading merged LoRA checkpoint"
  hf_download wsadwsad6779/fasterwam-a3-cookie-lora "$WORKSPACE/a3_train_runs/a3_lora_r1" \
    "step_002000_merged.pt"

  log "downloading Wan2.2 VAE/T5 and Wan2.1 tokenizer"
  VAE="$CKPT_DIR/Wan-AI/Wan2.2-TI2V-5B"
  TOK="$CKPT_DIR/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
  mkdir -p "$VAE" "$TOK"
  BASE_HF="${HF_ENDPOINT:-https://huggingface.co}"
  curl -L -C - -sS -o "$VAE/Wan2.2_VAE.pth" \
    "$BASE_HF/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth"
  curl -L -C - -sS -o "$VAE/models_t5_umt5-xxl-enc-bf16.pth" \
    "$BASE_HF/Wan-AI/Wan2.2-TI2V-5B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth"
  for f in special_tokens_map.json tokenizer_config.json spiece.model tokenizer.json; do
    curl -L -sS -o "$TOK/$f" \
      "$BASE_HF/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/google/umt5-xxl/$f"
  done

  verify_size "$CKPT_DIR/fasterwam_release/libero/step_021700.pt" 11117679929 || true
  verify_size "$WORKSPACE/a3_train_runs/a3_lora_r1/step_002000_merged.pt" 11117977659 || true
  verify_size "$CKPT_DIR/fasterwam_release/libero/dataset_stats.json" 40952 || true
  # The Wan/T5 files are opaque to this repo; only a size floor is meaningful.
  verify_min_size "$VAE/Wan2.2_VAE.pth" 1000000000 || true
  verify_min_size "$VAE/models_t5_umt5-xxl-enc-bf16.pth" 10000000000 || true

  if [ "$WITH_TRAINING" -eq 1 ]; then
    log "downloading training artefacts (~37 GB)"
    hf_download wsadwsad6779/fasterwam-a3-cookie-lora "$WORKSPACE/a3_train_runs/a3_lora_r1" \
      "trainer/step_002000.pt" "trainer/state/step_002000/*" "data/frames.dat"
    hf_download Eter0109/a3-front-close-left-100 "$WORKSPACE/a3_data/a3-front-close-left-100"
  fi
fi

# ---------------------------------------------------------------- 4. env file
ENV_FILE="$WORKSPACE/activate_fasterwam_a3.sh"
cat > "$ENV_FILE" <<EOF
# generated by deploy/bootstrap_pod.sh
export WORKSPACE=$WORKSPACE
export A3_CONFIG_PATH=$A3_DIR/configs/cookie_batch.yaml
export FASTERWAM_ROOT=$FW_DIR
export FASTERWAM_A3_CKPT=$WORKSPACE/a3_train_runs/a3_lora_r1/step_002000_merged.pt
export FASTERWAM_A3_INSTRUCTION='transfer 10 cookies into target box'
export DIFFSYNTH_MODEL_BASE_PATH=$CKPT_DIR
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTERWAM_A3_BRIDGE=$BRIDGE_DIR
export FASTERWAM_A3_SRC=$A3_DIR/src
export FASTERWAM_A3_RUNS=$WORKSPACE/a3_train_runs
export FASTERWAM_A3_DATA=$WORKSPACE/a3_data
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=$NEXT_DIR:$A3_DIR/src:$FW_DIR/src:$FW_DIR:$BRIDGE_DIR\${PYTHONPATH:+:\$PYTHONPATH}
EOF
log "wrote $ENV_FILE"

# ---------------------------------------------------------------- 5. smoke
if [ "$SMOKE" -eq 1 ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  log "action-chain audit (no weights, ~1 min)"
  python3 "$NEXT_DIR/run_teacher_forcing.py" --check-only --seeds 7001 \
    --out "$WORKSPACE/runs/bootstrap_check"
  if [ "$SKIP_WEIGHTS" -eq 0 ]; then
    for model in raw lora; do
      log "2-step smoke for $model (loads an 11 GB checkpoint)"
      python3 "$NEXT_DIR/run_matrix.py" --model "$model" --controls safety \
        --seeds 999 --max-steps 2 --out "$WORKSPACE/runs/_smoke_$model"
      python3 - "$WORKSPACE/runs/_smoke_$model/manifest_$model.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"  model={m['model']} gripper_mode={m['gripper_mode']} "
      f"action.min={m['normalizer']['action']['min']} action.max={m['normalizer']['action']['max']}")
PY
    done
  fi
fi

log "done"
cat <<EOF

Next, from a fresh shell:

  source $ENV_FILE

  # target-free local sanity check (mujoco only)
  python3 $NEXT_DIR/selftest.py

  # the low-intervention 2x2
  python3 $NEXT_DIR/run_matrix.py --model lora --controls safety,full \\
      --seeds 11 21 22 23 24 --out $WORKSPACE/runs/stageB

EOF
