#!/usr/bin/env bash
# Restore the FasterWAM / A3 working set onto a brand new GPU instance and prove
# the bytes match the archive. Read-only with respect to the archive: this script
# never deletes anything, never overwrites a file whose sha256 already matches,
# and never starts training or evaluation (those commands live in
# docs/RESTORE_NEW_INSTANCE.md).
#
#   HF_TOKEN=... bash scripts/restore_new_instance.sh --all      # code+deps+data+weights
#   bash scripts/restore_new_instance.sh --verify                # hash what is on disk
#   bash scripts/restore_new_instance.sh --weights weights/r4/light/step_004000/adapter.pt
#   bash scripts/restore_new_instance.sh --rest                  # eval/videos/replay/runs/tests/research
#   bash scripts/restore_new_instance.sh --parts                 # rebuild the three >2 GiB weights
#   bash scripts/restore_new_instance.sh --mirror                # use hf-mirror.com
set -uo pipefail

REPO='wsadwsad6779/fasterwam-a3-cookie-lora'
DATASET='Eter0109/a3-front-close-left-100'
DATASET_REV='8fdd8a7caf8275eba8be8a83b53a99176a084062'
UP_FASTERWAM='https://github.com/hustvl/FasterWAM'
UP_FASTERWAM_SHA='83667817df0d4f823f39d90700e61ea2f432ac45'
UP_A3SIM='https://github.com/Eter0109/a3_dual_arm_sim'
UP_A3SIM_SHA='08ebe8cb370856051d0b32b07e691cc9b056716b'
WS="${WS:-/workspace}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PIP_MIRROR="${PIP_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"
BASE='https://huggingface.co'
MANIFEST="${MANIFEST:-$HERE/archive/2026-11-01/p0p1_manifest.tsv}"
PARTS="${PARTS:-$HERE/archive/2026-11-01/parts60gz_manifest.tsv}"
MANIFEST_SHA="${MANIFEST_SHA:-54f4e3669a46b3af58331abee2b55c350c1292516b692c19cb4bb0520d596b08}"
PARTS_SHA="${PARTS_SHA:-307172a4f82d98f58391068c2ac4078ac4f8ff64efafa50470361456db6d0f04}"
JOBS="${JOBS:-6}"

actions=()
only_paths=()
for a in "$@"; do
  case "$a" in
    --mirror) BASE='https://hf-mirror.com' ;;
    --all) actions+=(code deps data weights rest parts verify) ;;
    --code) actions+=(code) ;;
    --deps) actions+=(deps) ;;
    --data) actions+=(data) ;;
    --weights) actions+=(weights) ;;
    --rest) actions+=(rest) ;;
    --parts) actions+=(parts) ;;
    --verify) actions+=(verify) ;;
    --*) echo "unknown flag: $a" >&2; exit 2 ;;
    *) only_paths+=("$a"); actions+=(weights) ;;
  esac
done

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
url_for() { printf '%s/%s/resolve/main/%s' "$BASE" "$REPO" "$1"; }
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
size_of() { stat -c %s "$1" 2>/dev/null || echo -1; }

# The two ledgers also live in the repo's MANIFEST/ directory. If the local copy
# is absent (e.g. the gz part ledger is not committed to git), fetch it and check
# the pinned sha256 before anything trusts it.
ensure_ledger() {
  local f="$1" name="$2" want="$3"
  if [ -f "$f" ]; then
    if [ -z "$want" ] || [ "$(sha_of "$f")" = "$want" ]; then return 0; fi
    log "WARN: local $name does not match the pinned sha256, refetching"
  fi
  mkdir -p "$(dirname "$f")"
  log "ledger $name <- $BASE/$REPO/resolve/main/MANIFEST/$name"
  curl -sSL --fail --retry 5 --retry-delay 3 -o "$f.part" \
    "$BASE/$REPO/resolve/main/MANIFEST/$name" || return 1
  if [ -n "$want" ] && [ "$(sha_of "$f.part")" != "$want" ]; then
    echo "LEDGER SHA MISMATCH $name got=$(sha_of "$f.part") want=$want" >&2; return 1
  fi
  mv "$f.part" "$f"
}

ensure_ledgers() {
  ensure_ledger "$MANIFEST" p0p1_manifest.tsv "$MANIFEST_SHA" || return 1
  ensure_ledger "$PARTS" parts60gz_manifest.tsv "$PARTS_SHA" || return 1
}

want_for()  { awk -F'\t' -v p="$1" '$2==p {print $5; exit}' "$MANIFEST"; }
bytes_for() { awk -F'\t' -v p="$1" '$2==p {print $4; exit}' "$MANIFEST"; }
# Column 3 is the path the file had on the old pod, relative to /workspace. It is
# the only correct destination: a3_train_runs/a3_lora_r3/checkpoints/weights/...
# and .../checkpoints/light/... both carry a "checkpoints" segment that a naive
# weights/<round>/<rest> rewrite drops.
rel_for()   { awk -F'\t' -v p="$1" '$2==p {print $3; exit}' "$MANIFEST"; }

# Parallel byte-range fetch. Measured against the mirror: one stream ~5.6 MB/s,
# six ranges ~26 MB/s; a straight huggingface.co stream is ~1.7 MB/s.
fetch_range6() {
  local url="$1" out="$2" total="$3"
  local per nr i f exp s e
  per=$(( (total + JOBS - 1) / JOBS ))
  nr=$(( (total + per - 1) / per ))
  local tmp="$out.ranges"
  mkdir -p "$tmp"
  local pieces=()
  for ((i = 0; i < nr; i++)); do
    f="$tmp/$(printf '%02d' "$i")"
    s=$((i * per)); e=$((s + per - 1)); [ "$e" -ge "$total" ] && e=$((total - 1))
    exp=$((e - s + 1))
    pieces+=("$f")
    if [ -f "$f" ] && [ "$(size_of "$f")" = "$exp" ]; then continue; fi
    log "  range $((i + 1))/$nr bytes $s-$e"
    curl -sSL --fail --retry 5 --retry-delay 3 -r "$s-$e" -o "$f.part" "$url" || return 1
    [ "$(size_of "$f.part")" = "$exp" ] || { log "  range $i truncated"; return 1; }
    mv "$f.part" "$f"
  done
  cat "${pieces[@]}" > "$out" || return 1
  [ "$(size_of "$out")" = "$total" ] || { log "  assembled size mismatch"; return 1; }
}

fetch_one() {
  local in_repo="$1" dest="$2"
  local want bytes got
  want="$(want_for "$in_repo")"; bytes="$(bytes_for "$in_repo")"
  if [ -z "$want" ]; then echo "not in manifest: $in_repo" >&2; return 1; fi
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ] && [ "$(sha_of "$dest")" = "$want" ]; then log "OK (already) $in_repo"; return 0; fi
  if [ "${bytes:-0}" -gt 2000000000 ]; then
    log "range fetch $in_repo (${bytes} B)"
    fetch_range6 "$(url_for "$in_repo")" "$dest" "$bytes" || return 1
  else
    log "get $in_repo"
    curl -sSL --fail --retry 5 --retry-delay 3 -o "$dest.part" "$(url_for "$in_repo")" || return 1
    mv "$dest.part" "$dest"
  fi
  got="$(sha_of "$dest")"
  if [ "$got" != "$want" ]; then echo "SHA MISMATCH $in_repo got=$got want=$want" >&2; return 1; fi
  log "OK $in_repo"
}

fetch_one_part() {
  local in_repo="$1" dest="$2" want="$3" bytes="$4"
  if [ -f "$dest" ] && [ "$(sha_of "$dest")" = "$want" ]; then log "OK (already) $in_repo"; return 0; fi
  log "part $in_repo"
  curl -sSL --fail --retry 5 --retry-delay 3 -o "$dest.part" "$(url_for "$in_repo")" || return 1
  [ "$(size_of "$dest.part")" = "$bytes" ] || { echo "PART SIZE MISMATCH $in_repo" >&2; return 1; }
  [ "$(sha_of "$dest.part")" = "$want" ] || { echo "PART SHA MISMATCH $in_repo" >&2; return 1; }
  mv "$dest.part" "$dest"
}

# The three >2 GiB weights are archived as ordered 60 MiB slices, each stored
# gzip-compressed (gzip -n -1). Two reasons, both measured on the pod:
#   1. The mirror's Xet uploader silently stalls (no socket, no progress, xet log
#      frozen) on any file where the server answers the chunk dedup query with
#      "found". Recompressed bytes never hit an existing chunk, so every slice
#      uploads at ~10 MB/s and is verified on read-back.
#   2. The slices are plain byte ranges of the original file, so decompressing in
#      part order reproduces it byte for byte. gzip members concatenate, so
#      `cat part_*.gz | gunzip` is the whole file; the whole-file sha256 listed in
#      p0p1_manifest.tsv proves the result.
assemble_part() {
  local whole="$1" dest="$2"
  local want rows tmp f got list=()
  command -v gunzip > /dev/null || { echo "gunzip not found" >&2; return 1; }
  want=$(awk -F'\t' -v p="$whole" '$2==p {print $5; exit}' "$MANIFEST")
  [ -z "$want" ] && { echo "no manifest row for $whole" >&2; return 1; }
  rows=$(awk -F'\t' -v w="$whole" 'index($2, "/" w "/part_") > 0 {print $2 "\t" $4 "\t" $5}' "$PARTS" | sort)
  [ -n "$rows" ] || { echo "no parts listed for $whole in $PARTS" >&2; return 1; }
  tmp="$WS/_parts60gz/$(basename "$whole")"
  mkdir -p "$tmp" "$(dirname "$dest")"
  while IFS=$'\t' read -r pr pb ps; do
    [ -n "$pr" ] || continue
    f="$tmp/${pr##*/}"
    fetch_one_part "$pr" "$f" "$ps" "$pb" || return 1
    list+=("$f")
  done <<< "$rows"
  cat "${list[@]}" | gunzip > "$dest.part" || { echo "gunzip failed for $whole" >&2; return 1; }
  got="$(sha_of "$dest.part")"
  if [ "$got" != "$want" ]; then echo "ASSEMBLED SHA MISMATCH $whole got=$got want=$want" >&2; return 1; fi
  mv "$dest.part" "$dest"
  log "OK (assembled) $whole -> $dest"
}

do_code() {
  mkdir -p "$WS"
  local url sha dir
  for spec in "$UP_FASTERWAM $UP_FASTERWAM_SHA FasterWAM" "$UP_A3SIM $UP_A3SIM_SHA a3_dual_arm_sim"; do
    set -- $spec
    url="$1"; sha="$2"; dir="$WS/$3"
    if [ -d "$dir/.git" ]; then
      log "already there: $dir ($(git -C "$dir" rev-parse HEAD))"
      [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || log "WARN: $dir is not at the pinned commit"
      continue
    fi
    git clone "$url" "$dir" && git -C "$dir" checkout "$sha"
  done
  mkdir -p "$WS/fasterwam_a3_bridge"
  ln -sfn "$WS/FasterWAM" "$WS/fasterwam_a3_bridge/FasterWAM"
  log "code done"
}

do_deps() {
  pip install -i "$PIP_MIRROR" --upgrade pip
  pip install -i "$PIP_MIRROR" \
    'mujoco==3.3.2' 'robosuite==1.4.0' transformers==4.57.6 accelerate==1.13.0 \
    robomimic libero bddl numpy opencv-python h5py imageio imageio-ffmpeg \
    tqdm pyyaml hydra-core omegaconf safetensors einops timm
  log "deps done (torch comes from the instance image; confirm python 3.12.3)"
}

do_data() {
  local in_repo rel
  while IFS=$'\t' read -r _b in_repo rel _n _s; do
    case "$in_repo" in
      data/*) fetch_one "$in_repo" "$WS/$rel" ;;
    esac
  done < "$MANIFEST"
  if [ "${WITH_DATASET:-0}" = 1 ]; then
    hf download "$DATASET" --repo-type dataset --revision "$DATASET_REV" \
      --local-dir "$WS/a3_data/a3-front-close-left-100"
  else
    log "skipped the third-party dataset (set WITH_DATASET=1 to pull it; redistribution needs the owner's OK)"
  fi
}

do_weights() {
  if [ ${#only_paths[@]} -gt 0 ]; then
    local p r
    for p in "${only_paths[@]}"; do
      r="$(rel_for "$p")"
      [ -n "$r" ] || { echo "not in manifest: $p" >&2; return 1; }
      fetch_one "$p" "$WS/$r"
    done
    return
  fi
  local in_repo rel n
  while IFS=$'\t' read -r _b in_repo rel n _s; do
    case "$in_repo" in
      large_parts/*|parts60/*|parts60gz/*) : ;;
      weights/*|trainer/*)
        if [ "${n:-0}" -gt 2000000000 ]; then log "skip $in_repo (giant, rebuild with --parts)"; continue; fi
        fetch_one "$in_repo" "$WS/$rel" ;;
    esac
  done < "$MANIFEST"
}

# Everything that is not code, data or weights: evaluation outputs, videos,
# replay frames, run configs/logs, light_state tests and research artefacts.
do_rest() {
  local in_repo rel
  while IFS=$'\t' read -r _b in_repo rel _n _s; do
    case "$in_repo" in
      eval/*|videos/*|replay/*|runs/*|tests/*|research/*) fetch_one "$in_repo" "$WS/$rel" ;;
    esac
  done < "$MANIFEST"
}

do_parts() {
  local w r
  while read -r w; do
    [ -n "$w" ] || continue
    r="$(rel_for "$w")"
    [ -n "$r" ] || { echo "no manifest row for $w" >&2; return 1; }
    assemble_part "$w" "$WS/$r"
  done < <(awk -F'\t' '{sub("^parts60(gz)?/[^/]+/", ""); sub("/part_[0-9]+(\\.gz)?$", ""); print}' "$PARTS" | sort -u)
}

do_verify() {
  [ -f "$MANIFEST" ] || { echo "manifest missing: $MANIFEST" >&2; return 1; }
  local rows=0 ok=0 missing=0 bad=0 dest in_repo rel s
  while IFS=$'\t' read -r _b in_repo rel _n s; do
    rows=$((rows + 1))
    dest="$WS/$rel"
    if [ ! -f "$dest" ]; then missing=$((missing + 1)); continue; fi
    if [ "$(sha_of "$dest")" = "$s" ]; then ok=$((ok + 1)); else bad=$((bad + 1)); echo "BAD $in_repo"; fi
  done < "$MANIFEST"
  local pw=0 pbad=0 whole wsha pdest
  if [ -f "$PARTS" ]; then
    while read -r whole; do
      [ -n "$whole" ] || continue
      wsha=$(awk -F'\t' -v p="$whole" '$2==p {print $5; exit}' "$MANIFEST")
      pdest="$WS/$(rel_for "$whole")"
      if [ ! -f "$pdest" ]; then pw=$((pw + 1)); log "missing (assembled) $whole"; continue; fi
      if [ "$(sha_of "$pdest")" = "$wsha" ]; then log "OK (assembled) $whole"; else pbad=$((pbad + 1)); echo "BAD $whole"; fi
    done < <(awk -F'\t' '{sub("^parts60(gz)?/[^/]+/", ""); sub("/part_[0-9]+(\\.gz)?$", ""); print}' "$PARTS" | sort -u)
  fi
  log "verify rows=$rows ok=$ok missing=$missing sha_bad=$bad giants_missing=$pw giants_bad=$pbad"
  [ "$bad" = 0 ] && [ "$pbad" = 0 ]
}

for act in "${actions[@]}"; do
  case "$act" in
    data|weights|rest|parts|verify) ensure_ledgers || exit 3 ;;
  esac
done

for act in "${actions[@]}"; do
  case "$act" in
    code) do_code ;;
    deps) do_deps ;;
    data) do_data ;;
    weights) do_weights ;;
    rest) do_rest ;;
    parts) do_parts ;;
    verify) do_verify ;;
    *) echo "unknown action: $act" >&2; exit 2 ;;
  esac
done
