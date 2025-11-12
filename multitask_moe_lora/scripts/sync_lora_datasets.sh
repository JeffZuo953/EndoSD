#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT=${SRC_ROOT:-/data/ziyi/multitask}
REMOTE_USER=${REMOTE_USER:-ziyi}
REMOTE_HOST=${REMOTE_HOST:-10.147.20.230}
REMOTE_ROOT=${REMOTE_ROOT:-/mnt/DATA/ziyi/multitask}
SSH_PASSWORD=${SSH_PASSWORD:-}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-/tmp/sync_lora_datasets_$$.sock}
SSH_CONTROL_PERSIST=${SSH_CONTROL_PERSIST:-600}
USE_SSH_CONTROL=${USE_SSH_CONTROL:-1}

SSH_PASS_PREFIX=()
if [[ -n "${SSH_PASSWORD}" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
        echo "[ERROR] sshpass is not installed but SSH_PASSWORD is set. Install sshpass or unset SSH_PASSWORD."
        exit 1
    fi
    SSH_PASS_PREFIX=(sshpass -p "${SSH_PASSWORD}")
fi

SSH_BASE_OPTS=()
if [[ "${USE_SSH_CONTROL}" == "1" ]]; then
    SSH_BASE_OPTS=(-o ControlMaster=auto -o ControlPersist="${SSH_CONTROL_PERSIST}" -o ControlPath="${SSH_CONTROL_PATH}")
    cleanup_control_socket() {
        ssh_with_opts -O exit "${REMOTE_USER}@${REMOTE_HOST}" >/dev/null 2>&1 || true
        rm -f "${SSH_CONTROL_PATH}"
    }
    trap cleanup_control_socket EXIT
fi

ssh_with_opts() {
    if [[ ${#SSH_PASS_PREFIX[@]} -gt 0 ]]; then
        "${SSH_PASS_PREFIX[@]}" ssh "${SSH_BASE_OPTS[@]}" "$@"
    else
        ssh "${SSH_BASE_OPTS[@]}" "$@"
    fi
}

rsync_with_opts() {
    if [[ ${#SSH_PASS_PREFIX[@]} -gt 0 ]]; then
        "${SSH_PASS_PREFIX[@]}" rsync -e "ssh ${SSH_BASE_OPTS[*]}" "$@"
    else
        rsync -e "ssh ${SSH_BASE_OPTS[*]}" "$@"
    fi
}

DATASETS=(
    "data/LS/EndoSynth"
    "data/LS/EndoNeRF"
    "data/LS/EndoVis2017"
    "data/LS/EndoVis2018"
    "data/NO/Kidney3D-CT-depth-seg"
    "data/NO/RIRS-SegC"
    "data/NO/RIRS-SegP"
    "data/NO/bkai-igh-neopolyp"
    "data/NO/clinicDB"
    "data/NO/CVC-EndoScene"
    "data/NO/kvasir-SEG-split"
    "data/NO/kvasir-seg"
    "data/NO/ETIS-LaribPolypDB"
)

TOTAL_SIZE=0
printf "[INFO] Dataset footprints under %s\n" "$SRC_ROOT"
for rel in "${DATASETS[@]}"; do
    if [[ -d "${SRC_ROOT}/${rel}" ]]; then
        du -sh "${SRC_ROOT}/${rel}"
    else
        echo "[WARN] Missing ${SRC_ROOT}/${rel}"
    fi
done

echo "[INFO] Ensuring remote root ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}"
ssh_with_opts "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}'"

for rel in "${DATASETS[@]}"; do
    SRC_PATH="${SRC_ROOT}/${rel}"
    DEST_PATH="${REMOTE_ROOT}/${rel}"
    echo "[SYNC] ${SRC_PATH} -> ${REMOTE_USER}@${REMOTE_HOST}:${DEST_PATH}"
    ssh_with_opts "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '$(dirname "${DEST_PATH}")'"
    # -L ensures we copy the contents of cache_pt (and other symlinked dirs) instead of the link itself
    rsync_with_opts -aL --partial --update --compress --info=progress2 \
        "${SRC_PATH}/" \
        "${REMOTE_USER}@${REMOTE_HOST}:${DEST_PATH}/"
done

echo "[DONE] LoRA datasets synced."
