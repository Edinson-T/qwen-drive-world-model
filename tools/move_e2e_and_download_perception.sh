#!/usr/bin/env bash
set -u

SRC=/home/fractal/datasets/waymo_e2e
DST=/run/media/fractal/Expansion/waymo_e2e
PERCEPTION=/home/fractal/datasets/waymo_perception_full
LOG=/tmp/move_and_download.log

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "start"
mkdir -p "$DST"

# Resume the copy; completed files are skipped, partial files are retried.
rsync -rt --modify-window=2 --partial --info=progress2 \
    --no-perms --no-owner --no-group "$SRC/" "$DST/" >> "$LOG" 2>&1
rc=$?
log "rsync exit=$rc"
if [ "$rc" -ne 0 ]; then
    log "abort: rsync failed, source kept"
    exit 1
fi

src_count=$(find "$SRC" -type f | wc -l)
dst_count=$(find "$DST" -type f | wc -l)
src_bytes=$(find "$SRC" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
dst_bytes=$(find "$DST" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
log "verify src_count=$src_count dst_count=$dst_count src_bytes=$src_bytes dst_bytes=$dst_bytes"

if [ "$src_count" -ne "$dst_count" ] || [ "$src_bytes" -ne "$dst_bytes" ]; then
    log "abort: verification failed, source kept"
    exit 2
fi

rm -rf "$SRC"
log "deleted source $SRC"

mkdir -p "$PERCEPTION"
log "download perception training"
gsutil -m cp -r -n \
    gs://waymo_open_dataset_v_1_4_2/individual_files/training/ \
    "$PERCEPTION/" >> "$LOG" 2>&1
log "training exit=$?"

log "download perception validation"
gsutil -m cp -r -n \
    gs://waymo_open_dataset_v_1_4_2/individual_files/validation/ \
    "$PERCEPTION/" >> "$LOG" 2>&1
log "validation exit=$?"

log "done"
