#!/bin/bash
# LoRA Dataset Studio, started inside the ComfyUI-Nvidia container.
#
# Installed as /userscripts_dir/50-lora-dataset-studio.sh, which upstream's
# init.bash runs in sorted order as the `comfy` user — AFTER the WANTED_UID/GID
# remap and ComfyUI's venv install, BEFORE ComfyUI itself launches. That hook is the
# whole seam between this app and mmartial/comfyui-nvidia-docker: no ENTRYPOINT
# override, one script, at their documented extension point.
#
# This script must ALWAYS exit 0 and must never `set -e`. Upstream calls it as
# `$userscript || error_exit "..."`, so a non-zero return aborts the boot — and a
# problem with the studio must never cost the user their ComfyUI.
set -u

STUDIO_DIR=/app
# Spelled out rather than built from STUDIO_DIR: the contract test asserts on it.
STUDIO_PY=/app/.venv/bin/python
DATA_DIR="${LDS_DATA_DIR:-/data}"

log() { echo "[studio] $*"; }

# Only /app/.venv, and never /app itself: the compose file bind-mounts ./.env at
# /app/.env, so a chown -R of /app would walk into that mount and re-own the
# user's real API-key file on the host.
#
# Backgrounded on purpose. This walks ~7 GB of torch, and upstream runs this hook
# BEFORE it launches ComfyUI — so doing it inline would hold up both services for
# minutes on every container recreate. The studio only needs to READ the venv to
# run; ownership matters solely so the in-app dependency installer can pip into it.
if [ ! -w "${STUDIO_DIR}/.venv" ]; then
  log "adopting ${STUDIO_DIR}/.venv for uid $(id -u):$(id -g) in the background"
  log "(installing dependencies from inside the app will not work until it finishes)"
  ( sudo chown -R "$(id -u):$(id -g)" "${STUDIO_DIR}/.venv" \
      && log "${STUDIO_DIR}/.venv adopted" \
      || log "chown of ${STUDIO_DIR}/.venv failed — the in-app installer will not work" ) &
fi

# A bind mount is created by the host before the container starts, so /data always
# exists. A path nested inside a mounted VOLUME does not: on a rented pod
# LDS_DATA_DIR lives under the network volume, and nothing ever creates it — while
# `[ ! -w ]` on a missing directory is true, so the check below would report a
# permission problem for a directory that simply was not there. Before the chown
# too, or the opt-in below reports a failure to adopt a directory that does not
# exist yet. Non-fatal on purpose: if the parent cannot be written either, the
# writability check says so, with the command to run.
mkdir -p "${DATA_DIR}" 2>/dev/null || true

# /data is the user's bind mount: the host owns it, so its ownership is not rewritten
# behind their back. Only on request, and otherwise say exactly what to run. The
# opt-in adopts the entire tree even when /data itself is writable: files below it
# may still be root-owned after an earlier container run.
if [ "${LDS_FORCE_CHOWN:-false}" = "true" ]; then
  log "LDS_FORCE_CHOWN=true — taking ownership of ${DATA_DIR}"
  sudo chown -R "$(id -u):$(id -g)" "${DATA_DIR}" || log "chown of ${DATA_DIR} failed"
fi

if [ ! -w "${DATA_DIR}" ]; then
  log "ERROR: ${DATA_DIR} is not writable by uid $(id -u):$(id -g), so the studio"
  log "ERROR: cannot start. On the host, run:"
  log "ERROR:   sudo chown -R $(id -u):$(id -g) <the folder mounted at ${DATA_DIR}>"
  log "ERROR: or set LDS_FORCE_CHOWN=true in the compose file. ComfyUI still starts."
  exit 0
fi

/usr/bin/python3 "${STUDIO_DIR}/packaging/docker/seed_comfy_config.py" \
  || log "config seeding failed — set the ComfyUI folders in Settings > Local tools"

# ComfyUI is the foreground process of upstream's init script; the studio is a
# background job, so nothing else would ever restart it. This loop is that
# supervisor. `env -u` clears ComfyUI's activated venv so backend/run.py's own
# interpreter check starts from a clean environment.
(
  while true; do
    env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \
      "${STUDIO_PY}" "${STUDIO_DIR}/backend/run.py" 2>&1 | sed -u 's/^/[studio] /'
    # `$?` here would be sed's result and would hide the backend's deliberate
    # restart code. PIPESTATUS must be captured before any other command runs.
    studio_status=${PIPESTATUS[0]}
    if [ "${studio_status}" -eq 75 ]; then
      log "restart requested — restarting immediately"
      continue
    fi
    log "the studio exited with status ${studio_status} — restarting it in 10s"
    sleep 10
  done
) &

log "starting on port ${LDS_PORT:-5050}; ComfyUI follows on 8188"
exit 0
