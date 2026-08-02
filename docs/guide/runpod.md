# RunPod guide

[← Documentation index](../README.md) · [Docker guide](docker.md) · [Settings reference](settings-reference.md) · [Security policy](../../SECURITY.md)

Run the whole studio on a rented NVIDIA GPU and reach it from any browser, with your datasets on a network volume that survives pod restarts.

> **Status: the configuration below is validated, the timings are not.** The first version of this page was written from the design and shipped three settings that could not work; the boot sequence has since been run end-to-end against the image this repository builds, on a GPU, with an empty root-owned volume standing in for a network volume — up to a studio that answers on its port behind the token gate. What is still unmeasured is *how long* things take on real RunPod hardware, and those numbers are marked where they appear. See the [open questions](#open-questions).

## What you get, and what you don't

| | |
|---|---|
| **Runs on the pod** | The studio UI, the Image Bank, captioning, scoring, watermark tools, and ComfyUI generation on the pod's GPU |
| **Still runs elsewhere** | LoRA training. The cloud lane rents a vast.ai instance exactly as it does locally — the pod does not train |
| **Not available** | Local ai-toolkit training. ai-toolkit is not in the image, so the "local" training lane has nothing to run |

If you only want a GPU for training, you do not need this page — the existing cloud training lane already rents one per run.

## Why pods and not serverless

RunPod's Load Balancer serverless endpoints do expose arbitrary HTTP, so hosting the Flask app there is not impossible in principle. Two platform limits rule it out for this app:

- Every request needs an `Authorization: Bearer <RUNPOD_API_KEY>` header, including health checks. A browser cannot set that header when you navigate to a URL, so the interface never loads.
- Requests and responses are capped at 30 MB. This app sets a 64 MB upload limit, and dataset ZIP export, full backup and checkpoint downloads routinely exceed 30 MB — a LoRA safetensors file alone is typically 20-600 MB.

Pods have neither limit.

## Step 1 — put the image in a registry

RunPod can only **pull** an image; it cannot build a Dockerfile. The pod runs exactly what `Dockerfile.gpu` already builds — there is no separate RunPod image — so push that somewhere RunPod can reach:

```bash
docker compose -f docker-compose.gpu.yml build
docker tag lora-dataset-studio-gpu <registry>/<name>:<tag>
docker push <registry>/<name>:<tag>
```

Budget for this. The built image is **32.1 GB uncompressed** (measured), so the build is long and the push is longer. It is a one-time cost per image version, not per pod.

## Step 2 — create the pod

| Field | Value |
|---|---|
| **Container image** | the tag you pushed |
| **Container start command** | **leave empty** |
| **Expose HTTP ports** | `5050` |
| **Network volume mount path** | `/comfy/mnt` |

The start command must stay empty. `Dockerfile.gpu` deliberately sets neither `ENTRYPOINT` nor `CMD` so that the base image's own `/comfyui-nvidia_init.bash` runs — it owns the UID/GID remap and the ComfyUI install cycle. Filling in a start command replaces it and nothing starts.

Environment variables:

```text
LDS_PUBLIC=1
LDS_DATA_DIR=/comfy/mnt/lds/data
LDS_CONFIG=/comfy/mnt/lds/data/config.json
LDS_HOST=0.0.0.0
LDS_PORT=5050
LDS_RUNTIME=docker-gpu
LDS_RESTART_MODE=supervisor
LDS_BIND_MANAGED=1
LDS_DOCKER_COMFY_MODE=bundled
LDS_DOCKER_HAS_COMFYUI=1
WANTED_UID=0
WANTED_GID=0
SECURITY_LEVEL=normal
USE_UV=true
USE_NEW_MANAGER=true
```

API keys go in the same list — `GEMINI_API_KEY`, `OPENAI_API_KEY`, `VAST_API_KEY` and the rest. The app reads secrets straight from the process environment, so the `.env` file that `docker-compose.gpu.yml` bind-mounts is not needed and has no equivalent here.

Two things there differ from `docker-compose.gpu.yml`, and each is the difference between a pod that serves and a pod that 404s:

- **No `BASE_DIRECTORY`, deliberately.** The compose file sets `BASE_DIRECTORY=/basedir` and mounts a host folder there. Set it on a pod and the boot dies on the first screen of the log: the base image *validates* that path and refuses to create it — `ERROR: BASE_DIRECTORY requested but not found or not a directory` — while a fresh network volume is empty, so there is nothing to find and no shell in which to make it. Omitting it is not a compromise. ComfyUI then keeps `models`, `input` and `output` inside `/comfy/mnt/ComfyUI`, which is on the volume and so exactly as persistent, and that is the path the studio's own config seeder looks for — point it at a `basedir` instead and the studio would not find your models even if the boot survived.
- **`WANTED_UID=0` / `WANTED_GID=0`.** A network volume arrives owned by root, and the base image refuses a run directory it does not own — `ERROR: Directory /comfy/mnt owned by unexpected user/group, expected 1024:1024, actual 0:0`, telling you to run a `chown` for which there is, again, no shell. `LDS_FORCE_CHOWN` cannot rescue this one either: it explicitly declines to chown a path mounted at container startup. Setting both to `0` makes root the expected owner, which on a single-tenant rented box is what you have anyway. If you would rather not run as root, the alternative is to attach the volume to a throwaway pod first and `chown -R 1024:1024` its mount path.

## Step 3 — open it

`LDS_PUBLIC=1` forces the access-token gate on. A pod's proxy URL is public — anyone who knows it can reach the service, with no RunPod login — and every route in this app can read API keys, launch GPU trainings and delete datasets. The gate is not optional there, and the switch in **Settings → Server & access** is locked with that reason shown.

The launcher generates a token on first boot and prints it to the pod log:

```text
[LDS] LDS_PUBLIC=1 -> this bind is reachable from the internet -> access token REQUIRED.
[LDS] Open with:  /?token=<token>
```

Open `https://<podid>-5050.proxy.runpod.net/?token=<token>` once. A signed session cookie takes over, so later requests need nothing. The bare URL without a token returns 403 — that is the gate working, not a fault.

The token is persisted in `config.json` on the network volume, so it survives restarts instead of rotating. **Settings → Server & access** shows it, with copy and regenerate controls.

`LDS_ALLOW_UNAUTHENTICATED=1` overrides all of this and serves the pod with no token at all. It exists for setups that supply their own authentication — a VPN, or a reverse proxy that authenticates — and is the wrong choice for a bare proxy URL.

## The proxy URL returns 404

A 404 from `*.proxy.runpod.net` is RunPod telling you **nothing is listening on that port** — it is not the app refusing you. A running studio that does not want to let you in answers **403**, because the token is missing or wrong. So a 404 means the studio did not start, and the pod log says why. Read it from the top:

| Log line | Cause |
|---|---|
| `ERROR: BASE_DIRECTORY requested but not found or not a directory` | `BASE_DIRECTORY` is set. Remove it — see step 2. Nothing starts at all, and the pod restart-loops on this line. |
| `ERROR: Directory /comfy/mnt owned by unexpected user/group` | The volume is root-owned. Set `WANTED_UID=0`/`WANTED_GID=0`. Nothing starts at all. |
| `[studio] ERROR: … is not writable by uid` | The studio's data directory could not be created or written, so only the studio half is missing — ComfyUI comes up beside it. |
| ComfyUI's `To see the GUI go to:` but no `[studio] starting on port 5050` | Same shape as the row above. The `[studio]` lines just before it name the reason; the launcher never aborts the container, so ComfyUI keeps running either way. |

A pod that is merely still installing is not a 404 with an error in the log — it is a 404 with the log still moving. First boot downloads several GB of CUDA wheels; the image allows 1200 seconds for it.

## Why the volume mounts at `/comfy/mnt`

`/comfy/mnt` is where the base image creates ComfyUI's virtualenv, source checkout and Hugging Face cache at runtime. Mounting the network volume there makes all of that persistent by construction, and the studio's own data is placed underneath it via `LDS_DATA_DIR`.

The alternative — mounting at `/workspace` and symlinking `/comfy/mnt` from a startup script — races the base image's init script, which touches that path on a schedule this project does not control.

The cost is cosmetic: your datasets live under a ComfyUI-named path. The benefit is that first boot's dependency install is paid once rather than on every pod start. The image allows **1200 seconds** for that first install in its own healthcheck, which is the right order of magnitude to expect.

## Limits

Every one of these is a real boundary, not a caveat:

- **The pod HTTP proxy times out at 100 seconds** (Cloudflare, reported as a 524). Dataset ZIP export and full backup build the whole archive before sending the first byte, so both can exceed it on a large dataset. The size at which this starts happening is **not yet measured**. Downloads that have already started streaming are not affected.
- **Training does not run on the pod.** It rents a vast.ai instance, as it does from a local install.
- **ai-toolkit is not in the image**, so local training is unavailable regardless of the pod's GPU.
- **A network volume pins the pod to one datacenter.** RunPod cannot schedule your pod elsewhere once a volume is attached, which can mean waiting for capacity.
- **Do not run two pods against one volume.** Two Flask processes must not share a SQLite database and a `config.json` — the same rule as the Docker guide's two-container warning.

## Open questions

Still unverified on real RunPod hardware. If you run this, please report back.

1. **First-boot and restart durations.** The design's whole argument for mounting at `/comfy/mnt` is that a restart skips the dependency install. Not yet timed on a pod.
2. **The 100-second export threshold**, as above.

**Closed:** *volume ownership*, which used to be the first entry here. The base image does not cope with a root-owned run directory and `LDS_FORCE_CHOWN` cannot help — it refuses to chown a path mounted at container startup. `WANTED_UID=0`/`WANTED_GID=0` in step 2 is the answer, and it is now part of the configuration rather than an open question.
