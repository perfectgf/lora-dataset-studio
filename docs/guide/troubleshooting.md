# Troubleshooting

Symptom-first, most-reported first. If your problem isn't here, the next
chapter (**Getting help**) shows how to report it with one click.

---

## "No Z-Image model available" in the Test Studio or training panel

**Why:** the Test Studio generates through ComfyUI, so the Z-Image *base model*
must physically live in your ComfyUI install — and the scanner only accepts it
inside a sub-folder whose name contains `z image` (or `zimage`). A file dropped
loose in `models/unet` is **not** detected.

**Fix:** lay the stack out like this inside your ComfyUI folder, then re-test:

```
models/unet/z image/<your Z-Image checkpoint>.safetensors
models/text_encoders/Z image/qwen_3_4b.safetensors
models/vae/z ae.safetensors
```

A Z-Image LoRA only works on a Z-Image base — a regular SD/SDXL graph
(20–30 steps, CFG 7) renders garbage; Z-Image-Turbo wants euler / simple /
**8 steps / CFG 1.0** (the app's workflows already do this).

## "No SDXL checkpoint found" on a fresh install

**Why:** the app derives the models folder from **Settings → Local tools →
ComfyUI install directory**. If only the API URL is set, there's nothing to scan.

**Fix:** point the install directory at the folder that contains `models/` and
`main.py` (the Setup wizard detects it for you), then hit **Test**. SDXL
checkpoints are scanned from `models/checkpoints`.

## The reference crop isn't centered on the face

**Why:** on a fresh clone the configured Ollama vision model isn't pulled yet,
so head detection silently falls back to a centered square crop. The app now
shows a warning toast naming the missing model when this happens.

**Fix:** **Setup → Ollama** — pull the vision model (use the **Instruct**
variant, not *Thinking*), or click the tile's crop button and frame it by hand.
**↺ Reset to auto** re-runs the auto-crop after the model is installed.

## Training log looks frozen for several minutes

**Why:** ai-toolkit's output is block-buffered during model load and latent
caching — nothing prints even though it's working. A "warming up" phase before
the first logged step is expected, and Krea-2-Raw runs are *hours* long by
design.

**Fix:** nothing to fix — check GPU utilization or watch the ai-toolkit output
folder for new files if you want proof of life. The cloud runs page has a
stall watchdog that kills genuinely stuck runs.

## ComfyUI shows as unreachable

Check **Settings → Local tools → ComfyUI API URL** (default
`http://127.0.0.1:8188`), confirm ComfyUI is actually running, and check that a
firewall or a different bind interface isn't blocking the connection. The
**Test** button answers immediately.

## Klein engine stays greyed out

Klein needs a reachable ComfyUI **and** the Klein model files (~16 GB VRAM
class). **Setup → ComfyUI** offers the download; the license-gated fp8 model
needs a Hugging Face token (Settings → Local tools).

## Port 5000 conflict on macOS

macOS reserves port 5000 for AirPlay Receiver. Change the port in
**Settings → Server & access** (e.g. 5050) and restart.

## Garbled characters in the Windows console

Cosmetic only — some UTF-8 text renders wrong on the legacy console codepage.
The app itself is unaffected.

## `npm install` fails with `Cannot find module @rollup/rollup-<platform>-...`

Only relevant if you rebuild the frontend yourself (the repo ships `dist/`
prebuilt). It's a known npm bug: delete `frontend/node_modules` +
`frontend/package-lock.json` and run `npm install` again on this machine.

## A cloud run seems stuck

Open the **Cloud** tab: every run shows its live phase, and the stall watchdog
(Settings → Training → stall timeout) rescues logs and kills the pod if no step
progress happens for too long. Orphaned pods are also destroyed automatically
at every app start — you never pay for a forgotten GPU.
