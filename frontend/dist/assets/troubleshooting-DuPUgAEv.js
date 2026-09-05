const e=`# Troubleshooting

Symptom-first, most-reported first. If your problem isn't here, the next
chapter (**Getting help**) shows how to report it with one click.

---

## The page is completely blank on Windows, in every browser

**Symptom:** the server log looks healthy, but the page is white and no browser
loads the interface.

**Why:** Windows stores a content type for each extension in
\`HKEY_CLASSES_ROOT\\<ext>\\Content Type\`. Another program can overwrite the
\`.js\` value with \`text/plain\`; older LDS builds then served the bundle with
that registry MIME type and browsers refused to execute it.

**Fix:** update LoRA Dataset Studio and restart it. Current builds set the MIME
type of served assets themselves and do not trust the Windows registry. If an
old build must be used temporarily, repair the \`.js\` Content Type to a
JavaScript MIME type, then restart the browser. Updating is safer than making a
registry edit by hand.

*(Reported and diagnosed in [GitHub #12](https://github.com/perfectgf/lora-dataset-studio/issues/12).)*

## "No Z-Image model available" in the Test Studio or training panel

**Why:** the Test Studio generates through ComfyUI, so the Z-Image *base model*
must physically live in your ComfyUI install — and the scanner only accepts it
inside a sub-folder whose name contains \`z image\` (or \`zimage\`). A file dropped
loose in \`models/unet\` is **not** detected.

**Fix:** lay the stack out like this inside your ComfyUI folder, then re-test:

\`\`\`
models/unet/z image/<your Z-Image checkpoint>.safetensors
models/text_encoders/Z image/qwen_3_4b.safetensors
models/vae/z ae.safetensors
\`\`\`

**The text encoder and the VAE are flexible** — only the base model needs that
sub-folder. The app resolves those two itself: any capitalisation, any separator
and any sub-folder work, so \`models/vae/z_ae.safetensors\`, \`models/vae/ae.safetensors\`
(the name ComfyUI's own Z-Image page uses), \`text_encoders/Z Image/qwen_3_4b.safetensors\`
and a bare \`text_encoders/qwen_3_4b.safetensors\` are all found, including under an
\`extra_model_paths.yaml\` root. **Do not rename your files to match the layout above.**
If the app still says one is missing, the message lists what it accepted and where it
looked; you can also pin either file by hand with the \`zimage.vae\` /
\`zimage.text_encoder\` settings (see *Settings reference → Config-file-only settings*).

A Z-Image LoRA only works on a Z-Image base — a regular SD/SDXL graph
(20–30 steps, CFG 7) renders garbage. The two Z-Image builds then want opposite
sampler settings, and the Test Studio proposes the right pair per base model:
**Z-Image-Turbo** is guidance-distilled and wants euler / simple / **8 steps /
CFG 1.0**, while the non-distilled **Z-Image Base** needs roughly **30–50 steps at
CFG 3–5** (ComfyUI's own recommendation) — run Base at CFG 1 and it renders mush.
Those are starting points on a sweepable axis, not measured optima: the Studio grid
exists to let you find yours.

## "No SDXL checkpoint found" on a fresh install

**Why:** the app derives the models folder from **Settings → Local tools →
ComfyUI install directory**. If only the API URL is set, there's nothing to scan.

**Fix:** point the install directory at the folder that contains \`models/\` and
\`main.py\` (the Setup wizard detects it for you), then hit **Test**. SDXL
checkpoints are scanned from \`models/checkpoints\`.

## The Krea 2 Turbo Test Studio says a custom node is missing

**Why:** the Krea grid rebalances the Qwen3-VL conditioning through a small
community node (class \`ConditioningKrea2Rebalance\`). It isn't a stock ComfyUI
node, so a ComfyUI that doesn't have it can't run the Krea pipeline and the
Studio stops before wasting a run — the same up-front check used for missing
model files.

**Fix:** install the **ComfyUI-Conditioning-Rebalance** pack (in ComfyUI-Manager,
search **"Krea 2 Conditioning"** — repo
\`https://github.com/nova452/ComfyUI-Conditioning-Rebalance\`), then restart
ComfyUI and relaunch the test. The Studio's error banner names this pack and
links it directly. Either that original pack or its \`comfyui-krea2-conditioning\`
fork works — the app pins the node so your rebalance-strength setting is applied
the same way on both.

## The reference crop isn't centered on the face

**Why:** on a fresh clone the configured Ollama vision model isn't pulled yet,
so head detection silently falls back to a centered square crop. The app now
shows a warning toast naming the missing model when this happens.

**Fix:** **Setup → Ollama** — pull the vision model (use the **Instruct**
variant, not *Thinking*), or click the tile's crop button and frame it by hand.
**↺ Reset to auto** re-runs the auto-crop after the model is installed.

## Ollama isn't detected (or is installed but stopped)

In Docker, host binary detection is not the deployment selector. Open **Setup → Ollama** and choose:

| Docker choice | Expected state | Fix |
|---|---|---|
| **No Ollama** | Disabled by choice | Choose another card only if you want the Ollama features |
| **Existing host Ollama** | API at \`http://host.docker.internal:11434\` | Start Ollama on the host, bind it so Docker can reach it, and restrict port 11434 to Docker/private networks |
| **Docker Ollama** | Companion API at \`http://ollama:11434\` | If the companion is absent, rerun the same LDS Docker launcher |

On a native install, LDS still distinguishes **not installed**, **installed but stopped**, and **running**. The **▶ Start Ollama** button applies only to a detected native binary.

**You do not have to install Ollama to finish Setup.** If JoyCaption is installed, captioning already works without it and the step is only a recommendation. With neither installed, the step offers **Continue without Ollama**, which lists what turns off (auto-classify framing, auto head-crop, Test Studio Describe & Enhance, the bank's "Describe filter", the vision route of watermark detection, short captions) before you commit, and then stops asking. Starting Ollama later cancels the skip on its own — nothing to undo.

No launcher or **Install everything** action pulls the large vision model. Once the selected service is reachable, use the explicit **Pull** button in LDS Setup; it shows progress and supports cancellation/resume. Keep the **Instruct** tag. The Thinking variant reasons instead of returning the compact captions these workflows expect.

## LM Studio is running but LDS says nothing is loaded

That is usually correct, not a bug. LM Studio ships with **JIT loading off**, so the server answers every request that lists models and refuses every request that generates one. Load a model in its **Developer** tab (a vision model if you want captioning, framing or head-crop) and the status turns green.

Three more things worth knowing when the two disagree:

| Symptom | Cause | Fix |
|---|---|---|
| Every call fails, and the message talks about Ollama holding the GPU | The URL carries a path — LM Studio's Developer tab shows \`http://localhost:1234/v1\` and that is what gets pasted | Nothing to do on recent builds: the \`/v1\` is stripped automatically. If you typed something else after the port, remove it. |
| **"No usable model is loaded"** | LM Studio ships with just-in-time loading OFF, and older LDS builds left the loading to you — then unloaded your own copy when their keep-warm expired, which read as "load it, again and again" | Update LDS: it now loads the model itself — automatically when a pass needs it, or from the **⏬ Load the vision model** button in Setup and Settings ▸ Local tools. A missing model can be **downloaded from Settings ▸ Local tools** as well — model id or huggingface.co URL; the job runs inside LM Studio, so it survives an LDS restart. |
| The card says the server answers but cannot tell what is loaded | Only the OpenAI-compatible API is answering; it reports neither model type nor residency | Name a model explicitly in **Settings ▸ Local tools ▸ LM Studio model**, or update LM Studio so its native API answers |
| Captioning works but framing/head-crop do not | The loaded model is a text model, not a vision one | Load a VLM (a model LM Studio lists with vision support) |

**In Docker, \`127.0.0.1\` is the container, not your machine.** LM Studio runs on the host, so a containerised LDS must be pointed at **\`http://host.docker.internal:1234\`** — the Settings card shows that address as the placeholder when it detects a container. LM Studio's server also has to be reachable from Docker (it listens on localhost only by default; enable serving on the local network in its Developer tab).

**▶ Start LM Studio** appears on the Local tools card and the Setup step when the server is down and LM Studio's command-line tool is present — it is installed the first time you open LM Studio, so an install that has never been launched gets the Developer-tab sentence instead of a button that could not work. Pressing it leaves a model alone if only the server had stopped; if LM Studio itself was closed, the server comes back empty and you load a model in its Developer tab. Either way it starts the server on the port your settings name. In Docker the button is not offered: the container cannot start an application on your desktop, whatever the URL says.

## Training log looks frozen for several minutes

**Why:** ai-toolkit's output is block-buffered during model load and latent
caching — nothing prints even though it's working. A "warming up" phase before
the first logged step is expected, and Krea-2-Raw runs are *hours* long by
design.

**Fix:** nothing to fix — check GPU utilization or watch the ai-toolkit output
folder for new files if you want proof of life. The cloud runs page has a
stall watchdog that kills genuinely stuck runs.

## Training dies immediately on an RTX 50-series card ("no kernel image is available")

**Why:** an RTX 50-series/Blackwell GPU reports compute capability \`sm_120\`.
An older torch build can still report \`torch.cuda.is_available() == True\` and
name the card correctly while carrying no kernel for that architecture. The run
then fails on its first real CUDA operation.

**Fix:** install a CUDA 12.8 torch build **inside ai-toolkit's own Python
environment**, not only in the LDS venv:

\`\`\`bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
\`\`\`

Run that command with the exact Python configured under **Settings → Local
tools → ai-toolkit Python interpreter**. The preflight/failure panel recognizes
this specific \`sm_120\` mismatch. For another architecture mismatch, use the
torch build appropriate to that card instead of copying this command blindly.

## Local training crawls: GPU near idle, RAM climbing, ETA in the hundreds of hours

The run started, the training panel shows a step counter, but the ETA reads
like \`ETA 300:12:45\`, the GPU sits at a few percent and the only number that
moved is system RAM. Nothing failed, and that is the problem.

**What it is.** The run no longer fits in the card's memory. On Windows the
NVIDIA driver, by default (Control Panel → *CUDA - Sysmem Fallback Policy*),
does not answer that with an out-of-memory error: it pages the overflow into
system RAM and keeps going, tens of times slower. Measured here: Krea 2 at 1024
on a saturated 24 GB card, about 180 s per step and an ETA of seven days; the
same run at 768, 3.5 s per step. Linux, or Windows with *Prefer No Sysmem
Fallback* set, fails loudly instead (\`CUDA out of memory\`), which is the kinder
failure.

**Why it happens on a card that should fit.** The Krea 2 and FLUX.1 recipes are
calibrated to fit a 24 GB card that is *empty*, and they do: measured with the
shipped Krea 2 recipe on an otherwise idle 24 GB card, the run peaks at about
21.6 GB, so a little under 3 GB is all the room there is. Anything else holding
that much tips them over, and the usual culprit on a machine that also runs ComfyUI is
ComfyUI itself: it keeps the models it last used resident (in VRAM while it
has the room, in system RAM once they leave the card) and does not let go on
its own. A
browser with many tabs, a second monitor and a game launcher add up too.

**Two readings settle it.** Task Manager → Performance → GPU: *Dedicated GPU
memory* pinned near the top with *Shared GPU memory* at several GB means the
driver is paging. The GPU % in the app's 📊 readout comes straight from
\`nvidia-smi\` and is the figure to trust.

**Fix.** ⏹ Stop the run, unfold the 📊 readout in the top bar and press
🧹 **Free memory** (ComfyUI unloads its cached models and the vision model is
released; the VRAM figure should fall to a couple of GB), or simply close
ComfyUI, then launch again. A healthy run shows the GPU at 90-100 % and a few
seconds per step once the first minutes are over. If it still crawls with an
empty card, ⚙️ Advanced options → **Resolution** → *768 only (low VRAM)*, and
leave Expert → **Memory saving** on: that trio of switches is what makes a 12B
model fit at all.

The app also asks ComfyUI to unload before every local run (the same request
the 🧹 button and the vision passes make), so with a reachable ComfyUI this
trap closes by itself; the app log's launch line says what the card held
before the request and at spawn.

**Not this.** The first minutes of an image run legitimately show a low GPU and
a high RAM: the weights are read into RAM and quantised, the text embeddings and
latents are cached, and the step-0 preview images are rendered before step 1 (a
video run skips the previews). Read the ETA after five minutes of steps, not
before.

## ai-toolkit isn't detected (conda / uv / no venv)

**Why:** the app auto-detects ai-toolkit's Python from a \`venv/\` or \`.venv/\`
folder next to its \`run.py\`. Installs that use conda, uv or the system Python
have no such folder, so the Test button can't find an interpreter — training
and JoyCaption stay hidden.

**Fix:** in **Settings → Local tools → ai-toolkit**, keep the directory pointing
at the ai-toolkit folder and fill the optional **Python interpreter** field with
the full path to the python that has ai-toolkit's dependencies (e.g.
\`C:\\miniconda3\\envs\\aitk\\python.exe\`), then hit **Test**. ComfyUI Desktop installs
are recognized automatically — no extra step.

## Reddit scan says "rate limiting requests, retry in Ns" (429)

**Why:** out of the box, Reddit scans authenticate with a **public client id
shared by many people** (the gallery-dl one). Reddit's quota — about 1000
requests per 10-minute window — is attached to that id, so other users can
exhaust it before your very first scan of the day. The "retry in Ns" number is
just the time left in the current 10-minute window.

**Fix:** get your own free client ID (one minute, no app secret involved):
**Settings → Scraping & sources** has the field plus a built-in step-by-step
guide. The one trap: on reddit.com/prefs/apps, pick the app type
**installed app** — a *web app* or *script* id comes with a client secret and
Reddit then rejects the anonymous login this app uses (every scan fails
with 401). Takes effect immediately, no restart needed.

## ComfyUI shows as unreachable

Check **Settings → Local tools → ComfyUI API URL** (default
\`http://127.0.0.1:8188\`), confirm ComfyUI is running, and check that a firewall
or a different bind interface is not blocking the connection.

The test has two different failure paths:

- **Connection refused/unreachable** returns quickly and tells you to start
  ComfyUI or correct the URL.
- **ComfyUI answered but its node/model inventory timed out** means the process
  may simply be slow. Large custom-node and model collections make
  \`/object_info\` expensive. LDS waits **45 seconds** by default; raise
  \`comfyui.object_info_timeout_s\` under **Settings → Local tools → ComfyUI**
  and test again.

Do not keep increasing the timeout for a refused connection: a genuinely stopped
ComfyUI is detected in seconds. If the URL test succeeds but generation still
fails, continue with the shared-filesystem case below.

## ComfyUI runs in another container (or WSL, or another machine) and generation fails

**Symptom:** setup goes green — the ComfyUI URL answers, the install directory is
accepted — and then every generation fails.

**Why:** the app talks to ComfyUI through **two** channels, and only one of them is
the network.

1. **The HTTP API** (\`Settings → Local tools → ComfyUI API URL\`). This is what the
   *Test* button and the Setup wizard check.
2. **The filesystem.** Every local engine (Klein, Krea 2 Edit, Klein watermark
   cleaning) hands ComfyUI its source image by **copying the file into ComfyUI's
   \`input/\` folder**, and the result comes back from its \`output/\` folder. There is
   no upload over the API on that path.

A ComfyUI in a separate container, in WSL, or on another host does not share those
folders with the app by default. The URL answers, so everything looks configured —
and then the copy writes into a folder ComfyUI cannot see, or fails outright.

**What it takes to work:**

- \`input/\` and \`output/\` must be visible **to both sides at the same path**. Not
  "an equivalent folder": the app writes \`<input>/edit_source_….png\` and then tells
  ComfyUI to load \`edit_source_….png\` from *its own* input folder — the two must be
  the same directory.
- The app's process must be able to **write** into \`input/\` (a read-only bind mount
  is not enough), and ComfyUI must be able to read it.
- If ComfyUI was started with \`--input-directory\` / \`--output-directory\`, set the
  matching paths in **Settings → Local tools → Advanced: ComfyUI folder overrides**.
  Those fields take the path **as seen by the app**.

**How you will know.** The app checks this itself, at the moment it stages a source
image and again in Settings and the Setup wizard: it asks the running ComfyUI, over
ComfyUI's own \`/view\`, whether it can see the file that was just written. When the
answer is no, the generation is refused with a message naming **the folder the app
used** and **what ComfyUI reported about its own** — quoted from the command line
ComfyUI echoes in \`/system_stats\`, so a second install or a \`--base-directory\` is
named rather than guessed. When ComfyUI also says where it reads (an absolute
\`--input-directory\` in that command line), the Setup wizard's ComfyUI card offers
that folder in one click. This replaced the failure that started it: ComfyUI
answering \`Invalid image file: krea_source_….png\` to its own console while the app
showed a tile that stopped instantly with no error at all (GitHub #64). If ComfyUI
cannot be asked — stopped, behind a proxy that refuses \`HEAD\`, too old — nothing is
refused and staging behaves exactly as before.

With Docker, that means bind-mounting the same host folders into both containers at
identical paths, e.g.:

\`\`\`yaml
# both services
volumes:
  - /srv/comfyui/input:/srv/comfyui/input
  - /srv/comfyui/output:/srv/comfyui/output
\`\`\`

and then pointing the two override fields at \`/srv/comfyui/input\` and
\`/srv/comfyui/output\`. The shipped \`docker-compose.yml\` deliberately does **not**
do this: it runs the app in API-only mode, where ComfyUI is out of scope.

**How you'll know:** the failure now says so. Settings flags an override folder it
cannot write into, the Setup wizard warns while you configure (a warning, never a
blocker — mounting volumes afterwards is fine), and a generation that cannot reach
the folder answers with the folder path and the reason instead of a bare \`500\`.
Those messages are path-redacted, so they are safe to paste in a help thread.

**Everything else keeps working without shared folders**: the API engines (Gemini,
ChatGPT, OpenRouter), scraping, curation, captioning through Ollama, training, and
Hugging Face publishing. Only the ComfyUI-local engines need the filesystem.

*(Reported by nofaceman on Discord.)*

## "Value not in list" on every model, on Linux (fixed)

**Symptom:** on a Linux install, nothing generated at all. ComfyUI's console showed,
for every workflow:

\`\`\`
Failed to validate prompt for output 28:
* UNETLoader 20:
  - Value not in list: unet_name: 'Krea\\krea2_turbo_fp8.safetensors'
    not in ['Krea/krea2_turbo_fp8.safetensors']
\`\`\`

**Why:** ComfyUI builds its model lists with the separator of **its own** host —
backslash on Windows, forward slash on Linux — and validates a model widget by
exact string match. The app spelled those names with a Windows backslash whatever
the platform, and it keeps every model in a subfolder (\`Krea\`, \`klein\`,
\`z image\`, and every LoRA you train), so on Linux the answer was "nothing works"
rather than "one model is missing".

**Fixed:** the app now reads the spelling from the ComfyUI it is actually talking
to and matches it. This also covers the reverse case — the app on Windows driving
a ComfyUI in WSL, Docker or on another machine, which needs forward slashes — so
there is nothing to configure either way.

*(Found and diagnosed by 1Tomber, [GitHub #21](https://github.com/perfectgf/lora-dataset-studio/issues/21).)*

## Klein engine stays greyed out

Klein needs a reachable ComfyUI **and** the Klein model files (~16 GB VRAM
class). **Setup → ComfyUI** offers the download; the license-gated fp8 model
needs a Hugging Face token (Settings → Local tools).

Whatever the cause, the greyed-out engine now **names it**, and the Setup wizard
shows the same sentence — the two screens read one verdict, so they cannot send
you to fix different things. The causes, and what each one means:

| What it says | What it means | What to do |
| --- | --- | --- |
| \`Configure ComfyUI in Settings\` | ComfyUI is not answering | Start it, or fix the API URL |
| \`Klein <file(s)> missing\` | that weight is not on disk | Download it in Setup ▸ Install components |
| \`… is on disk but cannot be loaded\` | the file is there but unreadable | See below |
| \`Your ComfyUI doesn't have <value>\` | the graph pins a widget value your ComfyUI doesn't offer | Install the named node pack, restart ComfyUI |
| \`disabled in Settings (engines)\` | you turned the engine off | Re-enable it in Settings ▸ Engines |

### Where the Klein model may live — you do **not** need \`models/unet/klein/\`

\`models/unet/klein/\` is only where the Setup button *downloads* to. It has never
been a requirement, and nothing needs to be copied, moved or symlinked to satisfy
it. The app resolves the Klein UNET exactly where a running ComfyUI would, in
this order:

| Layout | Example |
| --- | --- |
| a \`klein\`-named sub-folder of \`models/unet\` | \`models/unet/klein/flux-2-klein-9b-kv-fp8.safetensors\` |
| **any** sub-folder whose name contains \`klein\`, any capitalisation or spacing | \`models/unet/Flux2 Klein/…safetensors\` |
| the **top level** of \`models/unet\` | \`models/unet/flux-2-klein-9b-kv-fp8.safetensors\` |
| the same three, under \`models/diffusion_models\` | \`models/diffusion_models/flux2-klein-9b/…safetensors\` |
| any root declared in your \`extra_model_paths.yaml\` | a Stability Matrix / portable / A1111-shared tree |
| a relocated models folder | **Settings → Local tools → ComfyUI models folder** |

**The one limit:** the model has to be *nameable* as Klein — either the **file
name** or its **sub-folder name** must contain \`klein\`. A file called
\`model.safetensors\` sitting loose in \`diffusion_models/\` is invisible; put it in
a \`klein/\` folder (any file name then works) or rename the file. That rule is
what stops another family's checkpoint being wired into the Klein graph.

Every row of that table is covered by
\`backend/tests/test_klein_model_locations_documented.py\`, so it cannot quietly
stop being true.

*(Reported by CyberTod on Reddit, who duplicated the weights and built a symlink
to reclaim the disk space — neither was necessary.)*

### "On disk but cannot be loaded"

A \`.safetensors\` file declares the length of its JSON header in its first eight
bytes. A download that was **cut short or corrupted** leaves a file that is
shorter than it claims — plausible size, right name, right folder, and no loader
can open it. A licence or login page saved as \`.safetensors\` fails the same way.

Setup used to tick these as **✓ Installed** (the file existed) while the
generation page refused the engine. It now shows **⚠ On disk, unreadable** with
the file name and the reason, and **↻ Download again** replaces the bad file
instead of reporting "already present" and doing nothing.

If you placed the file by hand somewhere other than the folder Setup downloads
into, delete it yourself first — the app only replaces files at its own path.

*(Reported by zigzag4794 on Discord.)*

### "Not checked" is not "ready"

With ComfyUI stopped, the checks that need it cannot run, and the app reports no
gap rather than inventing one. Your files being on disk is therefore **not** a
clean bill of health, and Setup says so instead of showing a tick it did not
earn. Start ComfyUI and re-check.

## "Upscale & improve" makes my anime look realistic

**Why:** the improve pass sends Klein a fixed instruction, and the shipped one is
a *photographic* recipe — \`add detailed texture, add sharp details, add candid
shot, add soft focus effect\`. It is applied to every dataset, drawn ones
included, so on anime or illustration it does exactly what it says: it adds skin
texture and photo micro-detail your line art never had.

**Fix — two levers, both in Settings → Image engines → *Identity, Klein & Krea 2
prompts (advanced)*:**

1. **Rewrite the instruction.** The *Klein upscale & improve prompt* box holds
   the text in use; edit it to something that suits drawn art (e.g. "keep the
   drawn anime rendering, clean line art, flat cel shading, sharpen the lines").
   Clearing the box restores the shipped default — nothing is frozen.
2. **Or send no instruction at all.** The checkbox above the box — *Apply an
   improvement prompt on "Klein upscale & improve"* — turns it off, and the pass
   becomes a pure upscale.

Separately, **Settings → Image engines → *Upscale & improve — strength*** decides
how far the pass may move the image at all (output megapixels, sampler steps, the
enhancement LoRA, the consistency LoRA). Lower the *Enhancement LoRA* and raise
the *Consistency LoRA* if you want the pass to change less.

Both are now quoted and linked **from the ✨ Upscale & improve button itself** —
in the lightbox and in the grid's bulk toolbar — so the instruction currently in
force is readable where the action happens, and anime datasets get an explicit
warning there.

*(Reported by Qeeyana on Reddit.)*

## Port 5000 conflict on macOS

macOS reserves port 5000 for AirPlay Receiver. Change the port in
**Settings → Server & access** (e.g. 5050) and restart.

## Garbled characters in the Windows console

Cosmetic only — some UTF-8 text renders wrong on the legacy console codepage.
The app itself is unaffected.

## \`npm install\` fails with \`Cannot find module @rollup/rollup-<platform>-...\`

Only relevant if you rebuild the frontend yourself (the repo ships \`dist/\`
prebuilt). It's a known npm bug: delete \`frontend/node_modules\` +
\`frontend/package-lock.json\` and run \`npm install\` again on this machine.

## A cloud run seems stuck

Open the **Cloud** tab: every run shows its live phase, and the stall watchdog
(Settings → Training → stall timeout) rescues logs and kills the pod if no step
progress happens for too long. Orphaned pods are also destroyed automatically
at every app start — you never pay for a forgotten GPU.

**Decide without opening the [vast.ai](https://cloud.vast.ai/?ref_id=683073) console.** The longest phase of a fresh
run is the pod pulling its base weights — 26 GB for Krea 2, half an hour on a
healthy host. The card shows that download as it happens: how much has landed,
of how much, at what speed and its ETA. Two readings tell you everything:

- the figures move → it is working, however slow it looks. A crawling download
  is still a download; the *no training step in 45 min* watchdog will end it if
  it never gets there;
- the figures do not move, and the card says **Silent run — no progress
  reported for N min** → the pod is frozen, not slow. Stop it, or let the
  freeze watchdog terminate it at the configured limit. That counter is
  measured on the pod and survives restarting the app, so leaving and coming
  back does not reset the evidence.
`;export{e as default};
