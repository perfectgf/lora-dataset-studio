# Known limitations

[← Documentation index](../README.md) · [Settings reference](settings-reference.md) · [Troubleshooting](troubleshooting.md)

These are current boundaries, not setup failures.

| Area | Current boundary | Practical path |
|---|---|---|
| **Test Studio preparation** | All seven image training families have generation workflows; each needs matching local models and native ComfyUI support | Prepare missing FLUX.1, Anima or Qwen-Image 2.1 files in Settings → Image engines → Trained image models; update ComfyUI when the node check requests it |
| **Krea 2 img2img Studio mode** | `backend/workflows/krea2_turbo_img2img.json` exists but is not wired to a separate Studio mode | Krea 2 Edit remains available for dataset/reference generation; Studio uses its reachable text-to-image path |
| **Canvas comparison generation** | A single launch requires checkpoints from one model family | Run separate same-family comparisons; cross-family workflows do not share one base graph |
| **Dual captions** | Local training only; Krea 2 and Anima cache text embeddings and use the long caption alone | Keep the long caption complete for every family |
| **Concept datasets on SDXL** | Concept captions are only produced as **prose**; there is no booru-tag concept captioner, and the prose/booru selector is hidden on concept datasets. A concept dataset trained on a booru-native SDXL checkpoint therefore always trips the caption-style check | Train the concept on a prose family (Z-Image, Krea 2, FLUX.1, FLUX.2 Klein, Anima), or force the launch and accept the quality loss a booru-native base takes from prose. Character and Style datasets are unaffected — both reach booru captions normally |
| **Cloud training** | Support is family- and variant-specific, and full-state continuation is local-only | Trust the launch UI's live compatibility reason before renting a pod |
| **GPU Docker** | Ollama and local ai-toolkit training are not bundled; watermark inpainting is not currently available in this lane | Connect external Ollama/ai-toolkit, use cloud training, and use model-free watermark crop where suitable |
| **Remote access** | The app has no user accounts and ComfyUI's published port is not protected by the LDS access token | Keep loopback defaults or use tokens, firewall/VPN and an authenticated reverse proxy as described in [Security](../../SECURITY.md) |
| **Browser-local preferences** | Some convenience choices, such as the last generator, are remembered per browser | Explicitly select a configured engine after moving to a new browser/profile |

ComfyUI-dependent paths are covered extensively against a mocked API, but not every model/custom-node combination has been exercised on live third-party installations. A failed preflight should name the missing asset; use **Settings → Local tools → Test** and attach the diagnostic report when a supported layout is not detected.

Provider policies, moderation and service availability are outside this project's control. See the direct [Gemini](workflow.md#generate-from-references), [ChatGPT subscription](../../bundled/api_engines/README.md), [OpenRouter/image-engine](settings-reference.md#image-engines) and [Pexels](workflow.md#the-built-in-web-scraper) notes before depending on those lanes.
