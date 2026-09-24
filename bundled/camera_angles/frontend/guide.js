// Markdown ships with this product version. Lines stay editable without a Markdown loader.
export const GUIDE = {
  chapters: [],
  sections: [
    {
      chapter: "settings-reference",
      anchor: "camera-angles-local",
      markdown: [
        "## Camera angles (local)",
        "",
        "📷 **Camera angles** re-photographs an existing picture from another camera position: open it in the 🖼 Gallery, press **Camera angles**, pick where the camera stands on the dial, how high it is and how close. The subject stays where it is and **the background moves with the camera**, so what was behind them comes into view.",
        "",
        "**This is not the shot catalog's \"profile view\".** That one asks an edit model for another angle and the model answers by turning the *person* — measured on this app's own Klein lane, the room behind never moved, whatever the wording. Moving the viewpoint needs a model trained on real viewpoint changes, which is why this lane runs on **Qwen-Image-Edit 2511** with fal.ai's Multiple-Angles LoRA (trained on gaussian-splatting renders, Apache-2.0) rather than on the Klein weights you already have.",
        "",
        "**What it costs.** The base model is **~20.5 GB**, plus 295 MB for the angles LoRA and 850 MB for the optional 4-step speed LoRA. The text encoder and VAE are shared with lanes you may already have installed. Pressing 📷 with the weights absent starts those downloads and tells you so — nothing is fetched behind your back. Once the model is resident a view takes **12–16 s**; the first one of a session also pays for loading the model (~1 min).",
        "",
        "**The limits, stated up front:**",
        "",
        "- **Distance is approximate.** Close-up / medium / wide are hints the model mostly honours; several poses asked at *medium* come back tighter than the source.",
        "- **Off-camera detail is invented.** The part of the scene the original photo never showed is plausible, not real. Fine for a character dataset, wrong for anything that has to be a faithful record of a place.",
        "- **A camera view cannot be re-shot from another angle.** The second pass would re-invent what the first already invented and present it as the original scene, so the button is refused there and says why.",
        "- **Up to the whole vocabulary (96 views) in one run.** The count under the button is the product of the axes you ticked; it says what the run will cost before you spend it, turns amber past about five minutes, and every queued view can be dropped one at a time from the system queue.",
        "",
        "**Model files (optional).** Open **Plugins → Camera angles → Settings** to edit these preferences. Same contract as the shared model pins — empty means auto-detect (canonical download filename first, then a narrow token scan), a value pins one file.",
        "",
        "- **Diffusion model** → `camera.unet`. Default **empty**. Auto-detection prefers a **2511** build: the LoRA was trained on that generation and a 2509 build loads happily and quietly under-performs. This key also has a picker in the app — the **Model row of the 📷 panel** lists every qwen build on your disk (files in qwen-named folders under `diffusion_models`, plus root-level files with `qwen` in the name) and saves this same key, app-wide: pick a finetune or an NSFW merge there and every camera run uses it, on both surfaces, until you clear it back to the default. The angle grammar comes from the LoRA, so a different build changes the look, not the camera. A pinned file that later disappears is flagged in the row and the run falls back to auto-detection rather than refusing.",
        "- **Text encoder** → `camera.text_encoder`. Default **empty**. ⚠️ `models/text_encoders` can hold **three different Qwen encoders** — Klein's `qwen_3_8b`, Z-Image/Krea's `qwen3vl_4b`, and this lane's `qwen_2.5_vl_7b`. They are not interchangeable and a wrong one fails at sampling time with a shape error, so auto-detection is deliberately narrow and pinning is how you rescue a renamed file.",
        "- **VAE** → `camera.vae`. Default **empty**. The same file the **Krea 2 Edit** lane installs — one copy, one Setup button; this lane never downloads a second.",
        "- **Angles LoRA** → `camera.angles_lora`. Default **empty**. **Required**: without it the base model still edits, it just answers the camera vocabulary the way any edit model does — by turning the subject. A camera view with no camera in it would look like a success, so the lane refuses to run rather than render one.",
        "- **Speed LoRA** → `camera.speed_lora`. Default **empty**, and genuinely optional: absent, the graph raises its own step count from 4 to 20 and renders correctly, roughly five times slower. ⚠️ When the **Model row picks a build whose name says it is already distilled** (`rapid`, `lightning`, `turbo`, `aio`, `hyper`, `lcm`, `4step`…), runs **skip this LoRA and keep 4 steps** — chaining a speed LoRA onto an already-few-step merge is distillation applied twice, and it renders confetti-like patches over skin and tiles while every job reports success (measured, same seed, same pose). The picker's note says so when it applies, and **pinning a file here overrides the skip** — a pin is you saying you know better than the filename.",
      ].join("\n") + "\n",
    },
    {
      chapter: "using-the-app",
      anchor: "re-shoot-an-image-with-camera-angles",
      markdown: [
        "## Re-shoot an image with Camera angles",
        "",
        "Install Camera angles and configure ComfyUI, then install its weights in Plugins → Camera angles → Settings → Preparation.",
        "Open **Camera angles** from the navigation and import or drop a PNG, JPEG or WebP image (up to 32 MB and 40 megapixels). The source appears beside the camera controls; no dataset is needed.",
        "Shoot the selected views and watch results appear below. Open a result at full size or download it. Imported images and their results remain available when you return; choose a thumbnail to revisit an earlier source. Manage or cancel unfinished views in the system queue.",
        "You can also open an original image in Gallery or a kept dataset image and choose **Camera angles**.",
        "Choose camera sides, heights and distances; the displayed count is their combination.",
        "The results are new images beside their source, not edits to the original. Dataset views arrive as pending candidates with the angle in their caption.",
        "To work from a bank photo, first promote it into a dataset. Camera outputs cannot be used to recursively create more camera views.",
        "The Model row accepts compatible Qwen-Image-Edit builds; their appearance can differ while the camera controls stay the same.",
      ].join("\n") + "\n",
    },
  ],
}

const HELP_SECTIONS = {
  "action-camera-angles": [
    "using-the-app",
    "re-shoot-an-image-with-camera-angles"
  ],
  "action-dataset-camera-angles": [
    "using-the-app",
    "re-shoot-an-image-with-camera-angles"
  ],
  "action-camera-model": [
    "settings-reference",
    "camera-angles-local"
  ]
}
export function guideHelp(topic) {
  if (topic.app?.route?.startsWith('/settings/') || topic.app?.route?.startsWith('/setup')) {
    const { route, ...app } = topic.app
    topic = { ...topic, app: { ...app, route: '/plugins/camera_angles/settings',
      ...(route.startsWith('/settings/') ? { legacyRoute: route } : {}) } }
  }
  const target = HELP_SECTIONS[topic.id]
  return { ...topic, ...(target ? { guide: { chapter: target[0], anchor: target[1] } } : {}) }
}
