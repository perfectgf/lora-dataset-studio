// Historical IDs and dates are preserved when ownership moves out of the core feed.
export const MIGRATED_NEWS = [
  {
    id: '2026-09-24-camera-standalone',
    date: '2026-09-24',
    title: 'Camera angles has its own workspace',
    blurb: 'Open Camera angles from the navigation, import an image and choose your viewpoints directly. Originals and generated views stay in the workspace, with previews and downloads for each result.',
    to: '/camera-angles',
  },
{
    id: '2026-08-28-camera-model-choice',
    date: '2026-08-28',
    title: 'Camera angles can run on your own Qwen build',
    blurb:
      '📷 The camera-angles panel now has a Model row: pick any '
      + 'Qwen-Image-Edit build on your disk — a finetune, an NSFW merge — '
      + 'and every camera run uses it, on the Gallery and in datasets alike. '
      + 'Empty keeps the installed 2511 default. The angle grammar comes from '
      + 'the LoRA, so a different build changes the look, not the camera.',
    to: '/gallery',
  },
{
    id: '2026-08-26-camera-angles-dataset',
    date: '2026-08-26',
    title: 'Camera angles inside a dataset — with the angle already captioned',
    blurb:
      'Open any kept image of a dataset and press 📷 Camera angles: the views '
      + 'arrive as pending candidates in the ordinary keep/reject cycle, and '
      + 'each one is born knowing its own caption fragment — "seen from '
      + 'behind, low camera angle" — which the captioner then completes and '
      + 're-injects on every later pass. That phrase is the point: an angle '
      + 'left undescribed binds to the trigger word, and the angle is the one '
      + 'fact a vision model cannot reliably see while the app knows it '
      + 'exactly, because you asked for it. Imports and ✨ results are valid '
      + 'sources; camera views are not re-shot from camera views. The Bank '
      + 'deliberately does not carry the button — it is the reservoir of real '
      + 'photos; promote to a dataset first, and the views are born as '
      + 'candidates, never filed as real.',
    to: '/datasets?section=images',
    image: new URL('./assets/news/camera-angles-picker.png', import.meta.url).href,
  },
{
    id: '2026-08-26-camera-setup-card',
    date: '2026-08-26',
    title: 'Install 📷 Camera angles from Setup, before the first click',
    blurb:
      'The camera weights used to install only when you pressed 📷 with them '
      + 'missing. Setup now shows the lane properly: a one-click install card '
      + 'on the Install screen (~21.6 GB, shared parts skipped when another '
      + 'engine already brought them), a row per weight in the repair menu so '
      + 'a broken download can be fixed alone, and Camera angles is counted on '
      + 'the readiness screen — a machine without it reads "not ready, here is '
      + 'the install", never a shorter list that certifies completeness by '
      + 'leaving it out.',
    to: '/plugins/camera_angles/settings',
  },
{
    id: '2026-08-26-camera-angles',
    date: '2026-08-26',
    title: 'Walk around your subject: re-shoot any picture from another camera position',
    blurb:
      'Open a picture in the Gallery and press 📷 Camera angles: pick where the '
      + 'camera stands on the dial, how high it is and how close, and the app '
      + 'renders that scene from there — the subject stays put and the '
      + 'background moves with the camera, so what was behind them comes into '
      + 'view. This is not the "profile view" shot the catalog already had: '
      + 'that one turns the person and leaves the room where it was. Eight '
      + 'sides of a subject at eye level is one gesture and about two minutes. '
      + 'First use downloads the weights from Setup ▸ ComfyUI. The Gallery '
      + 'carries a Beta chip while this settles: distance is a hint the model '
      + 'mostly honours, and whatever the original photo never showed is '
      + 'plausible rather than real.',
    to: '/gallery',
  },
]
