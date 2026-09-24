import { MIGRATED_NEWS } from './migratedNews.js'
import { GUIDE, guideHelp } from './guide.js'
// 📷 Camera angles — the frontend descriptor of the bundled `camera_angles`
// plugin. Plain JS on purpose: node --test imports it (the parity and help
// contracts), Vite globs it at build time (src/plugins/bundled.js). The React
// panels are lazy imports, so each is its own chunk and only loads where a
// surface mounts it.
import { action, setupStep } from '@lds/plugin-sdk/help'
import { cameraCatalogItems, cameraSetupRows } from './lib/cameraInstall.js'

export default {
  guide: GUIDE,
  id: 'camera_angles',
  nav: [{ to: '/camera-angles', label: '📷 Camera angles' }],
  routes: [{ path: '/camera-angles', page: () => import('./panels/CameraStudio.jsx') }],
  slots: {
    'settings.group': [{ id: 'camera-models', section: 'engines',
      title: 'Camera model preferences', blurb: 'Choose installed model files or keep automatic selection.',
      keywords: ['camera', 'qwen', 'model', 'vae', 'text encoder', 'lora'],
      panel: () => import('./panels/CameraSettings.jsx') }],
    // The verb, on the two surfaces whose rows the lane can re-shoot: a
    // dataset image and a library (Gallery / Studio / Canvas) image. The Bank
    // is skipped on paper below, never silently.
    'lightbox.action': [
      {
        id: 'camera',
        panels: {
          dataset: () => import('./panels/DatasetCameraAction.jsx'),
          gallery: () => import('./panels/GalleryCameraAction.jsx'),
        },
      },
    ],
    // The one-click install card on the Setup install screen.
    'setup.card': [{ id: 'camera', panel: () => import('./panels/CameraInstallCard.jsx') }],
    // The counted capability row and the install-menu rows.
    'setup.step': [{ id: 'camera', rows: cameraSetupRows, catalog: cameraCatalogItems }],
  },
  hosts: [],
  help: ([
    /* 📷 Its own topic rather than a line under the Gallery's, because the
       question people arrive with is not "what can the Gallery do" — it is
       either "how do I get the back of this character" or, more often, "why did
       it turn the person instead of moving the camera". Both vocabularies are in
       the keywords, including the shot-catalog words, so someone who tried
       "profile view" first lands here. */
    action('action-camera-angles', 'Re-shoot a picture from another camera position',
      ['camera', 'camera angle', 'angles', 'multi-angle', 'multiple angles', 'around',
        'orbit', 'rotate camera', 'move camera', 'viewpoint', 'point of view',
        'other side', 'back of', 'behind', 'from behind', 'back view', 'profile',
        'side view', 'three-quarter', 'low angle', 'high angle', 'from below',
        'from above', 'turntable', 'coverage', 'sks', 'qwen', 'why did it turn the person',
        'background did not move', 'same scene different angle'],
      '/camera-angles', 'using-the-app', 're-shoot-an-image-with-camera-angles',
      { trigger: 'camera-angles-picker',
        text: 'Pick axes, not pictures: the sides you tick times the heights times '
          + 'the distances is the run — the count under the button is what it will cost.' }),
    /* The picker's Model row — a SETTING (app-wide camera.unet), so it owes a
       topic, and the words it is asked with ("can I run this on a finetune / an
       NSFW build") appear in none of the other camera topics. */
    action('action-camera-model', 'Run camera angles on another Qwen-Image-Edit build',
      ['camera model', 'qwen model', 'qwen build', 'qwen edit', '2511', 'swap model',
        'another model', 'different model', 'change model', 'custom model', 'finetune',
        'fine-tune', 'merge', 'aio', 'nsfw', 'uncensored', 'rapid', 'which model',
        'model not found', 'camera.unet', 'model row', 'camera picker model'],
      '/plugins/camera_angles/settings', 'settings-reference', 'image-engines'),
    /* In a dataset the same verb answers a different question — "how do I get
       training coverage of the back of my character" — and adds the captioning
       angle, so it earns its own topic with the dataset vocabulary. */
    action('action-dataset-camera-angles', 'Cover a dataset subject from more angles',
      ['camera angles dataset', 'multi-angle dataset', 'coverage', 'training coverage',
        'back of my character', 'more angles', 'angle caption', 'seen from behind',
        'caption angle', 'pending candidates', 'camera view candidate',
        'why is the caption pre-filled', 'bank camera', 'why not in the bank',
        'promote then camera', 're-shoot dataset image'],
      '/datasets?section=images', 'using-the-app',
      'the-character-walkthrough-reference-photo-trained-lora'),
    setupStep('setup-camera-install', 'install', 'Install 📷 Camera angles',
      ['camera angles', 'camera', 'angles', 'multi-angle', 'multiple angles', 're-shoot',
        'reshoot', 'other side', 'back view', 'viewpoint', 'qwen image edit', 'qwen 2511',
        'sks', 'camera lora', 'lightning', 'speed lora', 'install camera',
        'camera weights', 'camera model missing', '20 gb', 'gallery camera button',
        'models/diffusion_models/qwen', 'models/loras/qwen']),
  ]).map(guideHelp),
  whatsNew: MIGRATED_NEWS,
  paritySkip: [
    { slot: 'lightbox.action', surface: 'bank',
      reason: 'The Bank reviews photos it did not generate; a camera view is a generated '
        + 'candidate and lands in a dataset or the Gallery, never in a bank folder. '
        + 'Promote the picture to a dataset, then re-shoot it there.' },
  ],
}
