/* ✨ Neural render (DLSS 5) — its own section rather than keywords bolted onto
   the video lane's topics: the two verbs live on two pages (a video dataset's
   clips, the Test Studio's clip history) and the install lives in Setup, and
   someone whose button refuses is not helped by shot-detection troubleshooting.
   ORDER MATTERS across sections (helpRegistry.js concatenates them). */
import { action, setupStep } from '../topicBuilders.js';

export const NEURAL_RENDER_TOPICS = [
  // The verb, on both surfaces. `app.route` is the datasets library: the
  // workspace needs an id nobody can put in a static registry, and the library
  // always leads there in one click.
  action('video-neural-render', 'Neural render a video clip (DLSS 5)',
    ['neural render', 'neural rendering', 'dlss 5', 'dlss5', 'dlss', 'nvidia dlss',
     'neural', 'photoreal', 'more detail', 'skin detail', 'hair detail', 'fabric',
     'relight', 'tone', 'structure', 'automask', 'temporal', 'still mode',
     '704', 'width floor', 'clip too narrow', 'restore original', 'restore originals',
     'originals kept', 'backup', 'undo neural render', 'video enhance', 'enhance video',
     'enhance a clip', 'improve a clip', 'flat art', 'anime keep tones', 'greys the whites',
     'washed out', 'nvngx_dlssnr.dll', 'model not found', 'windows only', 'nvidia only',
     'not available in docker', 'render as a new clip', 'compare', 'compare with original',
     'side by side', 'before after', 'before and after', 'original vs render', 'swap sides',
     'in step', 'synchronised playback', 'synchronized playback', 'strength', 'detail strength',
     'not striking', 'too subtle', 'more effect', 'stronger', 'passes', 'two passes', 'render at 2x',
     '2x', 'zoom 1:1', 'one to one', 'real size', 'pixels'],
    '/datasets', 'using-the-app', 'neural-render-for-video-clips'),
  // The install, in Setup: the bridge is a button, the model is a folder.
  setupStep('setup-dlss5-install', 'install', 'Install DLSS 5 Neural Rendering',
    ['install dlss 5', 'dlss 5 bridge', 'neural rendering bridge', 'dlss5nr_bridge',
     'nvngx_dlssnr.dll', 'where to put the model', 'model folder', 'runtime folder',
     'dlss5nr', 'bridge missing', 'model missing', 'forwarder', '165 mb', 'sha256',
     'pinned release', 'ComfyUI-DLSS5-NR', 'nvidia driver', 'ngx', 'optical flow',
     'nvofapi64', 'windows only', 'no linux', 'no docker', 'rtx 50', 'rtx 40']),
];
