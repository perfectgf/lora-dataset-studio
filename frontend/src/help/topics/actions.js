/* One section of the help registry, moved verbatim (2026-08-24 split).
   ORDER MATTERS inside and across sections: helpRegistry.js concatenates
   the six section arrays in a fixed order, and for a given (chapter,
   anchor) the FIRST topic owns the "Open this screen →" button. */
import { action, setting } from '../topicBuilders.js';

export const ACTION_TOPICS = [
  // ---- Action topics (kind 'action') -------------------------------------
  action('action-edit-reference', 'Edit the reference photo',
    ['edit', 'reference', 'prompt', 'chatgpt', 'nano banana', 'openrouter', 'klein',
     'krea', 'krea 2 edit', 'local', 'free', 'comfyui', 'background',
     'glasses', 'retouch', 'before', 'after', 'keep', 'discard'],
    '/datasets?section=add', 'using-the-app',
    // No one-time tip on purpose: the modal states the cost, the reference
    // support and the availability gap PERMANENTLY, next to the control they
    // apply to. A tip that repeats what is already on screen is noise.
    'the-character-walkthrough-reference-photo-trained-lora'),
  action('action-watermark-clean', 'Find & clean watermarks',
    ['watermark', 'clean', 'find', 'lama', 'klein', 'crop', 'remove'],
    '/datasets?section=curation&panel=watermarks', 'settings-reference', 'captioning-quality',
    { trigger: 'watermark-batch-clean',
      text: 'Clean has two engines — LaMa (fast) and Klein (quality) — and auto-crop can be turned off.' }),
  action('action-dataset-reject-flagged', 'Reject every flagged image at once',
    ['reject all', 'reject flagged', 'bulk reject', 'watermark', 'flagged', 'shortcut',
     'undo reject', 'bring back', 'rejected', 'false positive', 'stop watermark scan',
     'cancel scan', 'rescan dismissed'],
    '/datasets?section=curation&panel=reject-flagged', 'using-the-app',
    'reject-every-flagged-image-at-once'),
  setting('setting-watermark-backend', 'captioning', 'wmdet-backend',
    'Which engine finds watermarks',
    ['watermark backend', 'watermark detection', 'detector', 'vision model', 'ollama',
     'siglip', 'auto', 'which detector', 'why is this flagged', 'watermark source',
     'not installed', 'fallback', 'extra']),
  action('action-bank-watermark-clean', 'Clean a bank\'s watermarks (2 levels)',
    ['watermark', 'bank', 'clean', 'crop', 'auto-crop', 'inpaint', 'lama', 'klein',
     'remove watermark', 'logo', 'url', 'undo cleaning', 'before after', 'original',
     // Asked in the panel's own words: "who decided this was a watermark?"
     'watermark source', 'detector', 'vision model', 'why is this flagged',
     'watermark score', 'sensitivity', 'threshold', 'false positive'],
    '/bank', 'using-the-app', 'clean-the-watermarks-a-bank-found'),
  action('action-find-text', 'Erase burned-in text (bubbles, subtitles, captions)',
    ['find text', 'text', 'ocr', 'burned-in', 'burned in', 'speech bubble',
     'bubble', 'subtitle', 'subtitles', 'caption', 'sound effect', 'sfx',
     'manga', 'comic', 'webtoon', 'lettering', 'erase text', 'remove text',
     'clean text', 'rapidocr', 'korean', 'japanese', 'meme'],
    '/bank', 'using-the-app', 'erase-burned-in-text-bubbles-subtitles-captions'),
  /* ✂ and ✨ share one guide section but get a topic EACH, for the same reason
     🎨 Medium and ⤢ Angle do below: they are two different gestures asked about
     in two different vocabularies ("how do I crop in the bank?" vs "can I
     upscale before promoting?"), and one topic would only ever be found by half
     the people looking. ↩ Revert gets its own because it is what people search
     for in a hurry, after the edit they regret. */
  action('action-image-repair', 'Repaint one detail without regenerating the image',
    ['repair', 'repaint', 'inpaint', 'inpainting', 'fix a detail', 'small fix',
     'remove jewelry', 'necklace', 'earrings', 'skin', 'blemish', 'imperfection',
     'custom prompt', 'free prompt', 'mask', 'zone', 'without regenerating',
     'keep the rest', 'byte identical', 'klein', 'dataset',
     'smeared', 'smeary', 'blurry repair', 'lanpaint', 'remove object'],
    '/datasets', 'using-the-app', 'repaint-one-detail-without-regenerating-the-image'),
  action('action-bank-crop', 'Crop an image inside a bank',
    ['crop', 'cropping', 'reframe', 'reframing', 'framing', 'cut', 'trim',
     'zoom in', 'recadrer', 'bank', 'review', 'lightbox', 'box', 'ratio',
     'aspect', 'square', 'resample', 'resolution', 'no resize', 'without dataset',
     'before promoting', 'edit image', 'C key'],
    '/bank', 'using-the-app', 'crop-and-upscale-inside-a-bank'),
  action('action-bank-improve', 'Upscale & improve images inside a bank',
    ['upscale', 'upscaling', 'improve', 'enhance', 'sharpen', 'super resolution',
     'super-resolution', 'klein', 'seedvr2', 'seedvr', 'low resolution', 'small',
     'blurry', 'soft', 'quality', 'gpu', 'comfyui', 'bank', 'batch', 'pass',
     'before promoting', 'without dataset', 'stop'],
    '/bank', 'using-the-app', 'crop-and-upscale-inside-a-bank'),
  action('action-bank-revert-edits', 'Undo a crop or an upscale made in a bank',
    ['revert', 'undo crop', 'undo upscale', 'undo improve', 'restore', 'original',
     'back to original', 'cancel edit', 'remove edit', 'edits', 'mistake',
     'wrong crop', 'redo', 'improve again', 'run it again', 'bank'],
    '/bank', 'using-the-app', 'crop-and-upscale-inside-a-bank'),
  // 🎨 Medium and ⤢ Angle share one guide section but get a topic EACH: they are
  // two separate chip rows, asked about in two very different words ("is this
  // anime?" vs "where are my profile shots?"), and one topic would only ever be
  // found by half the people looking for it.
  action('action-bank-medium', 'Tell photos, anime, 3D renders and illustrations apart',
    ['medium', 'mediums', 'photo', 'photograph', 'photographic', 'anime', 'manga',
     'cartoon', 'drawing', 'drawn', 'illustration', 'painting', 'painted', 'art',
     'artwork', '3d', '3d render', 'render', 'rendered', 'cgi', 'game', 'unsure',
     'classify medium', 'what is this made of', 'is this anime', 'is this a photo',
     'is this drawn', 'real photo', 'style', 'not a photo', 'mixed dump',
     'cosplay', 'zero-shot', 'clip', 'medium confidence', 'separate anime',
     'anime dataset', 'photo dataset', 'bank'],
    '/bank', 'using-the-app', 'sort-a-bank-by-medium-and-by-head-angle'),
  action('action-bank-angle', 'Find frontal, three-quarter and profile shots',
    ['angle', 'angles', 'head angle', 'yaw', 'pose', 'head pose', 'frontal',
     'front', 'facing camera', 'three quarter', 'three-quarter', '3/4', 'profile',
     'side view', 'sideways', 'turned', 'turned away', 'looking away',
     'from behind', 'back view', 'behind', 'no face', 'camera angle',
     'measure angles', 'missing angles', 'backfill', 'variety of angles',
     'angle coverage', 'too many frontal', 'need profiles', 'bank'],
    '/bank', 'using-the-app', 'sort-a-bank-by-medium-and-by-head-angle'),
  action('action-bank-relocate', 'Move a bank\'s folder to another disk',
    ['bank', 'move', 'moved', 'relocate', 'repoint', 'folder', 'new location', 'another disk',
     'other drive', 'external drive', 'unplugged', 'disconnected', 'renamed', 'drive letter',
     'path changed', 'source folder', 'unavailable', 'missing images', 'keep analysis',
     'keep decisions', 'lost my scores', 'rescan'],
    '/bank', 'using-the-app', 'move-a-bank-folder-to-another-disk'),
  action('action-scoring-python', 'Make ✨ Score use a GPU Python you already have',
    ['score', 'scoring', 'gpu', 'cuda', 'cpu', 'slow', 'hours', 'torch', 'pytorch',
     'open_clip', 'openclip', 'transformers', 'timm', 'interpreter', 'python',
     'ai-toolkit', 'comfyui', 'venv', 'environment', 'faster', 'speed up',
     'aesthetic', 'nsfw', 'borrow', 'reuse'],
    '/bank', 'using-the-app', 'make-score-use-a-gpu-python-you-already-have'),
  action('action-semantic-python', 'Build the SigLIP 2 index on a GPU Python you already have',
    ['siglip', 'siglip2', 'siglip 2', 'semantic', 'semantic index', 'index', 'embedding',
     'embeddings', 'gpu', 'cuda', 'cpu', 'slow', 'hours', 'torch', 'pytorch',
     'transformers', 'siglip2model', 'too old', 'interpreter', 'python',
     'ai-toolkit', 'comfyui', 'venv', 'environment', 'faster', 'speed up',
     'borrow', 'reuse', 'device', 'bank'],
    '/bank', 'using-the-app', 'build-the-siglip-2-index-on-a-gpu-python-you-already-have'),
  action('action-watermark-python', 'Run the watermark detector on a GPU Python you already have',
    ['watermark', 'watermarks', 'detector', 'find', 'scan', 'gpu', 'cuda', 'cpu',
     'slow', 'hours', 'torch', 'pytorch', 'transformers', 'grounding dino',
     'siglip', 'interpreter', 'python', 'ai-toolkit', 'comfyui', 'venv',
     'environment', 'faster', 'speed up', 'borrow', 'reuse', 'device', 'bank'],
    '/bank', 'using-the-app', 'run-the-watermark-detector-on-a-gpu-python-you-already-have'),
  action('action-score-resume', 'Stopping ✨ Score, and what a relaunch costs',
    ['score', 'scoring', 'stop', 'stopped', 'cancel', 'resume', 'relaunch', 'restart',
     'rerun', 're-run', 'again', 'cache', 'cached', 'skip', 'already scored',
     'lost my scores', 'starts over', 'from scratch', 'rescore', 'rescore all',
     'recompute', 'redo', 'style groups', 'style cluster', 'aesthetic', 'nsfw',
     'partial', 'interrupted', 'bank', 'triage'],
    '/bank', 'using-the-app', 'stopping-score-and-what-a-relaunch-costs'),
  action('action-grid-status-filter', 'Filter the grid by decision',
    ['filter', 'decision', 'undecided', 'awaiting', 'pending', 'kept', 'keep', 'rejected',
     'reject', 'improve', 'candidates', 'klein', 'isolate', 'triage', 'select all', 'grid'],
    '/datasets?section=images', 'dataset-guide', '2-how-many-images-and-which-ones'),
  // ✎ Edit this instruction here — the improve prompt, editable from the note
  // under the ✨ button instead of only from Settings. Its own topic because the
  // question it answers is "how do I change this sentence WITHOUT leaving my
  // images", and because the panel has a property the Settings card does not:
  // it writes the app-wide value from a per-dataset-looking screen, which is the
  // one thing a user must be told before they use it.
  action('action-edit-improve-instruction', 'Edit the improve instruction without leaving the images',
    ['improve', 'upscale', 'instruction', 'prompt', 'edit', 'edit here', 'inline', 'in place',
     'change the prompt', 'turn off', 'disable', 'toggle', 'no prompt', 'upscale only',
     'klein', 'anime', 'drawn', 'realistic', 'texture', 'skin', 'detail', 'lightbox',
     'reset to default', 'built-in default', 'global', 'app-wide', 'every dataset',
     'applies everywhere', 'same as settings',
     // The same note now also picks the LoRA preset the pass chains
     // (klein.improve_lora_preset) — one panel, all three improve knobs.
     'lora preset', 'improve preset', 'chain lora', 'preset', 'extra loras'],
    '/datasets?section=images', 'settings-reference', 'image-engines'),
  // ↩ On a ✨ result in the generated-image viewer (Gallery, checkpoint
  // galleries, Canvas): make future improves run the way THIS one did.
  action('action-use-improve-settings', 'Reuse the settings a ✨ result was made with',
    ['use these improve settings', 'reuse settings', 'restore settings', 'same settings',
     'copy settings', 'apply settings', 'like this result', 'run again like this',
     'improve settings back', 'previous settings', 'settings from image',
     'which settings made this', 'recorded settings', 'improve profile'],
    '/gallery', 'using-the-app', 'the-gallery-every-image-you-generated'),
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
    '/gallery', 'using-the-app', 'the-gallery-every-image-you-generated',
    { trigger: 'camera-angles-picker',
      text: 'Pick axes, not pictures: the sides you tick times the heights times '
        + 'the distances is the run — the count under the button is what it will cost.' }),
  /* 📷 In a dataset the same verb answers a different question — "how do I get
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
  action('action-reimprove-tile', 'Re-run Upscale & improve after changing its settings',
    ['improve', 'upscale', 'reimprove', 're-improve', 'rerun', 're-run', 'redo', 'again',
     'regenerate', 'no regenerate button', 'missing button', 'klein improve', 'candidate',
     'steps', 'megapixels', 'strength', 'try again', 'source image', 'parent'],
    '/datasets?section=images', 'settings-reference', 'image-engines'),
  // ⟨ / ⟩ in the dataset lightbox. The buttons are visible, but the ← → keys,
  // the fact that the walk follows the FILTERS, and the deliberate absence of a
  // wrap-around are all invisible — which is what earns this its own topic.
  action('action-inspect-next-previous', 'Move through a dataset without closing the image',
    ['next image', 'previous image', 'next', 'previous', 'navigate', 'navigation',
     'arrows', 'arrow keys', 'left right', 'keyboard', 'shortcut', 'shortcuts',
     'hotkey', 'browse', 'flip through', 'go through', 'one by one', 'review',
     'lightbox', 'full screen', 'fullscreen', 'inspect', 'zoom', 'slideshow',
     'close and reopen', 'have i seen everything', 'position', '12 / 340',
     'counter', 'first image', 'last image', 'wrap', 'loop', 'end of the list',
     'crosses pages', 'page', 'filters', 'sort'],
    '/datasets?section=images', 'using-the-app', 'move-through-a-dataset-without-closing-the-image'),
  // ✓ Keep / ✕ Reject / ⏭ Skip in the dataset lightbox — the Bank's review bar,
  // on the same K/R/S. Its own topic because the questions it raises are not the
  // arrows': does K delete anything, is it the same ✓ as the tile's, and why did
  // the picture move on by itself.
  action('action-lightbox-keep-reject', 'Keep or reject an image without leaving the picture',
    ['keep', 'reject', 'skip', 'verdict', 'decide', 'decision', 'judge', 'triage',
     'curate', 'curation', 'review', 'review one by one', 'one by one', 'fast',
     'k', 'r', 's', 'shortcut', 'shortcuts', 'keyboard', 'hotkey', 'key',
     'lightbox', 'full screen', 'fullscreen', 'inspect', 'zoom', 'grid',
     'tick', 'cross', 'green', 'red', 'undecided', 'pending', 'status',
     'moves on', 'next image by itself', 'auto advance', 'advance',
     'does it delete', 'delete', 'undo', 'take it back', 'same as the tile'],
    '/datasets?section=images', 'using-the-app', 'keep-or-reject-a-dataset-image-without-leaving-the-picture'),
  // ☰ Actions — the one button the whole action list moves behind on a phone.
  // Its own topic because the question it raises is "where did Crop go?", which
  // no other topic answers: the buttons are not hidden, they are one tap away,
  // and Esc now means two different things depending on what is open.
  action('action-lightbox-phone-actions', 'Inspect an image on a phone',
    ['phone', 'mobile', 'tablet', 'small screen', 'narrow', 'portrait mode',
     'actions', 'actions button', 'hamburger', 'menu', 'panel', 'drawer',
     'sheet', 'where is crop', 'no crop button', 'missing buttons', 'buttons gone',
     'image too small', 'thumbnail', 'tiny image', 'cannot see the image',
     'lightbox', 'full screen', 'inspect', 'compare on a phone', 'side by side',
     'escape', 'esc', 'close the panel', 'klein note', 'instruction editor'],
    '/datasets?section=images', 'using-the-app', 'inspect-an-image-on-a-phone'),
  // The lightbox's ⧉ Compare with original. Its whole point is that the two
  // panes are shown at the SAME scale — the guide section explains why, and why
  // 100 % zoom is deliberately off in that mode.
  action('action-compare-with-original', 'Compare an improved image with the original',
    ['compare', 'comparison', 'side by side', 'side-by-side', 'before after', 'before/after',
     'original', 'improved', 'improve', 'upscale', 'klein', 'candidate', 'rescue',
     'small image', 'judge', 'is it better', 'difference', 'a/b', 'lightbox',
     'same scale', 'zoom', '100 %', 'undecided', 'keep or reject',
     'keep improved image', 'keep candidate', 'unkeep original', 'un-keep original',
     'original pending', 'original undecided', 'automatic unkeep', 'keep both',
     'bulk keep', 'batch keep', 'nothing deleted', 'do not delete'],
    '/datasets?section=images', 'using-the-app', 'compare-an-improved-image-with-the-original'),
  // The lightbox's ◐ Compare with reference — a DIFFERENT question from the one
  // above ("same person?" vs "sharper?"), on a different set of images (all of
  // them, not just candidates), with a different promise about scale. Its own
  // topic on purpose: one topic answering both would have to hedge on the one
  // sentence that matters, which pane geometry guarantees what.
  action('action-compare-with-reference', 'Compare an image with the dataset reference photo',
    ['compare', 'comparison', 'reference', 'reference photo', 'ref', 'side by side',
     'side-by-side', 'same person', 'is it the same person', 'identity', 'likeness',
     'resemblance', 'face', 'drift', 'off model', 'off-model', 'does not look like',
     'doesn t look like', 'check the reference', 'show the reference', 'lightbox',
     'generated image', 'variation', 'imported photo', 'different framings',
     'no compare button', 'no reference', 'add a reference photo'],
    '/datasets?section=images', 'using-the-app', 'compare-an-image-with-the-dataset-reference-photo'),
  // ✨ in the CANVAS lightbox AND in the checkpoint / run gallery's. Its own
  // topic, not a variant of the dataset one: the result lands somewhere else
  // (the checkpoint's gallery, not the curation grid), and "where did my upscale
  // go" is the question this button actually raises on a screen where nothing
  // moves when you press it. ONE topic for both surfaces on purpose — it is the
  // same pass on the same row, and two topics would be two answers to drift.
  action('action-canvas-improve', 'Upscale a picture from the board or its gallery',
    ['canvas', 'board', 'improve', 'upscale', 'upscale & improve', 'enhance', 'klein',
     'seedvr2', 'sharpen', 'detail', 'resolution', 'megapixels', 'lightbox',
     'pinned image', 'generated image', 'where did it go', 'result', 'gallery',
     'checkpoint gallery', 'improve from canvas', 'no improve button',
     'improve an improvement', 'reference face', 'retry', 'failed upscale',
     'improve from the gallery', 'upscale from the gallery', 'run gallery',
     'gallery lightbox', 'improve a test image', 'improve a render',
     'gallery did not update', 'upscale not showing'],
    '/canvas', 'using-the-app', 'upscale-a-picture-straight-from-the-board'),
  action('action-grid-sort', 'Sort the dataset grid, or group it by shot type',
    ['sort', 'order', 'ordering', 'reorder', 'rank', 'ranking', 'best first',
     'worst first', 'face similarity', 'similarity', 'resemblance', 'looks like',
     'face score', 'closest', 'least alike', 'review faster', 'grid', 'unscored',
     'not scored', 'analyze faces', 'group', 'grouping', 'shot type', 'shot types',
     'framing', 'face bust body back', 'compare', 'side by side', 'mixed up',
     'all mixed', 'which to keep'],
    '/datasets?section=images', 'using-the-app', 'sort-a-grid-to-review-faster'),
  action('action-classify-framing', 'Classify framing of imported images',
    ['framing', 'classify', 'classify framing', 'shot type', 'shot types', 'unknown framing',
     'no framing', 'not classified', 'unclassified', 'composition', 'composition zero',
     'composition empty', 'bar at 0', 'counts nothing', 'missing from composition',
     'face', 'bust', 'body', 'back', 'sort shots', 'imported', 'import', 'drag and drop',
     'no crop', 'head crop off', 'after crop', 'after cropping', 'cropped',
     'still says body', 'wrong shot type', 'ollama', 'vision', 'qwen'],
    '/datasets?section=add', 'dataset-guide', '2-how-many-images-and-which-ones'),
  // The composition bar can be fully green on a set that is one pose, one outfit,
  // one light. This is the panel that says so — keyworded on the SYMPTOM ("all my
  // images look the same", "lora only makes one pose"), because that is what
  // someone types before they know a coverage panel exists.
  action('action-dataset-coverage', '🔍 Coverage: what your dataset never shows',
    ['coverage', 'variety', 'variation', 'diversity', 'diverse', 'missing',
     'what is missing', 'gaps', 'balanced', 'unbalanced', 'all the same',
     'look the same', 'same pose', 'same outfit', 'same background',
     'same lighting', 'one outfit', 'no profile', 'profile', 'side view',
     'angles', 'camera angle', 'view', 'lighting', 'outfit', 'clothes',
     'expression', 'setting', 'background', 'overfit', 'overfitting',
     'baked in', 'generalise', 'generalize', 'only makes one', 'captions',
     'caption keywords', 'composition green', 'target reached',
     'show me those', 'which images', 'click a chip', 'filter by caption',
     'see the profiles', 'find the ones'],
    '/datasets?section=add', 'dataset-guide', '9-coverage-what-your-set-never-showed'),
  // Krea's Fit path applies the selected card's frame to a reference that still
  // anchors identity. Keep the stable id/anchor so old help links continue to land.
  action('krea-reference-shape', 'Krea follows the selected shot framing',
    ['krea', 'krea 2', 'krea 2 edit', 'reference', 'reference photo', 'aspect',
     'aspect ratio', 'shape', 'square', 'portrait', 'landscape', 'crop', 'recrop',
     'body', 'back', 'full body', 'full length', 'framing', 'cropped', 'tight',
     'too close', 'zoomed in', 'bust instead of body', '3:4'],
    '/datasets?section=add', 'using-the-app', 'krea-and-the-shape-of-your-reference-photo'),
  // Rotation exists in TWO places with two different promises (a dataset file is
  // rewritten, a bank file never is), so it gets one topic that says both —
  // otherwise "does this touch my folder?" has no address. Idea by 1Tomber (#17).
  action('action-rotate-image', 'Rotate an image 90°',
    ['rotate', 'rotation', 'turn', 'sideways', 'upright', 'orientation', 'portrait',
     'landscape', 'straighten', 'quarter turn', '90', '180', '270', 'left', 'right',
     'clockwise', 'counter-clockwise', 'exif', 'upside down', 'lossless', 'mirror',
     'flip', 'bank', 'crop'],
    '/datasets', 'using-the-app', 'rotate-a-sideways-image'),
  action('action-caption-generate', 'Generate captions',
    ['caption', 'generate', 'joycaption', 'ollama', 'text',
     // Caption STYLE lives on this control: the prose/booru selector next to the
     // button. Anima accepts both forms, so a user searching "booru" or "anima"
     // must land here rather than conclude the app only does one of them.
     'prose', 'booru', 'tags', 'style', 'anima', 'hybrid'],
    '/datasets?section=captions&panel=generate', 'dataset-guide', '3-captions-the-make-or-break-step'),
  action('action-caption-options', 'Caption method options',
    ['caption', 'options', 'engine', 'model', 'ollama', 'pull', 'instructions', 'prompt',
     'method', 'vocabulary', 'explicit', 'clinical', 'nsfw', 'abliterated', 'uncensored'],
    '/datasets?section=captions&panel=generate', 'dataset-guide', '3-captions-the-make-or-break-step'),
  action('action-caption-stop', 'Stop a captioning batch',
    ['caption', 'stop', 'cancel', 'abort', 'interrupt', 'batch', 'graceful', 'halt'],
    '/datasets?section=captions&panel=generate', 'dataset-guide', '3-captions-the-make-or-break-step'),
  action('action-training-launch', 'Train the LoRA',
    ['train', 'training', 'launch', 'cloud', 'lora', 'start'],
    '/datasets?section=training&panel=launch', 'dataset-guide', '5-pre-flight-checklist'),
  action('action-training-stop', 'Stop a training run',
    ['train', 'training', 'stop', 'cancel', 'abort', 'interrupt', 'kill', 'halt', 'finish',
     'comfyui', 'gpu', 'release', 're-enable', 'checkpoints', 'queue'],
    '/datasets?section=training&panel=launch', 'dataset-guide', '5-pre-flight-checklist'),
  action('training-continue-anyway', 'Continue anyway (train a not-ready dataset)',
    ['continue', 'anyway', 'not ready', 'blocker', 'override', 'too few', 'overfit', 'readiness', 'force'],
    '/datasets?section=training&panel=launch', 'dataset-guide', '5-pre-flight-checklist'),
  action('parallel-cloud-runs', 'Compare settings with two runs at once',
    ['parallel runs', 'second run', 'two runs', 'same dataset twice', 'a/b',
     'ab test', 'compare settings', 'settings comparison', 'launch anyway',
     'second pod', 'run chip', 'different dataset generation'],
    '/datasets', 'using-the-app', 'compare-settings-with-two-runs-at-once'),
  action('action-scrape-scan', 'Scan a gallery URL',
    ['scrape', 'scan', 'gallery', 'url', 'import', 'concept'],
    '/datasets?section=scrape&panel=scan', 'using-the-app', 'concept-datasets-an-object-or-action-not-a-person'),
  action('action-scrape-websearch', 'Search the web for images by keyword',
    ['scrape', 'search', 'websearch', 'web images', 'keyword', 'duckduckgo', 'import', 'concept'],
    '/datasets?section=scrape&panel=scan', 'using-the-app', 'concept-datasets-an-object-or-action-not-a-person'),
  action('action-import-from-bank', 'Import images from a bank',
    ['bank', 'import from bank', 'promote', 'triaged', 'kept images', 'add images',
     'copy from bank', 'reuse bank', 'nothing to promote', 'already imported'],
    '/datasets?section=add', 'using-the-app', 'the-image-bank-triage-a-big-folder'),
  action('action-edit-identity-prompt', 'Edit the identity instruction (Extra refs ✎)',
    ['identity', 'instruction', 'prompt', 'extra refs', 'multiple references', 'identity lock',
     'edit prompt', 'face_multi', 'klein identity', 'stronger identity'],
    '/datasets?section=add', 'settings-reference', 'image-engines'),
  action('action-studio-open', 'Open Studio',
    ['studio', 'test', 'lora', 'checkpoint', 'open'],
    '/datasets?section=studio', 'dataset-guide', '6-after-training-pick-the-right-checkpoint'),
  action('continue-training', 'Continue a training run',
    ['continue', 'resume', 'more steps', 'epoch', 'checkpoint', 'restart', 'undercook', 'overcook',
     'learning rate', 'lr', 'half', 'tenth', 'gentle finish', 'polish', 'timestep', 'cadence',
     'lane', 'local', 'cloud', 'run it',
     // How a full model's 26 GB reaches the pod — the priced choice in this
     // same dialog. Searchable from the words a user would actually type when
     // they are staring at a GPU cost they did not expect.
     'send it via', 'transport', 'upload', 'uplink', 'hugging face copy',
     'gpu cost', 'how long', 'slice', 'resumable upload'],
    '/datasets?section=checkpoints', 'dataset-guide', '6-after-training-pick-the-right-checkpoint',
    { trigger: 'continue-any-epoch',
      text: 'Finished a run? ▶ Continue trains it further — for any number of steps, or resumed from an earlier, less-cooked epoch.' }),
  action('action-recaption-targeted', 'Re-caption leaking images',
    ['caption', 'recaption', 'leak', 'targeted', 'fix', 'review'],
    '/datasets?section=captions&panel=leak-review', 'dataset-guide', '3-captions-the-make-or-break-step',
    { trigger: 'leak-panel-visible',
      text: 'You can re-caption just one leaking image (or all of them) — no full re-run.' }),
  action('action-watermark-restore', 'Restore original',
    ['watermark', 'restore', 'original', 'undo', 'revert', 'clean'],
    '/datasets?section=curation&panel=review-flagged', 'settings-reference', 'captioning-quality',
    { trigger: 'watermark-clean-done',
      text: 'Not happy with a clean? Restore brings the original back — then try the other engine.' }),
];
