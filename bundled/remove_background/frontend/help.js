import { action } from '@lds/plugin-sdk/help'

const CHAPTER = 'remove_background.guide'
const WORDS = ['remove background', 'background removal', 'cut out', 'cutout', 'cut-out', 'transparent', 'alpha',
  'mask', 'matte', 'sam', 'sam 3', 'sam 3.1', 'birefnet', 'rmbg', 'rmbg 3.0', 'lucida', 'hair', 'fur', 'glass', 'logo']

export const HELP = [
  action('remove-background', 'Remove the background of a picture',
    [...WORDS, 'isolate subject', 'keep subject', 'select subject', 'white background', 'png', 'brush', 'touch up'],
    '/remove-background', CHAPTER, 'remove-the-background'),
  { ...action('remove-background-setup', 'Prepare Remove Background', [...WORDS, 'install', 'models', 'comfyui'],
    '/plugins/remove_background/settings', CHAPTER, 'models'),
  app: { route: '/plugins/remove_background/settings', focus: 'remove-background-preparation' } },
]
