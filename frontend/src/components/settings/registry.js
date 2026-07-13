// Data-driven section list for the Settings page: sidebar labels, deep-link
// ids, the mono eyebrow tag, and the keywords the sidebar search matches on.

export const SETTINGS_SECTIONS = [
  { id: 'overview', title: 'Overview', icon: '📊', eyebrow: 'status',
    description: 'What is configured and what to do next.',
    keywords: ['status', 'summary', 'capabilities', 'ready'] },
  { id: 'engines', title: 'Image engines', icon: '🎨', eyebrow: 'generation',
    description: 'API keys and engines used to generate dataset images.',
    keywords: ['gemini', 'openai', 'api key', 'chatgpt', 'nano banana', 'klein', 'engine', 'subscription', 'gpt-image'] },
  { id: 'local-tools', title: 'Local tools', icon: '🖥️', eyebrow: 'integrations',
    description: 'ComfyUI, Ollama and ai-toolkit — where they run and where they live.',
    keywords: ['comfyui', 'ollama', 'ai-toolkit', 'vision model', 'path', 'url', 'hugging face', 'hf token', 'directory', 'install'] },
  { id: 'captioning', title: 'Captioning & quality', icon: '✍️', eyebrow: 'pipeline',
    description: 'How captions are written and how face similarity is judged.',
    keywords: ['caption', 'joycaption', 'backend', 'face score', 'threshold', 'green', 'orange', 'similarity'] },
  { id: 'training', title: 'Training', icon: '🏋️', eyebrow: 'training',
    description: 'Default model family and cloud GPU guardrails.',
    keywords: ['family', 'zimage', 'sdxl', 'krea', 'cloud', 'vast', 'budget', 'price', 'stall', 'gpu'] },
  { id: 'server', title: 'Server & access', icon: '🌐', eyebrow: 'network',
    description: 'Port, LAN access and the access token.',
    keywords: ['port', 'host', 'lan', 'network', 'token', 'remote', 'phone', 'bind'] },
  { id: 'maintenance', title: 'Maintenance', icon: '🔧', eyebrow: 'housekeeping',
    description: 'Updates, server log and data location.',
    keywords: ['update', 'restart', 'log', 'diagnostic', 'data', 'storage', 'version', 'bug'] },
]

/* Sidebar LED per section — derived from live capabilities so the rail doubles
   as a health map of the rig: 'ready' | 'partial' | 'off' | null (no LED). */
export function sectionStatus(id, caps) {
  const c = caps || {}
  const e = c.engines || {}
  switch (id) {
    case 'engines':
      return (e.nanobanana || e.chatgpt || e.klein) ? 'ready' : 'off'
    case 'local-tools': {
      const parts = [
        !!(c.comfyui && c.comfyui.reachable),
        !!(c.ollama && c.ollama.reachable),
        !!(c.aitoolkit && c.aitoolkit.valid),
      ]
      const n = parts.filter(Boolean).length
      return n === 3 ? 'ready' : n > 0 ? 'partial' : 'off'
    }
    case 'captioning': {
      const cap = c.captioners || {}
      return (cap.joycaption || cap.ollama) ? 'ready' : 'off'
    }
    case 'training':
      return c.training_visible ? (c.cloud_training ? 'ready' : 'partial') : 'off'
    default:
      return null
  }
}

export function matchesQuery(section, q) {
  const needle = (q || '').trim().toLowerCase()
  if (!needle) return true
  return section.title.toLowerCase().includes(needle)
    || section.keywords.some((k) => k.includes(needle))
}
