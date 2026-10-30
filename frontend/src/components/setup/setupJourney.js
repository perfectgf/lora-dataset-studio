import { engineCatalog } from '../../engines/catalog.js'
import { capabilityDestination, deriveSetupSteps } from '../../hooks/useSetupSteps.js'
import { pluginActive, pluginDesired } from '../../plugins/lifecycle.js'
import { pluginSettingsPath } from '../../pages/pluginSettings.js'

export const SETUP_GOALS = [
  { id: 'dataset', title: 'Prepare a dataset', description: 'Import, sort and caption your own images. Nothing to install.', icon: 'images' },
  { id: 'captions', title: 'Automatically caption images', description: 'A local AI describes your images: Ollama or LM Studio.', icon: 'captions' },
  { id: 'images', title: 'Generate images', description: 'On your GPU with ComfyUI, or online with an API key.', icon: 'generate' },
  { id: 'plugins', title: 'Create videos & add tools', description: 'Add a plugin, then prepare what it needs.', icon: 'plugins' },
]

// Generating images starts with the one question that decides everything
// after it: the user's own ComfyUI, or an online engine.
export const IMAGE_PLACES = [
  { id: 'local', title: 'Local', description: 'ComfyUI on this computer. Free, uses your GPU.' },
  { id: 'api', title: 'API', description: 'An online service. Needs an account or an API key.' },
]

// Both servers are external apps LDS talks to; the choice is saved in config.
export const CAPTION_SERVICES = [
  { id: 'ollama', title: 'Ollama' },
  { id: 'lmstudio', title: 'LM Studio' },
]

const ID = /^[a-z][a-z0-9_-]{0,63}(\.[a-z][a-z0-9_-]{0,63})?$/
export function normalizeJourney(value) {
  if (!value || !SETUP_GOALS.some(goal => goal.id === value.goal)) return null
  const result = { goal: value.goal }
  for (const key of ['plugin', 'engine']) {
    if (typeof value[key] === 'string' && ID.test(value[key])) result[key] = value[key]
  }
  if (value.goal === 'images' && IMAGE_PLACES.some(place => place.id === value.where)) result.where = value.where
  if (typeof value.capability === 'string' && value.capability.length <= 200) result.capability = value.capability
  return result
}

/** The engines a place offers: ComfyUI engines for Local, every other one for API. */
export function enginesFor(where, engines = engineCatalog()) {
  return engines.filter(engine => (engine.kind === 'local') === (where === 'local'))
}

export function journeyPath(value) {
  const journey = normalizeJourney(value)
  return journey ? '/setup?' + new URLSearchParams(journey) : '/setup'
}

export function journeyToolPath(step, value) {
  const params = new URLSearchParams(normalizeJourney(value) || {})
  params.set('step', step)
  return '/setup?' + params
}

export function localJourneyPath(value) {
  return typeof value === 'string' && /^\/(?!\/)[^\s\\\u0000-\u001f]*$/.test(value) ? value : null
}

export function journeyProducts(catalog, installed = []) {
  const products = new Map()
  for (const product of catalog?.products || []) {
    const release = product.recommended || product.releases?.[0]
    if (release?.manifest) products.set(product.id, { id: product.id, name: release.manifest.name,
      description: release.manifest.description, experience: release.manifest.experience,
      issues: release.compatibility_issues || [], price: release.price, installed: false })
  }
  for (const plugin of installed) {
    products.set(plugin.id, { ...products.get(plugin.id), id: plugin.id, name: plugin.name,
      description: plugin.description, experience: plugin.package_contract?.experience,
      installed: true })
  }
  return [...products.values()]
}

const action = (label, to) => ({ label, to })
// A row's description is only shown while it is the next step, so it says
// what to do, in one short sentence.
const row = (id, title, ready, description, next) => ({ id, title, ready, description, action: next })
const finish = (title, description, to, label) => ({ title, description, links: [action(label, to)] })

function pluginLifecycle(plugin, product, journey) {
  const installedPath = '/plugins?tab=installed&plugin=' + encodeURIComponent(journey.plugin || '')
  const discoverPath = '/plugins?tab=discover&plugin=' + encodeURIComponent(journey.plugin || '')
  const activationReady = !!plugin && pluginActive(plugin) && pluginDesired(plugin) && !plugin.pending_action
  const rows = [
    row('package', 'Install ' + (product?.name || plugin?.name || 'the plugin'), !!plugin,
      'Review it in Plugins, then install it.', action('Open in Plugins', discoverPath)),
    row('activation', 'Turn on the plugin', activationReady,
      plugin?.pending_action ? 'Apply the waiting change in Plugins. LDS restarts once.'
        : plugin?.error || 'Turn it on in Plugins.',
      action(plugin?.pending_action ? 'Apply in Plugins' : 'Open Plugins', installedPath)),
  ]
  if (plugin?.environment) rows.push(row('environment', 'Prepare the plugin’s tools', plugin.environment.ready === true,
    plugin.environment.reason || 'Install the tools offered on its card in Plugins.',
    action('Prepare in Plugins', installedPath)))
  return rows
}

/** All readiness comes from the live probes. Clicks never mark a step ready. */
export function deriveJourney(value, { caps = {}, runtime = null, installed = [], catalog = null,
  productRows = [], interfaceProblem = '' } = {}) {
  const journey = normalizeJourney(value)
  if (!journey) return null
  const goal = SETUP_GOALS.find(item => item.id === journey.goal)
  const result = { title: goal.title, rows: [], ready: false, first: null }
  const tools = step => journeyToolPath(step, journey)
  if (journey.goal === 'dataset') {
    result.rows = [row('workspace', 'Your dataset workspace', true, '')]
    result.first = finish('Create your first dataset', 'Name it, then import a few images.',
      '/datasets', 'Open LDS')
  } else if (journey.goal === 'captions') {
    // The provider question is asked on this screen, above the steps: both are
    // always offered, whichever one the machine scan happened to find.
    const llm = deriveSetupSteps(caps, runtime).find(step => step.id === 'ollama')
    const name = llm.isLmStudio ? 'LM Studio' : 'Ollama'
    result.title = 'Automatically caption images'
    result.llmProvider = llm.llmProvider
    result.rows = [
      row('service', 'Start ' + name, llm.reachable,
        llm.managedInitializing ? name + ' is starting. Check again in a moment.'
          : name + ' is not answering. Install it or start it.',
        action('Set up ' + name, tools('ollama'))),
      row('model', 'Load a vision model', llm.reachable && llm.visionModelReady,
        'Pick a model that can read images.',
        action(llm.isLmStudio ? 'Load a vision model' : 'Choose a vision model', tools('ollama'))),
    ]
    result.first = finish('Caption a few images', 'Open a dataset, then Captions.',
      '/datasets?section=captions', 'Open Captions')
  } else if (journey.goal === 'images') {
    if (!journey.where) return { ...result, chooseWhere: true }
    const local = journey.where === 'local'
    const comfy = caps.comfyui || {}
    const managed = runtime?.comfyui?.mode
    result.where = journey.where
    // Local: the ComfyUI folder is asked first; its engines are offered once
    // that folder is a ComfyUI install (or a Docker install manages it). A
    // ComfyUI that merely answers is not enough: model downloads need the folder.
    result.comfyFound = comfy.dir_valid === true || managed === 'integrated' || managed === 'external-host'
    result.comfyFolder = comfy.base_dir || comfy.resolved_dir || ''
    result.title = local ? 'Generate images locally' : 'Generate images with an API'
    const engines = enginesFor(journey.where)
    const engine = engines.find(item => item.id === journey.engine)
    if (!engine || (local && !result.comfyFound)) return { ...result, chooseEngine: true, engines }
    result.title = 'Generate images with ' + engine.label
    if (engine.plugin) {
      const plugin = installed.find(item => item.id === engine.plugin)
      result.rows.push(...pluginLifecycle(plugin, null, { ...journey, plugin: engine.plugin }))
    }
    const ready = caps.engines?.[engine.id] === true
    if (local) {
      result.rows.push(row('service', 'Start ComfyUI', comfy.reachable === true,
        'ComfyUI is not answering. Start it, then check again.', action('Open ComfyUI setup', tools('comfyui'))))
      result.rows.push(row('engine', 'Download ' + engine.label, ready,
        'Download its model files. Sizes are shown before anything starts.',
        action('Download ' + engine.label, engine.plugin ? pluginSettingsPath(engine.plugin)
          : tools(engine.id === 'krea' ? 'install' : 'comfyui'))))
    } else {
      const settings = engine.plugin ? pluginSettingsPath(engine.plugin) : tools('image')
      result.rows.push(row('engine', 'Connect ' + engine.label, ready,
        'Add its API key, or connect your account, then save.',
        action('Connect ' + engine.label, engine.plugin && engine.secret
          ? settings + '?focus=' + encodeURIComponent(engine.secret) : settings)))
    }
    result.first = finish('Generate your first image', 'Open a dataset, add a reference photo, then Generate variations.',
      '/datasets?section=add&panel=generate', 'Open Generate variations')
  } else {
    const product = journeyProducts(catalog, installed).find(item => item.id === journey.plugin)
    if (!journey.plugin) return { ...result, choosePlugin: true }
    const plugin = installed.find(item => item.id === journey.plugin)
    result.title = 'Get started with ' + (product?.name || plugin?.name || journey.plugin)
    result.rows = pluginLifecycle(plugin, product, journey)
    if (result.rows.every(item => item.ready)) {
      if (interfaceProblem) result.rows.push(row('interface', 'Load the plugin interface', false,
        interfaceProblem, { label: 'Reload LDS', kind: 'reload' }))
      else if (productRows.length && !productRows.some(item => item.label === journey.capability)) {
        return { ...result, chooseCapability: true }
      } else if (productRows.length) {
        const selected = productRows.find(item => item.label === journey.capability)
        const destination = capabilityDestination({ ...selected, ok: selected.state === 'ready', pending: selected.state === 'pending' })
        let target = localJourneyPath(destination?.href) || pluginSettingsPath(plugin.id) + '?focus=plugin-preparation'
        if (target.startsWith('/setup')) {
          const query = new URL(target, 'http://lds.invalid').searchParams
          target = journeyToolPath(query.get('step') || 'tools', journey)
        }
        result.rows.push(row('function', selected.label, selected.state === 'ready',
          selected.note || selected.what || 'Prepare it from the plugin’s settings.',
          action('Prepare this function', target)))
      } else {
        result.manual = true
        result.rows.push(row('manual', 'Check the plugin’s setup instructions', false,
          product?.experience?.setup_hint || 'Follow the instructions in its settings.',
          action('Open plugin settings', pluginSettingsPath(plugin.id))))
      }
    } else result.rows.push(row('function', 'Prepare the function you want to use', false,
      'Choose a function once the plugin is on.',
      action('Open Plugins', '/plugins?tab=installed')))
    result.first = { title: 'Try your plugin', description: journey.capability
      ? 'Open the plugin and try ' + journey.capability + '.'
      : 'Start from the plugin’s own page.',
    links: (product?.experience?.entrypoints || []).filter(item => localJourneyPath(item.path))
      .map(item => action(item.label, item.path)) }
  }
  result.next = result.rows.find(item => !item.ready)
  result.ready = result.rows.length > 0 && !result.next
  return result
}
