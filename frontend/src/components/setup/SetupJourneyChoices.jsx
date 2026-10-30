import { useState } from 'react'
import { IMAGE_PLACES } from './setupJourney'

const CHOICE = 'w-full min-w-0 rounded-xl border border-border bg-surface p-4 text-left hover:border-primary disabled:cursor-not-allowed disabled:opacity-60'
const STATES = { ready: 'Ready', setup: 'Needs preparation', pending: 'Waiting for a service', unknown: 'Check unavailable' }

export default function SetupJourneyChoices({ plan, journey, products, catalog, rows, caps = {}, comfyuiFolder, onChoose, onBrowse }) {
  const [search, setSearch] = useState('')
  if (plan.chooseWhere) return <section className="space-y-4" aria-label="Choose where images are generated">
    <h2 className="text-lg font-semibold text-content">Where should images be generated?</h2>
    <div className="grid gap-3 sm:grid-cols-2">{IMAGE_PLACES.map(place => <button type="button" key={place.id} className={CHOICE}
      onClick={() => onChoose({ ...journey, where: place.id })}>
      <span className="block text-sm font-semibold text-content">{place.title}</span>
      <span className="mt-1 block text-xs text-content-muted">{place.description}</span>
    </button>)}</div>
  </section>
  if (plan.chooseEngine) {
    const local = plan.where === 'local'
    return <div className="space-y-6">
      {/* Local asks for the ComfyUI folder first; its engines appear once
          LDS has found ComfyUI there. */}
      {local && <section className="space-y-3" aria-label="ComfyUI folder">
        <h2 className="text-lg font-semibold text-content">Where is ComfyUI installed?</h2>
        {comfyuiFolder}
        {!plan.comfyFound && <a href="https://github.com/comfyanonymous/ComfyUI" target="_blank" rel="noreferrer"
          className="inline-flex min-h-10 items-center text-xs text-primary underline">No ComfyUI yet? Get it on GitHub</a>}
      </section>}
      {(!local || plan.comfyFound) && <section className="space-y-4" aria-label="Choose an image engine">
        <h2 className="text-lg font-semibold text-content">Choose an engine</h2>
        {plan.engines.length > 0 ? <div className="grid gap-3 sm:grid-cols-2">{plan.engines.map(engine => {
          const ready = caps.engines?.[engine.id] === true
          return <button type="button" key={engine.id} className={CHOICE}
            onClick={() => onChoose({ ...journey, engine: engine.id })}>
            <span className="block text-sm font-semibold text-content">{engine.label}</span>
            <span className={'mt-1 block text-xs ' + (ready ? 'text-emerald-400' : 'text-content-muted')}>
              {ready ? 'Ready' : local ? 'Needs its model files' : 'Needs an API key or account'}
            </span>
          </button>
        })}</div> : <p className="text-sm text-content-muted">No online engine is installed yet.</p>}
        <button type="button" onClick={() => onChoose({ goal: 'plugins' })} className="min-h-10 text-sm text-primary underline">
          More engines in the plugin Store
        </button>
      </section>}
    </div>
  }
  if (plan.chooseCapability) return <section className="space-y-4" aria-label="Choose a plugin function">
    <h2 className="text-lg font-semibold text-content">What would you like to use first?</h2>
    <div className="space-y-3">{rows.map((item, index) => <button type="button" key={item.label + index} className={CHOICE}
      onClick={() => onChoose({ ...journey, capability: item.label })}>
      <span className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-semibold text-content">{item.label}</span>
        <span className={'text-xs ' + (item.state === 'ready' ? 'text-emerald-400' : 'text-content-muted')}>{STATES[item.state]}</span>
      </span>
      {(item.what || item.note) && <span className="mt-1 block text-xs text-content-muted">{item.what || item.note}</span>}
    </button>)}</div>
  </section>
  const matches = products.filter(product => (product.name + ' ' + product.description).toLowerCase().includes(search.toLowerCase()))
  return <section className="space-y-4" aria-label="Choose a plugin">
    <h2 className="text-lg font-semibold text-content">Choose the tool you want to add</h2>
    <label className="block text-sm text-content">Find a plugin
      <input type="search" value={search} onChange={event => setSearch(event.target.value)} placeholder="Search by name or use…"
        className="mt-1 min-h-11 w-full rounded-lg border border-border-strong bg-surface-raised px-3 py-2 text-content" />
    </label>
    {catalog?.status !== 'ready' && <p role="status" className="text-sm text-content-muted">{catalog?.message || 'The Store is not available. Installed plugins are listed below.'}</p>}
    <div className="grid gap-3 sm:grid-cols-2">{matches.map(product => <button type="button" key={product.id} className={CHOICE}
      disabled={!product.installed && product.issues?.length > 0}
      onClick={() => onChoose({ goal: 'plugins', plugin: product.id })}>
      <span className="block text-sm font-semibold text-content">{product.name}</span>
      <span className="mt-1 block text-xs text-primary">{product.installed ? 'Installed' : product.price?.kind === 'paid' ? 'Paid plugin · review before purchase' : 'Review installation in the Store'}</span>
      <span className="mt-2 block text-xs leading-relaxed text-content-muted">{product.description}</span>
      {!product.installed && product.issues?.length > 0 && <span className="mt-2 block text-xs text-amber-300">{product.issues[0].message}</span>}
    </button>)}</div>
    {!matches.length && <p className="text-sm text-content-muted">{search ? 'No plugin matches this search.' : 'No plugin is available here yet. Open the Store to check its connection or install a plugin archive.'}</p>}
    <button type="button" onClick={onBrowse} className="min-h-10 text-sm text-primary underline">Open the plugin Store</button>
  </section>
}
