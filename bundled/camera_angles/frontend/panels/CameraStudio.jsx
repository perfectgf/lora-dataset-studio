import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router'
import { Camera, Download, ImagePlus, Loader2 } from 'lucide-react'
import { apiFetch, postForm, postJson } from '@lds/plugin-sdk'
import CameraAnglePicker from './CameraAnglePicker.jsx'

const API = '/api/camera/studio/images'
const active = view => ['queued', 'running', 'stalled', 'cancel_requested'].includes(view.status)
const originalUrl = image => `${API}/${image.id}/original`
const viewUrl = (image, view) => `${API}/${image.id}/views/${view.id}`
const stateLabel = { queued: 'Queued', running: 'Rendering…', stalled: 'Paused — check the system queue',
  cancel_requested: 'Cancelling…', failed: 'Failed' }

export default function CameraStudio() {
  const [images, setImages] = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [pollError, setPollError] = useState('')
  const [notice, setNotice] = useState('')
  const fileInput = useRef(null)
  const selectedId = selected?.id
  const pending = selected?.views.filter(active).length || 0

  useEffect(() => {
    let alive = true
    apiFetch(API).then(data => {
      if (!alive) return
      setImages(data.images || [])
      setSelected(data.images?.[0] || null)
    }).catch(err => { if (alive) setError(err.message || 'Could not load imported images.') })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [])

  // Reconcile once on selection, then only while this source has unfinished views.
  useEffect(() => {
    if (!selectedId || sending) return undefined
    let alive = true
    let timer
    setPollError('')
    const refresh = async () => {
      try {
        const data = await apiFetch(`${API}/${selectedId}`)
        if (!alive) return
        setSelected(data.image)
        setImages(items => items.map(item => item.id === selectedId ? data.image : item))
        setPollError('')
        if (data.image.views.some(active)) timer = setTimeout(refresh, 2500)
      } catch (err) {
        if (!alive) return
        setPollError(err.message || 'Could not refresh camera views. Retrying…')
        timer = setTimeout(refresh, 5000)
      }
    }
    refresh()
    return () => { alive = false; clearTimeout(timer) }
  }, [selectedId, sending])

  const upload = async file => {
    if (!file || uploading || sending || loading) return
    setError('')
    setNotice('')
    if (file.size > 32 * 1024 * 1024) {
      setError('Choose an image smaller than 32 MB.')
      return
    }
    setUploading(true)
    try {
      const form = new FormData()
      form.append('image', file)
      const data = await postForm(API, form)
      setImages(items => [data.image, ...items])
      setSelected(data.image)
    } catch (err) { setError(err.message || 'Could not import this image.') }
    finally { setUploading(false) }
  }

  const shoot = async poses => {
    if (!selected || sending || uploading || pending) return
    setSending(true)
    setError('')
    setNotice('')
    try {
      const data = await postJson(`${API}/${selected.id}/shoot`, { poses })
      setSelected(data.image)
      setImages(items => items.map(item => item.id === data.image.id ? data.image : item))
      setNotice(data.queued === data.requested
        ? `${data.queued} camera view${data.queued === 1 ? '' : 's'} queued. Results appear below.`
        : `${data.queued} of ${data.requested} views queued. Check the failed view below before trying again.`)
    } catch (err) { setError(err.message || 'Could not queue camera views.') }
    finally { setSending(false) }
  }

  return (
    <div className="mx-auto w-full max-w-7xl space-y-5 p-4 sm:p-6">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold text-gray-100"><Camera className="size-5 text-indigo-300" />Camera angles</h1>
          <p className="mt-1 text-sm text-gray-400">Import an image, choose your viewpoints and re-shoot the scene.</p>
        </div>
        <Link to="/plugins/camera_angles/settings" className="rounded-lg border border-white/10 px-3 py-2 text-sm text-gray-300 hover:border-white/30">Models &amp; setup</Link>
      </header>
      {error && <p role="alert" className="rounded-lg border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-200">{error}</p>}
      {notice && <p role="status" className="rounded-lg border border-indigo-400/30 bg-indigo-500/10 p-3 text-sm text-indigo-100">{notice}</p>}
      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,20rem)_minmax(0,1fr)]">
        <section aria-label="Source image" className="min-w-0 space-y-3">
          <input ref={fileInput} type="file" accept="image/png,image/jpeg,image/webp" className="sr-only"
            aria-label="Import image" disabled={loading || uploading || sending}
            onChange={event => { upload(event.target.files?.[0]); event.target.value = '' }} />
          <button type="button" disabled={loading || uploading || sending} onClick={() => fileInput.current?.click()}
            onDragOver={event => event.preventDefault()}
            onDrop={event => { event.preventDefault(); upload(event.dataTransfer.files?.[0]) }}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-indigo-400/40 bg-indigo-500/10 px-4 py-4 text-sm font-semibold text-indigo-100 hover:bg-indigo-500/20 disabled:opacity-50">
            {uploading || loading ? <Loader2 className="size-5 animate-spin" /> : <ImagePlus className="size-5" />}
            {loading ? 'Loading images…' : uploading ? 'Importing…' : 'Import image or drop it here'}
          </button>
          <p className="text-xs text-gray-500">PNG, JPEG or WebP · up to 32 MB / 40 MP</p>
          {selected ? <div className="overflow-hidden rounded-xl border border-white/10 bg-black/20">
            <a href={originalUrl(selected)} target="_blank" rel="noreferrer" title="Open original image">
              <img src={originalUrl(selected)} alt={selected.name} className="max-h-[28rem] w-full object-contain" />
            </a>
            <div className="space-y-1 p-3">
              <p className="break-words text-sm font-medium text-gray-200">{selected.name}</p>
              <p className="text-xs text-gray-500">Original · {selected.width} × {selected.height} · kept unchanged</p>
            </div>
          </div> : <div className="rounded-xl border border-white/10 bg-white/[0.02] p-6 text-sm leading-relaxed text-gray-400">
            Your source image appears here. Choose the camera positions on the right, then shoot your views.
          </div>}
          {images.length > 1 && <div>
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-400">Imported images</h2>
            <div className="flex max-h-48 flex-wrap gap-2 overflow-y-auto">
              {images.map(item => <button key={item.id} type="button" disabled={sending || uploading}
                onClick={() => { setSelected(item); setError(''); setNotice('') }} aria-label={`Use ${item.name}`}
                aria-pressed={selectedId === item.id} title={item.name}
                className={`overflow-hidden rounded-lg border-2 ${selectedId === item.id ? 'border-indigo-400' : 'border-transparent hover:border-white/30'}`}>
                <img src={originalUrl(item)} alt="" loading="lazy" className="size-16 object-cover" />
              </button>)}
            </div>
          </div>}
        </section>
        <CameraAnglePicker inline onShoot={shoot} busy={sending}
          disabled={!selected || loading || uploading || pending > 0} />
      </div>
      {selected && <section aria-label="Camera results" className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-base font-semibold text-gray-200">Camera views {selected.views.length > 0 && `(${selected.views.length})`}</h2>
          {pending > 0 && <p role="status" className="text-sm text-indigo-200">{pending} view{pending === 1 ? '' : 's'} in progress · manage or cancel in the system queue</p>}
        </div>
        {pollError && <p role="alert" className="text-sm text-amber-200">{pollError}</p>}
        {selected.views.length === 0 && <p className="rounded-xl border border-white/10 p-5 text-sm text-gray-500">Shoot your first views. Each result will appear here with its camera position and a download button.</p>}
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {selected.views.map(view => <article key={view.id} className="min-w-0 overflow-hidden rounded-xl border border-white/10 bg-white/[0.02]">
            {view.status === 'done'
              ? <a href={viewUrl(selected, view)} target="_blank" rel="noreferrer" title={`Open ${view.label}`}>
                <img src={viewUrl(selected, view)} alt={view.label} loading="lazy" className="aspect-square w-full object-contain bg-black/20" />
              </a>
              : <div className="flex aspect-square items-center justify-center gap-2 p-4 text-center text-sm text-gray-400">
                {active(view) && <Loader2 className="size-4 shrink-0 animate-spin" />}
                {stateLabel[view.status] || view.status}
              </div>}
            <div className="space-y-2 p-3">
              <p className="text-sm font-medium text-gray-200">{view.label}</p>
              {view.error && <p className="break-words text-xs text-red-200">{view.error}</p>}
              {view.status === 'done' && <a href={`${viewUrl(selected, view)}?download=1`} download
                className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-3 py-2 text-xs text-gray-300 hover:border-white/30"><Download className="size-3.5" />Download</a>}
            </div>
          </article>)}
        </div>
      </section>}
    </div>
  )
}
