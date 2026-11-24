import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router'
import { importFile, isRunning, readPictures, removePicture, thumbUrl } from '../lib/api.js'
import Editor from './Editor.jsx'

const PER_PAGE = 24
const MAX_BYTES = 48 * 1024 * 1024

export default function RemoveBackgroundPage() {
  const [pictures, setPictures] = useState([])
  const [selected, setSelected] = useState(null)
  const [page, setPage] = useState(0)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const input = useRef(null)
  const picture = pictures.find(item => item.id === selected) || null
  const pages = Math.max(1, Math.ceil(pictures.length / PER_PAGE))
  const shown = pictures.slice(page * PER_PAGE, (page + 1) * PER_PAGE)

  useEffect(() => {
    let alive = true
    readPictures().then(list => {
      if (!alive) return
      setPictures(list)
      setSelected(list[0]?.id || null)
    }).catch(problem => { if (alive) setError(problem.message || 'Your pictures could not be loaded.') })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [])
  useEffect(() => { if (page > pages - 1) setPage(pages - 1) }, [page, pages])

  const replace = next => setPictures(list => list.map(item => item.id === next.id ? next : item))
  const add = async file => {
    if (!file || uploading) return
    setError('')
    if (file.size > MAX_BYTES) { setError('Choose a picture smaller than 48 MB.'); return }
    setUploading(true)
    try {
      const next = await importFile(file)
      setPictures(list => [next, ...list.filter(item => item.id !== next.id)])
      setSelected(next.id)
      setPage(0)
    } catch (problem) { setError(problem.message || 'This picture could not be imported.') }
    finally { setUploading(false) }
  }
  const remove = async item => {
    setError('')
    try {
      await removePicture(item)
      setPictures(list => list.filter(other => other.id !== item.id))
      if (selected === item.id) setSelected(pictures.find(other => other.id !== item.id)?.id || null)
    } catch (problem) { setError(problem.message || 'This picture could not be removed.') }
  }

  return <div className="mx-auto w-full max-w-7xl space-y-4 sm:p-4" onDragOver={event => event.preventDefault()}
    onDrop={event => { event.preventDefault(); add(event.dataTransfer.files?.[0]) }}>
    <header className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <h1 className="m-0 text-xl font-semibold text-content">Remove background</h1>
        <p className="mt-1 text-sm text-content-muted">Say what stays, get a mask you can paint on and a clean cut-out. Everything runs on your own computer.</p>
      </div>
      <div className="flex flex-wrap gap-2">
        <input ref={input} type="file" accept="image/png,image/jpeg,image/webp,image/bmp,image/avif" className="sr-only" tabIndex={-1}
          aria-label="Import a picture" onChange={event => { add(event.target.files?.[0]); event.target.value = '' }} />
        <button type="button" disabled={loading || uploading} onClick={() => input.current?.click()}
          className="min-h-10 rounded-lg border border-primary/50 bg-primary/10 px-4 py-2 text-sm font-semibold text-primary hover:bg-primary/20 disabled:opacity-50">
          {uploading ? 'Importing…' : 'Import a picture'}</button>
        <Link to="/plugins/remove_background/settings" className="inline-flex min-h-10 items-center rounded-lg border border-border px-3 py-2 text-sm text-content hover:border-border-strong">Models</Link>
      </div>
    </header>
    {error && <p role="alert" className="m-0 rounded-lg border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-200">{error}</p>}
    {picture ? <Editor key={picture.id} picture={picture} onChange={replace} className="lg:h-[calc(100dvh-14rem)] lg:min-h-[30rem]" />
      : <button type="button" disabled={loading || uploading} onClick={() => input.current?.click()}
        className="flex min-h-[16rem] w-full flex-col items-center justify-center gap-1 rounded-xl border border-dashed border-border-strong p-6 text-center text-sm text-content-muted hover:border-primary disabled:opacity-60">
        <span className="text-base font-semibold text-content">{loading ? 'Loading…' : 'Import a picture or drop it here'}</span>
        <span>PNG, JPEG, WebP, BMP or AVIF, up to 48 MB and 40 megapixels</span>
      </button>}
    {pictures.length > 0 && <section aria-label="Your pictures" className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="m-0 text-sm font-semibold text-content">Your pictures ({pictures.length})</h2>
        {pages > 1 && <nav aria-label="Pages of pictures" className="flex items-center gap-2 text-sm text-content-muted">
          <button type="button" disabled={page === 0} onClick={() => setPage(page - 1)} className="min-h-10 rounded-lg border border-border px-3 disabled:opacity-40">Previous</button>
          <span>Page {page + 1} of {pages}</span>
          <button type="button" disabled={page >= pages - 1} onClick={() => setPage(page + 1)} className="min-h-10 rounded-lg border border-border px-3 disabled:opacity-40">Next</button>
        </nav>}
      </div>
      <ul className="m-0 grid list-none grid-cols-[repeat(auto-fill,minmax(6rem,1fr))] gap-2 p-0">
        {shown.map(item => <li key={item.id} className="relative">
          <button type="button" onClick={() => { setSelected(item.id); setError('') }} aria-pressed={selected === item.id} title={item.name}
            className={`block w-full overflow-hidden rounded-lg border-2 ${selected === item.id ? 'border-primary' : 'border-transparent hover:border-border-strong'}`}>
            <img src={thumbUrl(item)} alt={item.name} loading="lazy" className="block aspect-square w-full bg-black/30 object-cover" />
          </button>
          {!isRunning(item) && <button type="button" onClick={() => remove(item)} aria-label={`Remove ${item.name}`} title="Remove from this page. The cut-out goes to the LDS trash."
            className="absolute right-1 top-1 size-8 rounded-full bg-black/70 text-white hover:bg-red-600">×</button>}
        </li>)}
      </ul>
    </section>}
  </div>
}
