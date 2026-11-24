import { apiFetch, del, postForm, postJson } from '@lds/plugin-sdk'

export const API = '/api/remove-background'
export const ACTIVE = ['queued', 'running']
export const isRunning = picture => ACTIVE.includes(picture?.run?.status)

const at = picture => `${API}/pictures/${picture.id}`
const version = picture => picture.mask?.version || 0

export const viewUrl = picture => `${at(picture)}/view?v=1`
export const thumbUrl = picture => `${at(picture)}/thumb?v=1`
export const baseUrl = picture => `${at(picture)}/base?max=2048&v=${version(picture)}`
export const paintUrl = picture => `${at(picture)}/paint?v=${version(picture)}`
export const maskUrl = picture => `${at(picture)}/mask?v=${version(picture)}`

/** The finished picture; `max` is the editor's smaller look at it. */
export function cutoutUrl(picture, { background = 'transparent', clean = true, max = 0 } = {}) {
  const query = new URLSearchParams({ background, clean: clean ? '1' : '0', v: String(version(picture)) })
  if (max) query.set('max', String(max))
  return `${at(picture)}/cutout?${query}`
}

export const readStatus = () => apiFetch(`${API}/status`)
export const readDatasets = () => apiFetch(`${API}/datasets`).then(data => data.datasets || [])
export const readPictures = () => apiFetch(`${API}/pictures`).then(data => data.pictures || [])
export const readPicture = (picture, options) => apiFetch(at(picture), options).then(data => data.picture)
export const removePicture = picture => del(at(picture))
export const cancelCut = picture => postJson(`${at(picture)}/cancel`, {}).then(data => data.picture)

export function importFile(file, surface = '') {
  const form = new FormData()
  form.append('image', file, file.name || 'picture')
  if (surface) form.append('surface', surface)
  return postForm(`${API}/pictures`, form).then(data => data.picture)
}

/** A dataset or Gallery picture is read by LDS itself; a Bank one travels as a file. */
export async function importFrom({ surface, imageId, url, name }) {
  if (surface === 'dataset' || surface === 'gallery') {
    return postJson(`${API}/pictures`, { surface, image_id: imageId }).then(data => data.picture)
  }
  const response = await fetch(url, { credentials: 'same-origin' })
  if (!response.ok) throw new Error('This picture could not be read.')
  const blob = await response.blob()
  return importFile(new File([blob], name || 'picture', { type: blob.type }), surface)
}

export function startCut(picture, { keep, kind, phrase, points, selection }) {
  if (keep !== 'selection') {
    return postJson(`${at(picture)}/cut`, { keep, kind, phrase, points }).then(data => data.picture)
  }
  const form = new FormData()
  form.append('keep', keep)
  form.append('kind', kind)
  form.append('selection', selection, 'selection.png')
  return postForm(`${at(picture)}/cut`, form).then(data => data.picture)
}

/** The whole brush layer, or nothing to go back to the models' mask. */
export function savePaint(picture, blob) {
  const form = new FormData()
  if (blob) form.append('strokes', blob, 'strokes.png')
  return postForm(`${at(picture)}/paint`, form).then(data => data.picture)
}

export const addCandidate = (picture, { background, clean }) => postJson(`${at(picture)}/dataset`, { background, clean })

/** Any other dataset: the app's own import, with this cut-out as the file. */
export async function addToDataset(picture, datasetId, { background, clean }) {
  const response = await fetch(cutoutUrl(picture, { background, clean }), { credentials: 'same-origin' })
  if (!response.ok) throw new Error('The cut-out could not be read.')
  const form = new FormData()
  form.append('files', await response.blob(), 'cutout.png')
  form.append('crop', '0')
  return postForm(`/api/dataset/${datasetId}/import`, form)
}

export const startInstall = actions => postJson('/api/plugins/remove_background/preparation', { actions })
export const installStatus = actions => apiFetch(
  `/api/setup/install-all/status?${new URLSearchParams({ actions: actions.join(',') })}`, { background: true })
