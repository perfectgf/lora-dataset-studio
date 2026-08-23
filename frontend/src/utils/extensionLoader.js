// Optional backend-declared UI extensions. Every normal install gets an empty
// manifest and this whole file is a no-op. Extensions may never break the app:
// any failure here is swallowed after a console warning.

export function mountExtensionScripts(list, doc = document) {
  const mounted = []
  for (const ext of list || []) {
    if (!ext || !ext.frontend_entry) continue
    const el = doc.createElement('script')
    el.type = 'module'
    el.src = ext.frontend_entry
    el.dataset.extension = ext.name
    doc.head.appendChild(el)
    mounted.push(ext.name)
  }
  return mounted
}

export async function loadExtensions(doc = document) {
  try {
    const res = await fetch('/api/extensions/')
    if (!res.ok) return []
    const data = await res.json()
    return mountExtensionScripts(data.extensions, doc)
  } catch (err) {
    console.warn('extensions manifest unavailable:', err)
    return []
  }
}
