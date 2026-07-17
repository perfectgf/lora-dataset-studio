/* Pure formatter for /api/diagnostic — kept out of the .jsx so it can be unit-tested
   under `node --test` (which can't parse JSX). Renders the payload as a fenced-
   markdown block ready to paste into Discord/GitHub.

   Keys/paths never appear: the backend ships presence booleans, redacts the home dir
   out of every log/error line, and caps the variable-length sections. Sections are
   ordered most-discriminating first; empty ones are dropped so the healthy-case
   report stays short. */
export function formatDiagnostic(d) {
  const yn = (v) => (v ? 'yes' : 'no')
  const c = d.capabilities || {}
  const e = c.engines || {}
  const cf = d.config || {}
  const o = d.ollama || {}
  const tags = o.tags_seen || []
  const pml = d.python_ml || {}
  const pil = d.pillow || {}
  const disk = d.disk || {}
  const rt = d.comfyui_runtime || {}
  const ge = d.generation_errors || {}
  const errlog = d.error_log || []

  const L = []
  L.push('```')
  L.push(`LoRA Dataset Studio diagnostic — v${d.app_version}${d.git_sha ? ` (${d.git_sha})` : ''}`)
  L.push(`OS: ${d.os} · Python ${d.python}${pml.ml_supported === false ? ` (⚠ outside ML wheel range ${pml.ml_range})` : ''}`)

  L.push('── Engines ──')
  L.push(`default=${cf.default_engine} · nanobanana=${yn(e.nanobanana)} chatgpt=${yn(e.chatgpt)} klein=${yn(e.klein)}`)
  if ((c.klein_missing || []).length) L.push(`  klein missing assets: ${c.klein_missing.join(', ')}`)
  if (c.chatgpt_subscription) L.push('  chatgpt: subscription connected')
  L.push(`Keys set: ${Object.entries(d.secrets_present || {}).filter(([, v]) => v).map(([k]) => k).join(', ') || 'none'}`)

  L.push('── ComfyUI ──')
  const rtBits = []
  if (rt.version) rtBits.push(`version ${rt.version}`)
  if (rt.gpu) rtBits.push(rt.gpu)
  if (rt.vram_total_gb != null) rtBits.push(`VRAM ${rt.vram_total_gb}GB${rt.vram_free_gb != null ? ` (${rt.vram_free_gb} free)` : ''}`)
  if (rt.queue_running != null) rtBits.push(`queue ${rt.queue_running} running / ${rt.queue_pending ?? 0} pending`)
  L.push(`reachable=${yn(c.comfyui_reachable)} · klein_model=${yn(c.klein_model)}${rtBits.length ? ' · ' + rtBits.join(' · ') : ''}`)

  L.push('── Captioning (Ollama) ──')
  L.push(`reachable=${yn(c.ollama_reachable)} · vision_model_ready=${yn(c.vision_model_ready)}`)
  // The configured model + the tags Ollama actually reports: when vision_model=no
  // this shows whether the model is truly missing or just listed under a different
  // identifier (issue #7). abliterated vs vanilla is readable straight from the name.
  L.push(`  configured: ${o.vision_model || '(none)'} · tags: ${tags.length ? tags.join(', ') : '(none)'}`)

  L.push('── Environment ──')
  const envBits = []
  if (pil.version) envBits.push(`Pillow ${pil.version} (${pil.healthy === false ? 'MIXED ⚠' : 'healthy'})`)
  if (disk.free_gb != null) envBits.push(`disk ${disk.free_gb}GB free / ${disk.total_gb}GB`)
  envBits.push(`captioning=${cf.captioning_backend}`, `allow_crop=${yn(cf.watermark_allow_crop)}`, `LAN=${yn(cf.lan_enabled)}`)
  L.push(envBits.join(' · '))
  L.push(`ai-toolkit=${yn(c.aitoolkit_valid)} · face scoring=${yn(c.face_scoring)} · masks=${yn(c.masks)} · cloud=${yn(c.cloud_training)} · default family=${cf.training_default_family}`)

  const engErrs = Object.entries(ge.engines || {})
  if (engErrs.length || ge.studio) {
    L.push('── Recent generation failures ──')
    // fail_reason is stored as 'engine: …' on the generation path, so it already
    // names its engine — print it verbatim (re-prefixing gave 'chatgpt: chatgpt:').
    for (const [, reason] of engErrs) L.push(reason)
    if (ge.studio) L.push(`studio: ${ge.studio}`)
  }

  if (errlog.length) {
    L.push('── Last errors (with traceback) ──')
    L.push(...errlog)
  }

  // Raw tail for the surrounding sequence (warnings, the lead-up). Kept short —
  // the ERROR section above already carries the stack, so this is context only.
  L.push('── Last log lines ──')
  L.push(...(d.log_tail || []).slice(-18))
  L.push('```')
  return L.join('\n')
}
