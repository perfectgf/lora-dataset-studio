/**
 * ▶ Review — the Bank's fast-triage lightbox.
 *
 * One full-size image at a time with ✓ Keep / ✕ Reject / ⏭ Skip; every button
 * (and its keyboard shortcut) decides AND moves on, so a 3 000-image dump is
 * worked through without ever going back to the grid. "🎲 Random order"
 * shuffles what's left instead of walking the folder sequentially — on a dump
 * that means a representative sample straight away rather than 200 near-
 * identical frames in a row.
 *
 * All of the ordering logic (snapshot, shuffle, skip, end of pool) lives in the
 * JSX-free `bankReview.js` so `node --test` covers it. This file is the shell:
 * rendering, keyboard, and the one-POST-per-decision network lane.
 *
 * Decisions are sent IMMEDIATELY, one image per POST: closing after 50 calls
 * loses nothing. A POST that fails does NOT advance — the image stays under the
 * cursor with the error visible, so a decision is never silently dropped.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useFocusTrap } from '../../hooks/useFocusTrap'
import {
  createSession, currentId, isFinished, progress, back, decide, skip, setShuffle,
} from './bankReview.js'
import {
  DETAIL_CAVEAT, PROVENANCE_FLAG_LABEL, detailSummary, originHint, originLabel,
} from './bankProvenance.js'
import BankWatermarkMaskDialog from './BankWatermarkMaskDialog'
import CropModal from '../dataset/CropModal.jsx'
import { canEditMask, maskButtonLabel } from './bankWatermarkMask.js'
import { cropOutcomeMessage, imageVersionQuery } from './bankEdits.js'
import {
  REVIEW_SHORTCUT_HINT, ownsTypedKeys, reviewKeyAction,
} from '../shared/reviewShortcuts.js'
import ShortcutKey from '../shared/ShortcutKey'

// How many upcoming images we pull metadata for in one go (the decision helpers
// below the image). The grid page only holds the ids it rendered.
const META_WINDOW = 40

const FLAG_TEXT = {
  blur: '🌫 Blurry', noise: '📺 Noisy', uniform: '⬜ Flat', small: '📐 Small',
  unreadable: '❌ Unreadable', low_aesthetic: '💔 Low aesthetic', nsfw: '🔞 NSFW',
  watermark: '🚩 Watermark', ...PROVENANCE_FLAG_LABEL,
}

// Origin chip colours, one per state. 'unknown' is deliberately the quiet grey
// of a non-answer rather than a warning colour: it is an absence of evidence.
const ORIGIN_CLASS = {
  ai: 'bg-violet-500/25 text-violet-100',
  camera: 'bg-emerald-500/25 text-emerald-100',
  unknown: 'bg-white/10 text-white/60',
}

function Facts({ img }) {
  if (!img) return <span className="text-xs text-white/40">Reading image details…</span>
  const chip = (key, text, cls, title) => (
    <span key={key} title={title}
      className={`rounded px-1.5 py-px text-[11px] font-medium ${cls}`}>{text}</span>
  )
  const detail = detailSummary(img)
  const origin = originLabel(img)
  const hint = originHint(img)
  return (
    <div className="flex flex-wrap items-center justify-center gap-1.5">
      <span className="max-w-[22rem] truncate text-xs text-white/70" title={img.name}>{img.name}</span>
      {chip('res', `${img.width || '?'}×${img.height || '?'}`, 'bg-white/10 text-white/80')}
      {detail && chip('detail', detail.soft ? `🫧 ~${detail.real} px real` : '🫧 full detail',
        detail.soft ? 'bg-amber-500/20 text-amber-100' : 'bg-white/10 text-white/60',
        detail.soft ? `${detail.text}. ${DETAIL_CAVEAT}` : DETAIL_CAVEAT)}
      {origin && chip('origin', `${origin.icon} ${origin.label}`, ORIGIN_CLASS[origin.state],
        `${origin.detail}${hint ? ` ${hint}` : ''}`)}
      {img.aesthetic_score != null
        && chip('aes', `✨ ${img.aesthetic_score.toFixed(1)}`, 'bg-white/10 text-amber-200')}
      {img.nsfw_score != null
        && chip('nsfw', `🔞 ${Math.round(img.nsfw_score * 100)}%`, 'bg-white/10 text-rose-200')}
      {img.blur_score != null
        && chip('sharp', `sharpness ${Math.round(img.blur_score)}`, 'bg-white/10 text-white/70')}
      {img.face_cluster != null && chip('face', `👤 #${img.face_cluster}`, 'bg-white/10 text-sky-200')}
      {img.framing && chip('framing', `📐 ${img.framing}`, 'bg-white/10 text-teal-200')}
      {/* 🎨 the medium, with the margin BEHIND it in the tooltip: a verdict shown
          without its confidence is how a guess turns into a fact. 'unsure' is
          shown too — here, where the user is looking at the picture and can
          settle it, it is the most useful thing the classifier can say. */}
      {img.medium && chip('medium', `🎨 ${img.medium}`,
        img.medium === 'unsure' ? 'bg-white/10 text-white/50' : 'bg-white/10 text-lime-200',
        img.medium_margin != null
          ? `Zero-shot CLIP over the ✨ Score embedding — confidence gap ${img.medium_margin.toFixed(3)}.`
          : null)}
      {img.face_yaw != null && chip('yaw',
        `⤢ ${Math.round(Math.abs(img.face_yaw))}°`, 'bg-white/10 text-cyan-200',
        'How far the head is turned, measured by the 🎭 Faces pass.')}
      {img.dup_group != null && chip('dup', `≈ dup #${img.dup_group}`, 'bg-white/10 text-fuchsia-200')}
      {img.semantic_dup_group != null
        && chip('sdup', `✂ same shot #${img.semantic_dup_group}`, 'bg-white/10 text-orange-200')}
      {(img.flags || []).map((f) => chip(f, FLAG_TEXT[f] || f, 'bg-amber-500/20 text-amber-200'))}
      {img.status === 'keep' && chip('st', '✓ already kept', 'bg-emerald-500/25 text-emerald-200')}
      {img.status === 'reject' && chip('st', '✕ already rejected', 'bg-rose-500/25 text-rose-200')}
      {img.promoted_dataset_id != null && chip('pr', '⬆ promoted to a dataset', 'bg-indigo-500/25 text-indigo-200')}
      {img.promoted_bank_id != null && chip('prb', '⬆ promoted to another bank', 'bg-indigo-500/25 text-indigo-200')}
    </div>
  )
}

export default function BankReviewLightbox({
  bankId, ids, startId = null, seedImages = [], onDecided, onRotated, onEdited, onClose,
}) {
  const [session, setSession] = useState(() => createSession(ids, { startId }))
  const [meta, setMeta] = useState(() => {
    const m = {}
    for (const img of seedImages || []) m[img.id] = img
    return m
  })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  /* What an edit just did, said HERE rather than as a toast: the review runs
     full-screen over the workspace, and a toast fired behind it is a report the
     user never reads. Cleared as soon as the cursor moves — it describes the
     image on screen, not the session. */
  const [notice, setNotice] = useState(null)
  // 🚩 Which image's watermark mask is open, if any. Review is where a flagged
  // image is actually LOOKED at, so it is where a wrong detection box gets
  // fixed — until now that was only possible inside a dataset (Qeeyana, Reddit).
  const [maskId, setMaskId] = useState(null)
  /* ✂ Which image's crop editor is open. Review is the ONLY place the Bank shows
     an image big enough to draw a box on, so it is where the crop lives —
     the grid tile deliberately keeps its three hit targets and no more, which is
     the same reason the quarter-turn is a bulk action there and a button here. */
  const [cropId, setCropId] = useState(null)
  const dialogRef = useRef(null)
  const requested = useRef(new Set())

  // Hand the trap over while the mask or crop editor is open — two live traps
  // fight over the focus and the editor's own controls become unreachable.
  useFocusTrap(dialogRef, maskId == null && cropId == null)

  const id = currentId(session)
  const done = isFinished(session)
  const p = progress(session)
  const img = id == null ? null : meta[id]

  // Pull metadata for the current image and the ones just behind it, in one
  // request. Ids already asked for are never re-asked (an image deleted from the
  // bank simply stays without facts instead of looping).
  useEffect(() => {
    if (id == null) return
    const window = session.order.slice(session.pos, session.pos + META_WINDOW)
      .filter((x) => meta[x] === undefined && !requested.current.has(x))
    if (!window.length) return
    window.forEach((x) => requested.current.add(x))
    apiFetch(`/api/bank/${bankId}/images?ids=${window.join(',')}&limit=${window.length}`)
      .then((d) => setMeta((prev) => {
        const next = { ...prev }
        for (const row of d.images || []) next[row.id] = row
        return next
      }))
      .catch(() => { /* facts are an aid, not a gate — the image still shows */ })
  }, [bankId, id, session.order, session.pos, meta])

  const sendDecision = useCallback(async (status) => {
    const target = currentId(session)
    if (target == null || busy) return
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      await postJson(`/api/bank/${bankId}/images/status`, { ids: [target], status })
      setMeta((prev) => (prev[target] ? { ...prev, [target]: { ...prev[target], status } } : prev))
      setSession((s) => decide(s, status))
      onDecided?.(target, status)
    } catch (e) {
      // Stay on this image: a lost decision is worse than a stalled one.
      setError(e?.message || 'Could not save that decision — it was NOT recorded.')
    } finally {
      setBusy(false)
    }
  }, [bankId, busy, onDecided, session])

  // 🔄 Quarter turn of the image under the cursor (idea by 1Tomber, GitHub #17).
  // It does NOT advance — a sideways photo is fixed and then judged, which is the
  // whole point of noticing it here. The user's own file is never rewritten: the
  // angle is stored and the app rebuilds its view from the pristine source.
  const rotateCurrent = useCallback(async (degrees) => {
    const target = currentId(session)
    if (target == null || busy) return
    setBusy(true)
    setError(null)
    try {
      const d = await postJson(`/api/bank/${bankId}/rotate`, { ids: [target], degrees })
      const angle = d?.rotations?.[target] ?? d?.rotations?.[String(target)] ?? 0
      // A quarter turn transposes the frame; a half turn does not.
      const quarter = Math.abs(degrees) % 180 !== 0
      setMeta((prev) => (prev[target]
        ? { ...prev,
            [target]: { ...prev[target],
              rotation: angle,
              width: quarter ? prev[target].height : prev[target].width,
              height: quarter ? prev[target].width : prev[target].height } }
        : prev))
      onRotated?.(target, angle)
    } catch (e) {
      setError(e?.message || 'Could not rotate that image — it was NOT changed.')
    } finally {
      setBusy(false)
    }
  }, [bankId, busy, onRotated, session])

  /* ✂ Cut the image under the cursor down to the box just drawn, and ↩ throw
     that edit away. Both are asked for by nofaceman on Discord: the Bank has the
     filtering and the curation, so reframing a shot used to mean a round trip
     through a Dataset and an export into a NEW bank.

     Like the quarter turn, they DECIDE NOTHING and do not advance — a badly
     framed image is fixed and then judged, which is the whole reason you noticed
     it here. And like the quarter turn, your own file is never rewritten: the
     result is a copy the app keeps, and ↩ Revert deletes it.

     The box arrives in the NATURAL pixels of the image the editor loaded, which
     is the very image `/file/` resolves — cleaned and turned included. That is
     what the server crops, so what you drew is what you get. */
  const applyEdit = useCallback((target, state) => {
    if (!state) return
    setMeta((prev) => (prev[target]
      ? { ...prev,
          [target]: { ...prev[target],
            edit_method: state.edit_method ?? null,
            edit_generation: state.edit_generation ?? 0,
            rotation: state.rotation ?? 0,
            width: state.width ?? prev[target].width,
            height: state.height ?? prev[target].height } }
      : prev))
    onEdited?.(target, state)
  }, [onEdited])

  const cropCurrent = useCallback(async (box) => {
    const target = cropId
    setCropId(null)
    if (target == null || busy) return
    setBusy(true)
    setError(null)
    try {
      const d = await postJson(`/api/bank/${bankId}/image/${target}/crop`, box)
      applyEdit(target, d)
      setNotice(cropOutcomeMessage(d))
    } catch (e) {
      setError(e?.message || 'Could not crop that image — it was NOT changed.')
    } finally {
      setBusy(false)
    }
  }, [applyEdit, bankId, busy, cropId])

  const revertCurrent = useCallback(async () => {
    const target = currentId(session)
    if (target == null || busy) return
    setBusy(true)
    setError(null)
    try {
      const d = await postJson(`/api/bank/${bankId}/edits/revert`, { image_ids: [target] })
      // The route answers with the state of every row it touched, so the image
      // under the cursor is re-rendered from the ledger rather than from a guess
      // about what reverting should have produced.
      applyEdit(target, (d?.images || []).find((r) => r.id === target)
        || { edit_method: null, edit_generation: 0 })
      setNotice('Back to the image this bank started from — the rotation the edit'
        + ' had absorbed is yours again.')
    } catch (e) {
      setError(e?.message || 'Could not revert that image — it was NOT changed.')
    } finally {
      setBusy(false)
    }
  }, [applyEdit, bankId, busy, session])

  // Moving forward without judging IS a skip: it stays undecided in the DB and
  // is not proposed again in this session (→ and ⏭ are deliberately the same).
  const doSkip = useCallback(() => { setError(null); setNotice(null); setSession((s) => skip(s)) }, [])
  const goBack = useCallback(() => { setError(null); setNotice(null); setSession((s) => back(s)) }, [])
  const toggleShuffle = useCallback(() => setSession((s) => setShuffle(s, !s.shuffle)), [])

  useEffect(() => {
    const onKey = (e) => {
      // An open editor owns the keyboard: every letter here is one keystroke
      // away from a decision on the image being edited.
      if (maskId != null || cropId != null) return
      // K/R/S, ← and Esc come from the SHARED grammar (components/shared/
      // reviewShortcuts.js) — the same one the dataset lightbox reads, so the
      // two review surfaces cannot drift apart one refactor at a time. The keys
      // below are the Bank's own and are read off the same event.
      const action = reviewKeyAction(e)
      if (action === 'close') { onClose(); return }
      if (action === 'keep') { e.preventDefault(); sendDecision('keep'); return }
      if (action === 'reject') { e.preventDefault(); sendDecision('reject'); return }
      if (action === 'skip') { e.preventDefault(); doSkip(); return }
      if (action === 'back') { e.preventDefault(); goBack(); return }
      if (e.metaKey || e.ctrlKey || e.altKey || ownsTypedKeys(e.target)) return
      // [ and ] turn the image without deciding anything. Deliberately NOT
      // letters: every free letter here is one keystroke away from a decision.
      if (e.key === '[') { e.preventDefault(); rotateCurrent(-90) }
      else if (e.key === ']') { e.preventDefault(); rotateCurrent(90) }
      // M edits the watermark mask of the flagged image under the cursor.
      else if (e.key?.toLowerCase() === 'm' && canEditMask(img)) { e.preventDefault(); setMaskId(id) }
      // C opens the crop editor. A letter, like M and for the same reason: it
      // opens an EDITOR rather than deciding anything, so the "every free letter
      // is one keystroke from a decision" rule is not what is at stake.
      else if (e.key?.toLowerCase() === 'c' && id != null) { e.preventDefault(); setCropId(id) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, sendDecision, doSkip, goBack, rotateCurrent, maskId, cropId, img, id])

  // The key cap itself is shared with the dataset lightbox (ShortcutKey.jsx):
  // same letters, same look, one place.
  const shortcut = (label) => <ShortcutKey>{label}</ShortcutKey>

  const summary = useMemo(() => {
    const bits = [`${p.kept} kept`, `${p.rejected} rejected`]
    if (p.skipped) bits.push(`${p.skipped} skipped`)
    return bits.join(' · ')
  }, [p.kept, p.rejected, p.skipped])

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label="Review the bank image by image"
      data-probe-chrome="review" data-probe-layer
      className="fixed inset-0 z-[9996] flex flex-col bg-black/95">
      <div className="flex shrink-0 flex-wrap items-center gap-3 px-4 py-2 text-sm">
        <span className="font-semibold text-white">▶ Review</span>
        <span className="tabular-nums text-white/80">
          {done ? `${p.total} / ${p.total}` : `${p.position} / ${p.total}`}
        </span>
        <span className="text-white/50">{summary}</span>
        <label className="flex items-center gap-1.5 text-xs text-white/80"
          title="Walk what's left in random order instead of folder order — on a big dump that shows you a representative sample straight away instead of 200 near-identical shots. Nothing you have already seen comes back.">
          <input type="checkbox" checked={session.shuffle} onChange={toggleShuffle} />
          🎲 Random order
        </label>
        <button type="button" onClick={onClose} title="Close (Esc)" aria-label="Close review"
          className="ml-auto h-10 w-10 lg:h-9 lg:w-9 rounded-full bg-white/10 text-lg leading-none text-white hover:bg-white/20">✕</button>
      </div>

      {done ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-4 p-6 text-center">
          <p className="text-2xl font-bold text-white">🎉 All {p.total.toLocaleString()} image{p.total === 1 ? '' : 's'} reviewed</p>
          <p className="text-sm text-white/70">
            {p.kept} kept · {p.rejected} rejected
            {p.skipped ? ` · ${p.skipped} skipped (still undecided)` : ''}
          </p>
          <div className="flex flex-wrap items-center justify-center gap-2">
            {p.total > 0 && (
              <button type="button" onClick={goBack}
                className="rounded-lg border border-white/25 px-4 py-2 text-sm text-white hover:bg-white/10">
                ← Back to the last image
              </button>
            )}
            <button type="button" onClick={onClose}
              className="rounded-lg bg-gradient-primary px-5 py-2 text-sm font-semibold text-white">
              Back to the grid
            </button>
          </div>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 items-center justify-center p-3">
          {/* key= forces a fresh <img> per image so a slow load never shows the
              previous shot under the new one's buttons. */}
          {/* ?r=/?e= bust the browser cache after a turn or an edit — the bytes at
              this URL change while the URL itself does not. */}
          <img key={id} src={`/api/bank/${bankId}/file/${id}${imageVersionQuery(img)}`}
            alt={img?.name || `Bank image ${id}`}
            className="max-h-full max-w-full select-none object-contain" />
        </div>
      )}

      {!done && (
        <div className="shrink-0 space-y-2 bg-black/60 px-4 py-2.5">
          {error && (
            <p role="alert" className="text-center text-sm text-rose-300">
              ⚠️ {error} — try again, or close and check the app.
            </p>
          )}
          {notice && !error && (
            <p role="status" className="text-center text-sm text-sky-200">✂ {notice}</p>
          )}
          <Facts img={img} />
          <div className="flex flex-wrap items-center justify-center gap-2">
            <button type="button" onClick={goBack} disabled={session.pos === 0}
              title="Previous image (←) — navigation only, decides nothing"
              className="min-h-10 lg:min-h-0 rounded-lg border border-white/20 px-3 py-2 text-sm text-white disabled:opacity-35 hover:bg-white/10">
              ←
            </button>
            <button type="button" onClick={() => rotateCurrent(-90)} disabled={busy}
              aria-label="Rotate this image 90 degrees left"
              title="Rotate 90° left ([) — decides nothing. Your own file is never modified: the turn is stored and applied to what you see and to what gets promoted."
              className="min-h-10 lg:min-h-0 rounded-lg border border-white/20 px-3 py-2 text-sm text-white disabled:opacity-35 hover:bg-white/10">
              <span aria-hidden="true">↺</span><span className="sr-only">Rotate left</span>
            </button>
            <button type="button" onClick={() => rotateCurrent(90)} disabled={busy}
              aria-label="Rotate this image 90 degrees right"
              title="Rotate 90° right (]) — decides nothing. Your own file is never modified: the turn is stored and applied to what you see and to what gets promoted."
              className="min-h-10 lg:min-h-0 rounded-lg border border-white/20 px-3 py-2 text-sm text-white disabled:opacity-35 hover:bg-white/10">
              <span aria-hidden="true">↻</span><span className="sr-only">Rotate right</span>
            </button>
            {/* ✂ and ↩ sit with the rotate pair, not with ✓/✕/⏭: everything left
                of the decisions CHANGES THE IMAGE and advances nothing. */}
            <button type="button" onClick={() => setCropId(id)} disabled={busy || id == null}
              title="Crop this image (C) — decides nothing. Nothing is resampled: a Bank sits upstream of the training resolution, so the cut keeps its pixels and a dataset decides the size when it imports. Your own file is never modified, and ↩ Revert brings the original framing back."
              className="rounded-lg border border-sky-400/60 bg-sky-500/20 px-4 py-2 text-sm font-semibold text-sky-100 disabled:opacity-50 hover:bg-sky-500/30">
              ✂ Crop{shortcut('C')}
            </button>
            {img?.edit_method && (
              <button type="button" onClick={revertCurrent} disabled={busy}
                title="Throw away the ✂ crop / ✨ upscale made in this bank and go back to the image it started from. Only a copy made by the app is deleted — your own file was never modified."
                className="rounded-lg border border-white/25 px-4 py-2 text-sm text-white disabled:opacity-50 hover:bg-white/10">
                ↩ Revert edit
              </button>
            )}
            {canEditMask(img) && (
              <button type="button" onClick={() => setMaskId(id)} disabled={busy}
                title="Draw the watermark zones on this image (M) — decides nothing. Works even when the scan found nothing: what you draw becomes the flag, and 🧽 Inpaint then repaints exactly that."
                className="rounded-lg border border-amber-400/60 bg-amber-500/20 px-4 py-2 text-sm font-semibold text-amber-100 disabled:opacity-50 hover:bg-amber-500/30">
                🚩 {maskButtonLabel(img)}{shortcut('M')}
              </button>
            )}
            <button type="button" onClick={() => sendDecision('keep')} disabled={busy}
              title="Keep this image and move on (K)"
              className="rounded-lg border border-emerald-400/60 bg-emerald-500/20 px-5 py-2 text-sm font-semibold text-emerald-100 disabled:opacity-50 hover:bg-emerald-500/30">
              ✓ Keep{shortcut('K')}
            </button>
            <button type="button" onClick={() => sendDecision('reject')} disabled={busy}
              title="Reject this image and move on (R) — reversible, nothing is deleted from disk"
              className="rounded-lg border border-rose-400/60 bg-rose-500/20 px-5 py-2 text-sm font-semibold text-rose-100 disabled:opacity-50 hover:bg-rose-500/30">
              ✕ Reject{shortcut('R')}
            </button>
            <button type="button" onClick={doSkip}
              title="Decide later (S) — stays undecided and is not shown again in this review"
              className="rounded-lg border border-white/25 px-5 py-2 text-sm font-semibold text-white disabled:opacity-50 hover:bg-white/10">
              ⏭ Skip{shortcut('S')}
            </button>
          </div>
          <p className="text-center text-[11px] text-white/45">
            {REVIEW_SHORTCUT_HINT} · [ ] rotate · C crop · M watermark mask · ← → move
            without deciding · Esc close. Decisions are saved one by one — closing loses nothing.
          </p>
        </div>
      )}

      {maskId != null && meta[maskId] && (
        <BankWatermarkMaskDialog bankId={bankId} image={meta[maskId]}
          onSaved={(next) => setMeta((prev) => ({ ...prev, [next.id]: next }))}
          onClose={() => setMaskId(null)} />
      )}

      {/* The SAME editor the datasets use, unchanged: it already returns the box
          in natural image pixels, which is exactly the contract the bank's crop
          route expects. `onReset` is deliberately absent — it re-runs the
          dataset's automatic head-crop, and a bank has no such thing. */}
      {cropId != null && (
        <CropModal imageUrl={`/api/bank/${bankId}/file/${cropId}${imageVersionQuery(meta[cropId])}`}
          onCancel={() => setCropId(null)} onConfirm={cropCurrent} />
      )}
    </div>
  )
}
