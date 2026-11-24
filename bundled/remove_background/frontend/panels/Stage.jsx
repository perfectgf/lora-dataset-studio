import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { baseUrl, paintUrl, savePaint, viewUrl } from '../lib/api.js'
import { fit, held, onPicture, pinch, toPicture, toStage, zoomAt } from '../lib/geometry.js'
import { dirtyBox, fillOutline, strokeLayer, strokeMask, togglePoint } from '../lib/strokes.js'

const SAVE_AFTER = 700
const DOT = 7

const layer = (width, height) => Object.assign(document.createElement('canvas'), { width, height })

function load(url) {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('A part of this picture could not be loaded.'))
    image.src = url
  })
}

const toBlob = canvas => new Promise((resolve, reject) => canvas.toBlob(
  blob => blob ? resolve(blob) : reject(new Error('The drawing could not be read.')), 'image/png'))

/** The models' mask and the brush over it, at the editor's resolution. */
async function openMask(picture, size) {
  const base = layer(size.width, size.height)
  base.getContext('2d').drawImage(await load(baseUrl(picture)), 0, 0, size.width, size.height)
  let add = null
  let drop = null
  let painted = null
  if (picture.mask.painted) {
    painted = layer(size.width, size.height)
    const ctx = painted.getContext('2d', { willReadFrequently: true })
    ctx.drawImage(await load(paintUrl(picture)), 0, 0, size.width, size.height)
    // The saved layer is gray with transparency; the canvas needs what it adds and what it removes.
    const pixels = ctx.getImageData(0, 0, size.width, size.height)
    add = layer(size.width, size.height)
    drop = layer(size.width, size.height)
    const adding = new ImageData(size.width, size.height)
    const dropping = new ImageData(size.width, size.height)
    for (let i = 0; i < pixels.data.length; i += 4) {
      adding.data[i] = adding.data[i + 1] = adding.data[i + 2] = 255
      adding.data[i + 3] = Math.round(pixels.data[i + 3] * pixels.data[i] / 255)
      dropping.data[i + 3] = pixels.data[i + 3]
    }
    add.getContext('2d').putImageData(adding, 0, 0)
    drop.getContext('2d').putImageData(dropping, 0, 0)
  }
  const mask = layer(size.width, size.height)
  const brush = layer(size.width, size.height)
  const reset = () => {
    const m = mask.getContext('2d')
    const b = brush.getContext('2d')
    m.clearRect(0, 0, size.width, size.height)
    b.clearRect(0, 0, size.width, size.height)
    m.drawImage(base, 0, 0)
    if (painted) {
      m.globalCompositeOperation = 'destination-out'
      m.drawImage(drop, 0, 0)
      m.globalCompositeOperation = 'lighter'
      m.drawImage(add, 0, 0)
      m.globalCompositeOperation = 'source-over'
      b.drawImage(painted, 0, 0)
    }
  }
  reset()
  return { mask, brush, reset, forget() { painted = null; reset() }, touched: () => painted !== null }
}

/**
 * The picture, and what is drawn on it. `mode` says what a press does:
 * `look` moves the picture, `select` draws an outline, `points` places clicks,
 * `brush` paints the mask. Two fingers or the wheel always move and zoom.
 */
const Stage = forwardRef(function Stage({ picture, mode, result, background, points, onPoints, leaveOut,
  onSelection, brush, onPainted, onStrokes, onError, className = '' }, ref) {
  const canvasRef = useRef(null)
  const boxRef = useRef(null)
  const state = useRef({ view: null, stage: { width: 1, height: 1 }, pointers: new Map(), outline: null,
    stroke: null, strokes: [], hover: null })
  const art = useRef({ image: null, mask: null, tint: null, selection: null, shown: null })
  const frame = useRef(0)
  const saving = useRef({ timer: 0, work: null, dirty: false })
  const props = useRef({})
  props.current = { picture, mode, result, background, points, leaveOut, brush, onPainted, onStrokes, onError, onPoints, onSelection }
  const [ready, setReady] = useState(false)

  const draw = useCallback(() => {
    cancelAnimationFrame(frame.current)
    frame.current = requestAnimationFrame(() => {
      const canvas = canvasRef.current
      const { view, stage, outline, hover } = state.current
      const { image, mask, tint, selection, shown } = art.current
      const now = props.current
      if (!canvas || !view) return
      const ratio = window.devicePixelRatio || 1
      const ctx = canvas.getContext('2d')
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
      ctx.clearRect(0, 0, stage.width, stage.height)
      if (!image) return
      const box = [view.x, view.y, now.picture.width * view.scale, now.picture.height * view.scale]
      ctx.imageSmoothingQuality = 'high'
      if (now.mode === 'look' && shown) {
        if (now.background) { ctx.fillStyle = now.background; ctx.fillRect(...box) } else checker(ctx, box, stage)
        ctx.drawImage(shown, ...box)
      } else {
        ctx.drawImage(image, ...box)
        if (now.mode === 'brush' && mask) ctx.drawImage(tint, ...box)
      }
      if (now.mode === 'select') {
        if (selection) { ctx.globalAlpha = 0.4; ctx.drawImage(selection.shown, ...box); ctx.globalAlpha = 1 }
        if (outline?.length) {
          ctx.beginPath()
          outline.forEach((point, i) => {
            const at = toStage(view, point)
            if (i) ctx.lineTo(at.x, at.y)
            else ctx.moveTo(at.x, at.y)
          })
          ctx.setLineDash([6, 4])
          ctx.lineWidth = 2
          ctx.strokeStyle = now.leaveOut ? '#f87171' : '#fff'
          ctx.stroke()
          ctx.setLineDash([])
        }
      }
      if (now.mode === 'points') {
        for (const point of now.points) {
          const at = toStage(view, point)
          ctx.beginPath()
          ctx.arc(at.x, at.y, DOT, 0, Math.PI * 2)
          ctx.fillStyle = point.keep ? '#22c55e' : '#ef4444'
          ctx.fill()
          ctx.lineWidth = 2
          ctx.strokeStyle = '#fff'
          ctx.stroke()
        }
      }
      if (now.mode === 'brush' && hover) {
        ctx.beginPath()
        ctx.arc(hover.x, hover.y, Math.max(2, now.brush.size / 2), 0, Math.PI * 2)
        ctx.lineWidth = 3
        ctx.strokeStyle = 'rgba(0,0,0,.6)'
        ctx.stroke()
        ctx.lineWidth = 1.5
        ctx.strokeStyle = now.brush.keep ? '#86efac' : '#fca5a5'
        ctx.stroke()
      }
    })
  }, [])

  const size = useCallback(() => {
    const { image } = art.current
    return { width: image.naturalWidth, height: image.naturalHeight }
  }, [])
  const scaleOf = useCallback(() => size().width / props.current.picture.width, [size])

  const retint = useCallback((box = null) => {
    const { mask, tint } = art.current
    if (!mask) return
    const area = box || { x: 0, y: 0, ...size() }
    if (!area.width || !area.height) return
    const ctx = tint.getContext('2d')
    ctx.save()
    ctx.beginPath()
    ctx.rect(area.x, area.y, area.width, area.height)
    ctx.clip()
    ctx.clearRect(area.x, area.y, area.width, area.height)
    ctx.fillStyle = 'rgba(220, 38, 38, .62)'
    ctx.fillRect(area.x, area.y, area.width, area.height)
    ctx.globalCompositeOperation = 'destination-out'
    ctx.drawImage(mask.mask, 0, 0)
    ctx.restore()
  }, [size])

  // The picture itself.
  useEffect(() => {
    let alive = true
    setReady(false)
    art.current = { image: null, mask: null, tint: null, selection: null, shown: null }
    Object.assign(state.current, { view: null, strokes: [], stroke: null, outline: null })
    load(viewUrl(picture)).then(image => {
      if (!alive) return
      art.current.image = image
      const { width, height } = { width: image.naturalWidth, height: image.naturalHeight }
      art.current.selection = { area: layer(width, height), shown: layer(width, height) }
      state.current.view = fit(picture, state.current.stage)
      setReady(true)
      draw()
    }).catch(error => alive && props.current.onError?.(error.message))
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [picture.id])

  // A new answer of the models replaces whatever was painted on the previous one.
  const answer = picture.mask ? `${picture.id}:${picture.mask.run}` : ''
  useEffect(() => {
    if (!ready || !answer) { art.current.mask = null; return undefined }
    let alive = true
    openMask(picture, size()).then(mask => {
      if (!alive) return
      art.current.mask = mask
      art.current.tint = layer(size().width, size().height)
      state.current.strokes = []
      props.current.onStrokes?.({ count: 0, painted: mask.touched() })
      retint()
      draw()
    }).catch(error => alive && props.current.onError?.(error.message))
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, answer])

  useEffect(() => {
    let alive = true
    if (!result) { art.current.shown = null; draw(); return undefined }
    load(result).then(image => { if (alive) { art.current.shown = image; draw() } })
      .catch(error => alive && props.current.onError?.(error.message))
    return () => { alive = false }
  }, [result, draw])

  useEffect(() => { draw() }, [mode, background, points, leaveOut, brush, draw])

  useEffect(() => {
    const box = boxRef.current
    const observer = new ResizeObserver(() => {
      const { width, height } = box.getBoundingClientRect()
      if (!width || !height) return
      const canvas = canvasRef.current
      const ratio = window.devicePixelRatio || 1
      canvas.width = Math.round(width * ratio)
      canvas.height = Math.round(height * ratio)
      state.current.stage = { width, height }
      if (art.current.image) state.current.view = fit(props.current.picture, state.current.stage)
      draw()
    })
    observer.observe(box)
    return () => observer.disconnect()
  }, [draw])

  // --- painting is saved by itself, a moment after the last stroke ---
  const flush = useCallback(async () => {
    clearTimeout(saving.current.timer)
    while (saving.current.work) await saving.current.work
    if (!saving.current.dirty || !art.current.mask) return null
    saving.current.dirty = false
    const work = (async () => {
      const { mask } = art.current
      const empty = !state.current.strokes.length && !mask.touched()
      return savePaint(props.current.picture, empty ? null : await toBlob(mask.brush))
    })()
    saving.current.work = work.catch(() => null).finally(() => { saving.current.work = null })
    try {
      const saved = await work
      props.current.onPainted?.(saved)
      return saved
    } catch (error) {
      saving.current.dirty = true
      props.current.onError?.(error.message || 'The touch-up could not be saved.')
      throw error
    }
  }, [])
  const changed = useCallback(() => {
    saving.current.dirty = true
    props.current.onStrokes?.({ count: state.current.strokes.length, painted: art.current.mask?.touched() || false })
    clearTimeout(saving.current.timer)
    saving.current.timer = setTimeout(() => { flush().catch(() => {}) }, SAVE_AFTER)
  }, [flush])
  // Leaving right after a stroke saves it first.
  useEffect(() => () => { cancelAnimationFrame(frame.current); if (saving.current.dirty) flush().catch(() => {}) }, [flush])

  const replay = useCallback(() => {
    const { mask } = art.current
    if (!mask) return
    mask.reset()
    for (const stroke of state.current.strokes) {
      strokeMask(mask.mask.getContext('2d'), stroke, scaleOf(), 0)
      strokeLayer(mask.brush.getContext('2d'), stroke, scaleOf(), 0)
    }
    retint()
    draw()
    changed()
  }, [scaleOf, retint, draw, changed])

  useImperativeHandle(ref, () => ({
    flush,
    undo() { if (state.current.strokes.pop()) replay() },
    resetMask() { state.current.strokes = []; art.current.mask?.forget(); replay() },
    clearSelection() {
      for (const item of Object.values(art.current.selection || {})) item.getContext('2d').clearRect(0, 0, item.width, item.height)
      props.current.onSelection?.(false)
      draw()
    },
    selection: () => toBlob(art.current.selection.area),
    zoom(factor) {
      const { stage, view } = state.current
      if (!view) return
      state.current.view = zoomAt(view, factor, { x: stage.width / 2, y: stage.height / 2 }, props.current.picture, stage)
      draw()
    },
    fit() {
      if (!art.current.image) return
      state.current.view = fit(props.current.picture, state.current.stage)
      draw()
    },
  }), [flush, replay, draw])

  // --- the pointer ---
  const at = event => {
    const rect = canvasRef.current.getBoundingClientRect()
    return { x: event.clientX - rect.left, y: event.clientY - rect.top }
  }

  const dropGesture = () => {
    const { stroke } = state.current
    state.current.outline = null
    state.current.stroke = null
    // A second finger landed: what the first one started was a pan, not a stroke.
    if (stroke) { state.current.strokes = state.current.strokes.filter(item => item !== stroke); replay() }
    else draw()
  }

  const paintTo = (stroke, from) => {
    const { mask } = art.current
    strokeMask(mask.mask.getContext('2d'), stroke, scaleOf(), from)
    strokeLayer(mask.brush.getContext('2d'), stroke, scaleOf(), from)
    retint(dirtyBox(stroke, scaleOf(), from, size()))
    draw()
  }

  const onPointerDown = event => {
    if (!ready || (event.pointerType === 'mouse' && event.button > 1)) return
    event.preventDefault()
    canvasRef.current.setPointerCapture?.(event.pointerId)
    const point = at(event)
    const { pointers, view } = state.current
    pointers.set(event.pointerId, point)
    if (pointers.size === 2) { dropGesture(); return }
    if (pointers.size > 2) return
    const now = props.current
    const spot = toPicture(view, point)
    if (event.button === 1 || now.mode === 'look') return
    if (now.mode === 'points') {
      if (!onPicture(spot, now.picture)) return
      const keep = now.leaveOut === event.altKey
      now.onPoints(togglePoint(now.points, { ...spot, keep }, (DOT + 4) / view.scale))
    } else if (now.mode === 'select') {
      state.current.outline = [spot]
      draw()
    } else if (now.mode === 'brush' && art.current.mask) {
      const stroke = { keep: now.brush.keep !== event.altKey, size: now.brush.size / view.scale, points: [spot] }
      state.current.stroke = stroke
      state.current.strokes.push(stroke)
      paintTo(stroke, 0)
    }
  }

  const onPointerMove = event => {
    const point = at(event)
    const { pointers, view, stage } = state.current
    if (!view) return
    const before = pointers.get(event.pointerId)
    if (!before) {
      state.current.hover = event.pointerType === 'touch' ? null : point
      if (props.current.mode === 'brush') draw()
      return
    }
    if (pointers.size === 2) {
      const pair = [...pointers.values()]
      pointers.set(event.pointerId, point)
      state.current.view = pinch(view, pair, [...pointers.values()], props.current.picture, stage)
      draw()
      return
    }
    pointers.set(event.pointerId, point)
    const { outline, stroke } = state.current
    if (stroke) {
      // Every position the pointer went through, not only the last of the frame.
      const from = stroke.points.length - 1
      const events = event.nativeEvent.getCoalescedEvents?.() || []
      for (const item of events.length ? events : [event]) stroke.points.push(toPicture(view, at(item)))
      state.current.hover = event.pointerType === 'touch' ? null : point
      paintTo(stroke, from)
    } else if (outline) {
      outline.push(toPicture(view, point))
      draw()
    } else if (props.current.mode === 'look' || (event.buttons & 4)) {
      state.current.view = held({ ...view, x: view.x + point.x - before.x, y: view.y + point.y - before.y }, props.current.picture, stage)
      draw()
    }
  }

  const onPointerUp = event => {
    const { pointers, outline, stroke } = state.current
    if (!pointers.delete(event.pointerId)) return
    if (stroke) {
      state.current.stroke = null
      changed()
    }
    if (outline) {
      state.current.outline = null
      const { area, shown } = art.current.selection
      const drop = props.current.leaveOut !== event.altKey
      for (const item of [area, shown]) {
        const ctx = item.getContext('2d')
        ctx.globalCompositeOperation = drop ? 'destination-out' : 'source-over'
        fillOutline(ctx, outline, scaleOf())
        ctx.globalCompositeOperation = 'source-over'
      }
      const ctx = shown.getContext('2d')
      ctx.globalCompositeOperation = 'source-in'
      ctx.fillStyle = '#22c55e'
      ctx.fillRect(0, 0, shown.width, shown.height)
      ctx.globalCompositeOperation = 'source-over'
      props.current.onSelection?.(isDrawn(area))
      draw()
    }
  }

  // React's own wheel listener is passive: the page would scroll under the zoom.
  useEffect(() => {
    const canvas = canvasRef.current
    const onWheel = event => {
      const { view, stage } = state.current
      if (!view) return
      event.preventDefault()
      const rect = canvas.getBoundingClientRect()
      state.current.view = zoomAt(view, Math.exp(-event.deltaY * 0.0015),
        { x: event.clientX - rect.left, y: event.clientY - rect.top }, props.current.picture, stage)
      draw()
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  }, [draw])

  const cursor = mode === 'look' ? 'cursor-grab' : mode === 'brush' ? 'cursor-none' : 'cursor-crosshair'
  return <div ref={boxRef} className={`relative min-h-0 min-w-0 overflow-hidden ${className}`}>
    <canvas ref={canvasRef} role="img" aria-label={`${picture.name}, ${mode === 'look' ? 'result' : 'editor'}`}
      className={`absolute inset-0 h-full w-full touch-none select-none ${cursor}`}
      onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}
      onPointerLeave={() => { state.current.hover = null; draw() }} onContextMenu={event => event.preventDefault()} />
    {!ready && <p role="status" className="absolute inset-0 m-0 flex items-center justify-center text-sm text-content-muted">Loading…</p>}
  </div>
})

function checker(ctx, box, stage) {
  const [x, y, width, height] = box
  const left = Math.max(0, x)
  const top = Math.max(0, y)
  const right = Math.min(stage.width, x + width)
  const bottom = Math.min(stage.height, y + height)
  if (right <= left || bottom <= top) return
  ctx.save()
  ctx.beginPath()
  ctx.rect(left, top, right - left, bottom - top)
  ctx.clip()
  ctx.fillStyle = '#d4d4d4'
  ctx.fillRect(left, top, right - left, bottom - top)
  ctx.fillStyle = '#a3a3a3'
  for (let row = Math.floor(top / 12); row * 12 < bottom; row += 1) {
    for (let col = Math.floor(left / 12); col * 12 < right; col += 1) {
      if ((row + col) % 2) ctx.fillRect(col * 12, row * 12, 12, 12)
    }
  }
  ctx.restore()
}

/** Is anything left of the selection? Read small: the answer does not need every pixel. */
function isDrawn(canvas) {
  const small = layer(64, 64)
  const ctx = small.getContext('2d', { willReadFrequently: true })
  ctx.drawImage(canvas, 0, 0, 64, 64)
  return ctx.getImageData(0, 0, 64, 64).data.some((value, i) => i % 4 === 3 && value > 8)
}

export default Stage
