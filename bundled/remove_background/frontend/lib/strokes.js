// The brush. A stroke is {keep, size, points} in picture pixels, replayed at any
// resolution, so an undo is a replay without the last one.

export function trace(ctx, stroke, scale, from = 0) {
  const points = stroke.points
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.lineWidth = Math.max(1, stroke.size * scale)
  const start = Math.max(0, Math.min(from, points.length - 1))
  ctx.beginPath()
  ctx.moveTo(points[start].x * scale, points[start].y * scale)
  if (points.length - start < 2) ctx.lineTo(points[start].x * scale + 0.01, points[start].y * scale)
  for (let i = start + 1; i < points.length; i += 1) ctx.lineTo(points[i].x * scale, points[i].y * scale)
  ctx.stroke()
}

/** On the mask, whose transparency is what is kept. */
export function strokeMask(ctx, stroke, scale, from) {
  ctx.save()
  ctx.globalCompositeOperation = stroke.keep ? 'source-over' : 'destination-out'
  ctx.strokeStyle = '#fff'
  trace(ctx, stroke, scale, from)
  ctx.restore()
}

/** On the brush layer the server puts over the models' mask: white keeps, black removes. */
export function strokeLayer(ctx, stroke, scale, from) {
  ctx.save()
  ctx.globalCompositeOperation = 'source-over'
  ctx.strokeStyle = stroke.keep ? '#fff' : '#000'
  trace(ctx, stroke, scale, from)
  ctx.restore()
}

/** The part of the picture a stroke's newest segment touches, at `scale`, in whole pixels. */
export function dirtyBox(stroke, scale, from, size) {
  const points = stroke.points.slice(Math.max(0, from))
  const radius = stroke.size * scale / 2 + 2
  const xs = points.map(p => p.x * scale)
  const ys = points.map(p => p.y * scale)
  const x = Math.max(0, Math.floor(Math.min(...xs) - radius))
  const y = Math.max(0, Math.floor(Math.min(...ys) - radius))
  const right = Math.min(size.width, Math.ceil(Math.max(...xs) + radius))
  const bottom = Math.min(size.height, Math.ceil(Math.max(...ys) + radius))
  return { x, y, width: Math.max(0, right - x), height: Math.max(0, bottom - y) }
}

/** A drawn outline becomes a filled shape; fewer than three points is a slip, not a shape. */
export function fillOutline(ctx, outline, scale) {
  if (outline.length < 3) return false
  ctx.save()
  ctx.fillStyle = '#fff'
  ctx.beginPath()
  ctx.moveTo(outline[0].x * scale, outline[0].y * scale)
  for (const point of outline.slice(1)) ctx.lineTo(point.x * scale, point.y * scale)
  ctx.closePath()
  ctx.fill()
  ctx.restore()
  return true
}

/** The click, or the one it lands on: pressing a dot again takes it away. */
export function togglePoint(points, point, reach, limit = 32) {
  const index = points.findIndex(item => Math.hypot(item.x - point.x, item.y - point.y) <= reach)
  if (index >= 0) return points.filter((_, i) => i !== index)
  return points.length >= limit ? points : [...points, point]
}
