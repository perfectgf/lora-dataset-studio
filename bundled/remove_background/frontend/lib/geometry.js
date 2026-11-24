// Where the picture sits on the editor's stage. Pure: {scale, x, y} places a
// picture of {width, height} pixels on a stage of {width, height} CSS pixels.
export const MAX_ZOOM = 12

const clamp = (value, low, high) => Math.min(high, Math.max(low, value))

/** The whole picture, centred, never enlarged past its own pixels by more than 4. */
export function fit(picture, stage, margin = 8) {
  const room = { width: Math.max(1, stage.width - margin * 2), height: Math.max(1, stage.height - margin * 2) }
  const scale = Math.min(room.width / picture.width, room.height / picture.height, 4)
  return { scale, x: (stage.width - picture.width * scale) / 2, y: (stage.height - picture.height * scale) / 2 }
}

export const toPicture = (view, point) => ({ x: (point.x - view.x) / view.scale, y: (point.y - view.y) / view.scale })
export const toStage = (view, point) => ({ x: point.x * view.scale + view.x, y: point.y * view.scale + view.y })

export const onPicture = (point, picture) => point.x >= 0 && point.y >= 0
  && point.x < picture.width && point.y < picture.height

/** Keep some of the picture on the stage whatever the pan. */
export function held(view, picture, stage) {
  const width = picture.width * view.scale
  const height = picture.height * view.scale
  const keepX = Math.min(width, 64)
  const keepY = Math.min(height, 64)
  return { ...view, x: clamp(view.x, keepX - width, stage.width - keepX), y: clamp(view.y, keepY - height, stage.height - keepY) }
}

/** Zoom by `factor`, the picture point under `at` staying under it. */
export function zoomAt(view, factor, at, picture, stage) {
  const whole = fit(picture, stage)
  const scale = clamp(view.scale * factor, Math.min(whole.scale, 1) * 0.5, Math.max(whole.scale * MAX_ZOOM, 4))
  const point = toPicture(view, at)
  return held({ scale, x: at.x - point.x * scale, y: at.y - point.y * scale }, picture, stage)
}

/** Two fingers: pan with their middle, zoom with their distance. */
export function pinch(view, before, after, picture, stage) {
  const middle = pair => ({ x: (pair[0].x + pair[1].x) / 2, y: (pair[0].y + pair[1].y) / 2 })
  const apart = pair => Math.hypot(pair[0].x - pair[1].x, pair[0].y - pair[1].y) || 1
  const from = middle(before)
  const to = middle(after)
  const moved = { ...view, x: view.x + to.x - from.x, y: view.y + to.y - from.y }
  return zoomAt(moved, apart(after) / apart(before), to, picture, stage)
}
