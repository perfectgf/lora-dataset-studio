import { markdownHeadingId } from './headingId.js'

/** The H2 section for `anchor`, heading included, up to the next H2 — computed
 * with the SAME markdownHeadingId the Guide uses, so the modal and the Guide
 * can never disagree on where a section starts. Fenced code blocks are opaque:
 * a `## ` inside one is code, not a heading (same rule as guideTextIndex).
 * '' when the anchor is not found — the caller says "this section moved"
 * rather than rendering somebody else's section. */
export function sliceGuideSection(md, anchor) {
  const lines = String(md || '').split('\n')
  let start = -1
  let inFence = false
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]
    if (/^```/.test(line)) inFence = !inFence
    if (inFence || !/^##\s+/.test(line)) continue
    const id = markdownHeadingId(line.replace(/^##\s+/, ''))
    if (start >= 0) return lines.slice(start, i).join('\n').trim()
    if (id === anchor) start = i
  }
  return start >= 0 ? lines.slice(start).join('\n').trim() : ''
}
