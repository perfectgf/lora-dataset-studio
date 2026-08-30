/** ⓘ Which guide section explains each pass button — the map behind the little
 * "i" on the video bank's pass row.
 *
 * One entry PER BUTTON, deliberately: the request behind this ("what does Safe
 * zone even do?") came from the button row, so no button may lack its answer.
 * Passes with no section of their own (scan, thumbnails, the pipeline button)
 * open the row's overview section rather than nothing.
 *
 * PURE — imported by a node test that checks every id resolves in the help
 * registry, because these strings escape the source-scan the help contract
 * test runs (that scan only sees literal quoted topic attributes in JSX).
 */
export const VIDEO_PASS_TOPICS = {
  pipeline: 'video-bank-passes',
  probe: 'video-bank-passes',
  detect: 'video-shot-threshold',
  thumbs: 'video-bank-passes',
  measure: 'video-quality-cuts',
  embed: 'video-bank-search',
  caption: 'video-captions',
  dedup: 'video-duplicate-shots',
  watermark: 'video-watermark-flag',
  safezone: 'video-safe-zone',
  defects: 'video-defect-sweep',
  camera: 'video-camera-motion',
  aicheck: 'video-ai-check',
  promote: 'video-promote-target',
}
