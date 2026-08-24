// The one download-size formatter of the Setup cards. Four cards carried
// byte-identical copies (measured 2026-08-24); the CONTRACT is theirs
// alone - 2-decimal GB, whole MB, KB floored at 0 - and deliberately NOT
// shared with the other byte formatters: each of those carries its own
// load-bearing wording ('-', 'empty', '0 KB', kB vs KB, one even base
// 1024), and unifying them would silently change rendered strings.
export function fmtSize(b) {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`
  if (b >= 1e6) return `${(b / 1e6).toFixed(0)} MB`
  return `${Math.max(0, Math.round(b / 1e3))} KB`
}
