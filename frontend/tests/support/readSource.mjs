// One reader for the source-contract tests. Twenty-nine copies of `read`
// lived in tests/*.mjs with five signatures (test-relative URLs, a frontend
// join, `../src/` prefixes, one CRLF-normalizing); every file move meant
// hand-fixing paths in each. ONE contract now: paths resolve from the
// FRONTEND ROOT ('src/...', and '../docs/...' still reaches the repo docs),
// and text is CRLF-normalized so a Windows checkout regexes the same bytes
// CI does.
import { readFileSync } from 'node:fs'

const FRONTEND = new URL('../../', import.meta.url)

export const readSource = (rel) =>
  readFileSync(new URL(rel, FRONTEND), 'utf8').replace(/\r\n/g, '\n')
