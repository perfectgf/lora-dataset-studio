/* Pinokio update step: retire an env/ built on a Python this version cannot run.

   Until now Pinokio created env/ from its bundled Python 3.10. This version's
   requirements (Pillow 13, filelock 4.2) need Python 3.11 or newer, and Pinokio
   reuses an existing env/ as it is: `venv_python` only applies when it creates
   one. So an env/ older than 3.11 is moved aside here, in one rename, and the
   next step recreates it on Python 3.12 and installs into a fresh environment.
   A rename moves the whole folder or nothing: never a half-deleted env/. Only
   env/ is touched -- data/, config.json and the Image Bank live beside it.

   Usage: node scripts/pinokio_env_floor.js [install folder]   (default: the
   folder above this script). Exit 0: nothing to do, or env/ was moved aside.
   Exit 1: env/ is too old and could not be moved; nothing was changed. */
const fs = require('fs')
const path = require('path')

const MINIMUM = [3, 11]

function envVersion(env) {
  let text
  try {
    text = fs.readFileSync(path.join(env, 'pyvenv.cfg'), 'utf8')
  } catch {
    return null
  }
  // python -m venv writes `version = 3.10.11`; uv writes `version_info = 3.12.7`.
  const match = /^\s*version(?:_info)?\s*=\s*(\d+)\.(\d+)/m.exec(text)
  return match ? [Number(match[1]), Number(match[2])] : null
}

function older(version) {
  return version[0] < MINIMUM[0] || (version[0] === MINIMUM[0] && version[1] < MINIMUM[1])
}

function main(root) {
  const env = path.join(root, 'env')
  const version = envVersion(env)
  if (!version || !older(version)) {
    console.log(version ? `[LDS] env/ runs Python ${version.join('.')}: kept.` : '[LDS] no env/ yet: kept as is.')
    return 0
  }
  const aside = path.join(root, `env-python${version.join('.')}-retired`)
  try {
    fs.rmSync(aside, { recursive: true, force: true })
    fs.renameSync(env, aside)
  } catch (error) {
    console.log(`[LDS] env/ runs Python ${version.join('.')}, which this version no longer supports, and could ` +
                `not be moved aside (${error.code || error.message}). Nothing was changed. Stop the app, then ` +
                'click Update again.')
    return 1
  }
  console.log(`[LDS] env/ ran Python ${version.join('.')}; this version needs ${MINIMUM.join('.')} or newer. ` +
              'It was moved aside and is rebuilt on Python 3.12 next.')
  try {
    fs.rmSync(aside, { recursive: true, force: true, maxRetries: 3 })
  } catch {
    // Only disk space is at stake: the retired copy is removed again next time.
  }
  return 0
}

if (require.main === module) {
  process.exitCode = main(path.resolve(process.argv[2] || path.join(__dirname, '..')))
}

module.exports = { envVersion, older, main }
