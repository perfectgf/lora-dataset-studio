/* Pinokio update script.

   `--ff-only` on purpose: it is the exact command the in-app updater runs, so
   whichever surface the user reaches for, the checkout moves the same way and a
   local edit is reported as a refusal instead of being buried under a merge
   commit. The pip run afterwards is what picks up a new or bumped dependency;
   it is a no-op when nothing changed.

   The launcher's Update tab is offered only while the server is stopped (see
   pinokio.js) — updating a running tree then restarting from Pinokio is the
   clean order.

   A checkout the app moved onto the Patreon channel is AHEAD of the public
   branch, so the public pull may refuse to fast-forward. The fallback only
   accepts that refusal when the Patreon marker the app writes is part of HEAD;
   the requirements step then still installs what that release needs.

   Between the two, scripts/pinokio_env_floor.js moves aside an env/ built on
   Pinokio's bundled Python 3.10, which this version cannot run: Pinokio reuses
   an existing env/ as it is, and `venv_python` only applies when it creates one,
   so the requirements step then rebuilds env/ on Python 3.12. */
module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull --ff-only || git merge-base --is-ancestor refs/lds/patreon/installed HEAD"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "node scripts/pinokio_env_floor.js"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.12",
        message: "uv pip install -r backend/requirements.txt"
      }
    },
    {
      method: "notify",
      params: {
        html: "Up to date. Click <b>Start</b> to run the new version."
      }
    }
  ]
}
