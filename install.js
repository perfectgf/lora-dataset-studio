/* Pinokio install script — see pinokio.js for why this runs in place.

   Deliberately small. The core app is a Flask server plus a frontend that is
   committed already built (frontend/dist), so a working install is exactly one
   pip run: no torch, no node build, nothing GPU-specific. Everything heavy is
   OPTIONAL and installed later from the app's own Setup screen, which knows
   what this machine actually has — pinning that guesswork here would download
   gigabytes for users who only import images.

   The venv is created by Pinokio at `env/` on Python 3.12 (`venv_python`: uv
   creates it, downloading 3.12 if needed). The core needs 3.11 or newer
   (Pillow 13, filelock 4.2), so Pinokio's bundled 3.10 can no longer be used,
   and 3.12 is inside the window the optional ML extras publish wheels for, so
   the in-app Setup installs land in a supported interpreter.

   `requires: { bundle: "python" }` names the Pinokio toolkit this install needs
   — conda, git, uv, node — so a machine whose bin folder is missing one of them
   is routed to `/setup/python` rather than failing inside `uv pip install`.
   "python" and not "ai": we need no torch, no CUDA and no Visual Studio Build
   Tools, and asking for them would turn a two-minute install into a very long
   one for users who only import images. */
module.exports = {
  requires: {
    bundle: "python"
  },
  run: [
    {
      // An env/ left by an earlier install on Pinokio's Python 3.10 is reused as
      // it is by the step below; move it aside so that step builds one on 3.12.
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
        message: [
          "uv pip install -r backend/requirements.txt"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "Installed. Click <b>Start</b> — then use the app's <b>Setup</b> screen to connect ComfyUI, Ollama or the optional ML helpers."
      }
    }
  ]
}
