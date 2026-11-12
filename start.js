/* Pinokio start script — runs the Flask server as a Pinokio daemon.

   The URL is READ BACK from the terminal instead of being hardcoded: port 5050
   is a default, not a promise (backend/run.py advances to the next free port
   when something else already holds it), so `local.set url` uses whatever
   Werkzeug reports it bound. That is what fills the "Open Web UI" tab.

   Environment:
   - LDS_PORT=5050    same default as start.bat (5000 collides constantly).
   - LDS_NO_REEXEC=1  backend/run.py re-execs itself into a `.venv/` sitting
                      next to it when it finds one. Under Pinokio the pinned
                      interpreter is the `env/` venv Pinokio just activated, so
                      a `.venv/` left behind by an earlier start.bat run in the
                      same folder must not hijack the launch.
   - LDS_RUNTIME=pinokio  tells the app that START AND STOP belong to Pinokio.
                      Its Settings > Updates card then hands the update back
                      here (Stop / Update / Start) instead of offering its own
                      "Update & restart", which would relaunch the server in a
                      detached process this launcher no longer tracks.
   - LDS_OPEN_BROWSER is intentionally NOT set: Pinokio opens the app itself.

   `requires: { bundle: "python" }` asks Pinokio to make sure its own toolkit is
   there before running us: conda, git, uv, node. Pinokio 8 reads
   `requires.bundle`, and when a requirement is missing it routes to
   `/setup/python` instead of letting a command fail on a half-built machine. */
module.exports = {
  daemon: true,
  requires: {
    bundle: "python"
  },
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.12",
        env: {
          PYTHONUTF8: "1",
          LDS_PORT: "5050",
          LDS_NO_REEXEC: "1",
          LDS_RUNTIME: "pinokio"
        },
        message: [
          "python backend/run.py"
        ],
        on: [{
          // backend/run.py prints "[LDS] Ready on http://127.0.0.1:5050/" once
          // /api/health answers. NOT Werkzeug's " * Running on ..." banner:
          // create_app routes the root logger into data/app.log, so that banner
          // never reaches the terminal and a launcher waiting on it hangs.
          event: "/\\[LDS\\] Ready on (http:\\/\\/\\S+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
