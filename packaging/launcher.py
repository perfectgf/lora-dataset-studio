"""LoRA Dataset Studio - portable launcher.

Double-clicked by end users. Starts the bundled standalone Python running the Flask
server (no console window), waits until it answers, opens the browser, and shows a
tiny status window with Open / Quit. Everything writable (config.json, .env, the
datasets) lives under data/ next to this launcher, so the bundle stays fully portable.

Frozen with PyInstaller (--noconsole) into "LoRA Dataset Studio.exe" at the bundle root.
The APP runs under python/python.exe, which HAS pip -- that is the whole reason we ship
a real standalone Python instead of one frozen single-exe: the in-app Setup wizard's
`pip install -r backend/requirements-ml.txt` (face scoring, masks) keeps working.

Bundle layout the launcher expects (mirrors the repo so backend/config.py's
REPO_ROOT/FRONTEND_DIST resolve unchanged):

    LoRA Dataset Studio.exe   <- this, frozen
    python/python.exe         <- standalone CPython + core deps (has pip)
    backend/run.py            <- Flask entrypoint
    frontend/dist/            <- prebuilt UI
    data/                     <- created on first run (config.json, .env, datasets)
"""
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

APP_NAME = "LoRA Dataset Studio"
PREFERRED_PORT = 5050          # matches start.bat; only changed if already taken
CREATE_NO_WINDOW = 0x08000000  # Windows: no console window for the child server


def bundle_dir() -> Path:
    """Frozen: the exe sits at the bundle root. Dev (python packaging/launcher.py):
    the repo root is one level up from packaging/."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0   # nothing listening -> free


def pick_port() -> int:
    """Prefer 5050; if it's taken (another Flask app, a previous instance), let the OS
    hand out a free one so two people double-clicking never collide."""
    if _port_free(PREFERRED_PORT):
        return PREFERRED_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def python_exe(bundle: Path) -> Path:
    """The bundled standalone interpreter (python/python.exe on Windows; python/bin/python
    elsewhere, so the launcher can also be smoke-tested on a dev machine)."""
    win = bundle / "python" / "python.exe"
    return win if win.exists() else bundle / "python" / "bin" / "python"


def start_server(bundle: Path, port: int) -> subprocess.Popen:
    data = bundle / "data"
    data.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Keep every writable file under data/ so the bundle is portable (nothing written
    # next to the code, nothing in %APPDATA%). These overrides are read by config.py.
    env["LDS_CONFIG"] = str(data / "config.json")
    env["LDS_DATA_DIR"] = str(data)
    env["LDS_ENV"] = str(data / ".env")
    env["LDS_HOST"] = "127.0.0.1"
    env["LDS_PORT"] = str(port)
    flags = CREATE_NO_WINDOW if os.name == "nt" else 0
    log = open(data / "server.log", "ab", buffering=0)   # keep the server's own diagnostics
    return subprocess.Popen(
        [str(python_exe(bundle)), str(bundle / "backend" / "run.py")],
        cwd=str(bundle), env=env, stdout=log, stderr=log, creationflags=flags,
    )


def wait_until_up(health_url: str, proc: subprocess.Popen, timeout: float = 90.0) -> bool:
    """Poll /api/health until 200, or the server process dies, or we time out. First
    launch can be slow (SQLite init + additive migrations), hence the generous window."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:      # server exited during startup -> give up
            return False
        try:
            with urllib.request.urlopen(health_url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    import tkinter as tk
    from tkinter import ttk

    bundle = bundle_dir()
    py = python_exe(bundle)
    if not py.exists():
        _fatal(f"Bundled Python not found at {py}.\nThe download may be incomplete — "
               "re-extract the .zip.")
        return 1

    port = pick_port()
    url = f"http://127.0.0.1:{port}/"
    health = f"http://127.0.0.1:{port}/api/health"
    proc = start_server(bundle, port)

    root = tk.Tk()
    root.title(APP_NAME)
    root.resizable(False, False)
    ico = bundle / "icon.ico"
    if ico.exists():
        try:
            root.iconbitmap(str(ico))
        except Exception:
            pass

    frame = ttk.Frame(root, padding=20)
    frame.grid()
    ttk.Label(frame, text="🧬 " + APP_NAME, font=("Segoe UI", 12, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(0, 8))
    status = tk.StringVar(value="Starting the server…")
    ttk.Label(frame, textvariable=status, justify="center", font=("Segoe UI", 10)).grid(
        row=1, column=0, columnspan=2, pady=(0, 14))

    open_btn = ttk.Button(frame, text="Open", state="disabled",
                          command=lambda: webbrowser.open(url))
    open_btn.grid(row=2, column=0, padx=4, ipadx=10)

    def on_quit():
        try:
            proc.terminate()
        except Exception:
            pass
        root.destroy()

    ttk.Button(frame, text="Quit", command=on_quit).grid(row=2, column=1, padx=4, ipadx=10)
    root.protocol("WM_DELETE_WINDOW", on_quit)

    def ready_or_failed():
        up = wait_until_up(health, proc)

        def apply():
            if up:
                status.set(f"✅ Running\n{url}")
                open_btn.state(["!disabled"])
                webbrowser.open(url)
            else:
                status.set("⚠️ The server failed to start.\nSee data\\server.log for details.")
        try:
            root.after(0, apply)
        except Exception:
            pass

    threading.Thread(target=ready_or_failed, daemon=True).start()
    root.mainloop()
    return 0


def _fatal(message: str) -> None:
    """Best-effort error dialog when we can't even reach the Tk UI path."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        r = tk.Tk()
        r.withdraw()
        messagebox.showerror(APP_NAME, message)
        r.destroy()
    except Exception:
        sys.stderr.write(message + "\n")


if __name__ == "__main__":
    sys.exit(main())
