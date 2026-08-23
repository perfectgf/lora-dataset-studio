# Extensions guide

[← Documentation index](../README.md) · [Settings reference](settings-reference.md) · [Security policy](../../SECURITY.md)

Optional local packages dropped into `backend/extensions/` load when the app starts. Each one can register its own API routes and mount its own piece of UI. On a normal install the folder does not exist and nothing here runs.

## The trust model, stated plainly

An extension is **code you place on your own machine, in the app's own process**. It can do anything the app can do — read your datasets, your keys, your GPU. That is the same trust you already extend to the app itself and to any Python package you install; the loader adds no sandbox and pretends none. Three properties keep the mechanism honest:

- **Nothing loads that you did not put there.** The loader only scans the local directory; there is no marketplace, no download, no remote fetch.
- **Extensions load behind the access-token gate.** The network guard installs first, so a request the gate refuses is refused before any extension code sees it — on a LAN bind or an `LDS_PUBLIC=1` pod alike.
- **The directory can never ship.** `backend/extensions/` is gitignored, excluded from release bundles by the packaging script, and refused by the release-artifact checker if it ever reaches an archive — three separate locks, each pinned by its own test.

`LDS_EXTENSIONS=0` disables the loader entirely.

## Writing one

An extension is an ordinary Python package:

```text
backend/extensions/
  my_extension/
    __init__.py
```

`__init__.py` must expose `register(app, csrf)`, and may expose two optional attributes:

```python
from flask import Blueprint, jsonify

__version__ = '0.1.0'                       # shown in the manifest
FRONTEND_ENTRY = '/api/my-extension/ui.js'  # a module script the UI mounts

bp = Blueprint('my_extension', __name__, url_prefix='/api/my-extension')

@bp.get('/ping')
def ping():
    return jsonify({'ok': True})

def register(app, csrf):
    app.register_blueprint(bp)
```

What happens at boot:

- Every package under the directory (sorted by name) is imported and `register(app, csrf)` is called.
- A broken extension is skipped with a logged traceback — it never takes the app down.
- `GET /api/extensions/` answers the manifest of what loaded: `{"extensions": [{"name", "version", "frontend_entry"}]}`.
- The frontend fetches that manifest once at startup and appends one `<script type="module">` per extension that declared a `FRONTEND_ENTRY`. Serve that file from one of your own routes.

## Boundaries

- The app's own tests never load your extensions (the suite points the loader at an empty directory).
- Docker images built from your working tree would include the folder — images are built from clean checkouts in CI and never pushed, but keep it in mind if you build and publish your own.
- There is no versioned API contract yet: extensions reach into the same internals the app uses, and those internals move. Pin the app version you develop against.
