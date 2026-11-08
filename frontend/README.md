# frontend

React 19 + Vite + Tailwind SPA, served by Flask from `frontend/dist` at `/`.

```bash
npm install
npm run dev     # dev server on :5173, proxies /api to http://127.0.0.1:5050
npm run build   # outputs to dist/
```

## ⚠ `npm run dev` talks to a REAL backend

The dev server proxies `/api` to **`http://127.0.0.1:5050`** — which is where the
app you actually use is listening. Writes go through: deletions, imports,
training launches all land in real data, and nothing on screen says which backend
answered.

Point it at a throwaway instance with **`LDS_DEV_API_TARGET`**, in the shell or in
`frontend/.env.local` (gitignored):

```bash
LDS_DEV_API_TARGET=http://127.0.0.1:5051 npm run dev
```

Start that instance with its own data directory so it cannot touch yours:

```bash
LDS_PORT=5051 LDS_DATA_DIR=/tmp/lds-dev python backend/run.py
```

The default is unchanged on purpose — it is what most people want — so this is
opt-in, not a new required step. (`LDS_`, not `VITE_`: `VITE_*` variables are
inlined into the client bundle, and a dev-server setting has no business shipping
in built output.)

## Rollup optional-dependency gotcha

`package.json` deliberately does **not** list a platform-specific Rollup
binary (e.g. `@rollup/rollup-win32-x64-msvc`, `@rollup/rollup-linux-x64-gnu`).
npm resolves these as `optionalDependencies` of `rollup` itself and picks the
right one for the current OS/arch at install time — but a well-known npm bug
(https://github.com/npm/cli/issues/4828) can make `package-lock.json`
"remember" the platform it was generated on, so `npm install` on a different
platform fails with something like:

```
Error: Cannot find module @rollup/rollup-linux-x64-gnu
```

If this occurs, inspect the missing package and the lockfile for the target
platform. Reuse an existing compatible environment when it is sufficient for the
task. Do not delete shared `node_modules` or regenerate `package-lock.json` as an
automatic verification step for an unrelated change.

Changing the lockfile belongs to a dependency repair with its own reviewed diff.
If an isolated reinstall is necessary, preserve the shared installation and
other active work. Do not add an OS-specific `@rollup/rollup-*` package to the
application's direct dependencies.
