# frontend

React 18 + Vite + Tailwind SPA, served by Flask from `frontend/dist` at `/`.

```bash
npm install
npm run dev     # dev server on :5173, proxies /api to http://127.0.0.1:5000
npm run build   # outputs to dist/
```

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

If that happens: delete `node_modules` and `package-lock.json`, then run
`npm install` again **on the target platform** — that regenerates the lockfile
with the correct optional-dependency entries. Do not hand-add a specific
`@rollup/rollup-*` package to `dependencies`; that's what breaks installs on
every other platform.
