# frontend

React 18 + Vite + Tailwind SPA, served by Flask from `frontend/dist` at `/`.

```bash
npm install
npm run dev     # dev server on :5173, proxies /api to http://127.0.0.1:5000
npm run build   # outputs to dist/
```

## Interface languages

The frontend ships with English and Simplified Chinese. The language switcher
in the main navigation stores the choice in `localStorage` under `lds.locale`;
new browsers default to Simplified Chinese when the browser locale starts with
`zh`, otherwise English.

Language packs live in:

- `src/i18n/locales/en.json`
- `src/i18n/locales/zh-CN.json`

UI code reads messages through `useI18n().t('message.key', values)`. English is
the fallback catalog, so an incomplete additional language pack never renders
an empty label. Keep placeholders such as `{name}` and `{count}` identical
between catalogs.

### Adding another language

Adding a language is data-only: create `src/i18n/locales/<locale>.json`. Vite
discovers every JSON file in that directory automatically; no React import or
switcher edit is needed. Start the file with:

```json
{
  "_meta": {
    "locale": "fr",
    "label": "Français",
    "shortLabel": "FR",
    "documentTitle": "LoRA Dataset Studio",
    "direction": "ltr",
    "order": 30,
    "defaultForLanguage": true
  }
}
```

`label` is shown in the language menu. `direction` can be `ltr` or `rtl`.
`order` controls menu order. When several regional packs share a base language
(for example `pt-BR` and `pt-PT`), set `defaultForLanguage` on the one that
should handle a browser reporting only `pt`. All metadata fields except the
filename-derived locale code are optional.

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
