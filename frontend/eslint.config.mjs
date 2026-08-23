// ESLint — the frontend's lint gate (`npm run lint` from frontend/).
//
// Scope on purpose: correctness and dead code, not style. What is enabled
// here is what cannot be intended — a variable that is assigned and never
// read, a reference to a name that does not exist, a hook called under a
// condition, an object literal writing the same key twice — plus the
// exhaustive-deps rule, kept as a WARNING because a stale closure is a bug
// only sometimes, and a fix there changes behaviour. Formatting and naming
// stay out: a clean `npm run build` plus `node --test` are the bar for
// behaviour, this file only catches what those two cannot see.
//
// CI runs exactly `npm run lint`; the versions are pinned in package.json.

import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import react from "eslint-plugin-react";

export default [
  {
    ignores: ["dist/**", "node_modules/**"],
  },
  {
    files: ["**/*.{js,jsx,mjs}"],
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: "module",
      parserOptions: { ecmaFeatures: { jsx: true } },
      globals: { ...globals.browser, ...globals.node, ...globals.es2021 },
    },
    plugins: { "react-hooks": reactHooks, react },
    settings: { react: { version: "19.0" } },
    linterOptions: {
      // A disable comment for a rule nobody enables is a stale note, not a
      // suppression: report it so it gets cleaned up rather than trusted.
      reportUnusedDisableDirectives: "warn",
    },
    rules: {
      "no-unused-vars": ["error", {
        args: "after-used",
        ignoreRestSiblings: true,
        varsIgnorePattern: "^_",
        argsIgnorePattern: "^_",
        caughtErrors: "none",
      }],
      "no-undef": "error",
      "no-unreachable": "error",
      "no-dupe-keys": "error",
      "no-duplicate-case": "error",
      "no-empty": "warn",
      "no-constant-condition": "warn",
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      "react/jsx-uses-vars": "error",
      "react/jsx-uses-react": "off",
      "react/jsx-no-undef": "error",
      "react/jsx-key": "warn",
    },
  },
];
