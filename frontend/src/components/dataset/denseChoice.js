/*
 * The full-model base picker lives in the Cloud training plugin and only moves
 * the Training panel's state, so a base picked and not launched was gone after a
 * reload. The panel saves each choice itself, through the setters it hands the
 * plugin, which works with every version of the plugin.
 *
 * Kept in plain .js (no JSX) so `node --test` can exercise the decision itself.
 */
export const DENSE_PATH_PAUSE_MS = 800;

// The server only checks an ABSOLUTE path: a half-typed relative name would be
// stored as the base. So a typed path is saved once it reads as a complete file.
const COMPLETE_PATH = /^(?:[a-z]:[\\/]|[\\/]).*\.safetensors$/i;

/** When a choice is saved: 'now', 'pause' (a typed path, once the typing stops)
 * or null ("Custom weights…" without a complete path yet). */
export function denseChoiceTiming({ baseModel, customBase }) {
  if (!customBase) return 'now';
  return COMPLETE_PATH.test(String(baseModel ?? '').trim()) ? 'pause' : null;
}

/** One request at a time and the LAST choice wins: a quick Raw → Turbo → Raw
 * must reach the server in the order it was clicked. */
export function denseChoiceSaver(pauseMs = DENSE_PATH_PAUSE_MS) {
  let timer = null;
  let next = null;
  let running = false;
  const drain = async () => {
    if (running) return;
    running = true;
    try {
      while (next) {
        const job = next;
        next = null;
        await job();
      }
    } finally {
      running = false;
    }
  };
  const send = (job) => {
    next = job;
    return drain();
  };
  return {
    now(job) {
      clearTimeout(timer);
      return send(job);
    },
    afterPause(job) {
      clearTimeout(timer);
      timer = setTimeout(() => send(job), pauseMs);
    },
    drop() {
      clearTimeout(timer);
    },
  };
}

/** The three setters the plugin's picker receives. The plugin picks "Custom
 * weights…" with setCustomBase(true) then setBase(''), inside one event: `customRef`
 * carries the new mode to setBase before React renders it. `persist(selection,
 * { quiet })` saves one choice; a path still being typed fails quietly. */
export function densePickerSetters({
  variant, base, customRef, setVariant, setBase, setCustomBase, saver, persist,
}) {
  const choose = (choice) => {
    const selection = { variant, baseModel: base, ...choice };
    const timing = denseChoiceTiming({ ...selection, customBase: customRef.current });
    if (!timing) {
      saver.drop();
      return;
    }
    const job = () => persist(selection, { quiet: timing === 'pause' });
    if (timing === 'now') saver.now(job);
    else saver.afterPause(job);
  };
  return {
    setVariant: (next) => {
      setVariant(next);
      choose({ variant: next });
    },
    setBase: (next) => {
      setBase(next);
      choose({ baseModel: next });
    },
    setCustomBase: (next) => {
      customRef.current = next;
      setCustomBase(next);
    },
  };
}
