/* 🔌 Is this checkpoint DEPLOYED — i.e. can I generate from it right now?

   The pills under a run card (250, 500, … 3500) already knew the answer and
   never said it: `testable` rides on every pill and only surfaced as the words
   "· to deploy" inside the generation panel — AFTER the checkpoints were picked.
   The question is asked before picking, so the answer has to be on the pill.

   ⚠️ Two orthogonal dimensions live on that same 60×20 rectangle and both have
   to stay readable at once:

     • DEPLOYED / NOT DEPLOYED — a system fact, persistent. New here.
     • PICKED / NOT PICKED — the user's transient choice, already drawn as an
       indigo corner checkbox plus an indigo ring.

   Plus two meanings the shell already carries: EMERALD = this is the run's final
   checkpoint, DASHED = the file is gone from the disk. So deployment gets a
   channel of its own that touches none of them: a dedicated corner badge on the
   opposite corner from the checkbox, ROUND where the checkbox is square, in a
   colour used nowhere else on a pill.

   And never colour alone — roughly one man in twelve reads red and green the
   same, and the theme is dark graphite. Each state therefore carries THREE
   signals: a colour, a shape (filled disc vs hollow ring), and words in the
   pill's own title/aria text.

   Vocabulary, and it matters: NOT deployed does not mean missing. The file is on
   the disk (the card says "14 on disk"); it is simply not copied into ComfyUI
   yet, so nothing can generate from it until it is — which the board's own
   Generate button offers to do. Nothing here may read as "lost". */

/** 'deployed' | 'on-disk' | 'gone'. `gone` is the file the disk no longer has:
 *  it is not a deployment question at all, and gets no badge. */
export function deployState(pill) {
  if (!pill || pill.present === false) return 'gone';
  return pill.testable === true ? 'deployed' : 'on-disk';
}

/**
 * How to draw one state. `glyph` is the second channel (filled vs hollow, a
 * shape difference that survives any colour vision); `tone` names the palette
 * slot the component maps to a class; `text` is the third channel and goes into
 * the pill's title and aria-label, in words.
 */
export function deployBadge(state) {
  if (state === 'deployed') {
    return {
      show: true, tone: 'deployed', glyph: '●',
      short: 'Deployed',
      text: 'deployed to ComfyUI — you can generate from this checkpoint now',
    };
  }
  if (state === 'on-disk') {
    return {
      show: true, tone: 'on-disk', glyph: '○',
      short: 'Not deployed',
      // Says what to do about it, and names the button that does it — the board
      // already deploys the picks that need it, and a second mechanism would be
      // a second thing to keep true.
      text: 'on disk but not deployed to ComfyUI — tick it and 🎨 Generate will '
        + 'deploy it first',
    };
  }
  return { show: false, tone: null, glyph: '', short: '', text: '' };
}

/* How the state is DRAWN on a pill: a bar down its left edge.
   The edge, not a corner dot, because the corners are taken (the pick checkbox
   hangs off the top-left) and because a full-height bar stays visible when the
   board is zoomed out to fit a dozen lanes, where a 6-px dot is gone.
   Solid vs DASHED is the shape channel — it survives a greyscale screenshot and
   colour-blind vision alike; sky vs slate is the colour channel, and neither is
   a colour the pill already uses (emerald = final save, indigo = picked or
   resumed-from, amber = warning). */
export const DEPLOY_BAR_CLASS = {
  deployed: 'border-l-[3px] border-solid border-sky-400',
  'on-disk': 'border-l-[3px] border-dashed border-slate-500',
};

/** The words appended to a pill's tooltip. Empty for a checkpoint whose file is
 *  gone — that pill already says its own, more urgent thing. */
export function deployTitleSuffix(pill) {
  const badge = deployBadge(deployState(pill));
  return badge.show ? ` — ${badge.text}` : '';
}

/**
 * The legend the board needs next to the colours. A colour with no key is a
 * guess; this is one short line, in the same hint strip the other gestures are
 * explained in.
 */
/* 📱 `short` is the SAME key with the explanation clipped off. The full line is
   two sentences wide and, in a phone's board toolbar, it took a whole 40-px row
   of its own — a row spent above the board on every load. The colour still has
   a key at 400 px, which is the contract; what it loses is the half of the
   sentence the ☝ Gestures sheet gives back in full, one tap away. */
export const DEPLOY_LEGEND = [
  { tone: 'deployed', glyph: '●', short: 'deployed', label: 'deployed — ready to generate' },
  { tone: 'on-disk', glyph: '○', short: 'on disk', label: 'on disk only — Generate deploys it first' },
];

/**
 * One lane's deployment tally, for a summary that does not require counting
 * dots. Only pills whose file still exists are counted: a gone checkpoint is
 * neither deployed nor waiting to be.
 */
export function deployTally(pills) {
  let deployed = 0;
  let onDisk = 0;
  for (const p of (pills || [])) {
    const s = deployState(p);
    if (s === 'deployed') deployed += 1;
    else if (s === 'on-disk') onDisk += 1;
  }
  return { deployed, onDisk, total: deployed + onDisk };
}
