/** The Safelight selection mark — a grease-pencil stroke on the corner of a
 *  chosen frame, the way an editor marks a contact sheet.
 *
 *  It replaces the tint that used to sit ON the picture. A selected Bank tile
 *  wore `bg-indigo-500/30`: a 30 % accent film over the very image the user is
 *  judging, which is the one thing this theme exists to avoid — you cannot tell
 *  a warm photo from a neutral one through an orange filter. The mark says the
 *  same thing from the corner, over a dark disc that keeps it legible on a
 *  white dress or a black background alike.
 *
 *  Shared on purpose: the Bank grid and the dataset grid are two surfaces of
 *  one gesture, and a selection that looks different on each is a bug with a
 *  delay on it. */
export default function SelectionMark({ className = '' }) {
  return (
    <span aria-hidden="true"
      className={`pointer-events-none absolute left-1 top-1 grid h-6 w-6 place-items-center rounded-full bg-app/70 ${className}`}>
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none"
        stroke="rgb(var(--accent))" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round">
        {/* One unbroken stroke, drawn as a hand would: down into the valley,
            up through the tick, and round past the start — the overshoot is
            what makes it read as a pencil rather than a checkbox glyph. */}
        <path d="M5.5 12.4 L10 17.2 L18.8 6.4" />
        <path d="M18.8 6.4 C15.4 8.2 12.6 11.4 11.2 14.6" opacity="0.55" />
      </svg>
    </span>
  );
}
