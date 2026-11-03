// Members previews in the screens of an installation without an active, linked
// Patreon membership: a "Members options" block (or Test Studio rail section),
// a one-line mention, or a toolbar button for a members tool. Each preview
// shows its name, one sentence and a "Members" badge; one "Join on Patreon"
// link per block. Nothing else: no route is called and no members feature
// opens. memberPreviews.js holds the format and the checks.
import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { ExternalLink } from 'lucide-react';
import StudioSection from '../dataset/studio/StudioSection.jsx';
import { loadMemberPreviews, memberPreviewsSnapshot, previewsAt, subscribeMemberPreviews } from './memberPreviews.js';

const BADGE = 'inline-flex items-center rounded-full border border-primary/30 bg-primary/10 px-1.5 py-px '
  + 'text-[0.5625rem] font-medium leading-tight text-primary';
const JOIN = 'inline-flex items-center gap-1 rounded border border-border bg-surface px-2 py-0.5 '
  + 'text-[0.625rem] text-content-subtle no-underline hover:text-content';
const TOOL = 'inline-flex items-center gap-1 px-2 py-0.5 rounded border border-border bg-surface '
  + 'text-content-subtle text-[0.625rem] hover:text-content';
const LABEL = 'text-content-muted text-[0.625rem] uppercase tracking-wide font-semibold';

export function useMemberPreviews(placements) {
  const previews = useSyncExternalStore(subscribeMemberPreviews, memberPreviewsSnapshot, memberPreviewsSnapshot);
  useEffect(() => { loadMemberPreviews(); }, []);
  const key = placements.join('|');
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => previewsAt(previews.items, placements), [previews, key]);
}

export function MembersBadge() {
  return <span className={BADGE}>Members</span>;
}

function JoinOnPatreon({ items }) {
  return (
    <a href={items[0].url} target="_blank" rel="noopener noreferrer" className={JOIN}
      aria-label={`Join on Patreon to use ${items.map((item) => item.name).join(', ')}`}>
      Join on Patreon<ExternalLink aria-hidden="true" className="h-3 w-3" />
    </a>
  );
}

function Preview({ item }) {
  return (
    <li className="flex flex-col items-start gap-1" data-member-preview={item.id}>
      <span className="flex flex-wrap items-center gap-1.5 text-[0.75rem] font-medium text-content">
        {item.name}<MembersBadge />
      </span>
      <span className="text-[0.6875rem] leading-snug text-content-muted">{item.summary}</span>
    </li>
  );
}

/** The members previews of one place: `block` (a bordered "Members options"
 *  block), `section` (a Test Studio rail section) or `line` (one line naming
 *  them, each sentence in its tooltip). One Join link per block. */
export function MemberOptions({ placements, variant = 'block', storageKey }) {
  const items = useMemberPreviews(placements);
  if (!items.length) return null;
  if (variant === 'line') {
    // Running text, so a long sentence wraps like a sentence instead of stacking rows.
    return (
      <p data-member-options className="m-0 w-full text-[0.6875rem] leading-relaxed text-content-muted">
        <span className="mr-1.5 inline-block align-middle"><MembersBadge /></span>
        {items.map((item, index) => (
          <span key={`${item.placement}:${item.id}`} data-member-preview={item.id} title={item.summary}
            className="font-medium text-content">
            {index > 0 && <span aria-hidden="true" className="font-normal text-content-subtle"> · </span>}
            {item.name}
          </span>
        ))}
        {items.length === 1 && <span> — {items[0].summary}</span>}
        <span className="ml-1.5 inline-block align-middle"><JoinOnPatreon items={items} /></span>
      </p>
    );
  }
  const body = (
    <>
      <ul className="m-0 flex list-none flex-col gap-2.5 p-0">
        {items.map((item) => <Preview key={`${item.placement}:${item.id}`} item={item} />)}
      </ul>
      <span><JoinOnPatreon items={items} /></span>
    </>
  );
  if (variant === 'section') {
    return <StudioSection title="Members options" storageKey={storageKey} anchorId="st-members">{body}</StudioSection>;
  }
  return (
    <section aria-label="Members options" data-member-options
      className="flex flex-col gap-2.5 rounded-lg border border-border bg-surface px-3 py-2">
      <p className={`m-0 ${LABEL}`}>Members options</p>
      {body}
    </section>
  );
}

/** A members tool in a toolbar: the button opens its preview, which links to Patreon. */
export function MemberPreviewButton({ placement }) {
  const [item] = useMemberPreviews([placement]);
  const [open, setOpen] = useState(false);
  const wrap = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    const onDown = (e) => { if (!wrap.current?.contains(e.target)) setOpen(false); };
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onDown);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onDown);
    };
  }, [open]);
  if (!item) return null;
  const panelId = `member-preview-${item.id.replace(/\W+/g, '-')}`;
  return (
    <span ref={wrap} className="relative inline-flex">
      <button type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open} aria-controls={panelId}
        className={TOOL}>
        {item.name}<MembersBadge />
      </button>
      {open && (
        <div id={panelId} data-member-preview={item.id}
          className="absolute right-0 top-full z-30 mt-1 flex w-64 max-w-[calc(100vw-2rem)] flex-col items-start gap-1.5 rounded-lg border border-border bg-surface-raised p-3 text-left shadow-xl">
          <span className="flex flex-wrap items-center gap-1.5 text-[0.75rem] font-medium text-content">
            {item.name}<MembersBadge />
          </span>
          <span className="text-[0.6875rem] leading-snug text-content-muted">{item.summary}</span>
          <JoinOnPatreon items={[item]} />
        </div>
      )}
    </span>
  );
}
