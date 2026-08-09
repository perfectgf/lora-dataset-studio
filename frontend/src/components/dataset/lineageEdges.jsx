/* ◉ The edges of a lineage — the flowing bezier connectors that read parent→child.

   Extracted from RunLineageGraph.jsx so the LoRA Canvas draws the SAME edges as
   the in-card graph rather than a lookalike: same gradients, same trunk
   brightening, same dashed-and-dimmed superseded branch, same draw-in delay.

   ⚠️ SVG gradient/filter ids are DOCUMENT-global. `LineageEdgeDefs` must be
   rendered exactly ONCE per page: the in-card graph puts it inside its own
   <svg>, while the canvas — which draws one <svg> per dataset lane — renders it
   once at page level and every lane references the same four ids. Rendering it
   per lane would be N copies of the same id in one document; `url(#…)` would
   still resolve (to the first), but it would be a lie waiting to become a bug
   the day the definitions differ. */

/** The gradients + glow filter every lineage edge paints with. Render once per
 *  document (see the warning above). */
export function LineageEdgeDefs() {
  return (
    <defs>
      {/* edges flow left→right = parent→child, so a horizontal gradient in
          the path's own box paints the direction of descent. */}
      <linearGradient id="lds-edge-normal" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stopColor="rgb(148 163 184)" stopOpacity="0.15" />
        <stop offset="1" stopColor="rgb(203 213 225)" stopOpacity="0.4" />
      </linearGradient>
      <linearGradient id="lds-edge-spine" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stopColor="#6366f1" stopOpacity="0.6" />
        <stop offset="1" stopColor="#a5b4fc" stopOpacity="0.98" />
      </linearGradient>
      {/* A superseded branch is DIMMER than the trunk, not invisible. It used to
          fade in from 12% opacity — and the half that fades in is exactly the half
          that leaves the parent, so on a dark screen at reduced brightness (a phone)
          the edge read as absent and the continuation looked unlinked. It is the
          most informative edge on the graph (the parent kept saves past this point),
          so it now starts at a legible floor and ends nearly opaque. */}
      <linearGradient id="lds-edge-super" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stopColor="#f59e0b" stopOpacity="0.55" />
        <stop offset="1" stopColor="#fbbf24" stopOpacity="0.95" />
      </linearGradient>
      {/* 🧬 GENERATION PROVENANCE — "this picture was blended from that
          checkpoint". Its own violet, deliberately not the indigo trunk and not
          the neutral grey: it is a third kind of descent (a blend loads several
          LoRAs, so one image has several parents at once) and the board already
          spends indigo on training lineage and amber on superseded branches.
          Kept dimmer at the source and brighter at the picture, like the others,
          so it still reads left→right as "came from". */}
      <linearGradient id="lds-edge-blend" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stopColor="#a855f7" stopOpacity="0.45" />
        <stop offset="1" stopColor="#e9d5ff" stopOpacity="0.95" />
      </linearGradient>
      {/* 🔌 EXTERNAL LoRA PLUGIN NODE — "this picture used a file pinned on the
          board, not one made here". Its own cyan, matching the 🔌 accent used
          elsewhere for plugin nodes, so it reads as a third source distinct from
          both the indigo trunk and the violet blend provenance. */}
      <linearGradient id="lds-edge-external" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stopColor="#22d3ee" stopOpacity="0.45" />
        <stop offset="1" stopColor="#0891b2" stopOpacity="0.95" />
      </linearGradient>
      <filter id="lds-edge-glow" x="-20%" y="-40%" width="140%" height="180%">
        <feGaussianBlur stdDeviation="2.2" result="b" />
        <feMerge>
          <feMergeNode in="b" /><feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>
  );
}

/** Every edge of one graph: the glow halo underneath the trunk, then the crisp
 *  cores on top. `isLit(id)` says whether a node is on the hovered path — an
 *  edge whose two ends are both lit is drawn like the trunk, so hovering a run
 *  traces its whole descent back to the root. Pass `() => false` for a surface
 *  with no hover story. */
export function LineageEdges({ edges, isLit }) {
  const lit = typeof isLit === 'function' ? isLit : () => false;
  return (
    <>
      {/* Glow halo underneath the trunk (root→current), so even short hops read
          as a lit ribbon. Drawn first, then the crisp cores on top. */}
      <g fill="none" strokeLinecap="round" aria-hidden>
        {edges.map((e) => {
          if (!(e.onSpine || (lit(e.parentId) && lit(e.childId)))) return null;
          // A superseded edge on the trunk gets its halo too — in its OWN amber, so
          // it still reads "this branch left saves behind" while being as findable
          // as any other trunk hop. Skipping it entirely was why the one edge users
          // most need to see (a resume from below the parent's end) was the one they
          // could not find.
          return (
            <path key={`glow-${e.parentId}-${e.childId}`}
              d={e.d}
              stroke={`url(#${e.superseded ? 'lds-edge-super' : 'lds-edge-spine'})`}
              strokeWidth="5"
              opacity="0.5" filter="url(#lds-edge-glow)" />
          );
        })}
      </g>
      <g fill="none" strokeLinecap="round">
        {edges.map((e, i) => {
          const both = lit(e.parentId) && lit(e.childId);
          const spine = e.onSpine || both;
          // 🧬 A provenance edge keeps its violet whatever the hover is doing:
          // it is not part of the training spine, so lighting it like the trunk
          // would claim a descent that did not happen. 🔌 An external-LoRA edge
          // is checked first for the same reason, in its own cyan.
          const grad = e.external ? 'lds-edge-external'
            : e.blend ? 'lds-edge-blend'
            : e.superseded ? 'lds-edge-super' : spine ? 'lds-edge-spine' : 'lds-edge-normal';
          return (
            <path key={`${e.parentId}-${e.childId}`}
              className="lds-ledge"
              d={e.d}
              stroke={`url(#${grad})`}
              strokeWidth={e.external ? 2 : e.blend ? 2 : spine ? 2.6 : e.superseded ? 2.2 : 1.5}
              /* ⚠️ `.lds-ledge` sets stroke-dasharray in CSS for the draw-in
                 animation, and a CSS declaration beats a presentation attribute:
                 this dash never actually renders. A superseded branch therefore
                 reads by its amber colour, which is why that colour has to be
                 legible on its own (see the gradient above). */
              strokeDasharray={e.superseded ? '2 4' : undefined}
              pathLength="1"
              style={{ '--draw-delay': `${Math.min(i, 10) * 60 + 120}ms` }} />
          );
        })}
      </g>
    </>
  );
}
