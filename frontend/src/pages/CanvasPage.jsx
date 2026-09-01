import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiFetch, del, putJson } from '../api/fetchClient';
import { buildLineageGraph } from '../utils/lineageGraph';
import {
  readSelection, resolveSelection, toggleSelection, writeSelection,
} from '../utils/canvasSelection';
import {
  availableModelFamilies, availableStatusCategories, filterDatasetIdsByFamilies,
  filterLineageTree, readCanvasExtraFilters, readFamilySelection,
  resolveFamilySelection, toggleFamilySelection, writeCanvasExtraFilters,
  writeFamilySelection,
} from '../utils/canvasFamilyFilter';
import { toOverrideMap } from '../utils/canvasPlacement';
import { mergeLanePlacement, toLanePlacementMap } from '../utils/canvasLanePlacement';
import { pinWriteShortfall, toImageNodeMap, visibleImageNodes } from '../utils/canvasImageNodes';
import { tidyLaneRows } from '../utils/canvasPinBatch';
import { useToast } from '../components/common/Toast';
import CanvasDatasetFilter from '../components/canvas/CanvasDatasetFilter';
import LineageCanvas from '../components/canvas/LineageCanvas';
import CanvasUndeployPanel from '../components/canvas/CanvasUndeployPanel';
import { HelpBadge } from '../help/HelpMode';

/* ◉ LoRA Canvas — every dataset's training genealogy on one board.

   Until now a lineage graph was locked inside a single run's card: one dataset
   at a time, in a fixed frame. This page promotes it. Same trees, same cards,
   same edges — but all of them on one surface you can zoom and pan, with a
   filter to decide which datasets are on it.

   Loading is deliberately in TWO steps. The index (which datasets have runs) is
   a couple of grouped queries and arrives instantly, so the filter and the frame
   are usable straight away; each dataset's genealogy — which scans that run's
   saves on disk — is then fetched on its own and its lane fills in when it
   lands. One request for the whole library would have meant an empty screen for
   as long as the slowest disk scan took.

   Fetches are capped to a few at a time. A library of thirty trained datasets
   firing thirty concurrent disk-scanning requests would stall the very server
   that also has to answer whatever training or captioning is running. */

const MAX_PARALLEL_FETCHES = 3;
/* How long a coalesced geometry write waits for the burst to end. Same 500 ms
   as the external-LoRA list on the board, so the two feel alike; short enough
   that letting go of an arrow key and closing the tab still lands the write
   through the unmount flush. */
const COALESCE_WRITE_MS = 500;

export default function CanvasPage() {
  const toast = useToast();
  const [index, setIndex] = useState({ status: 'loading', datasets: [], error: null });
  const [stored, setStored] = useState(() => readSelection(
    typeof localStorage !== 'undefined' ? localStorage : null));
  const [storedFamilies, setStoredFamilies] = useState(() => readFamilySelection(
    typeof localStorage !== 'undefined' ? localStorage : null));
  const [query, setQuery] = useState('');
  // ⏏ Is the whole-install "what is deployed in ComfyUI" panel open.
  const [undeployOpen, setUndeployOpen] = useState(false);
  const [extraFilters, setExtraFilters] = useState(() => readCanvasExtraFilters(
    typeof localStorage !== 'undefined' ? localStorage : null));
  // dataset_id -> { status: 'loading'|'ready'|'error', tree, error }
  const [trees, setTrees] = useState({});
  const inflight = useRef(new Set());

  useEffect(() => {
    let alive = true;
    apiFetch('/api/train/canvas/datasets')
      .then((d) => { if (alive) setIndex({ status: 'ready', datasets: d?.datasets || [], error: null }); })
      .catch((e) => { if (alive) setIndex({ status: 'error', datasets: [], error: e?.message || 'Could not load your datasets' }); });
    return () => { alive = false; };
  }, []);

  // Where the user has MOVED cards: {datasetId: {record_id: {x, y}}}. One request
  // for the whole board — the lanes need their overrides before the first paint,
  // and these rows are tiny next to the genealogies they precede.
  const [positions, setPositions] = useState({});
  const loadPositions = useCallback(() => apiFetch('/api/train/canvas/positions')
    .then((d) => {
      const next = {};
      for (const [dsId, rows] of Object.entries(d?.positions || {})) next[dsId] = toOverrideMap(rows);
      setPositions(next);
    })
    // A board that opens in its automatic layout is a far better failure than
    // a board that does not open: nothing about the canvas may block it.
    .catch(() => {}), []);
  useEffect(() => { loadPositions(); }, [loadPositions]);

  /* Remember a lane's arrangement. Applied to the screen FIRST and sent
     afterwards, deliberately: a card must follow the finger at the speed of the
     finger, and a position is a display preference — if the write fails the move
     stays on screen and the user is not interrupted by a modal to be told the
     database did not like where they put a rectangle. The next drag re-sends the
     whole lane, so a lost write heals itself. */
  const onPinLane = useCallback((datasetId, rows) => {
    const map = toOverrideMap(rows);
    setPositions((p) => ({ ...p, [datasetId]: { ...(p[datasetId] || {}), ...map } }));
    putJson(`/api/dataset/${datasetId}/canvas/positions`, { positions: rows }).catch(() => {});
  }, []);

  /* 🛝 Where each LANE sits and how much room it keeps, {datasetId: {x?,y?,h?}}.
     Server-side like the card positions it travels with, and for the same
     reason: a board whose cards follow the dataset from one machine to the next
     while its lanes stay behind is an inconsistency you only discover after you
     have changed desk. A lane with no row is laid out by the stack — which is
     every lane until somebody drags one. */
  const [lanePlacements, setLanePlacements] = useState({});
  const loadLanePlacements = useCallback(() => apiFetch('/api/train/canvas/lanes')
    .then((d) => setLanePlacements(toLanePlacementMap(d?.lanes || [])))
    // Same rule as the positions: a board that opens in its automatic layout is
    // a far better failure than a board that does not open.
    .catch(() => {}), []);
  useEffect(() => { loadLanePlacements(); }, [loadLanePlacements]);

  /* Remember one lane's placement. Screen FIRST, database after — a lane must
     follow the finger at the speed of the finger, and where a block sits is a
     display preference: a failed write must never interrupt what the user is
     doing. The MERGE happens on both sides (here for the screen, in the service
     for the row) because each gesture only ever speaks for its own half — a
     move must not forget a height, a resize must not forget a position. */
  const onSaveLane = useCallback((datasetId, change) => {
    setLanePlacements((cur) => {
      const next = mergeLanePlacement(cur[datasetId], change);
      if (!next) {
        const { [datasetId]: _gone, ...rest } = cur;
        return rest;
      }
      return { ...cur, [datasetId]: next };
    });
    putJson(`/api/dataset/${datasetId}/canvas/lane`, change).catch(() => {});
  }, []);

  /* The images PINNED on the board, {datasetId: {imageId: node}}.
     Stored SERVER-SIDE, in canvas_image_node, deliberately next to the card
     positions rather than in localStorage: the cards already follow the dataset
     from one machine to the next, and a board whose cards travel while its
     pictures stay behind is an inconsistency you only discover after you have
     changed desk. Its own table, so nothing about the existing card rows
     changes shape -- a user updating into this version finds their cards
     exactly where they left them. */
  const [imageNodes, setImageNodes] = useState({});
  const loadImageNodes = useCallback(() => apiFetch('/api/train/canvas/images')
    .then((d) => {
      const next = {};
      for (const [dsId, rows] of Object.entries(d?.nodes || {})) {
        next[dsId] = toImageNodeMap(rows);
      }
      setImageNodes(next);
    })
    // Same rule as the positions: nothing about the canvas may stop the canvas
    // from opening.
    .catch(() => {}), []);
  useEffect(() => { loadImageNodes(); }, [loadImageNodes]);

  /* 💾 Re-read the whole arrangement from the server.
     Used after a layout preset is restored: the preset is applied SERVER-side
     through the live writers, so the browser's copy of where everything sits is
     now the stale one. Re-reading both maps is cheaper and far safer than
     replaying the preset locally — a local replay would be a second
     implementation of the restore, free to disagree with the one that actually
     wrote the rows. */
  const onReloadLayout = useCallback(
    () => Promise.all([loadPositions(), loadImageNodes(), loadLanePlacements()]),
    [loadPositions, loadImageNodes, loadLanePlacements]);

  /* The write itself, extracted so the immediate path and the coalesced one
     below cannot drift into two different requests. */
  const sendImageNodes = useCallback((datasetId, rows) => putJson(
    `/api/dataset/${datasetId}/canvas/images`, {
      // `image` is the client's own render payload; the server resolves it from
      // the id and must not be handed a copy to trust.
      nodes: rows.map(({ image, ...row }) => row),
    })
    // A row the SERVER refused (unusable geometry, an image from another lane)
    // comes back as a 200 with a smaller `saved` count. Swallowing that is how
    // a pin could appear, be dropped, and vanish on the next reload without a
    // word — see utils/canvasImageNodes.pinWriteShortfall. A dropped NETWORK
    // stays silent as before: that write heals on the next gesture.
    .then((d) => {
      const said = pinWriteShortfall(rows, d);
      if (said) toast.error(said);
    })
    .catch(() => {}), [toast]);

  /* The deferred half. `pending` is a dataset id → (image id → row) map, so a
     burst of thirty nudges of the same picture collapses to one row, and two
     pictures nudged in the same burst still both get written. */
  const pending = useRef(new Map());
  const flushTimer = useRef(null);
  const flushImageWrites = useCallback(() => {
    if (flushTimer.current) { clearTimeout(flushTimer.current); flushTimer.current = null; }
    const batches = pending.current;
    if (!batches.size) return;
    pending.current = new Map();
    for (const [datasetId, byImage] of batches) {
      sendImageNodes(datasetId, [...byImage.values()]);
    }
  }, [sendImageNodes]);
  const queueImageWrite = useCallback((datasetId, rows) => {
    const byImage = pending.current.get(datasetId) || new Map();
    for (const r of rows) byImage.set(r.image_id, r);
    pending.current.set(datasetId, byImage);
    if (flushTimer.current) clearTimeout(flushTimer.current);
    flushTimer.current = setTimeout(() => {
      flushTimer.current = null;
      flushImageWrites();
    }, COALESCE_WRITE_MS);
  }, [flushImageWrites]);
  // Leaving the board inside the coalescing window must not lose the last nudge
  // — the same trap the external-LoRA list had to be taught about.
  const flushRef = useRef(flushImageWrites);
  useEffect(() => { flushRef.current = flushImageWrites; }, [flushImageWrites]);
  useEffect(() => () => flushRef.current(), []);

  /* Pin, move, resize or CLOSE one or more images of a lane. Applied to the
     screen first and sent afterwards, exactly like a card position -- and for
     the same reason: a picture has to follow the finger at the speed of the
     finger, and a lost write heals on the next gesture rather than interrupting
     the user with a modal about a rectangle.

     A closed node keeps its row and its geometry (`visible: false`), which is
     what makes re-opening it land on the same spot at the same size.

     🖼🖼 `group_id`/`group_pos` travel the same way — which side-by-side strip
     this picture belongs to, and where in it. They are ADDITIVE and nullable:
     an install whose database predates them reads null and draws the board it
     always drew. A row that does not MENTION them keeps whatever it had, so a
     plain move or resize can never quietly dissolve a group.

     ⌨ `opts.coalesce` — the screen half ALWAYS happens immediately; only the
     PUT is deferred, and only when the caller says the write is one of a burst
     it is still in the middle of (a held arrow key repeats ~30×/s, and each
     repeat was a full PUT of the node). The pending rows are merged per
     dataset, so the request that finally goes out describes where the picture
     ENDED UP rather than every position it passed through. Flushed on unmount
     with `keepalive`, exactly like the external-LoRA list: a nudge followed
     immediately by leaving the page must not be the write that vanishes. */
  const onSaveImageNodes = useCallback((datasetId, rows, opts = null) => {
    setImageNodes((cur) => {
      const lane = { ...(cur[datasetId] || {}) };
      for (const r of rows) {
        const prev = lane[r.image_id];
        lane[r.image_id] = {
          imageId: r.image_id, x: r.x, y: r.y, w: r.w, h: r.h,
          visible: r.visible !== false,
          groupId: 'group_id' in r ? (r.group_id ?? null) : (prev?.groupId ?? null),
          groupPos: 'group_pos' in r ? (r.group_pos ?? null) : (prev?.groupPos ?? null),
          image: r.image || prev?.image,
        };
      }
      return { ...cur, [datasetId]: lane };
    });
    if (opts?.coalesce) { queueImageWrite(datasetId, rows); return; }
    sendImageNodes(datasetId, rows);
  }, [queueImageWrite, sendImageNodes]);

  /* 🗑 Forget pinned nodes LOCALLY — no write at all, deliberately.
     Used after the picture itself has been deleted: its canvas_image_node row
     now points at nothing, and the ordinary close would try to save geometry
     for an image the server can no longer validate (save_canvas_image_nodes
     checks the id against the dataset), so the user would delete a render and
     get a toast saying the board could not be saved. The orphan row is pruned
     server-side on the next read of the board — canvas_image_nodes does that
     already, for exactly this case. */
  const onForgetImageNodes = useCallback((datasetId, imageIds) => {
    const gone = new Set((imageIds || []).map(Number));
    if (!gone.size) return;
    setImageNodes((cur) => {
      const lane = cur[datasetId];
      if (!lane) return cur;
      const next = Object.fromEntries(
        Object.entries(lane).filter(([id]) => !gone.has(Number(id))));
      return { ...cur, [datasetId]: next };
    });
  }, []);

  const availableIds = useMemo(() => index.datasets.map((d) => d.id), [index.datasets]);
  const selected = useMemo(() => resolveSelection(availableIds, stored), [availableIds, stored]);
  const families = useMemo(() => availableModelFamilies(index.datasets), [index.datasets]);
  const selectedFamilies = useMemo(
    () => resolveFamilySelection(families, storedFamilies), [families, storedFamilies]);
  const visibleSelected = useMemo(
    () => filterDatasetIdsByFamilies(index.datasets, selected, selectedFamilies),
    [index.datasets, selected, selectedFamilies]);
  const statuses = useMemo(() => availableStatusCategories(trees), [trees]);
  const selectedStatuses = useMemo(() => (extraFilters.statuses == null
    ? statuses : statuses.filter((status) => extraFilters.statuses.includes(status))),
  [statuses, extraFilters.statuses]);
  const modelFilterActive = selectedFamilies.length !== families.length;
  const filteredTrees = useMemo(() => {
    const next = {};
    for (const [id, state] of Object.entries(trees)) {
      const row = index.datasets.find((dataset) => Number(dataset.id) === Number(id));
      next[id] = { ...state,
        tree: state?.tree ? filterLineageTree(state.tree, {
          families: selectedFamilies, statuses: selectedStatuses, query, datasetName: row?.name,
        }) : state?.tree };
    }
    return next;
  }, [trees, selectedFamilies, selectedStatuses, query, index.datasets]);
  const filteredImageNodes = useMemo(() => {
    if (!extraFilters.showPinned) return {};
    const treeFilterActive = modelFilterActive || selectedStatuses.length !== statuses.length
      || query.trim().length > 0;
    if (!treeFilterActive) return imageNodes;
    const next = {};
    for (const [id, map] of Object.entries(imageNodes)) {
      const allowed = new Set((filteredTrees[id]?.tree?.nodes || []).map((node) => node.record_id));
      next[id] = Object.fromEntries(Object.entries(map || {})
        .filter(([, node]) => allowed.has(node?.image?.record_id)));
    }
    return next;
  }, [imageNodes, filteredTrees, modelFilterActive, selectedStatuses, statuses,
    query, extraFilters.showPinned]);

  /* ✦ Tidy up: forget every moved card on the VISIBLE board. Scoped to what is
     on screen — a lane the user unticked is not on the board they are looking
     at, and silently flattening its arrangement too would be a surprise. */
  const onTidyUp = useCallback(() => {
    setPositions((p) => {
      const next = { ...p };
      for (const id of visibleSelected) delete next[id];
      return next;
    });
    for (const id of visibleSelected) del(`/api/dataset/${id}/canvas/positions`).catch(() => {});
    /* 🛝 And the LANES go back to the automatic stack. ✦ Tidy up is the escape
       hatch from every arrangement on this board — it already was for the cards
       and for the pictures, and a lane left parked at a position nobody
       remembers setting would be the one thing the button could not undo. */
    setLanePlacements((cur) => {
      const next = { ...cur };
      for (const id of visibleSelected) delete next[id];
      return next;
    });
    for (const id of visibleSelected) del(`/api/dataset/${id}/canvas/lane`).catch(() => {});
    /* And the pinned images RE-FLOW rather than being deleted or left behind.
       Neither of the obvious answers is right: deleting them would make a
       "rebuild the automatic tree" button destroy content the user placed
       deliberately, and leaving them alone would strand every picture at
       coordinates chosen next to a card that has just moved back. So each
       VISIBLE pin is recomputed to its default spot beside its own card, which
       is the tidy version of exactly what it was.

       CLOSED pins are not touched: their remembered geometry is a promise
       ("re-open it where I closed it") and Tidy up is not the place to break
       it. The re-flow needs the laid-out lane, so it is done here against the
       automatic tree the board is about to fall back to.

       It re-flows through the SAME function 📌 Pin all uses
       (utils/canvasPinBatch), on purpose: two placers would be two chances to
       disagree, and the one thing the user is promised on this board is that
       nothing lands on top of anything. Before this, the re-flow only avoided
       other PICTURES — it could park one squarely on a run card. */
    setImageNodes((cur) => {
      const next = { ...cur };
      for (const id of visibleSelected) {
        const map = next[id];
        if (!map) continue;
        const tree = trees[id]?.tree;
        const graph = tree ? buildLineageGraph(tree) : null;
        const lane = { ...map };
        const rows = [];

        /* 🖼🖼 STRIPS FIRST, and each as ONE object.
           A picture can now be parked anywhere on the board, its own lane's
           corner included — so "leave the groups alone", which was right while a
           strip could only ever be inside its lane, would now mean ✦ Tidy up
           walking past a whole assembled comparison stranded thousands of units
           off the board, with no way back short of hunting for it at 10 % zoom.
           A strip therefore comes home too. What the old rule was really
           protecting is untouched: only the ANCHOR's row is written, the strip
           is derived from it, and no membership is sent — so a tidy can move a
           group but can never take one apart.

           The strips and the loose pictures are placed by ONE function
           (utils/canvasPinBatch.tidyLaneRows), which the lane STACK also asks
           how much room to leave under this tree — otherwise the next dataset
           starts straight through the band this is about to lay down. */
        for (const r of tidyLaneRows({ graph, nodes: visibleImageNodes(map) }).rows) {
          lane[r.imageId] = { ...lane[r.imageId], x: r.x, y: r.y, w: r.w, h: r.h };
          rows.push({ image_id: r.imageId, x: r.x, y: r.y, w: r.w, h: r.h, visible: true });
        }

        next[id] = lane;
        // One write for the lane, not one per picture: a board carrying twenty
        // pins used to fire twenty requests at the server that is probably also
        // training something.
        if (rows.length) putJson(`/api/dataset/${id}/canvas/images`, { nodes: rows }).catch(() => {});
      }
      return next;
    });
  }, [visibleSelected, trees]);

  /* Re-read ONE lane from the server and put it back on the board. Used after a
     deploy launched from the canvas: the pills of that dataset have to come back
     `testable`, with the name of the copy that now sits in ComfyUI. Returns the
     fresh tree so the caller can read it directly instead of racing the state
     update it just triggered. */
  const onRefetchDataset = useCallback(async (id) => {
    const tree = await apiFetch(`/api/dataset/${id}/train/lineage`);
    setTrees((t) => ({ ...t, [id]: { status: 'ready', tree, error: null } }));
    return tree;
  }, []);

  const persist = useCallback((ids) => {
    setStored(ids);
    writeSelection(typeof localStorage !== 'undefined' ? localStorage : null, ids);
  }, []);

  const onToggle = useCallback((id) => {
    persist(toggleSelection(selected, id, availableIds));
  }, [persist, selected, availableIds]);

  const persistFamilies = useCallback((next) => {
    setStoredFamilies(next);
    writeFamilySelection(typeof localStorage !== 'undefined' ? localStorage : null, next);
  }, []);

  const onToggleFamily = useCallback((family) => {
    persistFamilies(toggleFamilySelection(selectedFamilies, family, families));
  }, [persistFamilies, selectedFamilies, families]);

  const persistExtraFilters = useCallback((next) => {
    setExtraFilters(next);
    writeCanvasExtraFilters(typeof localStorage !== 'undefined' ? localStorage : null, next);
  }, []);

  const onToggleStatus = useCallback((status) => {
    const next = selectedStatuses.includes(status)
      ? selectedStatuses.filter((item) => item !== status)
      : [...selectedStatuses, status];
    persistExtraFilters({ ...extraFilters, statuses: next });
  }, [selectedStatuses, persistExtraFilters, extraFilters]);

  const onResetFilters = useCallback(() => {
    setQuery('');
    persist(availableIds);
    persistFamilies(families);
    persistExtraFilters({ statuses: null, showPinned: true });
  }, [persist, availableIds, persistFamilies, families, persistExtraFilters]);

  // Fetch the genealogy of every selected dataset that has none yet, a few at a
  // time. A dataset unticked and re-ticked keeps its cached tree — the board must
  // not re-scan the disk for a filter click.
  useEffect(() => {
    const missing = visibleSelected.filter((id) => !trees[id] && !inflight.current.has(id));
    if (!missing.length) return;
    const room = MAX_PARALLEL_FETCHES - inflight.current.size;
    if (room <= 0) return;
    for (const id of missing.slice(0, room)) {
      inflight.current.add(id);
      setTrees((t) => ({ ...t, [id]: { status: 'loading', tree: null, error: null } }));
      apiFetch(`/api/dataset/${id}/train/lineage`)
        .then((tree) => setTrees((t) => ({ ...t, [id]: { status: 'ready', tree, error: null } })))
        .catch((e) => setTrees((t) => ({ ...t,
          [id]: { status: 'error', tree: null, error: e?.message || 'Could not load this lineage' } })))
        .finally(() => {
          inflight.current.delete(id);
          // Nudge the effect so the next batch starts.
          setTrees((t) => ({ ...t }));
        });
    }
  }, [visibleSelected, trees]);

  const entries = useMemo(() => visibleSelected.map((id) => {
    const row = index.datasets.find((d) => d.id === id);
    const state = filteredTrees[id];
    return {
      datasetId: id,
      name: row?.name || `Dataset ${id}`,
      runs: row?.runs || 0,
      families: row?.families || [],
      // The ★ pinned LoRA(s) of this dataset, so a delete from the board can warn
      // that it is about to break the saved winning combo — exactly like the
      // dataset panel does. Travels on the lane, and the canvas reads the lane of
      // whichever checkpoint popover is open.
      bestSettingsLoras: row?.best_settings_loras || [],
      // 🪪 The reference face this dataset was built around, so the board can
      // show WHO the renders are supposed to be. `kind` decides whether there
      // is one to show at all — a concept or a style dataset has no reference.
      refFilename: row?.ref_filename || null,
      kind: row?.kind || 'character',
      // 🧬 The dataset's trigger word, carried on the lane so a pick taken from
      // it knows what a blend will prepend to the prompt.
      triggerWord: row?.trigger_word || null,
      status: state?.status || 'loading',
      error: state?.error || null,
      // 🛝 What this lane was told about its own room and position, or null.
      // Consumed by utils/canvasLayout.stackLanes, straight through the two
      // memos in between.
      placement: lanePlacements[id] || null,
      // The RAW tree, not a laid-out graph: the canvas has to be able to lay it
      // out again when a run is removed from the board (see LineageCanvas).
      tree: state?.tree || null,
    };
  }).filter((entry) => entry.status !== 'ready' || (entry.tree?.nodes || []).length > 0),
  [visibleSelected, index.datasets, filteredTrees, lanePlacements]);
  const visibleRuns = useMemo(() => entries.reduce(
    (count, entry) => count + (entry.tree?.nodes || []).length, 0), [entries]);

  return (
    /* 📐 The page is a COLUMN one viewport tall (App.jsx pins the `/canvas`
       shell to `h-svh`), and everything above the board — the title, an error
       line — is fixed-size, so `min-h-0 flex-1` on the board's own wrapper
       hands it every remaining pixel. `min-h-0` is the load-bearing half: a
       flex child defaults to `min-height:auto`, refuses to shrink under its
       content, and the PAGE scrolls instead of the board. */
    <div className="flex min-h-0 flex-1 flex-col">
      {/* 📱 The blurb was the first thing a small screen could afford to lose:
          it explains the page once, and after that it is height spent above the
          board on every single load. It went first, hidden up to `lg`.

          The WHOLE HEADER goes now, at the same `lg`. Measured at 412×780 it was
          67 px — a title repeating the word the nav bar is already highlighting,
          a Beta badge, and a button — above a board that had 297 px to work in.
          Two things had to be re-homed rather than dropped for that to be
          honest, and both were: the <h1> stays as a screen-reader heading (a
          screen with no heading is a screen a reader lands in the middle of),
          and ⏏ Undeploy… together with the page's ? badge move onto the board's
          ⋯ shelf. Desktop, from `lg` up, is untouched. */}
      <h1 className="sr-only lg:hidden">LoRA Canvas</h1>
      <header className="mb-2 hidden sm:mb-3 lg:block">
        <h1 className="flex items-center gap-2 text-lg font-semibold text-content">
          <span aria-hidden>◉</span> LoRA Canvas
          <span className="px-1.5 py-0.5 rounded border border-amber-400/50 bg-amber-500/10 text-amber-300 text-[0.625rem] font-semibold uppercase tracking-wide">Beta</span>
          <HelpBadge topic="page-canvas" />
          {/* ⏏ An INSTALL-wide action, so it lives on the page chrome rather
              than on the board: what is deployed in ComfyUI has nothing to do
              with which lanes are currently in view. ml-auto keeps it off the
              title at every width. */}
          <button type="button" onClick={() => setUndeployOpen(true)}
            title="List every LoRA this app deployed into ComfyUI and remove the ones you tick. Your training saves are kept — each one can be deployed again."
            className="ml-auto rounded-md border border-border px-2 py-1 text-[0.6875rem] font-medium text-content-muted hover:bg-surface-raised hover:text-content">
            <span aria-hidden>⏏</span> Undeploy…
          </button>
        </h1>
        <p className="mt-1 hidden text-content-muted text-[0.75rem] lg:block">
          Every training run you have made, on one board: each dataset gets a lane, each run a card,
          and a continuation is joined to the exact checkpoint it resumed from.
        </p>
      </header>

      {index.status === 'error' && (
        <p className="mb-3 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-amber-100 text-[0.75rem]">
          {index.error}
        </p>
      )}

      {index.status === 'loading'
        ? <p className="text-content-subtle text-[0.75rem]">Loading your datasets…</p>
        : (
          <LineageCanvas entries={entries} positions={positions}
            /* The filter rides ON the board — see the overlay comment in
               LineageCanvas. Same component, moved, not a second copy. */
            filterSlot={(
              <CanvasDatasetFilter
              datasets={index.datasets}
              selected={selected}
              onToggle={onToggle}
              onAll={() => persist(availableIds)}
              onNone={() => persist([])}
              families={families}
              selectedFamilies={selectedFamilies}
              onToggleFamily={onToggleFamily}
              onAllFamilies={() => persistFamilies(families)}
              onNoFamilies={() => persistFamilies([])}
              query={query}
              onQueryChange={setQuery}
              statuses={statuses}
              selectedStatuses={selectedStatuses}
              onToggleStatus={onToggleStatus}
              showPinned={extraFilters.showPinned}
              onTogglePinned={() => persistExtraFilters({
              ...extraFilters, showPinned: !extraFilters.showPinned,
              })}
              onResetFilters={onResetFilters}
              visibleRuns={visibleRuns} />
            )}
            imageNodes={filteredImageNodes} allImageNodes={imageNodes}
            onSaveImageNodes={onSaveImageNodes}
            onForgetImageNodes={onForgetImageNodes}
            onReloadLayout={onReloadLayout}
            onPinLane={onPinLane} onTidyUp={onTidyUp}
            lanePlacements={lanePlacements} onSaveLane={onSaveLane}
            onRefetchDataset={onRefetchDataset}
            /* ⏏ Below `lg` the header that carries this button is not drawn, so
               the board's ⋯ shelf carries it instead. Same handler, same panel —
               the action moves, it is never lost. */
            onOpenUndeploy={() => setUndeployOpen(true)} />
        )}

      <CanvasUndeployPanel open={undeployOpen}
        onClose={() => setUndeployOpen(false)} />
    </div>
  );
}
