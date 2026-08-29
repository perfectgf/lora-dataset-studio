import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Archive, BarChart3, FolderInput, FolderOpen, Lightbulb, Palette, Rocket, Search, Target, Trash2, Type, Undo2, Wand2 } from 'lucide-react';
import { apiFetch, patchJson, postJson, putJson } from '../../api/fetchClient'
import { useFolderPersons } from './useFolderPersons'
import { useReviewLightbox } from './useReviewLightbox'
import { useCaptionOptions } from './useCaptionOptions'
import { useCurationLanes } from './useCurationLanes'
import { useToast } from '../common/Toast'
import { useCapabilities } from '../../context/CapabilitiesContext'
import { useConnectionStatus } from '../../hooks/useConnectionStatus'
import DupGroupsPanel from './DupGroupsPanel'
/* 🧪 The Caption Lab, shared with the Dataset side. Lazy for the same reason the
   dataset lazies it: both only exist behind a click, and the bank entry chunk already
   sits near vite's size warning. `bankLabSurface` is where the two surfaces differ —
   see captionLabSurface.js. */
import { bankLabSurface, imageDisplayName } from '../dataset/captionLabSurface'
import { imageVersionQuery } from './bankEdits.js'
const CaptionLabPicker = lazy(() => import('../dataset/CaptionLabPicker'))
const CaptionEditorDialog = lazy(() => import('../dataset/CaptionEditorDialog'))
// ── The pieces this screen used to define inline ────────────────────────────
// Split out by the Encre redesign: the top bar, the filter rail, the passes
// panel and the grid are four files now, and all four drew the same chips.
// The surface-inventory contract scans the whole Bank TREE, so moving a button
// between these files is free and losing one is loud (see bankSurfaces.js).
import { Stat } from './BankAtoms.jsx'
import { ProgressBar, UndoBar } from './BankProgress.jsx'
import Tile from './BankTile.jsx'
import CoveragePanel from './BankCoveragePanel.jsx'
// The two structural halves of the redesign: all of triage beside the grid, and
// the eight passes behind one button instead of across the top of the screen.
import BankFilterRail from './BankFilterRail.jsx'
import BankPassesPanel from './BankPassesPanel.jsx'
import {
  FLAG_HINT, FLAG_LABEL, FRAMING_BUCKETS, ORIGIN_BUCKETS, QUALITY_REJECT_FLAGS,
  RES_BUCKETS, SCORE_REJECT_FLAGS,
} from './bankFacets.js'
// Structure B's own logic — rail folding, facet groups, the passes panel.
import {
  loadRailOpen, passesButtonLabel, passesPanelStartsOpen, railIsColumn, saveRailOpen,
} from './bankLayout.js'
import PromoteDialog from './PromoteDialog'
import DeleteRejectedDialog from './DeleteRejectedDialog'
import LaunchAllDialog from './LaunchAllDialog'
import ScoringPythonDialog from './ScoringPythonDialog'
import PipelineReport from './PipelineReport'
import FolderSyncNote from './FolderSyncNote'
import RelocateBankDialog from './RelocateBankDialog'
import ForgetMissingDialog from './ForgetMissingDialog'
import BankReviewLightbox from './BankReviewLightbox'
import PersonPreflightDialog from './PersonPreflightDialog'
import { preflightNeeded, preflightWillSample } from './personPreflight.js'
import { folderSyncToast } from './bankSync.js'
import { undoOffer, undoResultMessage } from './bankUndo.js'
// An occupied bank refuses in OUR words, never in the server's (pure/testable).
import { busyRefusalLive } from './bankPassRun.js'
import { scoreDeviceNote } from './bankScoreDevice.js'
// Wording that adapts to the machine (a card-less box is never sold CUDA).
import { PICKER_PROFILES } from './scoringPython.js'
// Reuse the dataset's register list so the Bank lane never drifts from it — and the
// same ENGINE list, so "which engine" means the same thing on both surfaces.
import {
  CAPTION_LENGTH_OPTIONS, ENGINE_OPTIONS, VOCABULARY_OPTIONS,
} from '../dataset/CaptionOptionsPopover'
// Which pile the caption pass is aimed at, and the number the button quotes (pure).
import {
  captionIncludeAssertedLabel, captionRecaptionConfirmation,
  captionRecaptionDisabledReason, captionRecaptionLabel, captionRecaptionNote,
  captionScopeNote, captionScopeStatuses,
} from './bankCaptionScope.js'
// …and the one thing the captioners are known to do badly, said where the engine is
// still free to change. Pure/testable; stays silent unless the scope is measurably
// the material it was measured on (see the module header for the evidence base).
import { captionNsfwNotice } from './captionNsfwNotice.js'
import { passScopeOption } from './bankPassScope.js'
// 🎛 The launch window every pass now opens, and the two pure modules behind it:
// what each pass is (blocks, offered scopes, refusals) and how big a run is.
import PassDialog from './PassDialog.jsx'
import BankZonesPreview from './BankZonesPreview.jsx'
import { BANK_PASSES } from './bankPasses.js'
import {
  semanticEngineLabel, semanticEnginePatchBody, semanticEngineState,
  semanticPayloadMatches, semanticPrerequisite,
} from './bankSemanticEngine.js'
// Ordered zone model + the "what's next" accent, both pure/testable.
// Grid ordering menu (which sorts exist, and which ones have data) — pure/testable.
import { bankSortGroups, loadBankSort, saveBankSort } from '../../utils/gridSort.js'
// 🏷️ One image's caption → the chips you can filter by, and the same chips over a
// whole SELECTION with how often each was cited (pure/testable).
import { tagsParam, selectionTagCounts } from './bankTags.js'
// 🔤 Text search wording — "closest", never "matching" — plus the cold-start and
// CLIP-limitation copy. Pure/testable (node --test cannot parse this JSX).
import {
  PUSH_DOWN_STRENGTHS, pushDownCaveat, pushDownNote,
  limitsSentence, pendingLabel, readinessHint, suggestPushDown, summarize,
  withoutNegation,
} from './bankTextSearch.js'
// ⚖️ Balanced pick — the distribution obtained, in words and numbers. Pure logic
// on purpose: the repartition is what has to be provable (node --test, no JSX).
import {
  BALANCE_AXES, balanceNotes, balanceReadiness,
  balanceRows, summarizeBalance,
} from './bankBalance.js'
// 🎨 Medium + ⤢ Angle — buckets, tooltips and, above all, the LIMITS each row
// has to print. Pure/testable: what matters is that we never claim more than we
// measured (see bankMedium.js for the numbers behind the wording).
import {
  ANGLE_BUCKETS, MEDIUM_BUCKETS, angleReadiness, mediumLimits, shownBuckets,
} from './bankMedium.js'
// 🧹 Auto-reject — the number next to a checkbox must be the number the click
// moves, and a 0 has to say WHICH of its two meanings it carries.
import {
  flagCandidateLabel, flagPrereq, pickedCandidates, unscannedNotice,
} from './autoRejectReadiness.js'
import { chipCounts, facetDataKey, isFacetFiltered } from './bankFacetCounts.js'


const PAGE_SIZE = 120
/* The Curate row's five buttons, one style. They were the same gray text-xs as
   every utility control, and asked for from live use ("agrandir — ce sont des
   features principales"): the semantic curation is what the Bank is FOR once
   triage is done, and it dressed as a footnote. Same accent family as
   ▶ Review one by one, one size up. */
const CURATE_BTN = 'rounded-md border border-indigo-400/60 bg-indigo-500/20 '
  + 'px-3 py-1.5 text-sm font-semibold text-indigo-200 disabled:opacity-50 '
  + 'hover:bg-indigo-500/30'
/* How often the bank-wide counts refresh while a pass runs. The banner ticks
   every 2 s off /activity; the dashboard follows on this slower beat — plus
   immediately when the job lands, so the numbers you end on are exact. The
   payload is ~60 full-table aggregates: on a 50 000-image bank, asking for it
   every 2 s is what made the workspace unreadable and its Stop button
   unreachable in the first place.
   ⏱ A WALL-CLOCK deadline, deliberately, not "every Nth tick". The poll effect
   re-subscribes whenever the job's `detail` string changes — which is every
   couple of seconds during a real pass — so a counter living in the effect would
   reset before it ever reached N and the dashboard would never refresh at all. */
const FULL_PAYLOAD_MS = 10000
/* How many off-page captions the 🏷️ row will fetch for a selection.
   Not a taste call: `ids=` travels in the query string, and a few thousand
   integers build a request line the server refuses outright (the same limit the
   "show selected" view already lives with). 500 ids ≈ 3.5 kB, comfortably under
   it. Whatever the cap leaves out is DISCLOSED in the row — a count over 500 of
   3 200 images presenting itself as "your selection" is the very defect this
   surface exists to end. */
const TAG_CAPTION_FETCH_CAP = 500

/** Fetch EVERY image id matching a filter, page by page (used by the
 * cluster/flag "select all" actions — a cluster can exceed one grid page). */
async function fetchAllIds(bankId, params) {
  // ONE request for the ids of the whole filter (`ids_only=1`), in the order the
  // grid is showing. This used to walk the grid 500 rows at a time and keep only
  // `i.id` — 46 sequential round trips and 16 MB of image payloads to end up with
  // 23 000 integers, measured on a 22 940-image bank; with a measure sort active
  // each of those pages also re-ran the COUNT and the ORDER BY over the whole
  // table, which is what put seconds in front of ▶ Review.
  const qs = new URLSearchParams({ ...params, ids_only: '1' })
  const d = await apiFetch(`/api/bank/${bankId}/images?${qs}`)
  return d.ids || []
}

export default function BankWorkspace({ bankId, onBack, onGone }) {
  const toast = useToast()
  const { caps, loading: capsLoading, refresh: refreshCaps } = useCapabilities()
  // Updated on every render once the payload has been normalised. The unmount
  // cleanup must release the engine selected NOW, not the one from first mount.
  const semanticEngineRef = useRef('clip')
  const semanticModelKeyRef = useRef(null)
  // Every coverage response is tied to the request that produced it. Switching
  // engines invalidates an older request even if that request finishes last.
  const coverageRequestRef = useRef(0)
  const textStatusRequestRef = useRef(0)
  const [payload, setPayload] = useState(null)
  // The chip counters measured under the ACTIVE filter (null = nothing filtered,
  // so the payload's bank-wide numbers are the honest answer). See
  // bankFacetCounts.js for why these are two maps and not one.
  const [facets, setFacets] = useState(null)
  // `sort` opens on whatever order this bank was last reviewed in (per bank, not
  // global — see gridSort.bankSortStorageKey). Every other facet starts empty on
  // purpose: an order is a habit, a filter is a question you asked once.
  const [filter, setFilter] = useState(() => ({ status: null, flag: null, cluster: null,
    style: null, subfolder: null, search: null, exclude: null, tags: null,
    sort: loadBankSort(bankId), resBucket: null,
    origin: null,
    framing: null,
    // Dedicated keys, never folded into `flag` or the text lane.
    medium: null,
    angle: null }))
  const [searchText, setSearchText] = useState('')
  // 🚫 The inverse of the search box: hide what already carries a word. Session
  // state, deliberately NOT remembered like the sort — an order you can see in a
  // menu is a habit, images missing from a grid for a reason you set last week
  // reads as data loss.
  const [excludeText, setExcludeText] = useState('')
  // 🏷️ Tag chips lifted off ONE image's caption. `tagSource` is the image the
  // chips were read from (kept so the row can say WHOSE tags these are — chips
  // with no provenance are just mystery words), `tagPicked` the ticked subset.
  // Both are session state: this is a question you ask about one image now, not
  // a standing preference like the sort order.
  const [tagSource, setTagSource] = useState(null)
  const [tagPicked, setTagPicked] = useState(() => new Set())
  // 🏷️ …and the SELECTION's tags, which need no click at all. `tagFreeze` is the
  // snapshot taken the instant a chip is ticked: applying a filter clears the
  // selection (setF does, and must — the ids no longer match what is on screen),
  // so without this the row would compute the answer and then delete the question.
  const [tagFreeze, setTagFreeze] = useState(null)
  const [subfolders, setSubfolders] = useState([])
  // 👤 The preflight of the person pass: { plan, probing, run } — `run` is the
  // pass (or the whole 🚀 Launch all) the user actually asked for, held until
  // they have answered the folder question.
  const [preflight, setPreflight] = useState(null)
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState({ images: [], total: 0 })
  const [selected, setSelected] = useState(() => new Set())
  const [promoteOpen, setPromoteOpen] = useState(false)
  const [deleteRejectedOpen, setDeleteRejectedOpen] = useState(false)
  // ↩ one step back over the last bulk decision. `undoDismissedAt` remembers the
  // offer the user waved away by its timestamp, so the NEXT bulk action shows a
  // bar again instead of inheriting the dismissal.
  const [undoBusy, setUndoBusy] = useState(false)
  const [undoDismissedAt, setUndoDismissedAt] = useState(0)
  const [launchOpen, setLaunchOpen] = useState(false)
  const [semanticSwitching, setSemanticSwitching] = useState(false)
  /* 🎛 Which pass's launch window is open (a BANK_PASSES id, or null).
     ONE piece of state for every launch window: a pass button no longer fires, it opens
     the window that shows where the run applies, what the calculation reads and
     what is NOT decided there — then launches from the bottom of it. */
  const [passOpen, setPassOpen] = useState(null)
  // ✨ Score's interpreter picker — reuse a CUDA Python this machine already has
  // instead of downloading another torch. Opened from the CPU warning.
  // Which interpreter picker is open, if any: a PICKER_PROFILES key ('scoring'
  // for ✨ Score, 'semantic' for the SigLIP 2 index) or '' for none. One dialog
  // serves both — the profile decides the endpoint and the wording.
  const [pythonPickerFor, setPythonPickerFor] = useState('')
  const [dismissedReportAt, setDismissedReportAt] = useState(null)
  const [relocating, setRelocating] = useState(false)
  const [forgettingMissing, setForgettingMissing] = useState(false)
  const [openingSourceFolder, setOpeningSourceFolder] = useState(false)
  const [rejectFlags, setRejectFlags] = useState(() => new Set(['blur', 'uniform']))
  const [showAutoReject, setShowAutoReject] = useState(false)
  // 🎚 The threshold editor folds away: the chips are the daily gesture, the
  // numbers behind them are the occasional one.
  const [thresholdsOpen, setThresholdsOpen] = useState(false)

  /* ── Structure B's own view state ─────────────────────────────────────────
     The rail folds instead of overflowing, and the passes live in a panel
     opened on demand. Every DECISION behind these lives in bankLayout.js, where
     `node --test` can reach it; what is here is only the React wiring. */
  const viewportWidth = () => (typeof window === 'undefined' ? undefined : window.innerWidth)
  // Whether the rail can sit BESIDE the grid at this width, tracked live: a
  // window dragged narrow must turn the rail into a drawer, not clip it.
  const [railIsColumnNow, setRailIsColumnNow] = useState(() => railIsColumn(viewportWidth()))
  const [railOpen, setRailOpen] = useState(() => loadRailOpen(viewportWidth()))
  /* Open by default (asked for from live use): the six measured axes are what
     the passes were run FOR, and a closed disclosure hid them from exactly the
     people who had just paid for the measurements. Still collapsible — the
     fold exists for the 400 px drawer, not as the resting state. */
  const [moreOpen, setMoreOpen] = useState(true)
  const [passesOpen, setPassesOpen] = useState(false)
  useEffect(() => {
    const onResize = () => setRailIsColumnNow(railIsColumn(window.innerWidth))
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  /* The rail state is REMEMBERED (a screen preference — it belongs with view
     state, never with the settings that describe what a dataset produces), but
     only when the rail is a column. Closing the phone drawer is "I am done
     filtering", not "hide the filters on my desktop too". */
  const setRail = useCallback((open) => {
    setRailOpen(open)
    if (railIsColumn(viewportWidth())) saveRailOpen(open)
  }, [])
  const openRail = useCallback(() => setRail(true), [setRail])
  const closeRail = useCallback(() => setRail(false), [setRail])
  const togglePasses = useCallback(() => setPassesOpen((v) => !v), [])
  // "Show selected" VIEW: render ONLY the selected ids, in a chosen order.
  // showSelected flips the grid from the facet page to the selection; selectedOrder
  // holds the order to render them in — the similarity/diversity ranking after a
  // curate action (reference first, closest→farthest), else insertion order. It's a
  // VIEW, not a status: Keep/Reject/Promote still act on the selection itself.
  const [showSelected, setShowSelected] = useState(false)
  const [selectedOrder, setSelectedOrder] = useState(null)
  const [tileSize, setTileSize] = useState('M')
  const {
    captionVocab, setCaptionVocab, captionLength, setCaptionLength,
    captionEngine, setCaptionEngine, captionModel, setCaptionModel,
    captionIncludeAsserted, setCaptionIncludeAsserted,
    visionModel, visionModelLooksUncensored, ollamaPicksApply,
    captionModelChoices, captionRunOptions,
  } = useCaptionOptions({ caps })
  /* WHICH PILE each pass runs on, and whether it re-does rows that already have a
     result — kept HERE, not inside the windows, so closing one does not silently
     undo a choice. Keyed by pass id; '' is the historical scope (kept + undecided,
     and the request omits `statuses` entirely).

     🏷️ Caption reads its entry under the name it has always used, because the
     re-caption arithmetic beside it is written against that name. */
  /* 🧪 Caption Lab: which image to bench (picker), then the image it named. The
     Bank's half of the entry point the Dataset's Captions section carries. */
  const [labPickerOpen, setLabPickerOpen] = useState(false)
  const [labImage, setLabImage] = useState(null)
  const [passScopes, setPassScopes] = useState({})
  const [passRedo, setPassRedo] = useState({})
  const setPassScope = (id, v) => setPassScopes((p) => ({ ...p, [id]: v }))
  const setPassRedoFor = (id, v) => setPassRedo((p) => ({ ...p, [id]: v }))
  const captionScope = passScopes.caption || ''
  // Coverage advice (idea by @antonp) — a collapsible read-only panel, fetched
  // on demand (and refreshed whenever it's open and the bank changes).
  const [coverageOpen, setCoverageOpen] = useState(false)
  const [coverage, setCoverage] = useState(null)
  // started_at of the last FINISHED activity already announced (toast + grid
  // refresh). `undefined` = no payload seen yet — refreshPayload records the
  // first snapshot silently; the landing effect announces every one after it.
  const announcedActivityAt = useRef(undefined)
  // When the dashboard was last pulled in full. A ref, not effect-local state:
  // the poll effect re-subscribes on every change of the job's `detail`.
  const fullPayloadAt = useRef(0)
  // 📡 Drives the "we lost contact" note in the progress zone: a failed poll
  // must never render as "no job running".
  const connection = useConnectionStatus()

  const loadCoverage = useCallback(async () => {
    const requestId = ++coverageRequestRef.current
    const expectedEngine = semanticEngineRef.current
    const expectedModelKey = semanticModelKeyRef.current
    try {
      const next = await apiFetch(`/api/bank/${bankId}/coverage`)
      if (requestId !== coverageRequestRef.current
          || expectedEngine !== semanticEngineRef.current
          || !semanticPayloadMatches(next, expectedEngine, expectedModelKey)) return
      setCoverage(next)
    } catch { /* transient — the panel keeps its last read */ }
  }, [bankId])

  // The grid refresher, held in a ref: the folder walk below can add images at
  // any poll, and refreshImages is defined after refreshPayload.
  const refreshImagesRef = useRef(null)

  const refreshPayload = useCallback(async (opts = {}) => {
    try {
      // ?refresh=1 forces the source-folder re-walk (the bank was just opened);
      // a plain poll lets the server's cooldown decide, so the 2 s job poll
      // doesn't hammer the disk.
      const d = await apiFetch(`/api/bank/${bankId}${opts.force ? '?refresh=1' : ''}`,
        { background: !!opts.background })
      // First payload of this bank: record its finished-activity generation
      // WITHOUT announcing it — the snapshot can be a job up to 5 min old, and
      // toasting stale news on every open is worse than silence. From here on
      // the landing effect announces every NEW generation, including jobs that
      // finish before the first 2 s poll ever sees them running.
      if (announcedActivityAt.current === undefined) {
        announcedActivityAt.current = (d?.activity?.finished
          ? d.activity.started_at ?? null : null)
      }
      setPayload(d)
      // Images dropped in the folder show up on their own — say it, and pull
      // them into the grid, so the counters never move without a reason.
      const note = folderSyncToast(d.folder_sync)
      if (note) toast[note.type](note.text)
      if ((d.folder_sync?.added || 0) > 0) refreshImagesRef.current?.()
      return d
    } catch (e) {
      if (String(e?.message || '').includes('not found')) { onGone?.(); return null }
      return null
    }
  }, [bankId, onGone, toast])

  const filterParams = useCallback((f) => {
    const params = {}
    if (f.status) params.status = f.status
    if (f.flag) params.flag = f.flag
    if (f.cluster != null) params.cluster = String(f.cluster)
    if (f.style != null) params.style = String(f.style)
    // subfolder is a string facet where '' is meaningful (bank root) — send it
    // whenever it isn't null, empty string included.
    if (f.subfolder != null) params.subfolder = f.subfolder
    if (f.search) params.search = f.search
    // The exclude terms travel with the search on every surface that reads a
    // filter — the grid, "Select all in filter", ▶ Review and the curation picks.
    if (f.exclude) params.exclude = f.exclude
    // 🏷️ ticked chips — its own key, matched as WORDS and ANDed server-side.
    if (f.tags) params.tags = f.tags
    // Grid sort (any measured quantity, each way) — sent to the grid
    // AND to fetchAllIds so "Select all in filter" and > Review walk the SAME
    // order the user is looking at. 'default' keeps the server's flag order.
    if (f.sort && f.sort !== 'default') params.sort = f.sort
    // Resolution tier — a facet like the flags; also flows to fetchAllIds so
    // "Select all in filter" stays scoped to the active tier.
    if (f.resBucket) params.res_bucket = f.resBucket
    // Framing bucket (face/bust/body/back/unknown) — a facet like the flags.
    if (f.framing) params.framing = f.framing
    // Origin state (ai/camera/unknown) — a facet like the flags.
    if (f.origin) params.origin = f.origin
    // 🎨 what the picture is made of, and ⤢ where the head points. Their OWN
    // query keys, so they compose with search/exclude/sort instead of fighting
    // them for one.
    if (f.medium) params.medium = f.medium
    if (f.angle) params.angle = f.angle
    return params
  }, [])

  const refreshImages = useCallback(async (f = filter, off = offset, view) => {
    // `view` (optional) lets a caller drive the fetch with values it just set,
    // dodging state-closure lag; otherwise read the current selection view.
    const on = view ? view.on : showSelected
    const order = view ? view.order : selectedOrder
    const params = on
      ? { ids: (order || []).join(','), offset: String(off), limit: String(PAGE_SIZE) }
      : { ...filterParams(f), offset: String(off), limit: String(PAGE_SIZE) }
    try {
      const d = await apiFetch(`/api/bank/${bankId}/images?${new URLSearchParams(params)}`)
      setPage(d)
    } catch { /* transient — next poll retries */ }
  }, [bankId, filter, offset, filterParams, showSelected, selectedOrder])

  useEffect(() => { refreshImagesRef.current = refreshImages }, [refreshImages])

  // The chip counters, re-measured whenever the filter — or the data under it —
  // moves. Skipped entirely while nothing is filtered: there the payload's
  // bank-wide numbers ARE the answer, so the unfiltered bank costs exactly what
  // it did before. `sort` is stripped: reordering the same rows cannot change a
  // count, and leaving it in would re-fetch every time the user changes the
  // grid's order. A failed fetch keeps the last read rather than falling back to
  // the bank-wide totals — flashing 4 043 for one tick is the very bug this
  // replaces.
  const facetDeps = facetDataKey(payload?.counts, payload?.thresholds)
  useEffect(() => {
    if (!isFacetFiltered(filter)) { setFacets(null); return undefined }
    let alive = true
    const params = filterParams(filter)
    delete params.sort
    apiFetch(`/api/bank/${bankId}/facets?${new URLSearchParams(params)}`)
      .then((d) => { if (alive) setFacets(d) })
      .catch(() => { /* transient — the chips keep their last read */ })
    return () => { alive = false }
  }, [bankId, filter, filterParams, facetDeps])

  useEffect(() => {
    // Opening ANOTHER bank without unmounting (the workspace is not keyed by id)
    // has to pick up THAT bank's remembered order, not keep the previous one —
    // and the first fetch must already use it, hence the explicit filter here
    // instead of a setFilter that the fetch below would race.
    const f = { ...filter, sort: loadBankSort(bankId) }
    if (f.sort !== filter.sort) setFilter(f)
    // A new bank starts a new announcement history: without this reset, the
    // first payload of the next bank could re-toast its old finished snapshot.
    announcedActivityAt.current = undefined
    refreshPayload({ force: true }); refreshImages(f)
    apiFetch(`/api/bank/${bankId}/subfolders`)
      .then((d) => setSubfolders(d.subfolders || []))
      .catch(() => setSubfolders([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bankId])

  // Poll while a job runs; refresh the grid once when it lands.
  const live = payload?.activity && !payload.activity.finished
  useEffect(() => {
    if (!live) {
      /* The landing is announced PER JOB GENERATION (started_at), not per
         "the poll saw it running". The old gate was activityWasLive, and it
         silenced every job that finished before the first 2 s tick — a 👥 pass
         whose 27 images were already cached ran, landed and said absolutely
         nothing: no bar (too fast, fine), no toast, and the user launched it
         three more times looking for a reaction. `undefined` means "first
         payload after opening this bank": that snapshot can carry a finished
         job up to 5 min old, which is recorded silently — announcing it would
         toast stale news on every open. */
      const act = payload?.activity
      if (act?.finished && announcedActivityAt.current !== undefined
          && act.started_at !== announcedActivityAt.current) {
        announcedActivityAt.current = act.started_at ?? null
        refreshImages()
        if (act.error) toast.error(`Job failed — ${act.error}`)
        else if (act.cancelled && act.detail)
          toast.info(act.detail)   // Stopped — N cached (M remaining)…
        else if (act.detail) toast.success(act.detail)
      }
      return undefined
    }
    // ⏱ The 2 s tick asks for the JOB, not the dashboard. It used to re-fetch the
    // whole bank payload — ~60 bank-wide aggregates plus a per-image path walk —
    // which took 12.5 s at rest on a 50 397-image bank and 28.9 s while a pass
    // ran. Requests issued every 2 s that take 12 s stack up, and that payload is
    // what carries this banner: the bank went blank and Stop stopped answering,
    // at the one moment both were needed. /activity reads one indexed row.
    // The dashboard still refreshes, on its own slower beat, and IMMEDIATELY when
    // the job lands so the final counts are never a poll behind.
    let dropped = false
    const tick = async () => {
      let next = null
      try {
        // background: this tick is the one that stacked ten "Connection lost"
        // banners over the whole app when the phone's connection dropped.
        next = await apiFetch(`/api/bank/${bankId}/activity`, { background: true })
      } catch { return }               // transient — the next tick retries
      if (dropped) return
      setPayload((p) => (p ? { ...p, activity: next.activity } : p))
      const landed = !next.activity || next.activity.finished
      if (landed || Date.now() - fullPayloadAt.current >= FULL_PAYLOAD_MS) {
        fullPayloadAt.current = Date.now()
        refreshPayload({ background: true })
      }
    }
    const t = setInterval(tick, 2000)
    return () => { dropped = true; clearInterval(t) }
    // Les TRANCHES de activity, jamais l'objet : son identite change a
    // chaque poll et relancerait l'intervalle toutes les 2 s.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live, bankId, refreshPayload, refreshImages, toast,
      payload?.activity?.error, payload?.activity?.cancelled,
      payload?.activity?.detail, payload?.activity?.started_at])

  // Keep the coverage panel current: refetch when it opens, and whenever the kept
  // set or the framing classification changes (a keep/reject or the framing pass).
  useEffect(() => {
    if (coverageOpen) loadCoverage()
  }, [coverageOpen, loadCoverage, payload?.counts?.keep,
      payload?.counts?.framing_classified,
      // The panel now also reads captions and embeddings, so it must refresh
      // when those passes land — otherwise it keeps showing "no captions yet"
      // after the 🏷️ pass finished.
      payload?.counts?.captioned, payload?.counts?.scored,
      payload?.counts?.semantic_indexed, payload?.semantic?.engine])

  const {
    folderPersons, folderPersonInfo, folderPersonBusy, loadFolderPersons,
    assertFolderPerson, revokeFolderPerson, checkFolderPerson,
    scanFolderPersons,
  } = useFolderPersons({
    bankId, live, filter, toast, refreshImages, refreshPayload,
  })


  const openSourceFolder = async () => {
    if (openingSourceFolder) return
    setOpeningSourceFolder(true)
    try {
      await postJson(`/api/bank/${bankId}/open-source-folder`, {})
    } catch (e) {
      toast.error(e?.message || 'Could not open the bank source folder')
    } finally {
      setOpeningSourceFolder(false)
    }
  }


  // Leaving the selection view: back to the facet grid.
  const exitSelectionView = () => { setShowSelected(false); setSelectedOrder(null) }

  // The "Show selected (N)" / "Show all" toggle. Entering keeps any curate ranking
  // (selectedOrder); a plain manual selection shows in insertion order.
  const toggleSelectionView = () => {
    if (showSelected) {
      exitSelectionView(); setOffset(0); refreshImages(filter, 0, { on: false })
    } else {
      const order = (selectedOrder && selectedOrder.length) ? selectedOrder : [...selected]
      setSelectedOrder(order); setShowSelected(true); setOffset(0)
      refreshImages(filter, 0, { on: true, order })
    }
  }

  const setF = (patch) => {
    const f = { ...filter, ...patch }
    setFilter(f); setOffset(0); setSelected(new Set()); exitSelectionView()
    refreshImages(f, 0, { on: false })
  }

  // Sort only reorders — the same rows match, so the selection (a set of ids)
  // is kept; just jump back to page 1 to read the new order top-down. A facet
  // sort has no meaning inside the selection view, so it drops back to the grid.
  const setSort = (sort) => {
    const f = { ...filter, sort }
    saveBankSort(bankId, sort)
    setFilter(f); setOffset(0); exitSelectionView()
    refreshImages(f, 0, { on: false })
  }

  // 🏷️ Open the chip row on an image, with nothing ticked yet: reading the tags
  // is not the same act as filtering by them, and auto-applying all of them would
  // usually return that one image alone.
  //
  // It CLEARS the selection, because the selection now drives the same row: with
  // one live, this button would set a source the row never reads and read as dead.
  const openTagPicker = (img) => {
    setTagSource(img)
    setTagPicked(new Set())
    setTagFreeze(null)
    setSelected(new Set())
    if (filter.tags) setF({ tags: null })
  }

  // Tick / untick one chip and re-filter immediately — the grid IS the feedback,
  // so an Apply button would only add a click between the question and its answer.
  //
  // `basis` is the row the chip was clicked in. When it is the LIVE selection it
  // is frozen first: setF empties the selection on the next line, and a row that
  // vanished the moment you used it would read as a crash.
  const toggleTag = (tag, basis = null) => {
    const next = new Set(tagPicked)
    if (next.has(tag)) next.delete(tag); else next.add(tag)
    setTagPicked(next)
    if (basis && basis.kind === 'selection' && !basis.frozen) {
      setTagFreeze({ ...basis, frozen: true })
    }
    setF({ tags: tagsParam(next) })
  }

  const clearTags = () => {
    setTagSource(null)
    setTagFreeze(null)
    setTagPicked(new Set())
    if (filter.tags) setF({ tags: null })
  }

  // --- 🏷️ The captions the tag row counts over ------------------------------
  // Every image the grid has ever rendered leaves its caption here, so ticking
  // tiles on the page costs ZERO requests. Only ids the grid never showed — what
  // "Select all in filter" produces — are fetched, and only those.
  const captionCache = useRef(null)
  if (captionCache.current === null) captionCache.current = new Map()
  const [captionsSeen, setCaptionsSeen] = useState(0)
  useEffect(() => {
    let added = false
    for (const im of page.images || []) {
      if (!captionCache.current.has(im.id)) added = true
      captionCache.current.set(im.id, im.caption || '')
    }
    if (added) setCaptionsSeen((n) => n + 1)
  }, [page])

  // The ids we never rendered. Fetched in ONE request and capped: `ids=` rides in
  // the QUERY STRING, and a selection of a few thousand integers builds a request
  // line the server rejects outright. What the cap leaves out is stated in the row
  // rather than folded silently into the denominator.
  useEffect(() => {
    const ids = [...selected]
    const missing = ids.filter((id) => !captionCache.current.has(id))
    if (!missing.length) return undefined
    const batch = missing.slice(0, TAG_CAPTION_FETCH_CAP)
    let cancelled = false
    const t = setTimeout(async () => {
      try {
        const qs = new URLSearchParams({
          ids: batch.join(','), limit: String(batch.length),
        })
        const d = await apiFetch(`/api/bank/${bankId}/images?${qs}`, { background: true })
        if (cancelled) return
        for (const im of d.images || []) captionCache.current.set(im.id, im.caption || '')
        setCaptionsSeen((n) => n + 1)
      } catch { /* the row then counts what it has and says how many it could not read */ }
    }, 300)
    return () => { cancelled = true; clearTimeout(t) }
  }, [selected, bankId])

  // A new selection replaces a frozen one: a snapshot of last question's images
  // sitting above a fresh selection is a row that describes something else.
  useEffect(() => { if (selected.size) setTagFreeze(null) }, [selected])

  const selectionTags = useMemo(() => {
    if (!selected.size) return null
    const ids = [...selected]
    const known = ids.filter((id) => captionCache.current.has(id))
    return {
      kind: 'selection',
      ...selectionTagCounts(known.map((id) => captionCache.current.get(id))),
      unread: ids.length - known.length,
      size: ids.length,
    }
    // captionsSeen is the dependency that matters — the cache is a ref, so React
    // cannot see it change on its own.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, captionsSeen])

  /* WHICH row is on screen, in priority order. A frozen selection outranks a live
     one (it IS the live one, held still while its filter runs); a selection
     outranks the 🏷️ button, because it is the more recent gesture. */
  const tagRow = tagFreeze || selectionTags || (tagSource ? {
    kind: 'image',
    name: tagSource.name,
    ...selectionTagCounts([tagSource.caption]),
    unread: 0,
    size: 1,
  } : null)

  // Debounce the search box, then apply it as a filter (page 1, selection cleared).
  useEffect(() => {
    const term = searchText.trim()
    if ((filter.search || '') === term) return undefined
    const t = setTimeout(() => setF({ search: term || null }), 300)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchText])

  // Same debounce for the exclude box. It is a FILTER like any other, so it goes
  // through setF: page 1, selection cleared, and it rides to the curation
  // endpoints too (filterParams) — hiding an image in the grid must also keep it
  // out of "pick 60 diverse".
  useEffect(() => {
    const term = excludeText.trim()
    if ((filter.exclude || '') === term) return undefined
    const t = setTimeout(() => setF({ exclude: term || null }), 300)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [excludeText])
  const goto = (off) => { setOffset(off); refreshImages(filter, off) }

  /* `onRefusal` is for a caller that OWNS a surface for the refusal — today the
     🚀 Launch all dialog, which draws it next to the checkboxes it is about and
     stays open so they survive. When it is given, the message goes THERE instead
     of to a toast: two copies of the same sentence is not twice as clear. Every
     other button keeps the toast, because it has nowhere else to put it. */
  /* ⏱ The CHEAP liveness read, merged straight into the payload. The full
     payload is the ~60-aggregate dashboard, measured at ~25 s AT REST on a
     36 921-image bank and worse while a pass writes — so between a click and
     that landing, the page believed the bank was idle: no progress bar, while
     every further click 409'd about a pass "in the progress bar at the top of
     the bank" that was not on screen. /activity is one in-memory registry
     lookup; adopting its answer is what flips `live`, starts the 2 s poll and
     puts the bar up within a beat of the click instead of half a minute later.
     The merge is guarded on an existing payload: with none, nothing that could
     show a bar has rendered yet, and the payload on its way carries activity. */
  const adoptActivity = async () => {
    try {
      const next = await apiFetch(`/api/bank/${bankId}/activity`, { background: true })
      setPayload((p) => (p ? { ...p, activity: next.activity } : p))
      return next.activity
    } catch { return null }
  }

  const act = async (fn, okMsg, { onRefusal } = {}) => {
    try {
      const d = await fn()
      if (okMsg) toast.success(okMsg)
      // The bar first, the dashboard after: a 202 means a job is running RIGHT
      // NOW, and the heavy payload it rides in can be half a minute away.
      await adoptActivity()
      await refreshPayload(); await refreshImages()
      return d
    } catch (e) {
      // ONE bank, ONE background job: every action here can be refused because
      // another pass owns the bank. That refusal used to reach the user as the
      // server's own sentence — "a scan job is already running on this bank" —
      // which names no progress and no way out. The route now labels the 409
      // with `busy_kind`, so it is rewritten HERE, once, for every button that
      // goes through act(): the ✨ Analyze, the ↻ re-runs in the threshold
      // panel, 🗑 Delete rejected, ⬆ Promote, 🚀 Launch all. Anything else keeps
      // its own message — only a refusal that identified itself is reworded.
      // The refusal reads /activity itself (and adoptActivity merges it), so
      // the sentence carries the blocker's real progress AND the bar it points
      // at actually appears — a 409 is proof the page's picture was stale.
      const kind = e?.body?.busy_kind
      const message = e?.status === 409 && kind
        ? await busyRefusalLive({ kind, fetchActivity: adoptActivity,
          fallback: payload?.activity })
        : (e?.message || 'Action failed.')
      if (onRefusal) onRefusal(message)
      else toast.error(message)
      return null
    }
  }

  const runFacesPass = () => act(() => postJson(`/api/bank/${bankId}/faces`, {}), null)

  /* 👤 THE PREFLIGHT GATE.
     The person pass is the bank's most expensive one, and on scraped material
     most of it is spent rediscovering what the folder names already said. The
     app knew how to sample folders and offer the obvious ones — but only from a
     panel the user who presses 🚀 Launch all never opens. So the sampling now
     runs HERE, in front of the pass, whichever button started it.

     `gate` returns:
       false      — nothing worth asking (no subfolder, or all declared already);
                    the caller runs its pass immediately, exactly as before.
       'refused'  — the bank is busy. The refusal is raised at THIS point on
                    purpose: it is the same 409 the pass itself would produce, so
                    the Launch all dialog still gets it while its checkboxes are
                    alive, instead of losing them to a preflight that dies later.
       true       — the dialog is up and owns the decision. */
  const gate = async (run, onRefusal) => {
    let plan = null
    try { plan = await apiFetch(`/api/bank/${bankId}/person-preflight`) } catch { plan = null }
    if (!preflightNeeded(plan)) return false
    if (preflightWillSample(plan)) {
      try {
        await postJson(`/api/bank/${bankId}/person-preflight`, {})
      } catch (e) {
        const kind = e?.body?.busy_kind
        const message = e?.status === 409 && kind
          ? await busyRefusalLive({ kind, fetchActivity: adoptActivity,
            fallback: payload?.activity })
          : (e?.message || 'Could not check the folders.')
        if (onRefusal) onRefusal(message); else toast.error(message)
        return 'refused'
      }
      // The 2 s poll only ticks while a job is live, and the payload we hold
      // predates the one we just started — refresh so the dialog can watch it.
      refreshPayload({ force: true })
    }
    setPreflight({ plan, probing: preflightWillSample(plan), run })
    return true
  }


  /* The answer. Accepted folders become ORDINARY assertions (same endpoint
     family, same revoke) and only then does the pass the user asked for run. */
  const proceedPreflight = async ({ accept }) => {
    const run = preflight?.run
    setPreflight(null)
    if (accept && accept.length) {
      const d = await act(
        () => postJson(`/api/bank/${bankId}/folder-persons/accept`, { subfolders: accept }),
        null)
      if (d) {
        const missed = (d.failed || []).length
        toast.success(`${d.accepted.length} folder(s) · ${d.images} image(s) grouped `
          + 'as one person each — the pass will skip them'
          + (missed ? ` · ${missed} could not be grouped` : ''))
      }
      loadFolderPersons()
    }
    if (run) await run()
  }
  // ✨ Score always covers the WHOLE bank and skips what is already computed —
  // `rescore` is the explicit "recompute it all" lane (new model, scores you no
  // longer trust), exactly like the quality pass's "Rescan all".
  /* 🎛 ONE launcher for every pass window.
     The window hands back WHERE the run applies ({statuses} for a pile, the
     'selection' marker for the images the user ticked, `redo` for the "do it
     again on rows that already have a result" line). Everything is spread-if-set,
     so a run that changes nothing posts the SAME body the button posted before
     the windows existed — the byte-identical contract the caption options already
     followed, now the rule for all of them.

     It answers {ok,error} rather than toasting: the window owns its own refusal
     surface and stays open with the choices intact (utils/submitOutcome.js). */
  const passBody = (passId, { statuses, imageIds, redo } = {}, extra = {}) => {
    const spec = BANK_PASSES[passId]
    return {
      ...(spec?.redo?.explicit
        ? { [spec.redo.key]: !!redo }
        : (spec?.redo && redo ? { [spec.redo.key]: true } : {})),
      ...(statuses ? { statuses } : {}),
      ...(imageIds === 'selection' && selected.size ? { image_ids: [...selected] } : {}),
      ...extra,
    }
  }

  const runPass = async (passId, run, extra = {}) => {
    const spec = BANK_PASSES[passId]
    if (!spec) return { ok: false, error: 'Unknown pass.' }
    let error = null
    const d = await act(
      () => postJson(`/api/bank/${bankId}/${spec.endpoint}`, passBody(passId, run, extra)),
      null, { onRefusal: (m) => { error = m } })
    return d ? { ok: true } : { ok: false, error }
  }

  /* 👥 is the one pass whose launch is not a POST: the folder preflight can take
     ownership of it. When it does, this window has nothing left to do and closes
     on a success — the run is now the preflight dialog's to start. */
  const launchFacesFromDialog = async () => {
    let error = null
    const gated = await gate(runFacesPass, (m) => { error = m })
    if (gated === 'refused') return { ok: false, error }
    if (gated === true) return { ok: true }
    return runPass('faces', {})
  }
  const startCaption = (run) => runPass('caption', run, captionRunOptions())
  const cancelJob = () => act(() => postJson(`/api/bank/${bankId}/cancel`, {}), null)

  const changeSemanticEngine = async (engine) => {
    if (engine === semanticState.engine || semanticSwitching || semanticOperationBusy || live) return
    const previousEngine = semanticState.engine
    setSemanticSwitching(true)
    try {
      const d = await act(
        () => patchJson(`/api/bank/${bankId}/semantic-engine`, semanticEnginePatchBody(engine)),
        `Semantic engine changed to ${engine === 'siglip2' ? 'SigLIP 2' : 'CLIP'} — both caches were kept.`,
      )
      if (d) {
        // A text search may have left the previous engine warm. It no longer
        // belongs to this Bank view after a successful switch.
        releaseTextEncoder(previousEngine)
        // Close engine-specific explanatory overlays; the user's selection,
        // images and both caches stay exactly where they are.
        setCurateOpen(null)
        setTextStatus(null)
        setTextResult(null)
        setBalanceResult(null)
        // A curated order and coverage response belong to one vector space. Keep
        // the selection itself, but return to the ordinary grid and invalidate
        // every old-engine coverage request/result before showing the new engine.
        exitSelectionView()
        setOffset(0)
        refreshImages(filter, 0, { on: false })
        coverageRequestRef.current += 1
        textStatusRequestRef.current += 1
        setCoverage(null)
      }
    } finally {
      setSemanticSwitching(false)
    }
  }
  /* 🔄 THE DESTRUCTIVE TWIN. Same endpoint, same options, plus `force:true` — which
     drops the server's "no caption yet" filter and rewrites the whole pile. It exists
     because 🏷️ Caption greys out at zero uncaptioned rows and takes the engine/model
     selects down with it, leaving a fully captioned bank with no way to redo its
     captions with a better model.

     THREE RULES, all of them about not lying:
     - it ASKS FIRST, with the count of captions it will destroy, in the Dataset's own
       wording (dataset/captionCategory.js) so the app has one way of asking this;
     - it never carries `image_ids`. A selection can span pages that were never loaded,
       so how many selected rows already have a caption is unknowable client-side, and
       this button does not run on a number it cannot state (see
       captionRecaptionDisabledReason). 🏷️ Caption still honours selections;
     - `statuses` rides WITHOUT the `!selected.size` guard the normal pass needs,
       precisely because a selection makes this button inert instead. */
  const startRecaption = async () => {
    if (captionRecaptionDisabledReason(selected.size, live, counts, captionScope,
                                       captionIncludeAsserted)) {
      return { ok: false, error: null }
    }
    if (!window.confirm(captionRecaptionConfirmation(counts, captionScope,
                                                     captionIncludeAsserted))) {
      // A declined confirmation is not a failure: the window stays as it is and
      // says nothing (the question they just answered IS the explanation).
      return { ok: false, error: null }
    }
    let error = null
    const d = await act(() => postJson(`/api/bank/${bankId}/caption`, {
      force: true,
      // Sent ONLY when ticked. Omitting the key is the protected reading, on this
      // side as on the server's — the destructive one is never the default shape
      // of the request.
      ...(captionIncludeAsserted ? { include_asserted: true } : {}),
      ...captionRunOptions(),
      ...(captionScopeStatuses(captionScope)
        ? { statuses: captionScopeStatuses(captionScope) } : {}),
    }), null, { onRefusal: (m) => { error = m } })
    return d ? { ok: true } : { ok: false, error }
  }
  /* Posts with the dialog still OPEN and answers {ok,error}: a refused launch —
     "a scan job is already running on this bank" is the usual one — used to close
     the dialog first and reset all seven pass checkboxes and the reject flags to
     their defaults. The dialog decides what to do with the answer.

     🚀 Launch all is also the FIRST thing a new user presses, so it is the path
     the folder check has to be on — otherwise that saving only ever reaches the
     people who went looking for it. The question is asked BEFORE the run starts,
     which is the only honest place for it: the point of Launch all is walking
     away, and a dialog that woke the user three passes in would defeat that.
     The gate posts too, so a busy bank is refused HERE, while the checkboxes
     are still alive — the rule this whole contract is about. */
  const startPipeline = async (config) => {
    if ((config?.steps || []).includes('faces')) {
      let error = null
      const gated = await gate(() => runPipeline(config), (m) => { error = m })
      if (gated === 'refused') return { ok: false, error }
      // The preflight dialog owns the launch now; the Launch all card can close.
      if (gated === true) { setLaunchOpen(false); return { ok: true } }
    }
    return runPipeline(config)
  }

  const runPipeline = async (config) => {
    let error = null
    const d = await act(() => postJson(`/api/bank/${bankId}/pipeline`, config),
      'Launch all started — you can walk away; Stop any time.',
      { onRefusal: (m) => { error = m } })
    if (!d) return { ok: false, error }
    setLaunchOpen(false)
    return { ok: true }
  }

  const batchStatus = async (ids, status) => {
    if (!ids.length) return
    await act(() => postJson(`/api/bank/${bankId}/images/status`, { ids, status }),
      `${ids.length} image(s) → ${status}`)
    setSelected(new Set())
    // The selection is gone, so its view has nothing left to show — return to the grid.
    if (showSelected) { exitSelectionView(); setOffset(0); refreshImages(filter, 0, { on: false }) }
  }

  // ↩ Put the last bulk decision back. The reply is a ledger, not an "ok": it
  // is reported verbatim, so a restore that only got 340 of 400 rows back says
  // which ones it could not take and why.
  const undoLast = async () => {
    setUndoBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/undo`, {})
      const msg = undoResultMessage(d)
      toast[msg.type === 'error' ? 'error' : msg.type](msg.text)
      setSelected(new Set())
      if (showSelected) exitSelectionView()
      await refreshPayload(); await refreshImages()
    } catch (e) {
      toast.error(e?.message || 'Undo failed — nothing was changed.')
      await refreshPayload()
    } finally {
      setUndoBusy(false)
    }
  }

  // 🔄 Quarter turns on the selection (idea by 1Tomber, GitHub #17). A whole
  // scraped folder can come out sideways, so this is a BULK action here rather
  // than a per-tile button — the tile already carries the selection gesture, the
  // status badges and two hit targets.
  const rotateSelection = async (degrees) => {
    const ids = [...selected]
    if (!ids.length) return
    const d = await act(() => postJson(`/api/bank/${bankId}/rotate`, { ids, degrees }),
      null)
    if (d?.rotated) {
      toast.success(`${d.rotated} image(s) rotated 90° ${degrees === 90 ? 'right' : 'left'}`
        + ' — your own files are untouched.')
      await refreshImages()
    }
  }

  const applyAutoReject = async () => {
    setShowAutoReject(false)
    const flags = [...rejectFlags]
    const d = await act(() => postJson(`/api/bank/${bankId}/apply-flags`, { flags }), null)
    if (d?.rejected) {
      const n = Object.values(d.rejected).reduce((a, b) => a + b, 0)
      // Zero is not a success. It used to be announced in green as "0 image(s)
      // rejected", which on a second pass is the exact shape of a broken
      // feature — the pass had nothing left to do because the first one had
      // already rejected them all, and nothing said so.
      if (n === 0) {
        toast.info(`Auto-reject: nothing to do (${flags.join(', ')}) — every image carrying`
          + ' these flags has already been decided. Manual ✓/✕ are never re-flipped.')
      } else {
        toast.success(`Auto-reject: ${n} image(s) rejected (${flags.join(', ')}). Manual ✓/✕ untouched.`)
      }
    }
  }

  const {
    review, reviewLoading, openReview, onReviewDecided, onReviewRotated,
    onReviewEdited, closeReview,
  } = useReviewLightbox({
    bankId, filter, filterParams, fetchAllIds, showSelected, selected,
    selectedOrder, setPage, toast, refreshPayload, refreshImages,
  })

  const selectAllCurrent = async () => {
    try {
      const ids = await fetchAllIds(bankId, filterParams(filter))
      setSelected(new Set(ids))
      toast.info(`${ids.length} image(s) selected (whole filter, all pages).`)
    } catch (e) {
      toast.error(e?.message || 'Selection failed.')
    }
  }

  // --- Curation selectors (read the selected semantic index) -----------------
  // Both build a SELECTION the user then reviews with the existing ✓/✕/Promote
  // bar — nothing is auto-kept or deleted. The candidate pool is the current
  // filter (composable), so "60 most diverse of this subfolder" just works.
  // Curate a selection AND switch to the "show selected" view so the result is
  // actually visible (60 ids scattered across a 24k-image bank are invisible as
  // mere checkmarks). `order` is the ids in the order the grid should render them.
  const showCuratedSelection = (order) => {
    setSelected(new Set(order))
    setSelectedOrder(order)
    setShowSelected(true)
    setOffset(0)
    refreshImages(filter, 0, { on: true, order })
  }

  // The typicality guard reads the whole pool's neighbourhood before sampling, so
  // on a big bank this click is no longer instant — say so instead of looking dead.
  const semanticState = semanticEngineState(payload, capsLoading ? null : caps)
  semanticEngineRef.current = semanticState.engine
  semanticModelKeyRef.current = semanticState.modelKey
  const semanticReady = semanticState.ready
    && !(semanticState.engine === 'siglip2' && capsLoading)
  const semanticIndexed = semanticState.indexed
  const semanticBlocked = semanticState.engine === 'siglip2' && capsLoading
    ? 'Checking whether the SigLIP 2 Quality tool is installed…'
    : semanticPrerequisite(semanticState)
  const {
    curateOpen, setCurateOpen, diverseN, setDiverseN, diverseTypicality,
    setDiverseTypicality, diverseBusy, balanceN, setBalanceN, balanceAxis,
    setBalanceAxis, balanceBusy, balanceResult, setBalanceResult, similarN,
    setSimilarN, similarBusy, similarLast, similarAddN, setSimilarAddN,
    addMoreSimilar, textQuery, setTextQuery, textN, setTextN,
    textExclude, setTextExclude, textExcludeW, setTextExcludeW, textStatus,
    setTextStatus, textPending, textResult, setTextResult, pickDiverse,
    pickBalanced, findSimilar, openTextSearch, releaseTextEncoder,
    runTextSearch,
  } = useCurationLanes({
    bankId, filter, filterParams, selected, toast, showCuratedSelection,
    semanticState, semanticEngineRef, textStatusRequestRef,
  })
  const semanticOperationBusy = diverseBusy || balanceBusy || similarBusy || textPending

  const counts = payload?.counts
  /* A freshly imported dump has nothing measured: the grid is a wall of
     unscanned images and every useful action is inside the passes panel, so it
     opens ITSELF once. Guarded on the computed boolean rather than on `counts`,
     so the first scan flipping it to false never re-runs this — and so a user
     who closes the panel on such a bank is not fought by the effect. */
  const passesShouldOpen = passesPanelStartsOpen(counts, viewportWidth())
  useEffect(() => { if (passesShouldOpen) setPassesOpen(true) }, [passesShouldOpen])
  // The Sort menu greys an entry out when its pass has measured NOTHING. Face
  // confidence is the one whose progress the payload reports outside `counts`
  // (faces_scanned, a sibling key), so it is folded in here rather than by
  // changing the payload shape every other reader depends on.
  const sortGroups = bankSortGroups(
    counts ? { ...counts, faces: payload?.faces_scanned } : counts)
  // ↩ the live offer, minus the one the user already waved away.
  const offer = undoOffer(payload)
  const undoBar = offer && offer.at !== undoDismissedAt ? offer : null
  // `print` = what each chip SHOWS, measured under the filters in force; `wide` =
  // the bank-wide truth, which is what decides a chip is offered at all. Keeping
  // visibility on `wide` is deliberate: chips that disappeared as a filter
  // emptied them would leave no way back to the values you just excluded.
  const { print: chipPrint, wide: chipWide, filtered: chipsFiltered } =
    chipCounts(payload, facets)
  const flags = chipPrint.flags
  // A THIRD question, kept as its own map and deliberately NOT filtered:
  // `flags` is "images this chip would show" and `flags_actionable` is "what
  // 🧹 Auto-reject would flip" — a pass that runs over the whole bank, not over
  // the current view, so narrowing its number to the filter would under-promise
  // exactly as badly as the chips used to over-promise.
  const flagsActionable = payload?.flags_actionable || {}
  const resBuckets = chipPrint.resBuckets
  // Only surface tiers that actually hold scanned images (plus the active one,
  // so a tier you're filtering on never vanishes mid-review).
  const shownResBuckets = RES_BUCKETS.filter(
    (b) => (chipWide.resBuckets[b.id] || 0) > 0 || filter.resBucket === b.id)
  const clusters = payload?.clusters || []
  const styleClusters = payload?.style_clusters || []
  const framingCounts = chipPrint.framing
  // Only surface framing chips once the pass has classified something (plus the
  // active one, so a chip you're filtering on never vanishes mid-review).
  const shownFramings = FRAMING_BUCKETS.filter(
    (b) => (chipWide.framing[b.id] || 0) > 0 || filter.framing === b.id)
  // Origin chips appear as soon as the quality scan has measured anything. All
  // three are shown together once any is non-zero: hiding 'ai' at 0 would read as
  // "no AI images here", when what it means is "none that still carry metadata".
  const originCounts = chipPrint.origins
  const originMeasured = ORIGIN_BUCKETS.reduce(
    (n, b) => n + (chipWide.origins[b.id] || 0), 0)
  // 🎨 Medium / ⤢ Angle — same "only show what holds something, plus the active
  // one" rule as the framing and resolution rows.
  const mediumCounts = chipPrint.mediums
  const shownMediums = shownBuckets(MEDIUM_BUCKETS, chipWide.mediums, filter.medium)
  // The limits sentence is about the MEASUREMENT, not about the current view: it
  // weighs the buckets against how many rows the pass classified bank-wide, so
  // it reads the wide map or it would announce a fake blind spot on every filter.
  const mediumNote = mediumLimits(chipWide.mediums, counts?.medium_classified)
  const angleCounts = chipPrint.angles
  const shownAngles = shownBuckets(ANGLE_BUCKETS, chipWide.angles, filter.angle)
  const angleState = angleReadiness(payload)
  const visionReady = !!caps.ollama?.vision_model_ready
  // The explicit lane only spells acts out with an uncensored (abliterated) vision
  // model. We can't prove abliteration, but the common builds name themselves — a soft
  // heuristic drives an honest "may soften" hint (never a hard block: a differently
  // named abliterated model still works).
  // 🔄 Re-caption: inert (and why), plus the sentence that names what it destroys.
  const includeAssertedLabel = captionIncludeAssertedLabel(counts, captionScope)
  const recaptionInert = captionRecaptionDisabledReason(
    selected.size, live, counts, captionScope, captionIncludeAsserted)
  const recaptionNote = captionRecaptionNote(selected.size, live, counts, captionScope,
                                             captionIncludeAsserted)

  /* 🏷️ THE FIVE CAPTION OPTIONS, now INSIDE the caption window — literally the
     maintainer's example of what these windows are for ("this way we can gather
     all the caption options"). Nothing was dropped in the move: engine, model,
     register, length, the pile (which became the window's THIS RUN block) and
     🔄 Re-caption with both its figures and its confirmation.

     The state stays out here, in the workspace, so closing the window does not
     reset a choice the user made. Each select keeps the width rules measured at a
     400 px viewport: only the MODEL one is capped at 11rem (Ollama refs run long
     and overflow on their own); the rest are max-w-full with a 16rem ceiling from
     sm up, because capping them by symmetry truncated them into nonsense
     ("Standard — the prompt as⌄"). */
  const captionSelectClass = 'mt-0.5 w-full rounded-lg border border-border bg-app/60 '
    + 'px-2 py-1 text-xs text-content disabled:opacity-40'
  /* Rendered UNDER the engine picker, because it is about the engine and the picker
     is the lever it names. It re-computes as the engine changes, so ticking
     JoyCaption turns the warning into its own confirmation instead of leaving an
     alarm on screen about a half that will no longer run. */
  /* 🔤 Find text — the launch window's own dials. The sample is PER RUN
     (session state: a tryout is a gesture, not a preference); the sensitivity
     is the STORED text_scan.score_min both surfaces read, edited here because
     this window is where its effect is judged (scan a sample → ▶ Review the
     zones → adjust → re-read the same sample). Written through on release,
     like the dataset's allow-crop switch. */
  const [textSampleOn, setTextSampleOn] = useState(false)
  const [textSampleSize, setTextSampleSize] = useState(20)
  const storedTextSensitivity = Number.isFinite(Number(caps.text_scan_score_min))
    ? Number(caps.text_scan_score_min) : 0.5
  const [textSensitivity, setTextSensitivity] = useState(null)
  const effectiveTextSensitivity = textSensitivity ?? storedTextSensitivity
  const saveTextSensitivity = async (value) => {
    try {
      await putJson('/api/settings', { config: { text_scan: { score_min: value } } })
    } catch { /* the run still uses the stored value; the slider shows what failed */ }
  }
  const textScanRunOptions = () => (textSampleOn
    ? { limit: Math.max(1, Math.min(10000, Math.round(Number(textSampleSize) || 20))) }
    : {})
  /* 🚩 Find watermarks — brought to the SAME standard as 🔤 on the
     maintainer's ask ("the same kind of menu — tuning and visualisation —
     dataset and bank alike"): a per-run sample, the STORED detector threshold
     edited where its effect is judged (write-through; the dataset window edits
     the same value), and the flagged pages drawn in the window. */
  const [wmSampleOn, setWmSampleOn] = useState(false)
  const [wmSampleSize, setWmSampleSize] = useState(20)
  const storedWmThreshold = Number.isFinite(Number(caps.watermark_detect_threshold))
    ? Number(caps.watermark_detect_threshold) : 0.94
  const [wmThreshold, setWmThreshold] = useState(null)
  const effectiveWmThreshold = wmThreshold ?? storedWmThreshold
  const saveWmThreshold = async (value) => {
    try {
      await putJson('/api/settings', { config: { watermark_detect: { threshold: value } } })
    } catch { /* the run still uses the stored value; the slider shows what failed */ }
  }
  const wmScanRunOptions = () => (wmSampleOn
    ? { limit: Math.max(1, Math.min(10000, Math.round(Number(wmSampleSize) || 20))) }
    : {})
  const watermarkScanControls = (
    <div className="space-y-2 rounded-md border border-border bg-surface-raised p-2">
      <p className="m-0 text-[11px] font-semibold uppercase tracking-wide text-content-muted">
        Options for this run
      </p>
      <label className="flex items-start gap-2 text-[11px] text-content-subtle">
        <input type="checkbox" className="mt-0.5" checked={wmSampleOn}
          onChange={(e) => setWmSampleOn(e.target.checked)} disabled={live} />
        <span>
          <span className="font-medium text-content">Try on a sample first</span>
          {' — judge only the first '}
          <input type="number" min="1" max="10000" value={wmSampleSize}
            onChange={(e) => setWmSampleSize(e.target.value)}
            disabled={live || !wmSampleOn} aria-label="Sample size"
            className="mx-1 w-16 rounded border border-border bg-app px-1 py-0.5 text-content" />
          {' images of the scope. Judge the boxes below, then run again for the '}
          {'rest — or tick “re-check” above to re-judge the SAME sample at '}
          {'another threshold.'}
        </span>
      </label>
      {caps.watermark_detect ? (
        <label className="block text-[11px] text-content-subtle">
          <span className="font-medium text-content">Detector threshold</span>
          {' — the score an image needs to be flagged as watermarked. Lower '}
          {'flags fainter marks at the cost of false flags; higher keeps only '}
          {'the confident ones. Stored: the dataset scan reads the same value.'}
          <span className="mt-1 flex items-center gap-2">
            <input type="range" min="0.50" max="0.99" step="0.01"
              value={effectiveWmThreshold} disabled={live}
              aria-label="Detector threshold"
              onChange={(e) => setWmThreshold(Number(e.target.value))}
              onMouseUp={() => saveWmThreshold(effectiveWmThreshold)}
              onTouchEnd={() => saveWmThreshold(effectiveWmThreshold)}
              onKeyUp={() => saveWmThreshold(effectiveWmThreshold)}
              className="w-40" />
            <span className="tabular-nums text-content">{effectiveWmThreshold.toFixed(2)}</span>
            <span className="text-content-subtle">(default 0.94)</span>
          </span>
        </label>
      ) : (
        <p className="m-0 text-[11px] leading-snug text-content-subtle">
          This run takes the vision route (the dedicated detector is not
          installed), which answers yes/no with no score — so there is no
          threshold to tune here. Install “Watermark detector” from Setup for
          the ~10× faster scored route.
        </p>
      )}
      {/* The run's RESULT: the 🚩-family flagged pages (text-flagged ones live
          in the 🔤 window) with their boxes, filling in live while the scan
          runs — the window stays open on launch, like 🔤. */}
      <BankZonesPreview bankId={bankId} kind="watermark" live={live} />
    </div>
  )
  const textScanControls = (
    <div className="space-y-2 rounded-md border border-border bg-surface-raised p-2">
      <p className="m-0 text-[11px] font-semibold uppercase tracking-wide text-content-muted">
        Options for this run
      </p>
      <label className="flex items-start gap-2 text-[11px] text-content-subtle">
        <input type="checkbox" className="mt-0.5" checked={textSampleOn}
          onChange={(e) => setTextSampleOn(e.target.checked)} disabled={live} />
        <span>
          <span className="font-medium text-content">Try on a sample first</span>
          {' — read only the first '}
          <input type="number" min="1" max="10000" value={textSampleSize}
            onChange={(e) => setTextSampleSize(e.target.value)}
            disabled={live || !textSampleOn} aria-label="Sample size"
            className="mx-1 w-16 rounded border border-border bg-app px-1 py-0.5 text-content" />
          {' images of the scope. Judge the zones in ▶ Review (flagged), then '}
          {'run again for the rest — or tick “re-read” above to try the SAME '}
          {'sample at another sensitivity.'}
        </span>
      </label>
      <label className="block text-[11px] text-content-subtle">
        <span className="font-medium text-content">Sensitivity</span>
        {' — the OCR confidence a line needs to become a zone. Lower catches '}
        {'fainter or stylised lettering, at the cost of false zones. Stored: '}
        {'the dataset scan reads the same value.'}
        <span className="mt-1 flex items-center gap-2">
          <input type="range" min="0.30" max="0.70" step="0.05"
            value={effectiveTextSensitivity} disabled={live}
            aria-label="Text sensitivity"
            onChange={(e) => setTextSensitivity(Number(e.target.value))}
            onMouseUp={() => saveTextSensitivity(effectiveTextSensitivity)}
            onTouchEnd={() => saveTextSensitivity(effectiveTextSensitivity)}
            onKeyUp={() => saveTextSensitivity(effectiveTextSensitivity)}
            className="w-40" />
          <span className="tabular-nums text-content">{effectiveTextSensitivity.toFixed(2)}</span>
          <span className="text-content-subtle">(default 0.50)</span>
        </span>
      </label>
      {/* The run's RESULT, in the window that launched it: the flagged pages
          with their zones, filling in live while the scan runs (the window
          stays open on launch — stayOpenOnLaunch below). Judge, adjust the two
          dials above, re-run — without leaving. */}
      <BankZonesPreview bankId={bankId} live={live} />
    </div>
  )
  const captionNsfw = captionNsfwNotice({
    payload,
    scopeId: captionScope,
    piles: passScopeOption(captionScope).piles,
    engineId: captionEngine,
  })
  /* WHICH images the bench offers. A dataset hands its whole KEPT pile to the picker;
     a bank cannot — it pages over SQL and can hold six figures of rows, so a picker
     rendering "every non-rejected image" would be a different feature. It offers what
     you are actually looking at: your selection when there is one, otherwise this page.
     Same behaviour ("pick one of these and bench it"), in this surface's mechanics —
     the parity rule's "two surfaces, two mechanics, ONE contract". */
  const labPile = useMemo(() => {
    const rows = page.images || []
    /* A selection SURVIVES paging and sorting (only a filter change clears it), so the
       intersection can legitimately be empty while rows are right there on screen — and
       the button would then grey out saying "nothing on this page to bench", which is
       false. The page is the honest fallback. */
    const scoped = rows.filter((im) => selected.has(im.id))
    return selected.size && scoped.length ? scoped : rows
  }, [page.images, selected])
  /* ⚙️ "Use for the next run" — what ⚙️ Make default means on a surface with nothing
     to persist to. The four dials ARE the bank's caption method (useCaptionOptions:
     "a bank has no caption_options row"), so loading them keeps the dataset button's
     promise: the next pass runs the config that won the bench. */
  const applyLabConfig = useCallback((config) => {
    setCaptionEngine(config.backend || '')
    setCaptionModel(config.ollama_model || '')
    setCaptionVocab(config.vocabulary || '')
    setCaptionLength(config.length || '')
  }, [setCaptionEngine, setCaptionModel, setCaptionVocab, setCaptionLength])
  const captionRunControls = (
    <div className="space-y-2 rounded-md border border-border bg-surface-raised p-2">
      <p className="m-0 text-[11px] font-semibold uppercase tracking-wide text-content-muted">
        Options for this run
      </p>
      <p className="m-0 text-[11px] leading-snug text-content-subtle">
        These override your Settings for this run only — the global values are never
        written from here.
      </p>
      {/* 🧪 The bench, in the window that launches the pass — the Bank's half of the
          entry point the Dataset's Captions section carries. Same bench, same shared
          definition of a candidate (one backend function, pinned by
          test_bank_caption_lab.py); what it offers to bench and what "use this config"
          means are this surface's own. */}
      <div className="flex flex-wrap items-center gap-2 border-b border-border pb-2">
        <button type="button" onClick={() => setLabPickerOpen(true)}
          disabled={live || labPile.length === 0}
          aria-label="Open the Caption Lab"
          title={labPile.length === 0
            ? 'Nothing on this page to bench — clear a filter, or select the images you want.'
            : 'Try up to four caption configs (engine, vision model, register, length) on '
              + 'one image and read them side by side. Nothing is written until you keep a '
              + 'result, and the winning config can be loaded into the dials above.'}
          className="inline-flex min-h-10 items-center rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-semibold text-content disabled:opacity-40 lg:min-h-0">
          🧪 Caption Lab
        </button>
        <span className="text-[11px] leading-snug text-content-subtle">
          Compare engines, models and registers on ONE image before paying for a pass
          over the pile.
        </span>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        <label className="block text-[11px] text-content-subtle">
          Engine
          <select value={captionEngine} onChange={(e) => setCaptionEngine(e.target.value)}
            disabled={live} aria-label="Caption engine"
            title="Which engine writes this run's captions, without changing your Settings. Auto is a CHAIN, not a choice between two: JoyCaption drafts, then Ollama covers whatever it missed."
            className={`${captionSelectClass} sm:max-w-[16rem]`}>
            {/* 'none' is dropped on purpose: "caption with nothing" is not a pass. */}
            {ENGINE_OPTIONS.filter((o) => o.id !== 'none')
              .map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
        </label>
        <label className="block text-[11px] text-content-subtle">
          Vision model
          <select value={captionModel} onChange={(e) => setCaptionModel(e.target.value)}
            disabled={live || !ollamaPicksApply} aria-label="Caption vision model"
            title={ollamaPicksApply
              ? 'Which pulled Ollama vision model writes this run. Your Settings model stays the default and is not changed. Which model writes a caption is not a matter of taste: one that describes things in evasive terms produces captions that are about something slightly other than the images.'
              : 'Only used when the engine can reach Ollama.'}
            className={`${captionSelectClass} sm:max-w-[11rem]`}>
            <option value="">Configured model</option>
            {captionModelChoices.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          {!ollamaPicksApply && (
            <span className="mt-0.5 block text-[11px] leading-snug text-amber-300/90">
              The engine you picked does not reach Ollama, so this choice would change
              nothing — disabled rather than quietly ignored.
            </span>
          )}
        </label>
        <label className="block text-[11px] text-content-subtle">
          Register
          <select value={captionVocab} onChange={(e) => setCaptionVocab(e.target.value)}
            disabled={live} aria-label="Caption vocabulary register"
            title="How captions name nude or sexual content. Explicit needs an uncensored (abliterated) Ollama vision model. Richer, more explicit captions also make the search find more."
            className={`${captionSelectClass} sm:max-w-[16rem]`}>
            {VOCABULARY_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
        </label>
        <label className="block text-[11px] text-content-subtle">
          Length
          <select value={captionLength} onChange={(e) => setCaptionLength(e.target.value)}
            disabled={live} aria-label="Caption length"
            title="How much the captioner writes. Concise aims for one short sentence, Detailed for several - a target the model follows loosely, not a hard cap. Standard leaves the prompt untouched. Longer captions give the search more to match on."
            className={`${captionSelectClass} sm:max-w-[16rem]`}>
            {CAPTION_LENGTH_OPTIONS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
          </select>
        </label>
      </div>
      {captionNsfw && (
        <div role="note" aria-label="What this run is about to caption"
          className={`space-y-1 rounded-md border px-2 py-1.5 ${
            captionNsfw.tone === 'warn'
              ? 'border-amber-400/40 bg-amber-500/10' : 'border-sky-400/40 bg-sky-500/10'}`}>
          <p className={`m-0 text-[11px] font-semibold leading-snug ${
            captionNsfw.tone === 'warn' ? 'text-amber-200' : 'text-sky-200'}`}>
            {captionNsfw.tone === 'warn' ? '⚠ ' : 'ℹ ' }{captionNsfw.heading}
          </p>
          {captionNsfw.paragraphs.map((line) => (
            <p key={line} className={`m-0 text-[11px] leading-snug ${
              captionNsfw.tone === 'warn' ? 'text-amber-100/90' : 'text-sky-100/90'}`}>
              {line}
            </p>
          ))}
        </div>
      )}
      <p className="m-0 text-[11px] leading-snug text-content-subtle">
        {captionScopeNote(selected.size, counts, captionScope)}
      </p>
    </div>
  )

  /* 🔄 THE DESTRUCTIVE TWIN, in the window's footer beside the normal launch —
     a SECOND launch button, not a pass of its own. It keeps everything it had:
     the number it rewrites, the number it spares, its amber warning and its
     window.confirm. And it keeps greying out WITH ITS REASON rather than
     disappearing: on a bank whose only captions are ones you wrote by hand, that
     disabled reason is the only place the protection is visible at all. */
  const captionSecondary = (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={startRecaption} disabled={!!recaptionInert}
          aria-label="Re-caption"
          title={recaptionInert || recaptionNote}
          className="rounded-md border border-amber-400/40 bg-amber-500/10 px-3 py-1.5 text-xs font-semibold text-amber-200 disabled:opacity-40">
          {captionRecaptionLabel(counts, captionScope, recaptionInert,
                                 captionIncludeAsserted)}
        </button>
        {includeAssertedLabel && (
          <label className="flex items-center gap-1 text-[11px] text-amber-400/90">
            <input type="checkbox" checked={captionIncludeAsserted} disabled={live}
              onChange={(e) => setCaptionIncludeAsserted(e.target.checked)}
              aria-label="Also re-caption the captions I wrote by hand"
              className="accent-amber-500" />
            {includeAssertedLabel}
          </label>
        )}
      </div>
      {recaptionInert && (
        <p className="m-0 text-[11px] leading-snug text-amber-300/90">{recaptionInert}</p>
      )}
      {recaptionNote && (
        <p className="m-0 text-[11px] leading-snug text-amber-400/90">{recaptionNote}</p>
      )}
    </div>
  )

  const scored = counts?.scored || 0
  // ⚖️ Can a balanced pick even run? Answered BEFORE the click when we already
  // know (semantic index missing; coverage says nothing is classified) — otherwise the
  // backend answers it with the exact pass and the numbers.
  const balanceReady = balanceReadiness({
    semanticReady, coverage, engineLabel: semanticState.label,
    prerequisite: semanticBlocked,
  })
  // What ✨ Score will really run on — the pass no longer holds the GPU when it
  // computes on the CPU, and the UI must say which of the two is happening.
  const scoreDevice = payload?.score_device
  const scoreNote = scoreDeviceNote(scoreDevice, Boolean(caps.bank_scoring))
  // Does this machine have an NVIDIA card AT ALL? Reported by the same
  // score_device probe, and only meaningful while the pass would run on the
  // CPU (a GPU pass answers gpu:true and stops looking). Undefined until the
  // payload lands — assume a card, so we never flash "no NVIDIA card" at
  // someone who has one.
  const scoreGpuPresent = scoreDevice ? scoreDevice.gpu || !!scoreDevice.gpu_present : true
  const watermarkScanned = counts?.watermark_scanned || 0
  // Score flags only make sense once their pass ran; watermark is its own pass.
  const availableScoreFlags = SCORE_REJECT_FLAGS.filter(
    (f) => (f === 'watermark' ? watermarkScanned : scored) > 0)
  // 🧹 Auto-reject readiness. The never-scanned pile is unreachable by EVERY
  // quality flag (they are all gated on quality_state=='ok'), so the popover
  // says so and names the gesture; the per-flag numbers below come from
  // flags_actionable, which is what the click actually flips.
  const autoRejectNotice = unscannedNotice(counts)
  const autoRejectPicked = pickedCandidates(rejectFlags, flagsActionable)
  const canPromote = (counts?.keep || 0) > 0 || selected.size > 0
  // Is any facet narrowing the grid? Drives the "N shown of TOTAL" readout AND
  // the chip counters, which is why it is one shared predicate: two lists of
  // facet keys drift, and the one here used to be missing 🔎 Origin, 🚫 exclude
  // and the 🏷️ tag chips — so a grid narrowed by those said it was showing
  // everything.
  const isFiltered = isFacetFiltered(filter)


  return (
    /* ── Structure B ──────────────────────────────────────────────────────
       Layout only. An "Encre" skin was trialled here — a scoped token override
       that re-coloured the whole subtree — and dropped: the app keeps its own
       palette, and the Bank must not be the one screen that looks foreign.
       Every surface below therefore speaks the app's semantic utilities
       (bg-surface, border-border, text-content) and nothing else.

       The children are in READING ORDER — top bar, rail, grid — which is also
       the tab order. Nothing here carries a tabindex; the DOM is the
       accessibility contract, and a rail that tabbed out of order would be a
       regression the eye cannot see. */
    <div className="space-y-3">
      {/* ── The top bar ──────────────────────────────────────────────────
          Bank identity, the counters, and the DECISIVE actions. Everything
          here is bank-wide; anything that narrows the grid lives in the rail. */}
      {/* A phone held sideways has ~390 px of fold: the card drops its path and
          counter rows and most of its padding, and becomes a one-line toolbar. */}
      <header data-probe-chrome="header"
        className="space-y-2 rounded-xl border border-border bg-surface px-4 py-3 [@media(max-height:500px)]:flex [@media(max-height:500px)]:flex-nowrap [@media(max-height:500px)]:items-center [@media(max-height:500px)]:gap-3 [@media(max-height:500px)]:space-y-0 [@media(max-height:500px)]:py-1">
        <div className="flex flex-wrap items-center gap-2 [@media(max-height:500px)]:min-w-0 [@media(max-height:500px)]:shrink">
          <button type="button" onClick={onBack}
            className="min-h-10 lg:min-h-0 rounded-md border border-border px-2 py-1 text-xs text-content-muted hover:text-content hover:bg-surface-raised">
            ← Banks
          </button>
          <h1 className="flex items-center gap-2 text-lg text-content"><Archive aria-hidden="true" className="h-4 w-4" /> {payload?.name || `Bank #${bankId}`}</h1>
          {payload?.source_path && (
            /* hidden below sm: opening or moving the folder is a gesture on the
               machine that serves the app, and on a 360-px screen this row alone
               cost 60 px of a fold the header was already taking 38 % of. */
            <div className="hidden min-w-0 grow items-center gap-2 sm:flex [@media(max-height:500px)]:!hidden">
              <p className="min-w-0 grow truncate font-mono text-xs text-content-subtle"
                title={payload.source_path}>
                {payload.source_path}
              </p>
              <button type="button" onClick={openSourceFolder}
                disabled={openingSourceFolder} aria-busy={openingSourceFolder}
                title="Open this Bank's source folder in the system file explorer."
                className="min-h-10 lg:min-h-0 shrink-0 rounded border border-border px-2 py-0.5 text-xs text-content-muted hover:bg-surface-raised hover:text-content disabled:cursor-wait disabled:opacity-60">
                <FolderOpen aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />{openingSourceFolder ? 'Opening…' : 'Open folder'}
              </button>
              {/* Cold path. The folder-sync note below offers this too, but only once
                  the folder is already gone — and the real move is PLANNED: you look
                  for the option before you drag 30 000 files to another drive, not
                  after breaking the bank to discover it could have been repaired. */}
              <button type="button" onClick={() => setRelocating(true)}
                title="Moving this folder to another disk? Point the bank at its new location."
                className="min-h-10 lg:min-h-0 shrink-0 rounded border border-border px-2 py-0.5 text-xs text-content-muted hover:bg-surface-raised hover:text-content">
                <FolderInput aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Move folder…
              </button>
            </div>
          )}
        </div>
        {counts && (
          /* One scrolling line below sm: eight counters wrapping to four rows were
             a quarter of a phone's fold, before a single image. */
          <div className="flex flex-nowrap items-baseline gap-x-4 gap-y-1 overflow-x-auto border-t border-border pt-2 text-sm sm:flex-wrap sm:overflow-visible [@media(max-height:500px)]:hidden">
            <Stat label="images" value={counts.total} />
            <Stat label="scanned" value={counts.scanned} />
            {scored > 0 && <Stat label="scored" value={scored} />}
            {semanticState.hasStatus && (
              <Stat label={`${semanticState.label} semantic-ready`}
                value={semanticState.total > 0
                  ? `${semanticIndexed.toLocaleString()}/${semanticState.total.toLocaleString()}`
                  : semanticIndexed}
                tone={semanticReady ? 'emerald' : undefined} />
            )}
            {watermarkScanned > 0 && <Stat label="watermark-checked" value={watermarkScanned} />}
            <Stat label="undecided" value={counts.pending} />
            <Stat label="kept" value={counts.keep} tone="emerald" />
            <Stat label="rejected" value={counts.reject} tone="rose" />
            <Stat label="promoted" value={counts.promoted} tone="indigo" />
          </div>
        )}
        {/* The decisive actions. ⚙ Passes opens the analysis panel; the other
            three are the ones that change what leaves this bank. */}
        <div className="flex min-w-0 flex-nowrap items-center gap-2 overflow-x-auto border-t border-border pt-2 sm:flex-wrap sm:overflow-visible [@media(max-height:500px)]:ml-auto [@media(max-height:500px)]:flex-nowrap [@media(max-height:500px)]:overflow-x-auto [@media(max-height:500px)]:border-t-0 [@media(max-height:500px)]:pt-0">
          {/* ☰ exists only where the rail cannot sit beside the grid — at 400 px
              it is the ONLY way back to the filters, so it is a real button and
              never a CSS-hidden one. */}
          {!railIsColumnNow && (
            <button type="button" onClick={openRail}
              aria-expanded={railOpen} aria-controls="bank-filter-rail"
              className="min-h-10 lg:min-h-0 rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content hover:bg-surface">
              ☰ Filters
            </button>
          )}
          <button type="button" onClick={togglePasses}
            aria-expanded={passesOpen} aria-controls="bank-passes-panel"
            title="Open the analysis passes — scan, score, group by person, framing, medium, crops, watermarks and captions."
            className="min-h-10 lg:min-h-0 rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content hover:bg-surface">
            {passesButtonLabel(live)}
          </button>
          <button type="button" onClick={() => setLaunchOpen(true)} disabled={live || !(counts?.total > 0)}
            title={`Run the whole triage in one go — scan, auto-reject, Score${semanticState.engine === 'siglip2' ? ', SigLIP 2 semantic index' : ''}, crops/variants, watermarks, group by person and (optionally) caption. Start it and walk away. If the person pass is in, it checks your folders first and asks once, before the run.`}
            className="min-h-10 lg:min-h-0 rounded-md bg-gradient-primary px-4 py-2 text-sm font-bold text-gray-950 shadow disabled:opacity-50">
            <Rocket aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Launch all…
          </button>
          <span className="ml-auto" />
          <button type="button" onClick={() => setPromoteOpen(true)} disabled={live || !canPromote}
            title={canPromote
              ? 'Copy the kept selection into a dataset — or into a brand-new bank, to keep working on a shortlist apart'
              : 'Keep some images first'}
            className="min-h-10 lg:min-h-0 rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-gray-950 disabled:opacity-50">
            ⬆ Promote…
          </button>
          {/* Disabled outright when this bank's folder belongs to a dataset: the
              banner below says it is, and a button that still opened a dialog only
              to be refused there would make that sentence a lie. */}
          <button type="button" onClick={() => setDeleteRejectedOpen(true)}
            disabled={live || !(counts?.reject > 0) || !!payload?.dataset_conflict}
            title={payload?.dataset_conflict
              ? 'This bank sits on a dataset’s image folder — deleting these files would delete the dataset’s images.'
              : (counts?.reject > 0)
                ? 'Delete the rejected images from your disk (OS trash when available). Irreversible — asks you to type DELETE first. Kept images are untouched.'
                : 'No rejected images to delete'}
            className="min-h-10 lg:min-h-0 rounded-md border border-rose-500/50 px-3 py-1.5 text-sm text-rose-300 disabled:opacity-40 hover:bg-rose-500/10">
            <Trash2 aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Delete rejected from disk{(counts?.reject > 0) ? ` (${counts.reject})` : ''}
          </button>
        </div>
      </header>

      {/* A bank created before this was refused can still point at a dataset's
          own image folder. Nothing is repaired behind the user's back — the
          bank stays fully readable — but the fact is stated every time it is
          opened, because the click that hurts (🗑 Delete rejected) is right
          here on this page. */}
      {payload?.dataset_conflict && (
        <div role="alert"
          className="rounded-md border border-rose-500/70 bg-rose-500/15 p-3 text-sm text-rose-100 space-y-1">
          <p className="font-semibold">⛔ This bank sits on a dataset’s image folder</p>
          <p className="text-rose-100/90">{payload.dataset_conflict.message}</p>
          <p className="text-rose-100/90">
            Delete rejected is disabled here. Use Move folder… to point this
            bank at a folder of its own, or remove the bank — removing a bank never
            touches files.
          </p>
        </div>
      )}
      <FolderSyncNote sync={payload?.folder_sync}
        onRelocate={() => setRelocating(true)}
        onForget={() => setForgettingMissing(true)} />

      <ProgressBar activity={payload?.activity} onCancel={cancelJob} offline={!connection.online} />

      <UndoBar offer={undoBar} busy={undoBusy} onUndo={undoLast}
        onDismiss={() => setUndoDismissedAt(undoBar?.at || 0)} />

      {!live && payload?.pipeline_report
        && payload.pipeline_report.finished_at !== dismissedReportAt && (
        <PipelineReport report={payload.pipeline_report}
          onDismiss={() => setDismissedReportAt(payload.pipeline_report.finished_at)} />
      )}

      {/* ⚙ The analysis passes, opened on demand. All eight are here with their
          own dialogs untouched — only the door changed. */}
      {passesOpen && (
        /* data-probe-reading: the panel is what you asked for when you pressed ⚙,
           one tap puts it away, and nothing in it is used against the grid behind
           it (every pass opens its own window) — so it is measured for targets,
           truncation and fill, and charged to no fold budget. Plain comment: this
           is an EXPRESSION position (inside `{passesOpen && ( … )}`). */
        <div id="bank-passes-panel" data-probe-chrome="passes" data-probe-panel="passes" data-probe-reading>
          <BankPassesPanel
            bankId={bankId} payload={payload} counts={counts} live={live}
            compact={!railIsColumnNow}
            caps={caps} capsLoading={capsLoading}
            semanticState={semanticState} semanticReady={semanticReady}
            semanticBlocked={semanticBlocked} semanticSwitching={semanticSwitching}
            semanticOperationBusy={semanticOperationBusy}
            scoreGpuPresent={scoreGpuPresent} scoreDevice={scoreDevice}
            scoreNote={scoreNote} visionReady={visionReady}
            selected={selected} captionScope={captionScope}
            captionVocab={captionVocab} visionModel={visionModel}
            visionModelLooksUncensored={visionModelLooksUncensored}
            onPassOpen={setPassOpen} onPassRedo={setPassRedoFor}
            onPickPython={setPythonPickerFor}
            onSemanticEngineChange={changeSemanticEngine}
            onChanged={async () => { await refreshPayload(); await refreshImages() }} />
        </div>
      )}

      {/* ── The rail, beside the grid it filters ─────────────────────────
          Below `sm` it cannot share this row, so it becomes a drawer OVER the
          grid instead of squeezing it: at 400 px both would be unusable. */}
      <div className={`grid gap-3 ${railOpen && railIsColumnNow
        ? 'lg:grid-cols-[17rem_minmax(0,1fr)]' : 'grid-cols-1'}`}>
        {railOpen && !railIsColumnNow && (
          <div className="fixed inset-0 z-40 bg-black/60" onClick={closeRail} aria-hidden />
        )}
        {railOpen && (
          /* ⚠️ Plain block comment, NOT {/* … *!/}: this is the inside of
             `{railOpen && ( … )}`, an EXPRESSION position where braces are read
             as an object literal, not JSX children. `node --test` cannot parse
             JSX, so nothing in the suite would have caught it.

             ⚠️ STICKY is not decoration — it is what makes layout B keep its own
              promise. The rail is ~500 px tall; the grid it filters is thousands
              (20 000 images). As a plain grid column the rail scrolls away after
              one screen, so changing a filter means scrolling back UP — the exact
              round trip this layout was built to remove, reintroduced by a
              missing property. Pinned, the filters are beside the grid at every
              scroll position, and the empty band under a short rail is never on
              screen.
              It carries its own overflow: a rail taller than the viewport (many
              people, many tags) must scroll INSIDE its pin, otherwise its lower
              half becomes unreachable — worse than not pinning at all.
              Only when it IS a column: as a drawer it is `fixed` already, and
             sticky on a drawer does nothing. */
          <div id="bank-filter-rail"
            className={railIsColumnNow
              ? 'min-w-0 lg:sticky lg:top-[calc(var(--app-header-h)+0.75rem)] lg:max-h-[calc(100vh-var(--app-header-h)-1.5rem)] lg:overflow-y-auto lg:self-start'
              : 'min-w-0'}>
            <BankFilterRail
              bankId={bankId} filter={filter} setF={setF}
              clusters={clusters} styleClusters={styleClusters}
              searchText={searchText} setSearchText={setSearchText}
              excludeText={excludeText} setExcludeText={setExcludeText}
              subfolders={subfolders} folderPersonInfo={folderPersonInfo}
              folderPersons={folderPersons} folderPersonBusy={folderPersonBusy}
              assertFolderPerson={assertFolderPerson} revokeFolderPerson={revokeFolderPerson}
              checkFolderPerson={checkFolderPerson} scanFolderPersons={scanFolderPersons}
              tagRow={tagRow} tagPicked={tagPicked} toggleTag={toggleTag} clearTags={clearTags}
              chipsFiltered={chipsFiltered} flags={flags}
              statusCounts={chipPrint.status}
              availableScoreFlags={availableScoreFlags} payload={payload}
              shownResBuckets={shownResBuckets} resBuckets={resBuckets}
              originMeasured={originMeasured} originCounts={originCounts}
              shownFramings={shownFramings} framingCounts={framingCounts}
              shownMediums={shownMediums} mediumCounts={mediumCounts} mediumNote={mediumNote}
              shownAngles={shownAngles} angleCounts={angleCounts} angleState={angleState}
              live={live} setPassOpen={setPassOpen}
              thresholdsOpen={thresholdsOpen} setThresholdsOpen={setThresholdsOpen}
              connection={connection} refreshPayload={refreshPayload} refreshImages={refreshImages}
              onRunPass={(endpoint, body) => act(
                () => postJson(`/api/bank/${bankId}/${endpoint}`, body || {}),
                null)}
              sortGroups={sortGroups} setSort={setSort}
              tileSize={tileSize} setTileSize={setTileSize}
              moreOpen={moreOpen} setMoreOpen={setMoreOpen}
              isDrawer={!railIsColumnNow} onClose={closeRail} />
          </div>
        )}

        {/* ── The grid, full height ── with everything that acts on a SELECTION
            directly above it, where the effect of the click is visible. */}
        <main className="min-w-0 space-y-3">
        {/* Results readout + selection actions */}
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-content-muted">
            <span className="font-semibold tabular-nums text-content">{(page.total ?? 0).toLocaleString()}</span> shown
            {isFiltered && (
              <span className="text-content-subtle"> of {(counts?.total ?? 0).toLocaleString()}</span>
            )}
          </span>
          <span aria-hidden className="h-4 w-px bg-border" />
          {/* The fast lane: full-size, one image at a time, Keep/Reject/Skip with
              K/R/S. Deliberately a button of its own rather than a tile gesture —
              the tile click stays the bulk-selection gesture it has always been. */}
          <button type="button" onClick={() => openReview(null)} disabled={reviewLoading}
            title="Review the images of this filter one at a time, full size: ✓ Keep / ✕ Reject / ⏭ Skip (K/R/S) each move to the next. Optional random order."
            className="rounded-md border border-indigo-400/60 bg-indigo-500/20 px-2.5 py-0.5 text-xs font-semibold text-indigo-200 disabled:opacity-50 hover:bg-indigo-500/30">
            {reviewLoading ? '▶ Preparing…' : '▶ Review one by one'}
          </button>
          <span aria-hidden className="h-4 w-px bg-border" />
          <span className="text-content-muted">{selected.size} selected</span>
          <button type="button" onClick={selectAllCurrent}
            className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content hover:bg-surface-raised">
            Select all in filter
          </button>
          {/* Bulk-reject undecided images by quality flag — a triage shortcut that
              leaves your manual ✓/✕ untouched and deletes nothing off disk. */}
          <div className="relative">
            <button type="button" onClick={() => setShowAutoReject((v) => !v)} disabled={live}
              aria-expanded={showAutoReject}
              title="Bulk-reject the still-undecided images carrying the chosen quality flags"
              className="min-h-10 lg:min-h-0 rounded-md border border-border bg-surface-raised px-2 py-0.5 text-xs text-content disabled:opacity-50 hover:bg-surface">
              <Wand2 aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Auto-reject…
            </button>
            {showAutoReject && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setShowAutoReject(false)} aria-hidden />
                {/* Anchored under the button on a desktop, a bottom sheet on a
                    phone. It used to be `absolute left-0` at every width: the
                    button sits mid-toolbar, so at 400 px a 288-px panel ran off
                    the right edge and clipped its own text — including, now, the
                    counts this panel exists to show. Capped height + internal
                    scroll so "Reject them" is reachable however long the caveats
                    get. */}
                <div data-probe-chrome="auto-reject" data-probe-panel="auto-reject" data-probe-layer
                  className="fixed inset-x-3 bottom-3 z-50 max-h-[70vh] overflow-y-auto rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2 sm:absolute sm:inset-x-auto sm:bottom-auto sm:left-0 sm:mt-1 sm:w-72">
                  <p className="text-xs text-content-muted">
                    Rejects the UNDECIDED images with these flags. Your manual ✓/✕ are never changed;
                    everything stays reversible (nothing is deleted from disk).
                  </p>
                  {/* The pile no flag can reach. Every quality flag is gated on a
                      completed scan, so a never-scanned image is invisible to all
                      of them — and "0 blurry" would otherwise read as good news
                      when it means nothing was ever measured. */}
                  {autoRejectNotice && (
                    <p className="m-0 rounded-md border border-amber-400/40 bg-amber-500/10 px-2 py-1 text-[0.6875rem] leading-snug text-amber-200">
                      ⚠ {autoRejectNotice.text} {autoRejectNotice.action}
                      {autoRejectNotice.caveat ? ` ${autoRejectNotice.caveat}` : ''}
                    </p>
                  )}
                  {/* The caveat is printed, not left in a title= tooltip: soft_detail
                      and bars are provenance HINTS, not verdicts, and this is the one
                      screen that offers to act on them in bulk. A tooltip is invisible
                      on a phone and to anyone who does not hover. */}
                  {[...QUALITY_REJECT_FLAGS, ...availableScoreFlags].map((f) => {
                    // A 0 has two opposite meanings and used to render identically:
                    // "nothing left to reject" (clean) vs "this flag's pass never
                    // ran, so it cannot catch anything" (a missing prerequisite).
                    const prereq = flagPrereq(f, counts, originMeasured)
                    return (
                      <label key={f} className="block text-sm text-content">
                        <span className="flex items-center gap-2">
                          <input type="checkbox" checked={rejectFlags.has(f)}
                            onChange={(e) => setRejectFlags((prev) => {
                              const next = new Set(prev)
                              if (e.target.checked) next.add(f); else next.delete(f)
                              return next
                            })} />
                          {FLAG_LABEL[f]}{' '}
                          <span className="text-content-subtle">({flagCandidateLabel(f, flagsActionable)})</span>
                        </span>
                        {prereq && (
                          <span className="mt-0.5 block pl-6 text-[0.6875rem] leading-snug text-sky-200/90">
                            ⓘ {prereq}
                          </span>
                        )}
                        {FLAG_HINT[f] && (
                          <span className="mt-0.5 block pl-6 text-[0.6875rem] leading-snug text-amber-200/80">
                            ⚠ {FLAG_HINT[f]}
                          </span>
                        )}
                      </label>
                    )
                  })}
                  {/* What the button is about to do, before it is pressed — the
                      whole point of the fix. Amber when the answer is zero, so
                      "nothing will happen" is visible rather than discovered. */}
                  {autoRejectPicked && (
                    <p className={`m-0 text-[0.6875rem] leading-snug ${
                      autoRejectPicked.sum ? 'text-content-muted' : 'text-amber-200'}`}>
                      {autoRejectPicked.text}
                    </p>
                  )}
                  <button type="button" onClick={applyAutoReject} disabled={!rejectFlags.size}
                    className="min-h-10 lg:min-h-0 w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-gray-950 disabled:opacity-50">
                    Reject them
                  </button>
                </div>
              </>
            )}
          </div>
          {(selected.size > 0 || showSelected) && (
            <button type="button" onClick={toggleSelectionView}
              aria-pressed={showSelected}
              title={showSelected
                ? 'Back to the full grid with its filters'
                : 'Show only the selected images (as their own view), so a scattered curation/similarity result is visible in one place'}
              className={`rounded-md border px-2 py-0.5 text-xs font-medium ${showSelected
                ? 'border-indigo-400/60 bg-indigo-500/20 text-indigo-200'
                : 'border-border text-content-muted hover:text-content hover:bg-surface-raised'}`}>
              {showSelected ? <><Undo2 aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Show all</> : <><Search aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Show selected ({selected.size})</>}
            </button>
          )}
          {selected.size > 0 && (
            <>
              <button type="button" onClick={() => { setSelected(new Set()); if (showSelected) { exitSelectionView(); setOffset(0); refreshImages(filter, 0, { on: false }) } }}
                className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">Clear</button>
              <button type="button" onClick={() => batchStatus([...selected], 'keep')}
                className="rounded-md border border-emerald-400/50 bg-emerald-500/10 px-2 py-0.5 text-xs font-semibold text-emerald-200 hover:bg-emerald-500/20">✓ Keep</button>
              <button type="button" onClick={() => batchStatus([...selected], 'reject')}
                className="rounded-md border border-rose-400/50 bg-rose-500/10 px-2 py-0.5 text-xs font-semibold text-rose-200 hover:bg-rose-500/20">✕ Reject</button>
              <button type="button" onClick={() => batchStatus([...selected], 'pending')}
                className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:text-content">↺ Undecided</button>
              {/* 🔄 Rotate the selection. Your own files are never rewritten: the
                  angle is stored on the row and applied to what the app shows and
                  to what it promotes — so four turns cost the original nothing. */}
              <button type="button" onClick={() => rotateSelection(-90)}
                aria-label={`Rotate the ${selected.size} selected image(s) 90 degrees left`}
                title="Rotate 90° left (counter-clockwise). Your own files are never modified — the turn is stored and applied to what you see and to what gets promoted."
                className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:bg-surface-raised hover:text-content">
                <span aria-hidden="true">↺</span> Rotate left
              </button>
              <button type="button" onClick={() => rotateSelection(90)}
                aria-label={`Rotate the ${selected.size} selected image(s) 90 degrees right`}
                title="Rotate 90° right (clockwise). Your own files are never modified — the turn is stored and applied to what you see and to what gets promoted."
                className="rounded-md border border-border px-2 py-0.5 text-xs text-content-muted hover:bg-surface-raised hover:text-content">
                <span aria-hidden="true">↻</span> Rotate right
              </button>
            </>
          )}
        </div>

        {/* Curation — build a good LoRA subset out of a big dump from this Bank's
            selected semantic index. Diversity coverage + reference similarity, both
            producing a SELECTION the user reviews above. */}
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-xs font-semibold uppercase tracking-wide text-content-subtle">Curate</span>
          <div className="relative">
            <button type="button" disabled={live || !semanticReady || diverseBusy}
              onClick={() => setCurateOpen((v) => (v === 'diverse' ? null : 'diverse'))}
              aria-expanded={curateOpen === 'diverse'}
              title={semanticReady
                ? `Pick the N images that best COVER the visual variety of the current filter (varied angles/outfits/scenes) using the ${semanticState.label} semantic index.`
                : semanticBlocked}
              className={CURATE_BTN}>
              <Palette aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Pick diverse…{!semanticReady && ` (needs ${semanticState.label})`}{diverseBusy && ' (sampling…)'}
            </button>
            {curateOpen === 'diverse' && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setCurateOpen(null)} aria-hidden />
                <div data-probe-chrome="curate-diverse" data-probe-panel="curate-diverse" data-probe-layer
                  className="absolute z-50 mt-1 w-72 rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2">
                  <p className="text-xs text-content-muted">
                    Selects the most <strong>varied</strong> images of the current filter — the best
                    coverage of the visual space, not N look-alikes. Reviews as a normal selection
                    (nothing is kept or deleted yet).
                  </p>
                  <label className="flex items-center gap-2 text-sm text-content">
                    How many
                    <input type="number" min={1} max={2000} value={diverseN}
                      onChange={(e) => setDiverseN(Math.max(1, Math.min(2000, Number(e.target.value) || 1)))}
                      className="w-20 rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content" />
                  </label>
                  {/* Typicality guard — "most diverse" used to mean "most isolated",
                      which is how a meme or a photo of someone else won the first
                      picks. Exposed rather than silently changed: 0 is the old
                      behaviour, to the pick. */}
                  <label className="block space-y-1 text-sm text-content">
                    <span className="flex items-center justify-between gap-2">
                      <span>Skip the odd ones out</span>
                      <span className="tabular-nums text-xs text-content-muted">
                        {diverseTypicality === 0 ? 'off' : `${Math.round(diverseTypicality * 100)}%`}
                      </span>
                    </span>
                    <input type="range" min={0} max={1} step={0.05} value={diverseTypicality}
                      onChange={(e) => setDiverseTypicality(Number(e.target.value))}
                      className="w-full accent-primary" />
                  </label>
                  <p className="text-[11px] leading-snug text-content-muted">
                    {diverseTypicality === 0
                      ? 'Off — pure coverage, exactly like before. The most isolated images win the first picks, so memes, wrong-person shots and botched frames tend to come up first.'
                      : 'Images that look like nothing else in the bank (memes, screenshots, someone else) stop winning on isolation alone. Variety inside your subject is untouched.'}
                  </p>
                  <button type="button" onClick={pickDiverse} disabled={diverseBusy}
                    className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-gray-950 disabled:opacity-60">
                    {diverseBusy ? 'Sampling…' : `Select ${diverseN} most diverse`}
                  </button>
                </div>
              </>
            )}
          </div>
          {/* ⚖️ Balanced pick — a DIFFERENT question from 🎨 Pick diverse: not "is
              my set varied?" but "does it cover the framings I want to generate?".
              Kept as its own button rather than a mode of the other one, because a
              bank with no 📐 Framing pass can still use diversity. */}
          <div className="relative">
            <button type="button" disabled={live || balanceBusy || !balanceReady.ready}
              onClick={() => setCurateOpen((v) => (v === 'balanced' ? null : 'balanced'))}
              aria-expanded={curateOpen === 'balanced'}
              title={balanceReady.ready
                ? `Select N images SPREAD OVER the framings (face / bust / body / back), then diversify each bucket with the ${semanticState.label} semantic index.`
                : balanceReady.reason}
              className={CURATE_BTN}>
              ⚖️ Balanced pick…{balanceBusy && ' (sampling…)'}
            </button>
            {curateOpen === 'balanced' && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setCurateOpen(null)} aria-hidden />
                {/* Bottom sheet below sm (measured at 400 px, an anchored w-80 panel
                    pushes the page sideways), normal popover from sm up. */}
                <div data-probe-chrome="curate-balanced" data-probe-panel="curate-balanced" data-probe-layer
                  className="fixed inset-x-4 bottom-4 z-50 max-h-[75vh] overflow-y-auto rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2 sm:absolute sm:inset-x-auto sm:bottom-auto sm:left-0 sm:mt-1 sm:max-h-none sm:w-80 sm:overflow-visible">
                  <p className="text-xs text-content-muted">
                    Splits your pick <strong>evenly across the framings</strong> — “20 face, 20 bust,
                    20 body” — and fills each bucket with the same most-varied sampling.
                    Nothing is kept or deleted; you get a selection to review.
                  </p>
                  <label className="flex items-center gap-2 text-sm text-content">
                    How many
                    <input type="number" min={1} max={2000} value={balanceN}
                      onChange={(e) => setBalanceN(Math.max(1, Math.min(2000, Number(e.target.value) || 1)))}
                      className="w-20 rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content" />
                  </label>
                  <fieldset className="space-y-1">
                    <legend className="text-sm text-content">Balance on</legend>
                    {BALANCE_AXES.map((a) => (
                      <label key={a.id} className="flex items-start gap-2 text-xs text-content-muted">
                        <input type="radio" name="bank-balance-axis" value={a.id}
                          checked={balanceAxis === a.id}
                          onChange={() => setBalanceAxis(a.id)}
                          className="mt-0.5 accent-primary" />
                        <span><span className="text-content">{a.label}</span> — {a.hint}</span>
                      </label>
                    ))}
                  </fieldset>
                  <p className="text-[11px] leading-snug text-content-muted">
                    Framing is the reliable axis on a one-subject bank: person groups there tend to be
                    few, sparse and arbitrary. It uses the same “Skip the odd ones out” setting as
                    Pick diverse ({diverseTypicality === 0 ? 'off' : `${Math.round(diverseTypicality * 100)}%`}).
                  </p>
                  <button type="button" onClick={pickBalanced} disabled={balanceBusy}
                    className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-gray-950 disabled:opacity-60">
                    {balanceBusy ? 'Sampling…' : `Select ${balanceN}, balanced`}
                  </button>
                </div>
              </>
            )}
          </div>
          <div className="relative">
            <button type="button"
              disabled={live || !semanticReady || similarBusy
                || (selected.size !== 1 && !similarLast)}
              onClick={() => setCurateOpen((v) => (v === 'similar' ? null : 'similar'))}
              aria-expanded={curateOpen === 'similar'}
              title={!semanticReady
                ? semanticBlocked
                : selected.size === 1
                  ? `Rank the current filter against the ONE selected image with the ${semanticState.label} semantic index and select the closest N.`
                  : similarLast
                    ? 'Extend the last run — add the next closest images from the same ranking.'
                    : 'Select exactly one image to use as the reference'}
              className={CURATE_BTN}>
              <Target aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Similar to selected…{similarBusy && ' (ranking…)'}
            </button>
            {curateOpen === 'similar' && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setCurateOpen(null)} aria-hidden />
                <div className="absolute z-50 mt-1 w-72 rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2">
                  <p className="text-xs text-content-muted">
                    Ranks the current filter by {semanticState.label} similarity to your one selected image and selects
                    the closest — a fast way to extract one person or look. The reference is kept in
                    the selection.
                  </p>
                  {selected.size === 1 && (
                    <>
                      <label className="flex items-center gap-2 text-sm text-content">
                        How many
                        <input type="number" min={1} max={2000} value={similarN}
                          onChange={(e) => setSimilarN(Math.max(1, Math.min(2000, Number(e.target.value) || 1)))}
                          className="w-20 rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content" />
                      </label>
                      <button type="button" onClick={findSimilar} disabled={similarBusy}
                        className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-gray-950 disabled:opacity-60">
                        {similarBusy ? 'Ranking…' : `Select ${similarN} most similar`}
                      </button>
                    </>
                  )}
                  {similarLast && (
                    /* The follow-up the first run makes impossible otherwise:
                       the selection now holds N images, so the one-reference
                       gate cannot re-open — this block extends the LAST
                       ranking instead. Re-ranking is deterministic, so images
                       hand-unselected since then come back; said here rather
                       than discovered. */
                    <div className={`space-y-2 ${selected.size === 1 ? 'border-t border-border pt-2' : ''}`}>
                      <p className="m-0 text-xs text-content-muted">
                        Last run selected {similarLast.taken} around its reference.
                        Add the NEXT closest from the same ranking — anything you
                        unselected by hand since then is selected again.
                      </p>
                      <label className="flex items-center gap-2 text-sm text-content">
                        Add
                        <input type="number" min={1} max={2000} value={similarAddN}
                          onChange={(e) => setSimilarAddN(Math.max(1, Math.min(2000, Number(e.target.value) || 1)))}
                          className="w-20 rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content" />
                        more
                      </label>
                      <button type="button" onClick={addMoreSimilar} disabled={similarBusy}
                        className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-gray-950 disabled:opacity-60">
                        {similarBusy ? 'Ranking…' : `Add ${similarAddN} more`}
                      </button>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
          <div className="relative">
            <button type="button" disabled={live || !semanticReady}
              onClick={openTextSearch}
              aria-expanded={curateOpen === 'text'}
              title={semanticReady
                ? `Describe what you are looking for in words ("brunette outdoors, wide shot") and rank the current filter with ${semanticState.label}.`
                : semanticBlocked}
              className={CURATE_BTN}>
              <Type aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Find by text…{!semanticReady && ` (needs ${semanticState.label})`}
            </button>
            {curateOpen === 'text' && (
              <>
                <div className="fixed inset-0 z-40"
                  onClick={() => { setCurateOpen(null); releaseTextEncoder() }} aria-hidden />
                {/* Below sm this is a BOTTOM SHEET, not a dropdown. Anchored to its
                    trigger it is a w-80 panel hanging off a button that sits
                    mid-row: measured on a 400 px viewport it reached x=517 and made
                    the whole page scroll sideways. Pinning only the horizontal
                    gutters was not enough either — a fixed box with `top: auto`
                    resolves to its static position and lands off-screen once the
                    page is scrolled. So the vertical anchor is explicit, and the
                    sheet scrolls internally when the copy is long. From sm up it
                    behaves exactly like its two sibling popovers. */}
                <div data-probe-chrome="curate-text" data-probe-panel="curate-text" data-probe-layer
                  className="fixed inset-x-4 bottom-4 z-50 max-h-[75vh] overflow-y-auto rounded-lg border border-border bg-surface-overlay p-3 shadow-xl space-y-2 sm:absolute sm:inset-x-auto sm:bottom-auto sm:left-0 sm:mt-1 sm:max-h-none sm:w-80 sm:overflow-visible">
                  <p className="text-xs text-content-muted">
                    Ranks the <strong>current filter</strong> by how close each image is to your
                    words. It refines what the grid is showing — it does not search the whole bank.
                  </p>
                  <label htmlFor="bank-text-search" className="block text-sm text-content">
                    What are you looking for?
                  </label>
                  <input id="bank-text-search" type="search" value={textQuery}
                    placeholder="brunette outdoors, wide shot"
                    onChange={(e) => setTextQuery(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !textPending) runTextSearch() }}
                    className="w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-content" />
                  {/* The pedagogy that matters most, because the failure it
                      prevents is INVISIBLE: "without a hat" comes back full of
                      hats, confidently, with no signal. Offered, never applied on
                      its own — a wrong guess acted on silently would be the same
                      class of bug. */}
                  {suggestPushDown(textQuery) && !textExclude.trim() && (
                    <p className="text-xs text-amber-300/90">
                      “without” is ignored by the search.{' '}
                      <button type="button"
                        onClick={() => {
                          setTextExclude(suggestPushDown(textQuery))
                          setTextQuery(withoutNegation(textQuery))
                        }}
                        className="underline underline-offset-2 hover:text-amber-200">
                        Push “{suggestPushDown(textQuery)}” down instead?
                      </button>
                    </p>
                  )}
                  <label htmlFor="bank-text-exclude" className="block text-sm text-content">
                    Push down (optional)
                  </label>
                  <input id="bank-text-exclude" type="search" value={textExclude}
                    placeholder="hat, sunglasses"
                    onChange={(e) => setTextExclude(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter' && !textPending) runTextSearch() }}
                    className="w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-content" />
                  <p className="text-xs text-content-subtle">{pushDownCaveat()}</p>
                  {textExclude.trim() && (
                    <label className="flex flex-wrap items-center gap-2 text-sm text-content">
                      How hard
                      <select value={textExcludeW}
                        onChange={(e) => setTextExcludeW(Number(e.target.value))}
                        className="rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content">
                        {PUSH_DOWN_STRENGTHS.map((s) => (
                          <option key={s.value} value={s.value}>{s.label}</option>
                        ))}
                      </select>
                      <span className="w-full text-xs text-content-subtle">
                        {PUSH_DOWN_STRENGTHS.find((s) => s.value === textExcludeW)?.hint}
                      </span>
                    </label>
                  )}
                  <label className="flex items-center gap-2 text-sm text-content">
                    How many
                    <input type="number" min={1} max={2000} value={textN}
                      onChange={(e) => setTextN(Math.max(1, Math.min(2000, Number(e.target.value) || 1)))}
                      className="w-20 rounded-md border border-border bg-surface px-2 py-0.5 text-sm text-content" />
                  </label>
                  {/* The cost, BEFORE the click — a cold CLIP load is ~10 s and an
                      unexplained wait is exactly how this reads as a hang. */}
                  <p className="text-xs text-content-subtle">
                    {readinessHint(textStatus, semanticState.engine)}
                  </p>
                  <p className="text-xs text-amber-300/80">
                    {limitsSentence(semanticState.engine)}
                  </p>
                  <button type="button" onClick={runTextSearch}
                    disabled={textPending || !textQuery.trim()}
                    className="w-full rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-gray-950 disabled:opacity-50">
                    {textPending ? pendingLabel(textStatus) : `Rank the closest ${textN}`}
                  </button>
                </div>
              </>
            )}
          </div>
          {!semanticReady && (
            <span className="text-xs text-content-subtle">{semanticBlocked}</span>
          )}
          <button type="button" onClick={() => setCoverageOpen((v) => !v)}
            aria-expanded={coverageOpen}
            title="See what your kept set leans on and what's thin for a good LoRA — advice only, nothing is kept or rejected."
            className={CURATE_BTN}>
            <BarChart3 aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />Coverage advice{coverageOpen ? ' ▲' : ' ▼'}
          </button>
        </div>

        {/* 🔤 What the grid is currently showing, in words. This is the whole
            honesty of the feature: the grid alone would read as "these match",
            when it is a RANKING in which everything scores something. The range,
            the strength wording and the unsearchable count live here. aria-live so
            a screen reader hears the outcome — the grid change is silent. */}
        <div aria-live="polite">
          {textResult && (
            <div className="mt-2 space-y-1 rounded-lg border border-border bg-surface-raised px-3 py-2 text-xs text-content">
              <div className="flex flex-wrap items-start gap-x-2 gap-y-1">
                <Type aria-hidden="true" className="h-3.5 w-3.5" />
                <span className="min-w-0 flex-1">
                  {summarize(textResult, semanticState.engine)}
                </span>
                <button type="button"
                  onClick={() => { setTextResult(null); setShowSelected(false); refreshImages(filter, 0, { on: false }) }}
                  className="shrink-0 rounded-md border border-border px-2 py-0.5 text-xs text-content hover:bg-surface-raised">
                  Clear search
                </button>
              </div>
              {/* What the push-down did HERE, measured on this bank for this
                  pair of phrases — including "it changed nothing", which is the
                  outcome the user would otherwise never detect. */}
              {pushDownNote(textResult) && (
                <p className="text-content-muted">{pushDownNote(textResult)}</p>
              )}
              {textResult.cached === false && (
                <p className="text-content-subtle">
                  This phrase is now cached — searching it again is instant, even after a restart.
                </p>
              )}
              <p className="text-amber-300/80">{limitsSentence(semanticState.engine)}</p>
            </div>
          )}
        </div>

        {/* ⚖️ What the balanced pick actually GAVE you. A repartition the user
            cannot see is indistinguishable from an unbalanced one, so this is
            numbers first — the bar is decoration over a list that reads out. */}
        <div aria-live="polite">
          {balanceResult && (
            <div className="mt-2 space-y-2 rounded-lg border border-emerald-400/40 bg-emerald-500/10 px-3 py-2 text-xs text-content">
              <div className="flex flex-wrap items-start gap-x-2 gap-y-1">
                <span aria-hidden>⚖️</span>
                <span className="min-w-0 flex-1">{summarizeBalance(balanceResult)}</span>
                <button type="button" onClick={() => setBalanceResult(null)}
                  className="shrink-0 rounded-md border border-border px-2 py-0.5 text-xs text-content hover:bg-surface-raised">
                  Dismiss
                </button>
              </div>
              <ul className="flex flex-wrap gap-x-4 gap-y-1 tabular-nums">
                {balanceRows(balanceResult).map((r) => (
                  <li key={r.key} className={r.short ? 'text-amber-200' : 'text-content-muted'}>
                    <span className="text-content">{r.selected}</span> {r.label}
                    <span className="text-content-subtle"> of {r.available}</span>
                    {r.short && <span> · wanted {r.fairShare}</span>}
                  </li>
                ))}
              </ul>
              {balanceNotes(balanceResult).map((note, i) => (
                <p key={i} className={note.tone === 'warn' ? 'text-amber-300/90' : 'text-content-subtle'}>
                  {note.tone === 'warn' ? <AlertTriangle aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" /> : <Lightbulb aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />}{note.text}
                </p>
              ))}
            </div>
          )}
        </div>

        {coverageOpen && (
          <CoveragePanel coverage={coverage} semanticEngine={coverage?.engine || semanticState.engine}
            semanticLabel={semanticEngineLabel(coverage?.engine || semanticState.engine)}
            onClose={() => setCoverageOpen(false)}
            onBalance={balanceReady.ready ? () => setCurateOpen('balanced') : null}
            balanceReason={balanceReady.reason} />
        )}

          {filter.flag === 'dups' ? (
            <DupGroupsPanel bankId={bankId} live={live} kind="exact"
              onChanged={async () => { await refreshPayload(); await refreshImages() }} />
          ) : filter.flag === 'semantic_dups' ? (
            <DupGroupsPanel bankId={bankId} live={live} kind="semantic"
              semanticLabel={semanticState.label}
              onChanged={async () => { await refreshPayload(); await refreshImages() }} />
          ) : (
            <>
            {/* The grid owns the width now, so it earns denser columns on a big
                monitor than it could when it shared the row with three zones. */}
            <ul className={`grid gap-2 ${tileSize === 'S'
              ? 'grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 2xl:grid-cols-10'
              : 'grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 2xl:grid-cols-8'}`}>
              {page.images.map((img) => (
                <Tile key={img.id} img={img} bankId={bankId} size={tileSize}
                  selected={selected.has(img.id)}
                  onReview={() => openReview(img.id)}
                  onTags={() => openTagPicker(img)}
                  onToggle={() => setSelected((prev) => {
                    const next = new Set(prev)
                    if (next.has(img.id)) next.delete(img.id); else next.add(img.id)
                    return next
                  })} />
              ))}
            </ul>
            {page.total === 0 && (
              <p className="text-sm text-content-muted">Nothing matches this filter.</p>
            )}
            {page.total > PAGE_SIZE && (
              <nav className="flex items-center gap-3 text-sm" aria-label="Grid pages">
                <button type="button" disabled={offset === 0} onClick={() => goto(Math.max(0, offset - PAGE_SIZE))}
                  className="rounded-md border border-border px-2 py-1 text-content disabled:opacity-40">← Prev</button>
                <span className="text-content-muted">
                  {offset + 1}–{Math.min(offset + PAGE_SIZE, page.total)} of {page.total}
                </span>
                <button type="button" disabled={offset + PAGE_SIZE >= page.total}
                  onClick={() => goto(offset + PAGE_SIZE)}
                  className="rounded-md border border-border px-2 py-1 text-content disabled:opacity-40">Next →</button>
              </nav>
            )}
            </>
          )}
        </main>
      </div>


      {promoteOpen && (
        <PromoteDialog bankId={bankId}
          selectedIds={[...selected]}
          onClose={() => setPromoteOpen(false)}
          // refreshImages too: a promotion marks the rows it carried, and the ⬆
          // badge lives on the TILES — refreshing only the counters left the
          // header saying "3 promoted" over a grid showing none.
          onStarted={() => { setPromoteOpen(false); refreshPayload(); refreshImages() }} />
      )}

      {deleteRejectedOpen && (
        <DeleteRejectedDialog bankId={bankId} count={counts?.reject || 0}
          sourcePath={payload?.source_path}
          onClose={() => setDeleteRejectedOpen(false)}
          onDone={() => { setDeleteRejectedOpen(false); setSelected(new Set()); refreshPayload(); refreshImages() }} />
      )}

      {/* 🎛 THE PASS LAUNCH WINDOW — one component for every pass. 👥 routes through
          the folder preflight, 🏷️ Caption carries its five options and its
          destructive twin; everything else is the shared three blocks. */}
      {passOpen && (
        <PassDialog passId={passOpen} payload={payload} live={live}
          semanticEngine={semanticState.engine}
          selectionSize={selected.size}
          detectorReady={!!caps.watermark_detect}
          scope={passScopes[passOpen] || ''}
          onScope={(v) => setPassScope(passOpen, v)}
          redo={!!passRedo[passOpen]}
          onRedo={(v) => setPassRedoFor(passOpen, v)}
          onClose={() => setPassOpen(null)}
          stayOpenOnLaunch={passOpen === 'text_scan' || passOpen === 'watermark'}
          onLaunch={(run) => (passOpen === 'faces'
            ? launchFacesFromDialog()
            : (passOpen === 'caption'
              ? startCaption(run)
              : (passOpen === 'text_scan'
                ? runPass('text_scan', run, textScanRunOptions())
                : (passOpen === 'watermark'
                  ? runPass('watermark', run, wmScanRunOptions())
                  : runPass(passOpen, run)))))}
          secondary={passOpen === 'caption' ? captionSecondary : null}>
          {passOpen === 'caption' ? captionRunControls
            : passOpen === 'text_scan' ? textScanControls
              : passOpen === 'watermark' ? watermarkScanControls : null}
        </PassDialog>
      )}

      {launchOpen && (
        <LaunchAllDialog caps={caps} visionReady={visionReady}
          semanticEngine={semanticState.engine}
          counts={counts} flagsActionable={flagsActionable}
          onClose={() => setLaunchOpen(false)} onLaunch={startPipeline} />
      )}

      {preflight && (
        <PersonPreflightDialog plan={preflight.plan} probing={preflight.probing}
          activity={payload?.activity}
          reload={() => apiFetch(`/api/bank/${bankId}/person-preflight`)}
          onProceed={proceedPreflight}
          onStopProbe={() => postJson(`/api/bank/${bankId}/cancel`, {})}
          onCancel={() => setPreflight(null)} />
      )}

      {pythonPickerFor && (
        <ScoringPythonDialog profile={PICKER_PROFILES[pythonPickerFor]}
          onClose={() => setPythonPickerFor('')}
          onChanged={async () => {
            // The passes read bank_scoring_gpu_available() /
            // bank_siglip2_gpu_available(); force both probes so the CPU warning
            // and the "holds the GPU" tooltip agree at once.
            await refreshCaps(true)
            await refreshPayload()
          }} />
      )}

      {review && (
        <BankReviewLightbox bankId={bankId} ids={review.ids} startId={review.startId}
          seedImages={page.images} onDecided={onReviewDecided}
          onRotated={onReviewRotated} onEdited={onReviewEdited} onClose={closeReview} />
      )}

      {relocating && (
        <RelocateBankDialog bankId={bankId} bankName={payload?.name || `Bank #${bankId}`}
          sourcePath={payload?.source_path} onClose={() => setRelocating(false)}
          onDone={() => { refreshPayload({ force: true }); refreshImages() }} />
      )}

      {forgettingMissing && (
        <ForgetMissingDialog bankId={bankId} bankName={payload?.name || `Bank #${bankId}`}
          onClose={() => setForgettingMissing(false)}
          onDone={() => { refreshPayload({ force: true }); refreshImages() }} />
      )}

      {/* 🧪 Caption Lab — pick the subject, then the shared caption editor opens on
          its bench tab. ‹ Another image comes back to the picker instead of closing:
          benching is a comparison across rows. */}
      <Suspense fallback={null}>
        {labPickerOpen && (
          <CaptionLabPicker images={labPile}
            thumbUrl={(img) => `/api/bank/${bankId}/thumb/${img.id}${imageVersionQuery(img)}`}
            emptyNote="Nothing on this page to bench — clear a filter, or select the images you want."
            onClose={() => setLabPickerOpen(false)}
            onPick={(img) => { setLabPickerOpen(false); setLabImage(img) }} />
        )}
        {labImage && (
          <CaptionEditorDialog initialMode="lab"
            initialCaption={labImage.caption || ''}
            imageUrl={`/api/bank/${bankId}/file/${labImage.id}${imageVersionQuery(labImage)}`}
            imageLabel={imageDisplayName(labImage)}
            /* A bank caption is a plain description whose job is SEARCH — the dataset's
               "without the face" rule is about binding identity to a trigger and does
               not apply here. */
            captionPlaceholder="Caption — a plain description, used for search…"
            labSurface={bankLabSurface({
              bankId, imageId: labImage.id, onApplyRunConfig: applyLabConfig })}
            captionOrigin={labImage.caption_origin}
            onPickAnotherImage={() => { setLabImage(null); setLabPickerOpen(true) }}
            onClose={() => setLabImage(null)}
            /* Awaited, and the refusal handed BACK so the dialog draws it beside the
               text instead of closing on top of an unsaved caption — the same contract
               the dataset editor has had since a refused save destroyed one. */
            onSave={async (nextCaption) => {
              if (nextCaption === (labImage.caption || '')) return { ok: true }
              try {
                await putJson(`/api/bank/${bankId}/image/${labImage.id}/caption`,
                              { caption: nextCaption })
              } catch (e) {
                return { ok: false, error: e.message || 'Could not save the caption' }
              }
              setLabImage((cur) => (cur ? { ...cur, caption: nextCaption } : cur))
              // BOTH: the rows carry the caption, but the 🏷️ Caption window's figures
              // (and the re-caption arithmetic beside them) read payload.counts.
              refreshPayload({ force: true })
              refreshImages()
              return { ok: true }
            }} />
        )}
      </Suspense>
    </div>
  )
}
