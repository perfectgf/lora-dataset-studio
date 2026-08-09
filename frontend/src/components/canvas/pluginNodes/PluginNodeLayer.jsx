import { Suspense, lazy, useCallback, useEffect, useRef, useState } from 'react';
import { nodeKey } from './registry';
import { useBoardDrag } from './useBoardDrag';

// A type is "known" when the CALLER's own `types` map declares it — not the
// registry singleton. A caller that only wires up a subset of the registry
// (or, in a future test, a stub map) gets exactly the types it declared.
const isKnownType = (types, type) => Boolean(types && types[type]);

// One React.lazy() per loader function, cached for the module's lifetime —
// the loader (`type.Card`/`type.AddFlow`) is a stable reference from the
// registry singleton, so this never re-triggers the dynamic import.
const lazyByLoader = new WeakMap();
function lazyFor(loader) {
  if (!lazyByLoader.has(loader)) lazyByLoader.set(loader, lazy(loader));
  return lazyByLoader.get(loader);
}

// A persisted node whose type isn't in the registry (a name typed by a
// future version, or a stray value) must never crash the board — skip it,
// warn once per type so it doesn't spam the console on every render.
const warnedTypes = new Set();
function warnUnknown(type) {
  if (warnedTypes.has(type)) return;
  warnedTypes.add(type);
  // eslint-disable-next-line no-console
  console.warn('unknown plugin node type', type);
}

const idOf = (node) => node?.filename ?? node?.id;

function PluginNodeCard({ typeDef, node, store, boardScale, onGeometry }) {
  const posRef = useRef({ x: node.x, y: node.y });
  const [pos, setPos] = useState({ x: node.x, y: node.y });
  const sizeRef = useRef({ w: 0, h: 0 });
  const roRef = useRef(null);

  // The node's committed x/y (from the store) is the source of truth once a
  // drag isn't in flight — a reload or an external update must win over any
  // stale in-progress position.
  useEffect(() => {
    posRef.current = { x: node.x, y: node.y };
    setPos({ x: node.x, y: node.y });
  }, [node.x, node.y]);

  const report = useCallback((x, y) => {
    onGeometry?.(nodeKey(typeDef.type, node), { x, y, w: sizeRef.current.w, h: sizeRef.current.h });
  }, [onGeometry, typeDef.type, node]);

  useEffect(() => { report(pos.x, pos.y); }, [pos.x, pos.y, report]);

  // A CALLBACK ref, not a ref + effect with `[]` deps: the Card is behind
  // `React.lazy`/`Suspense`, so it isn't mounted yet on the render where an
  // effect with empty deps would run — that effect would fire once while
  // the card is still suspended, find `rootRef.current` null, and never run
  // again, leaving `h` at 0 forever. A callback ref is invoked by React the
  // moment the real DOM node appears (i.e. once the lazy component actually
  // resolves and mounts), so the observer always gets attached and an
  // immediate measurement is reported even if the card's size never changes
  // again after that.
  const attachRoot = useCallback((el) => {
    if (roRef.current) { roRef.current.disconnect(); roRef.current = null; }
    if (!el) return;
    sizeRef.current = { w: el.offsetWidth, h: el.offsetHeight };
    report(posRef.current.x, posRef.current.y);
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => {
      sizeRef.current = { w: el.offsetWidth, h: el.offsetHeight };
      report(posRef.current.x, posRef.current.y);
    });
    ro.observe(el);
    roRef.current = ro;
  }, [report]);

  const onMove = useCallback((dx, dy) => {
    posRef.current = { x: posRef.current.x + dx, y: posRef.current.y + dy };
    setPos(posRef.current);
  }, []);
  const onCommit = useCallback(() => {
    const { x, y } = posRef.current;
    const id = idOf(node);
    store.onNodesChange(store.nodes.map((n) => (idOf(n) === id ? { ...n, x, y } : n)));
  }, [store, node]);
  const dragHandlers = useBoardDrag(boardScale, onMove, onCommit);

  const onUpdate = useCallback((id, patch) => {
    store.onNodesChange(store.nodes.map((n) => (idOf(n) === id ? { ...n, ...patch } : n)));
  }, [store]);
  const onRemove = useCallback((id) => {
    store.onNodesChange(store.nodes.filter((n) => idOf(n) !== id));
    if (store.checked?.has(id)) {
      const next = new Set(store.checked);
      next.delete(id);
      store.onCheckedChange(next);
    }
  }, [store]);
  const onToggleChecked = useCallback((id) => {
    const next = new Set(store.checked || []);
    if (next.has(id)) next.delete(id); else next.add(id);
    store.onCheckedChange(next);
  }, [store]);

  const Card = lazyFor(typeDef.Card);
  return (
    <Suspense fallback={null}>
      <Card ref={attachRoot} node={{ ...node, x: pos.x, y: pos.y }} dragHandlers={dragHandlers}
        onUpdate={onUpdate} onRemove={onRemove}
        checked={!!store.checked?.has(idOf(node))}
        onToggleChecked={onToggleChecked}
        {...(store.extra || {})} />
    </Suspense>
  );
}

function PluginAddFlow({ typeDef, store }) {
  const AddFlow = lazyFor(typeDef.AddFlow);
  return (
    <Suspense fallback={null}>
      <AddFlow nodes={store.nodes} onNodesChange={store.onNodesChange}
        onClose={store.extra?.onClosePicker} {...(store.extra || {})} />
    </Suspense>
  );
}

/* Generic renderer for every board-space plugin-node type: the absolute-
   positioning loop, the `data-canvas-control` card, drag, and the add
   popover now live here instead of duplicated per type. A type only
   supplies markup (`Card`/`AddFlow`) and a payload shape via
   `pluginNodes/registry.js`; this component owns drag, geometry reporting
   and persistence wiring against whatever store the caller passes it. */
export default function PluginNodeLayer({ types, stores, boardScale = 1, onGeometry }) {
  return (
    <>
      {Object.entries(stores || {}).map(([type, store]) => {
        if (!isKnownType(types, type)) {
          warnUnknown(type);
          return null;
        }
        const typeDef = types[type];
        return (
          <div key={type}>
            {(store.nodes || []).map((node) => (
              <PluginNodeCard key={idOf(node)} typeDef={typeDef} node={node} store={store}
                boardScale={boardScale} onGeometry={onGeometry} />
            ))}
            {store.extra?.pickerOpen && <PluginAddFlow typeDef={typeDef} store={store} />}
          </div>
        );
      })}
    </>
  );
}
