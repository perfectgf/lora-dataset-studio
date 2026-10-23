// Coalesce repeated clicks while an asynchronous user action is still pending.
export function singleFlight() {
  let pending = null;
  return (action) => {
    if (pending) return pending;
    pending = Promise.resolve().then(action).finally(() => { pending = null; });
    return pending;
  };
}
