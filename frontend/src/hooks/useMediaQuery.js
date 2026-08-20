import { useEffect, useState } from 'react';

/* Read a CSS breakpoint from JS.

   Tailwind can HIDE a control at a width, but it cannot MOVE one: a chip that
   belongs in the toolbar on a desktop and in the ⋯ sheet on a phone would have
   to be written twice, and two copies of a control drift the first time one of
   them gains a prop. So the board reads the breakpoint instead and renders each
   control exactly once, in the one place it belongs at this width.

   `matchMedia` and not `innerWidth`: a resize listener fires on every pixel of a
   drag and on every scroll that moves a mobile URL bar, and re-renders a board
   of hundreds of cards each time. A media query fires twice — once crossing the
   breakpoint in each direction. */
export function useMediaQuery(query) {
  const [matches, setMatches] = useState(() => (
    typeof window !== 'undefined' && typeof window.matchMedia === 'function'
      ? window.matchMedia(query).matches
      : false));
  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return undefined;
    const mq = window.matchMedia(query);
    const onChange = () => setMatches(mq.matches);
    onChange();
    // addListener: Safari below 14 has no addEventListener on MediaQueryList,
    // and this app is opened from phones over Tailscale.
    if (mq.addEventListener) mq.addEventListener('change', onChange);
    else mq.addListener(onChange);
    return () => {
      if (mq.removeEventListener) mq.removeEventListener('change', onChange);
      else mq.removeListener(onChange);
    };
  }, [query]);
  return matches;
}

export default useMediaQuery;
