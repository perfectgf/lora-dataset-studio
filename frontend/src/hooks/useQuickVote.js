import { useEffect, useRef, useState } from 'react';

/**
 * File de vote rapide du Studio : queue + swipe tactile + clavier (←/→/Échap).
 * Espace n'est PAS intercepté (sinon preventDefault global volerait l'activation
 * native du bouton focalisé — correctif a11y). `rate(imageId, rating)` vient de
 * useLoraTestStudio (référence stable, useCallback).
 */
export function useQuickVote(rate) {
  const [voteQueue, setVoteQueue] = useState(null);
  const [voteIdx, setVoteIdx] = useState(0);
  // Titre de mode optionnel (ex. « Reconfirmer les 👍 ») : distingue visuellement
  // une 2e passe sur les votés d'une 1re passe sur les non-votés (sinon le modal
  // est identique et on risque de 👎 par erreur en croyant voter des nouvelles).
  const [voteTitle, setVoteTitle] = useState(null);
  const touchRef = useRef(null);

  const startVoting = (queue, title = null) => {
    if (queue.length) { setVoteQueue(queue); setVoteIdx(0); setVoteTitle(title); }
  };
  const close = () => { setVoteQueue(null); setVoteIdx(0); setVoteTitle(null); };
  const advanceVote = () => {
    if (!voteQueue || voteIdx + 1 >= voteQueue.length) close();
    else setVoteIdx((i) => i + 1);
  };
  const voteCurrent = (rating) => {
    const c = voteQueue && voteQueue[voteIdx];
    if (c) rate(c.id, rating);
    advanceVote();
  };
  const onTouchStart = (e) => {
    const t = e.touches && e.touches[0];
    if (t) touchRef.current = { x: t.clientX, y: t.clientY };
  };
  const onTouchEnd = (e) => {
    const st = touchRef.current; touchRef.current = null;
    const t = e.changedTouches && e.changedTouches[0];
    if (!st || !t) return;
    const dx = t.clientX - st.x; const dy = t.clientY - st.y;
    if (Math.abs(dx) < 50 || Math.abs(dx) < Math.abs(dy) * 1.5) return;
    voteCurrent(dx > 0 ? 1 : -1); // droite = 👍, gauche = 👎
  };

  useEffect(() => {
    if (!voteQueue) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') { close(); return; }
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return; // Espace volontairement ignoré (a11y)
      e.preventDefault();
      voteCurrent(e.key === 'ArrowRight' ? 1 : -1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // rate stable (useCallback) → hors deps
  }, [voteQueue, voteIdx]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    voteQueue, voteIdx, voteTitle, current: voteQueue ? voteQueue[voteIdx] : null,
    startVoting, close, advanceVote, voteCurrent, onTouchStart, onTouchEnd,
  };
}
