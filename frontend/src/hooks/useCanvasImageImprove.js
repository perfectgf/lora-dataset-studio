import { useCallback } from 'react';
import { postJson } from '../api/fetchClient';
import { useToast } from '../components/common/Toast';
import { canvasImproveLaunchMessage } from '../utils/canvasImprove';
import { improveEngine } from '../utils/improveEngines';

/* ✨ Start an Upscale & improve on ONE picture of the library — the handler the
   shared lightbox's `onImprove` expects, in ONE place.

   Two surfaces raise it and both are the SAME action on the same row: the ◉
   Canvas lightbox (a picture pinned on the board) and the checkpoint / run
   gallery lightbox (the place the result actually lands). Living in a hook
   rather than in each host is not tidiness — the route is the part that cannot
   be allowed to drift. `imageId` is a `lora_test_image.id`, while
   `/api/dataset/image/<id>/improve` resolves a `face_dataset_image`: two tables,
   two INDEPENDENT id spaces, so a copy of this handler that reached for the
   dataset route would not 404, it would improve a real but unrelated picture and
   say it worked. One implementation, asserted character by character in
   utils/canvasImprove.test.js.

   Nothing on either surface moves when the pass is queued — the improvement
   arrives minutes later as its own row of the checkpoint's gallery — so the
   toast NAMES where it will appear (canvasImproveLaunchMessage). A bare
   "started" would read as a dead click on both screens.

   `launchMessage` lets a host whose delivery address DIFFERS say its own —
   the 🖼 Gallery page's result lands at the head of the very feed the user is
   looking at, so promising "this checkpoint's gallery" there would send them
   somewhere else to find a picture already in front of them. The ROUTE is not
   configurable, on purpose: the address of the toast may vary by host, the
   address of the request may not. */
export function useCanvasImageImprove({ launchMessage = null } = {}) {
  const toast = useToast();
  return useCallback(async (imageId, engineId) => {
    try {
      const d = await postJson(`/api/canvas/image/${imageId}/improve`,
        engineId ? { engine: engineId } : {});
      if (!d?.ok) {
        toast.error(d?.error || 'Could not start the improvement');
        return;
      }
      // The engine the SERVER echoes names the toast, so a stale tab cannot
      // claim the wrong pass ran.
      const say = launchMessage || canvasImproveLaunchMessage;
      toast.success(say(improveEngine(d.engine).label));
    } catch (err) {
      toast.error(err?.message || 'Could not start the improvement');
    }
  }, [toast, launchMessage]);
}
