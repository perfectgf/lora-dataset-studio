import { useCallback } from 'react';
import { postJson } from '../api/fetchClient';
import { useToast } from '../components/common/Toast';
import { cameraLaunchMessage } from '../utils/cameraAngles';

/* 📷 Queue camera views of ONE picture of the library — the handler the picker
   expects, in one place.

   `imageId` is a `lora_test_image.id`, and the route is `/api/canvas/image/
   <id>/camera` for exactly the reason ✨ improve has its own: the dataset
   tables have an INDEPENDENT id space, so a copy of this handler that reached
   for a dataset route would not 404 — it would re-shoot a real but unrelated
   picture and report success. One implementation, one route.

   Nothing moves on screen when the jobs start: the views arrive minutes later
   as their own rows of the feed. So the toast NAMES where they will appear
   (cameraLaunchMessage) — a bare "started" reads as a dead click, which is the
   lesson the improve handler already carries.

   The 409 this route can answer is not an error to swallow: it means the
   weights are missing AND the download has been started. Its `error` string is
   written to be read by a person, so it goes to the toast as-is rather than
   being replaced by a generic failure. */
export function useCameraAngles() {
  const toast = useToast();
  return useCallback(async (imageId, poses) => {
    try {
      const d = await postJson(`/api/canvas/image/${imageId}/camera`, { poses });
      if (!d?.ok) {
        toast.error(d?.error || 'Could not queue the camera views');
        return false;
      }
      toast.success(cameraLaunchMessage(d.queued));
      return true;
    } catch (err) {
      toast.error(err?.message || 'Could not queue the camera views');
      return false;
    }
  }, [toast]);
}
