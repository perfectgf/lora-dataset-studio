import { useCallback } from 'react';
import { postJson } from '../api/fetchClient';
import { useToast } from '../components/common/Toast';
import { datasetCameraLaunchMessage } from '../utils/cameraAngles';

/* 📷 Queue camera views of ONE dataset image — the dataset twin of
   useCameraAngles, and a separate hook for the same reason ✨ improve keeps
   separate routes: `imageId` here is a `face_dataset_image.id`, and the canvas
   route resolves a `lora_test_image`. Two tables, two independent id spaces —
   the wrong route would not 404, it would re-shoot a real but unrelated
   picture and report success.

   The results land as PENDING candidates of the same dataset, each born with
   its angle phrase as the caption seed; the toast says so, because nothing
   moves on the grid for the first minute and "queued" alone reads as a dead
   click. The 409 (weights missing, downloads started) is written for a human
   and goes to the toast as-is. */
export function useDatasetCameraAngles() {
  const toast = useToast();
  return useCallback(async (imageId, poses) => {
    try {
      const d = await postJson(`/api/dataset/image/${imageId}/camera`, { poses });
      if (!d?.ok) {
        toast.error(d?.error || 'Could not queue the camera views');
        return false;
      }
      toast.success(datasetCameraLaunchMessage(d.queued));
      return true;
    } catch (err) {
      toast.error(err?.message || 'Could not queue the camera views');
      return false;
    }
  }, [toast]);
}
