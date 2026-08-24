// react-frontend/src/components/dataset/useBestEpoch.js
// The training panel's best-epoch cluster (jandordoe's feature), moved
// VERBATIM from TrainingPanel.jsx (2026-08-24, hook series wave 3).
import { useEffect, useState } from 'react';
import { trainingRunSelection } from '../../utils/checkpointBrowser';

export function useBestEpoch({
  ds, postTrain, toastTrainError, checkpointBase, checkpointTrainType,
  checkpointVariant,
}) {
  // Best-epoch (jandordoe): score the run's samples vs the reference, recommend
  // the checkpoint closest to the best-scoring step. Result cleared on base change.
  const [bestEpoch, setBestEpoch] = useState(null);
  const [bestEpochBusy, setBestEpochBusy] = useState(false);
  useEffect(() => { setBestEpoch(null); }, [checkpointBase, checkpointTrainType, checkpointVariant, ds.currentId]);
  const findBestEpoch = async () => {
    setBestEpochBusy(true);
    try {
      const d = await postTrain(`/api/dataset/${ds.currentId}/train/best-epoch`,
        trainingRunSelection(checkpointBase, checkpointTrainType, checkpointVariant));
      if (d && d.ok === false) { toastTrainError(d, 'best-epoch scoring failed'); return; }
      setBestEpoch(d);
    } finally {
      setBestEpochBusy(false);
    }
  };
  return { bestEpoch, bestEpochBusy, findBestEpoch };
}
