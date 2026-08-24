// react-frontend/src/components/bank/useFolderPersons.js
// The bank's folder-level person assertions (👤 "single person here"):
// state, the loader that also re-fires when a background job lands, and
// the assert / revoke / check / scan actions — moved VERBATIM from
// BankWorkspace.jsx (2026-08-24, hook series wave 4).
import { useCallback, useEffect, useState } from 'react';
import { apiFetch, del, postJson } from '../../api/fetchClient';

export function useFolderPersons({
  bankId, live, filter, toast, refreshImages, refreshPayload,
}) {
  // 👤 Folder-level person assertions ("this subfolder is one person").
  const [folderPersons, setFolderPersons] = useState([])
  // The whole folder-person payload: assertions PLUS the suggestions the app
  // probed by itself, and what a scan would cost.
  const [folderPersonInfo, setFolderPersonInfo] = useState(null)
  const [folderPersonBusy, setFolderPersonBusy] = useState(false)

  // 👤 "Single person here" — the folder-level person assertions. Reloaded when
  // a job LANDS too: the sample check writes its verdict from the background.
  const loadFolderPersons = useCallback(() => {
    apiFetch(`/api/bank/${bankId}/folder-persons`)
      .then((d) => { setFolderPersons(d.assertions || []); setFolderPersonInfo(d) })
      .catch(() => { setFolderPersons([]); setFolderPersonInfo(null) })
  }, [bankId])

  useEffect(() => { loadFolderPersons() }, [loadFolderPersons, live])

  const runFolderPerson = async (call, success) => {
    setFolderPersonBusy(true)
    try {
      const d = await call()
      if (success) toast.success(success(d))
      // The payload too, not only the grid: an assertion creates (or dissolves)
      // a person cluster, and the PEOPLE row above would otherwise keep showing
      // a group that no longer exists until the next poll.
      loadFolderPersons(); refreshImages(); refreshPayload({ force: true })
    } catch (e) {
      toast.error(e?.message || 'That did not work')
    } finally { setFolderPersonBusy(false) }
  }

  const assertFolderPerson = () => runFolderPerson(
    () => postJson(`/api/bank/${bankId}/folder-person`, { subfolder: filter.subfolder }),
    (d) => `${d.images} image(s) grouped as person #${d.cluster_id} — the face pass `
      + 'will skip them',
  )

  const revokeFolderPerson = () => runFolderPerson(
    () => del(`/api/bank/${bankId}/folder-person`
      + `?subfolder=${encodeURIComponent(filter.subfolder ?? '')}`),
    (d) => `${d.cleared} image(s) back to normal clustering`,
  )

  const checkFolderPerson = () => runFolderPerson(
    () => postJson(`/api/bank/${bankId}/folder-person/check`,
      { subfolder: filter.subfolder }),
    (d) => `Checking ${d.sample_size} images of this folder…`,
  )

  const scanFolderPersons = () => runFolderPerson(
    () => postJson(`/api/bank/${bankId}/folder-scan`, {}),
    () => 'Sampling the folders — nothing is grouped until you confirm',
  )
  return {
    folderPersons, folderPersonInfo, folderPersonBusy, loadFolderPersons,
    assertFolderPerson, revokeFolderPerson, checkFolderPerson,
    scanFolderPersons,
  };
}
