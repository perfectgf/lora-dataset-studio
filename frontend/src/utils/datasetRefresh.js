/**
 * Fetch a dataset and commit it only while that dataset is still active.
 *
 * Keeping the freshness check on both sides of the asynchronous JSON parse is
 * important: navigation can happen while either the request or body is pending.
 * The hook supplies refs/setters; this small orchestration function stays pure
 * enough to exercise the real race with a deferred request in Node tests.
 */
export async function refreshDatasetIfActive({
  datasetId,
  getActiveDatasetId,
  request,
  commitData,
  clearActiveDataset,
}) {
  if (!datasetId) return { status: 'skipped' };

  try {
    const response = await request(datasetId);

    if (response.ok) {
      const payload = await response.json();
      if (getActiveDatasetId() !== datasetId) return { status: 'stale' };
      commitData(payload);
      return { status: 'applied', data: payload };
    }

    // A late 404 for dataset A must not close dataset B after navigation.
    if (response.status === 404) {
      if (getActiveDatasetId() !== datasetId) return { status: 'stale' };
      commitData(null);
      clearActiveDataset();
      return { status: 'not_found' };
    }

    return { status: 'http_error', httpStatus: response.status };
  } catch (error) {
    return { status: 'network_error', error };
  }
}
