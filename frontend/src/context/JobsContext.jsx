/**
 * Global jobs context — aggregates all generation jobs (image, edit, video)
 * across pages so the GlobalJobsDock can show them regardless of navigation.
 *
 * Page-level hooks (useEditJobTracking, useIndexGeneration, video queue) keep
 * their existing local state for page-specific concerns; they mirror to this
 * context via `upsert()` / `remove()` so the dock always sees the truth.
 *
 * Job shape:
 *   {
 *     jobId: string,            // unique across types
 *     type: 'image' | 'edit' | 'video',
 *     status: 'processing' | 'completed' | 'failed' | 'queued',
 *     label?: string,           // e.g. 'Image Edit', 'WAN 2.2 video'
 *     prompt?: string,          // short prompt preview
 *     resultUrl?: string,       // for completed jobs
 *     error?: string,
 *     createdAt: number,        // epoch ms — sort order
 *   }
 */
import { createContext, useContext, useReducer, useCallback, useMemo } from 'react';

const JobsContext = createContext(null);

function reducer(state, action) {
  switch (action.type) {
    case 'UPSERT': {
      const next = new Map(state);
      const existing = next.get(action.job.jobId);
      next.set(action.job.jobId, {
        createdAt: existing?.createdAt ?? Date.now(),
        ...existing,
        ...action.job,
      });
      return next;
    }
    case 'REMOVE': {
      if (!state.has(action.jobId)) return state;
      const next = new Map(state);
      next.delete(action.jobId);
      return next;
    }
    case 'CLEAR_FINISHED': {
      const next = new Map();
      for (const [id, job] of state) {
        if (job.status === 'processing' || job.status === 'queued') next.set(id, job);
      }
      return next.size === state.size ? state : next;
    }
    default:
      return state;
  }
}

export function JobsProvider({ children }) {
  const [jobs, dispatch] = useReducer(reducer, new Map());

  const upsert = useCallback((job) => {
    if (!job?.jobId) return;
    dispatch({ type: 'UPSERT', job });
  }, []);

  const remove = useCallback((jobId) => {
    dispatch({ type: 'REMOVE', jobId });
  }, []);

  const clearFinished = useCallback(() => {
    dispatch({ type: 'CLEAR_FINISHED' });
  }, []);

  const value = useMemo(() => ({
    jobs: [...jobs.values()].sort((a, b) => b.createdAt - a.createdAt),
    activeCount: [...jobs.values()].filter(
      (j) => j.status === 'processing' || j.status === 'queued'
    ).length,
    upsert,
    remove,
    clearFinished,
  }), [jobs, upsert, remove, clearFinished]);

  return <JobsContext.Provider value={value}>{children}</JobsContext.Provider>;
}

export function useJobs() {
  const ctx = useContext(JobsContext);
  if (!ctx) throw new Error('useJobs must be used within JobsProvider');
  return ctx;
}
