"""HTTP driver for the ai-toolkit web UI API running on a cloud pod.

Endpoint contract verified against the ai-toolkit UI source (Next.js routes):
bearer auth on /api/* except /api/img/ and /api/files/ (public, path-restricted);
job_config is stored verbatim and executed by the pod's worker, so the config
built by lora_training.build_job_config() is submitted as-is (with cloud
overrides applied by the orchestrator)."""
import os
from urllib.parse import quote

import requests

_TIMEOUT = 30
_UPLOAD_TIMEOUT = 300
_UPLOAD_BATCH = 8
_DATA_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.txt')


class RemoteError(RuntimeError):
    pass


class RemoteAiToolkit:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip('/')
        self.token = token

    # -- plumbing ---------------------------------------------------------
    def _request(self, method, path, *, timeout=_TIMEOUT, **kwargs):
        headers = kwargs.pop('headers', {})
        headers.setdefault('Authorization', f'Bearer {self.token}')
        return requests.request(method, f'{self.base_url}{path}',
                                headers=headers, timeout=timeout, **kwargs)

    def _json(self, method, path, **kwargs):
        r = self._request(method, path, **kwargs)
        if r.status_code != 200:
            raise RemoteError(f'{method} {path} -> HTTP {r.status_code}: {r.text[:200]}')
        return r.json()

    # -- readiness / settings ---------------------------------------------
    def is_ready(self) -> bool:
        try:
            return self._request('GET', '/api/auth', timeout=8).status_code == 200
        except Exception:
            return False

    def get_settings(self) -> dict:
        return self._json('GET', '/api/settings')

    def ensure_settings(self, hf_token=None) -> dict:
        """POST /api/settings requires all three keys — echo back the folders
        read from GET so only HF_TOKEN actually changes. Only POSTs when a
        token is provided: a None hf_token must never clear a token already
        set on the pod (GET may omit secrets). Returns the applied state."""
        st = self.get_settings()
        if hf_token:
            self._json('POST', '/api/settings', json={
                'HF_TOKEN': hf_token,
                'TRAINING_FOLDER': st.get('TRAINING_FOLDER') or '',
                'DATASETS_FOLDER': st.get('DATASETS_FOLDER') or '',
            })
            st = {**st, 'HF_TOKEN': hf_token}
        return st

    # -- dataset upload -----------------------------------------------------
    def upload_dataset(self, name: str, folder: str) -> int:
        names = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith(_DATA_EXTS))
        total = 0
        for i in range(0, len(names), _UPLOAD_BATCH):
            batch = names[i:i + _UPLOAD_BATCH]
            handles = [open(os.path.join(folder, fn), 'rb') for fn in batch]
            try:
                files = [('files', (fn, fh)) for fn, fh in zip(batch, handles)]
                r = self._request('POST', '/api/datasets/upload', files=files,
                                  data={'datasetName': name}, timeout=_UPLOAD_TIMEOUT)
                if r.status_code != 200:
                    raise RemoteError(f'dataset upload -> HTTP {r.status_code}: {r.text[:200]}')
                total += len(batch)
            finally:
                for fh in handles:
                    fh.close()
        return total

    # -- jobs ----------------------------------------------------------------
    def create_job(self, name: str, job_config: dict, gpu_ids: str = '0') -> str:
        r = self._request('POST', '/api/jobs',
                          json={'name': name, 'gpu_ids': gpu_ids, 'job_config': job_config})
        if r.status_code != 200:
            raise RemoteError(f'create_job -> HTTP {r.status_code}: {r.text[:200]}')
        return str(r.json().get('id'))

    def start_job(self, job_id: str, gpu_ids: str = '0') -> None:
        self._json('GET', f'/api/jobs/{job_id}/start')
        self._json('GET', f'/api/queue/{gpu_ids}/start')

    def stop_job(self, job_id: str) -> None:
        self._json('GET', f'/api/jobs/{job_id}/stop')

    def get_job(self, job_id: str) -> dict:
        return self._json('GET', f'/api/jobs?id={job_id}')

    def get_log(self, job_id: str) -> str:
        return (self._json('GET', f'/api/jobs/{job_id}/log') or {}).get('log') or ''

    def get_samples(self, job_id: str) -> list:
        return (self._json('GET', f'/api/jobs/{job_id}/samples') or {}).get('samples') or []

    def list_files(self, job_id: str) -> list:
        return (self._json('GET', f'/api/jobs/{job_id}/files') or {}).get('files') or []

    # -- downloads (public, path-restricted routes) ---------------------------
    def _download(self, route: str, remote_path: str, dest_path: str) -> None:
        url_path = f'{route}{quote(remote_path, safe="")}'
        with self._request('GET', url_path, stream=True, timeout=_UPLOAD_TIMEOUT) as r:
            if r.status_code != 200:
                raise RemoteError(f'download {remote_path} -> HTTP {r.status_code}')
            tmp = dest_path + '.part'
            with open(tmp, 'wb') as fh:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        fh.write(chunk)
            os.replace(tmp, dest_path)

    def download_public_file(self, remote_path: str, dest_path: str) -> None:
        self._download('/api/files/', remote_path, dest_path)

    def download_sample(self, remote_path: str, dest_path: str) -> None:
        self._download('/api/img/', remote_path, dest_path)
