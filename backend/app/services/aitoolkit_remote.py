"""HTTP driver for the ai-toolkit web UI API running on a cloud pod.

Endpoint contract verified against the ai-toolkit UI source (Next.js routes):
bearer auth on /api/* except /api/img/ and /api/files/ (public, path-restricted);
job_config is stored verbatim and executed by the pod's worker, so the config
built by lora_training.build_job_config() is submitted as-is (with cloud
overrides applied by the orchestrator)."""
import logging
import os
import posixpath
import uuid
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 30
_UPLOAD_TIMEOUT = 300
_UPLOAD_BATCH = 8
# What upload_dataset ships to the pod. `.mp4` is here because a video dataset is
# a flat folder of clips with homonym .txt captions, and without it the upload
# succeeded while carrying ONLY the captions — the pod would then train on a
# folder with zero samples, after the GPU had been rented. An image dataset folder
# is written by the exporter and holds no video, so nothing else changes.
_DATA_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.txt', '.mp4')
_STREAM_BLOCK = 1024 * 1024


class _StreamedPart:
    """One multipart/form-data body, produced as it is sent instead of built.

    THE SAME DEFECT AS c09dba7b AND d062606c, ON THE SENDING SIDE. `requests`
    with ``files=`` hands the payload to urllib3's ``encode_multipart_formdata``,
    which calls ``read()`` on every file object and concatenates the result: the
    whole body exists in memory before the first byte leaves. That is fine for
    an 85 MB LoRA and is an OOM for a 26 GB checkpoint — and it is why a full
    model could not be sent to a pod at all. The remedy has the same shape it
    had the other two times: seek to an offset and read one block at a time.

    ``__len__`` is not decoration, it is the whole reason this works. requests'
    ``prepare_body`` treats any non-mapping iterable as a stream and then asks
    ``super_len`` for its size; a size means a real ``Content-Length`` header,
    no size means ``Transfer-Encoding: chunked``. We want the former: the
    receiving route is ai-toolkit's Next.js ``request.formData()``, and a body
    whose length is declared up front is the shape it is known to accept.

    Iterating twice would send a truncated body (the file handle has moved), so
    a redirect or a retry that replays the body is a silent corruption waiting
    to happen. ``__iter__`` therefore refuses a second pass rather than
    producing a short one, and the caller retries by building a new instance.
    """

    def __init__(self, field_name, filename, path, fields=None,
                 offset=0, length=None, boundary=None, on_block=None,
                 should_cancel=None):
        self.path = path
        self.offset = int(offset or 0)
        size = os.path.getsize(path)
        remaining = max(0, size - self.offset)
        self.length = remaining if length is None else min(int(length), remaining)
        self.boundary = boundary or uuid.uuid4().hex
        self.on_block = on_block
        self.should_cancel = should_cancel
        self._spent = False
        crlf = b'\r\n'
        dash = b'--' + self.boundary.encode('ascii')
        head = []
        for key, value in (fields or {}).items():
            head += [dash, crlf,
                     b'Content-Disposition: form-data; name="'
                     + str(key).encode('utf-8') + b'"', crlf, crlf,
                     str(value).encode('utf-8'), crlf]
        head += [dash, crlf,
                 b'Content-Disposition: form-data; name="'
                 + str(field_name).encode('utf-8') + b'"; filename="'
                 + str(filename).encode('utf-8') + b'"', crlf,
                 b'Content-Type: application/octet-stream', crlf, crlf]
        self._head = b''.join(head)
        self._tail = crlf + dash + b'--' + crlf

    @property
    def content_type(self) -> str:
        return f'multipart/form-data; boundary={self.boundary}'

    def __len__(self) -> int:
        return len(self._head) + self.length + len(self._tail)

    def __iter__(self):
        if self._spent:
            raise RemoteError('a streamed upload body cannot be sent twice — '
                              'build a new one for the retry')
        self._spent = True
        yield self._head
        left = self.length
        with open(self.path, 'rb') as fh:
            fh.seek(self.offset)
            while left > 0:
                if self.should_cancel is not None and self.should_cancel():
                    raise TransferCancelled('upload cancelled mid-body')
                block = fh.read(min(_STREAM_BLOCK, left))
                if not block:
                    # The file shrank under us. Padding to Content-Length would
                    # land a file that is the right SIZE and the wrong BYTES,
                    # which auto-resume would then happily train from.
                    raise RemoteError(
                        f'{os.path.basename(self.path)} shrank while it was '
                        f'being sent ({left} bytes short)')
                left -= len(block)
                yield block
                if self.on_block:
                    self.on_block(len(block))
        yield self._tail


class RemoteError(RuntimeError):
    pass


class TransferCancelled(RemoteError):
    """The caller asked for the transfer to stop (a user stop, a closing run).

    Its own type because the partial file is deliberately KEPT: a cancelled
    26 GB download that threw away 20 GB of progress would make "cancel" the
    most expensive button in the app.
    """


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
        """Put HF_TOKEN on the pod without dropping anything else.

        This endpoint is NOT a patch: it upserts every key it destructures out
        of the body, so a key the request omits is written as `undefined` and
        the whole call fails — one 500 `Failed to update settings`, with no
        word about which key. That is exactly how a body written against the
        2026-07-12 image died on the 2026-08-27 one (measured on a rented pod,
        run #164): `MODELS_PATH` had joined the schema in between, three keys
        were no longer all of them, and the run ended before the dataset was
        even uploaded — money spent, nothing trained.

        So the body is the GET payload echoed back with HF_TOKEN swapped in,
        rather than a list of key names kept in step by hand. A key added
        upstream tomorrow rides along; an image pinned to an older UI simply
        ignores the extra field it does not destructure. None values are sent
        as empty strings — the same undefined that breaks the upsert.

        Only POSTs when a token is provided: a None hf_token must never clear a
        token already set on the pod (GET may omit secrets). Returns the
        applied state."""
        st = self.get_settings()
        if hf_token:
            body = {k: ('' if v is None else v) for k, v in st.items()}
            body['HF_TOKEN'] = hf_token
            self._json('POST', '/api/settings', json=body)
            st = {**st, 'HF_TOKEN': hf_token}
        return st

    # -- dataset upload -----------------------------------------------------
    def upload_dataset(self, name: str, folder: str, on_progress=None) -> int:
        """Push a staged dataset folder to the pod, eight files per POST.

        on_progress(files_done, files_total, bytes_done, bytes_total) is called
        once before the first batch and after every batch that lands, for the
        same reason _download takes one: a caller has to be able to prove to
        its own watchdogs that a long transfer is alive. That is not a
        theoretical need here — a dataset of 12 422 files and 24 GB is 1 553
        sequential POSTs, and with no callback the whole thing was a single
        blocking call that reported nothing for hours (run #138). Same
        contract as the download side: the callback is never allowed to break
        the upload — a raising one is logged and disabled.
        """
        names = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith(_DATA_EXTS))
        sizes = {}
        for fn in names:
            try:
                sizes[fn] = os.path.getsize(os.path.join(folder, fn))
            except OSError:
                sizes[fn] = 0        # counted as a file, worth zero bytes
        total_bytes = sum(sizes.values())

        def notify(done, sent):
            nonlocal on_progress
            if not on_progress:
                return
            try:
                on_progress(done, len(names), sent, total_bytes)
            except Exception:
                on_progress = None
                logger.debug('upload progress callback disabled', exc_info=True)

        total = 0
        sent_bytes = 0
        notify(0, 0)
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
                sent_bytes += sum(sizes.get(fn, 0) for fn in batch)
            finally:
                for fh in handles:
                    fh.close()
            notify(total, sent_bytes)
        return total

    def upload_file_slice(self, datasets_folder: str, dest_dir: str,
                          remote_name: str, local_path: str, *, offset=0,
                          length=None, timeout=None, on_progress=None,
                          should_cancel=None) -> int:
        """Send BYTES [offset, offset+length) of a local file to the pod as
        ``dest_dir/remote_name``, without ever holding them in memory.

        The addressing trick is ``seed_checkpoint``'s and is unchanged: the
        /api/datasets/upload route joins its ``datasetName`` onto
        DATASETS_FOLDER with Node's ``path.join`` (which normalises ``..``), so
        a relative path lands the file EXACTLY in dest_dir, and the route
        sanitises the FILENAME to [A-Za-z0-9._-]. What changed is only the body:
        see ``_StreamedPart``.

        The slice arguments are what make a 26 GB transfer RESUMABLE. A single
        request that dies at 80% has nothing to offer the next attempt; a
        request that carries one numbered slice leaves every earlier slice
        sitting on the pod, where the next attempt can see it and skip it.

        on_progress(bytes_sent_so_far_in_this_slice) is called as the blocks go
        out — same contract as the download side, and for the same reason: a
        transfer measured in hours has to be able to prove it is alive. A
        raising callback is disabled, never fatal.

        Returns the number of payload bytes sent."""
        rel = posixpath.relpath(dest_dir, datasets_folder.rstrip('/'))
        sent = [0]
        cb = [on_progress]

        def on_block(n):
            sent[0] += n
            if cb[0]:
                try:
                    cb[0](sent[0])
                except Exception:
                    cb[0] = None
                    logger.debug('upload progress callback disabled', exc_info=True)

        body = _StreamedPart('files', remote_name, local_path,
                             fields={'datasetName': rel}, offset=offset,
                             length=length, on_block=on_block,
                             should_cancel=should_cancel)
        r = self._request('POST', '/api/datasets/upload', data=body,
                          headers={'Content-Type': body.content_type},
                          timeout=timeout or _UPLOAD_TIMEOUT)
        if r.status_code != 200:
            raise RemoteError(f'upload {remote_name} -> HTTP {r.status_code}: '
                              f'{r.text[:200]}')
        return sent[0]

    def seed_checkpoint(self, datasets_folder: str, dest_dir: str,
                        remote_name: str, local_path: str) -> None:
        """Pre-place a WHOLE file into an ARBITRARY pod directory (dest_dir —
        e.g. a job's save_root <TRAINING_FOLDER>/<job_name>) so ai-toolkit's
        auto-resume picks it up on the next job start. Raises RemoteError on a
        non-200 so a 'continue' that cannot seed fails loudly rather than
        silently training from scratch.

        Kept as its own name because three callers mean "the whole file, in one
        request" by it (a LoRA save, the Hub token, the fp8 helper). It goes
        through the streamed body like everything else: nothing about a small
        file needs the payload in memory either."""
        self.upload_file_slice(datasets_folder, dest_dir, remote_name, local_path)

    # -- jobs ----------------------------------------------------------------
    def create_job(self, name: str, job_config: dict, gpu_ids: str = '0') -> str:
        r = self._request('POST', '/api/jobs',
                          json={'name': name, 'gpu_ids': gpu_ids, 'job_config': job_config})
        if r.status_code != 200:
            raise RemoteError(f'create_job -> HTTP {r.status_code}: {r.text[:200]}')
        return str(r.json().get('id'))

    def find_job_by_name(self, name: str):
        """The job row whose `name` matches exactly, or None.

        `GET /api/jobs` with no `id` query param returns `{'jobs': [...]}` —
        every job row, newest first, with its `id`, `name`, `status` and
        `step`. This is what makes a create/adopt retry possible: the pod's
        job `name` carries a UNIQUE constraint (POST /api/jobs answers
        `409 {"error":"Job name already exists"}` on the violation), so a name
        we already submitted can be resolved back to its id instead of being a
        dead end."""
        jobs = (self._json('GET', '/api/jobs') or {}).get('jobs') or []
        for job in jobs:
            if isinstance(job, dict) and job.get('name') == name:
                return job
        return None

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
    def _download(self, route: str, remote_path: str, dest_path: str,
                  timeout=None, expected_size=None, attempts=3,
                  on_progress=None, resume=False, should_cancel=None) -> None:
        """Stream to dest_path.part, then rename. RESUME-CAPABLE: some vast
        hosts' proxies cut the stream every ~0.5-2 MB (observed live
        2026-07-13 on 2 of 3 pods — an 85 MB checkpoint needed ~100 resumed
        connections); each retry continues from the current offset with an
        HTTP Range header, as long as the previous attempt made progress.
        With expected_size, completion means EXACTLY that many bytes (a clean
        EOF short of it is just another resume point); without it, completion
        is a stream that ends without error (small files: samples).
        on_progress(bytes_so_far, expected_size) is called as the bytes land,
        so a caller can prove to its own watchdogs that a long transfer is
        alive; it is throttled by the caller, and never allowed to break the
        download (a raising callback is logged and disabled).

        resume=True ADOPTS a ``.part`` left by an earlier call instead of
        deleting it, and only makes sense with expected_size (the size is what
        turns a leftover into a valid offset). A LoRA save is small enough that
        restarting it costs nothing, which is why the default stays False; a
        dense checkpoint is 26 GB, and losing that to an app restart is the
        difference between a resumable transfer and a lost evening.

        should_cancel() is polled as the bytes land: a true answer raises
        TransferCancelled and KEEPS the partial file, so the next attempt
        continues where this one stopped."""
        url_path = f'{route}{quote(remote_path, safe="")}'
        tmp = dest_path + '.part'
        if resume and expected_size:
            try:
                # Never adopt something bigger than the target: that is not a
                # prefix of the file, it is garbage from a different save.
                if os.path.getsize(tmp) > int(expected_size):
                    os.remove(tmp)
            except OSError:
                pass
        else:
            try:
                os.remove(tmp)                # stale leftover from a past run
            except OSError:
                pass
        got = os.path.getsize(tmp) if (resume and os.path.exists(tmp)) else 0
        want = int(expected_size or 0)
        for _ in range(max(1, int(attempts))):
            before = got
            clean = False
            if should_cancel is not None and should_cancel():
                raise TransferCancelled(
                    f'download of {remote_path} cancelled ({got} bytes kept)')
            try:
                headers = {'Range': f'bytes={got}-'} if got else {}
                with self._request('GET', url_path, stream=True, headers=headers,
                                   timeout=timeout or _UPLOAD_TIMEOUT) as r:
                    if got and r.status_code == 416:
                        clean = True          # nothing left to serve
                    else:
                        if r.status_code not in (200, 206):
                            raise RemoteError(
                                f'download {remote_path} -> HTTP {r.status_code}')
                        if got and r.status_code == 200:
                            got = 0           # Range ignored -> full restart
                        written = got
                        with open(tmp, 'ab' if got else 'wb') as fh:
                            for chunk in r.iter_content(chunk_size=1024 * 256):
                                if should_cancel is not None and should_cancel():
                                    fh.flush()
                                    raise TransferCancelled(
                                        f'download of {remote_path} cancelled '
                                        f'({written} bytes kept)')
                                if chunk:
                                    fh.write(chunk)
                                    written += len(chunk)
                                    if on_progress:
                                        try:
                                            on_progress(written, want)
                                        except Exception:
                                            on_progress = None
                                            logger.debug('download progress '
                                                         'callback disabled',
                                                         exc_info=True)
                        clean = True          # stream ended without exception
            except RemoteError:
                raise                          # HTTP-level refusal: no point retrying
            except requests.RequestException:
                clean = False                  # cut mid-stream -> resume below
            got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if want:
                if got == want:
                    os.replace(tmp, dest_path)
                    return
                if got > want or got == before:
                    break                      # garbage, or no progress -> dead
            else:
                if clean:
                    os.replace(tmp, dest_path)
                    return
                if got == before:
                    break
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise RemoteError(f'download {remote_path} incomplete '
                          f'({got}{f"/{want}" if want else ""} bytes after resume attempts)')

    def download_public_file(self, remote_path: str, dest_path: str,
                             timeout=None, expected_size=None, attempts=3,
                             on_progress=None, resume=False,
                             should_cancel=None) -> None:
        # timeout/attempts overrides: the OPPORTUNISTIC mid-run checkpoint sync
        # fails fast (few attempts, short timeout — the monitor loop must not
        # hang); the FINAL end-of-run download passes a large attempts budget
        # so a sick-proxy host still delivers via many resumed connections.
        # resume/should_cancel are the dense harvest's: a 26 GB transfer has to
        # survive an app restart and has to be interruptible.
        self._download('/api/files/', remote_path, dest_path, timeout=timeout,
                       expected_size=expected_size, attempts=attempts,
                       on_progress=on_progress, resume=resume,
                       should_cancel=should_cancel)

    def download_sample(self, remote_path: str, dest_path: str) -> None:
        self._download('/api/img/', remote_path, dest_path)
