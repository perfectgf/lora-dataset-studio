"""Describe what HAPPENS in a shot — Qwen3-VL, as a warm worker.

The fourth model worker in this project and the first that reads TIME. ✨ Score
embeds one image, the CLIP towers encode one image or one phrase; this one is
handed a sequence of frames and asked what the movement is.

⛔ WHY NOT OLLAMA, pinned here because it is the mistake available for free.
Ollama fails SILENTLY on video on this machine: the request returns an EMPTY
response with no error, the fence swallowing everything (`text generate skipped`
in its logs). A captioner that returns "" and reports success fills a bank with
empty sidecars — and an empty sidecar is not a neutral default, ai-toolkit trains
it as an empty prompt and says nothing. So this lane talks to transformers
directly, where a failure is an exception.

VERIFIED ON THIS MACHINE BEFORE THIS FILE EXISTED — the video lane's absolute
rule, no unverified model claim:

    Qwen/Qwen3-VL-4B-Instruct   already in the HF cache, 8.3 GB, both shards
    ai-toolkit venv             python 3.12.9, transformers 5.5.3, torch 2.9.1+cu128
    Qwen3VLForConditionalGeneration   exposed NATIVELY by that transformers
    Qwen3VLProcessor            accepts `videos=`, carries a chat template
    qwen_vl_utils               NOT installed

That last line is a design constraint, not a footnote. Every Qwen-VL example
imports `qwen_vl_utils` to sample a video; it is not here, and adding a
dependency to an environment we did not build is something this project refuses
to do (see scoring_python.py). It is also unnecessary: the PARENT already
extracts frames with PyAV for the embedding pass, so it samples them here too
and hands over a list of JPEG paths. The processor takes them as `videos=[[...]]`
with no helper at all.

NATIVE CLASS, NOT trust_remote_code. The lesson is recorded in
watermark_detect_infer.py: Florence-2 ships its modelling code in its repo,
written against an old transformers, and pinning the whole shared environment
back for one unmaintained remote file was the wrong price. Qwen3-VL is integrated
in transformers itself — no remote code, no version roulette.

Protocol — a WARM WORKER, because loading 8.3 GB costs far more than captioning
one shot, and the parent commits per clip so a stopped pass loses nothing:

  stdin,  once      : {"models_root": path|null, "device": "auto"|"cpu",
                       "max_new_tokens": 96}
  stdout, once ready: {"ok": true, "ready": true, "device": "cuda"}
  stdin,  per clip  : {"frames": ["abs.jpg", ...], "prompt": "..."}\n
  stdout, per clip  : {"ok": true, "caption": "..."}\n
                      {"ok": false, "error": "..."}\n   ← one bad clip, worker lives
  EOF on stdin      : clean exit.
A fatal load failure prints {"ok": false, "ready": false, "error": ...} and exits.

An EMPTY generation is reported as an error, never as a caption. That is the
Ollama failure mode defended against structurally rather than by trusting a
model to always speak.
"""
from __future__ import annotations
import json
import os
import sys
from _harness import _emit, _log

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_report  # noqa: E402

# The DEFAULT checkpoint — verified present in the local HF cache, see the module
# docstring. The parent normally sends an explicit `model` in the handshake (from
# the config key `video_caption.model`); this is what a bare invocation falls back
# to, and it is what keeps an install that configures nothing captioning exactly
# as it did before the setting existed.
#
# Any checkpoint of the SAME architecture is a drop-in: nothing below names a
# Qwen-specific behaviour beyond the class, and the class is what transformers
# resolves from the repo's own config. A different architecture fails loudly at
# load, which is the correct outcome rather than a silent wrong answer.
MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'


def main() -> int:
    raw = sys.stdin.readline()
    try:
        req = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        _emit({'ok': False, 'ready': False, 'error': f'bad json: {e}'})
        return 1
    models_root = req.get('models_root') or None
    want = str(req.get('device') or 'auto').lower()
    max_new_tokens = int(req.get('max_new_tokens') or 96)
    # Explicit dtype, for the one case the default gets wrong: a CPU fallback in
    # float32 needs ~16 GB for a 4B model, which is more than most machines have
    # free — and "captioning is impossible here" is a worse answer than "slower
    # and half the memory". Unset keeps the safe default below.
    want_dtype = str(req.get('dtype') or '').lower()
    model_id = str(req.get('model') or '').strip() or MODEL_ID

    if want == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except Exception as e:  # noqa: BLE001 — clean JSON, never a mute traceback
        _emit({'ok': False, 'ready': False,
               'error': import_report.import_failure(e)})
        return 1

    device = 'cuda' if (want != 'cpu' and torch.cuda.is_available()) else 'cpu'
    _log(f'[caption] loading {model_id} ({device})…')
    try:
        kwargs = {'cache_dir': models_root} if models_root else {}
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=({'bfloat16': torch.bfloat16, 'float16': torch.float16,
                    'float32': torch.float32}.get(want_dtype)
                   or (torch.bfloat16 if device == 'cuda' else torch.float32)),
            **kwargs)
        model.to(device).eval()
    except Exception as e:  # noqa: BLE001
        _emit({'ok': False, 'ready': False,
               'error': f'caption model load failed: {type(e).__name__}: {e}'})
        return 1

    def _video_metadata(n_frames, span_s):
        """Honest timestamps for pre-sampled frames, or None when unavailable.

        Without metadata the processor warns and DEFAULTS TO 24 fps, then writes
        the resulting timestamps straight into the prompt
        (processing_qwen3_vl.py: `<{curr_time:.1f} seconds>`): eight frames of a
        five-second shot read as <0.0 s>…<0.3 s>, and the model literally
        believes the whole action took a third of a second — every judgement of
        speed and duration is wrong at the source. fps = (n-1)/span puts the
        last frame AT the span, which is what evenly-spread sampling means.

        None (fall back to today's behaviour) when the interpreter's
        transformers predates the VideoMetadata dataclass — a degraded caption
        beats a crashed worker on an install we do not control."""
        if not span_s or span_s <= 0 or n_frames < 2:
            return None
        try:
            from transformers.video_utils import VideoMetadata
        except Exception:  # noqa: BLE001 — older transformers: no class, no fix
            return None
        fps = (n_frames - 1) / float(span_s)
        return VideoMetadata(total_num_frames=n_frames, fps=fps,
                             duration=float(span_s),
                             frames_indices=list(range(n_frames)))

    def _caption(frame_paths, prompt, span_s=None):
        images = [Image.open(p).convert('RGB') for p in frame_paths]
        try:
            messages = [{'role': 'user', 'content': [
                {'type': 'video'},
                {'type': 'text', 'text': prompt}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            # `videos` takes a LIST OF SEQUENCES — one sequence per video in the
            # batch. Passing the frames flat silently reads as several one-frame
            # videos, which is how a "video" captioner ends up describing a still.
            meta = _video_metadata(len(images), span_s)
            try:
                if meta is not None:
                    inputs = processor(text=[text], videos=[images],
                                       video_metadata=[meta],
                                       return_tensors='pt')
                else:
                    inputs = processor(text=[text], videos=[images],
                                       return_tensors='pt')
            except TypeError:
                # A processor built before the kwarg existed: same fallback as a
                # missing dataclass — degrade to the old behaviour, loudly here
                # rather than dying per clip.
                _log('[caption] processor rejected video_metadata — '
                     'falling back to default timestamps')
                inputs = processor(text=[text], videos=[images],
                                   return_tensors='pt')
            inputs = inputs.to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False)
            # Only the NEW tokens: the prompt is echoed back in the sequence and
            # decoding all of it returns the instruction as the caption.
            trimmed = out[0][inputs['input_ids'].shape[1]:]
            return processor.decode(trimmed, skip_special_tokens=True).strip()
        finally:
            for im in images:
                try:
                    im.close()
                except Exception:  # noqa: BLE001
                    pass

    # The id is echoed so the parent records what REALLY loaded rather than what
    # it asked for — the two differ the moment any fallback creeps in.
    _emit({'ok': True, 'ready': True, 'device': device, 'model': model_id})
    _log('[caption] ready')

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({'ok': False, 'error': f'bad json: {e}'})
            continue
        frames = [str(p) for p in (msg.get('frames') or [])]
        prompt = str(msg.get('prompt') or '')
        span_s = msg.get('span_s')
        if not frames or not prompt:
            _emit({'ok': False, 'error': 'no frames or no prompt'})
            continue
        try:
            caption = _caption(frames, prompt, span_s)
        except Exception as e:  # noqa: BLE001 — one bad clip never kills the worker
            _emit({'ok': False, 'error': f'{type(e).__name__}: {e}'})
            continue
        if not caption.strip():
            # THE Ollama failure mode, refused structurally. An empty caption
            # stored as a caption becomes an empty sidecar, and an empty sidecar
            # is trained as an empty prompt with nothing said anywhere.
            _emit({'ok': False, 'error': 'the model returned an empty caption'})
            continue
        _emit({'ok': True, 'caption': caption})
    _log('[caption] stdin closed — exiting')
    return 0


if __name__ == '__main__':
    sys.exit(main())
