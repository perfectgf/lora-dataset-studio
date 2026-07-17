import test from 'node:test';
import assert from 'node:assert/strict';
import { formatDiagnostic } from './diagnosticFormat.js';

// A rich, everything-populated payload — the shape /api/diagnostic returns.
function fullPayload() {
  return {
    app_version: '2026.07.17.1',
    git_sha: 'abc1234',
    os: 'Windows 11',
    python: '3.12.4',
    python_ml: { version: '3.12.4', ml_supported: true, ml_range: '3.10–3.12' },
    pillow: { version: '12.2.0', healthy: true, incompatible_plugins: [] },
    disk: { free_gb: 45.2, total_gb: 931.0 },
    secrets_present: { GEMINI_API_KEY: true, OPENAI_API_KEY: false, HF_TOKEN: true },
    capabilities: {
      engines: { nanobanana: true, chatgpt: true, klein: false },
      comfyui_reachable: true,
      klein_model: false,
      klein_missing: ['klein_vae', 'klein_text_encoder'],
      chatgpt_subscription: true,
      ollama_reachable: true,
      vision_model_ready: false,
      face_scoring: false,
      masks: true,
      aitoolkit_valid: false,
      training_visible: false,
      studio_visible: true,
      cloud_training: false,
    },
    comfyui_runtime: {
      version: '0.3.30',
      gpu: 'cuda:0 NVIDIA GeForce RTX 4090',
      vram_total_gb: 24.0,
      vram_free_gb: 12.3,
      queue_running: 1,
      queue_pending: 2,
    },
    config: {
      captioning_backend: 'auto',
      default_engine: 'chatgpt',
      enabled_engines: ['nanobanana', 'chatgpt', 'klein'],
      training_default_family: 'zimage',
      comfyui_base_dir_set: true,
      aitoolkit_dir_set: false,
      watermark_allow_crop: true,
      lan_enabled: false,
    },
    ollama: {
      vision_model: 'huihui_ai/qwen3-vl-abliterated:8b-instruct',
      tags_seen: ['gemma4:e2b', 'qwen3-vl:8b-instruct'],
    },
    generation_errors: {
      engines: {
        klein: 'klein: ComfyUI 409 — klein_vae missing (auto-download queued)',
        chatgpt: 'chatgpt: 429 quota exceeded',
      },
      studio: 'node error: KSampler received an invalid model',
    },
    error_log: [
      '2026-07-17 12:00:00,000 ERROR app.routes.face_dataset: generation failed',
      'Traceback (most recent call last):',
      '  File "app/services/x.py", line 10, in run',
      'ValueError: boom',
    ],
    log_tail: ['line one', 'line two', 'line three'],
    generated_at: 1750000000,
  };
}

test('renders every section header, most-discriminating first', () => {
  const out = formatDiagnostic(fullPayload());
  const headers = ['── Engines ──', '── ComfyUI ──', '── Captioning (Ollama) ──',
    '── Environment ──', '── Recent generation failures ──',
    '── Last errors (with traceback) ──', '── Last log lines ──'];
  let last = -1;
  for (const h of headers) {
    const idx = out.indexOf(h);
    assert.ok(idx !== -1, `missing section: ${h}`);
    assert.ok(idx > last, `section out of order: ${h}`);
    last = idx;
  }
});

test('surfaces the discriminating fields the support cases needed', () => {
  const out = formatDiagnostic(fullPayload());
  // antonp: the exact missing Klein assets, by name.
  assert.match(out, /klein missing assets: klein_vae, klein_text_encoder/);
  // the vision-model trap (abliterated readable from the name) + issue #7 tags.
  assert.match(out, /huihui_ai\/qwen3-vl-abliterated:8b-instruct/);
  assert.match(out, /tags: gemma4:e2b, qwen3-vl:8b-instruct/);
  // environment health (self-heal verdict, disk).
  assert.match(out, /Pillow 12\.2\.0 \(healthy\)/);
  assert.match(out, /disk 45\.2GB free \/ 931GB/);
  // live ComfyUI runtime.
  assert.match(out, /version 0\.3\.30/);
  assert.match(out, /VRAM 24GB \(12\.3 free\)/);
  assert.match(out, /queue 1 running \/ 2 pending/);
  // the new member's real cause: per-engine failure + traceback.
  assert.match(out, /klein: ComfyUI 409 — klein_vae missing/);
  // fail_reason already names its engine — it must not be double-prefixed.
  assert.ok(!out.includes('klein: klein:') && !out.includes('chatgpt: chatgpt:'));
  assert.match(out, /studio: node error: KSampler/);
  assert.match(out, /Traceback \(most recent call last\):/);
  assert.match(out, /ValueError: boom/);
});

test('keys are listed by NAME only — a set key is named, an unset one is not', () => {
  const out = formatDiagnostic(fullPayload());
  assert.match(out, /Keys set: GEMINI_API_KEY, HF_TOKEN/);
  assert.ok(!out.includes('OPENAI_API_KEY'), 'unset key must not be listed');
});

test('outside-ML-range Python gets a visible warning; in-range does not', () => {
  const warn = formatDiagnostic({ ...fullPayload(),
    python_ml: { version: '3.14.0', ml_supported: false, ml_range: '3.10–3.12' } });
  assert.match(warn, /⚠ outside ML wheel range 3\.10–3\.12/);
  assert.ok(!formatDiagnostic(fullPayload()).includes('outside ML wheel range'));
});

test('a MIXED Pillow is flagged', () => {
  const out = formatDiagnostic({ ...fullPayload(),
    pillow: { version: '12.2.0', healthy: false, incompatible_plugins: ['PngImagePlugin.py'] } });
  assert.match(out, /Pillow 12\.2\.0 \(MIXED ⚠\)/);
});

test('empty optional sections are dropped so the healthy case stays short', () => {
  const lean = {
    app_version: '2026.07.17.1', git_sha: null, os: 'Linux', python: '3.12.4',
    python_ml: { ml_supported: true, ml_range: '3.10–3.12' },
    pillow: { version: null, healthy: null }, disk: {},
    secrets_present: {}, capabilities: {}, comfyui_runtime: {}, config: {},
    ollama: {}, generation_errors: {}, error_log: [], log_tail: [],
  };
  const out = formatDiagnostic(lean);
  assert.ok(!out.includes('── Recent generation failures ──'));
  assert.ok(!out.includes('── Last errors (with traceback) ──'));
  // core sections still render.
  assert.ok(out.includes('── Engines ──') && out.includes('── ComfyUI ──'));
  // no git sha -> no trailing parenthesis on the header line.
  assert.match(out, /diagnostic — v2026\.07\.17\.1\n/);
  // Keys set line degrades to 'none' rather than blowing up.
  assert.match(out, /Keys set: none/);
});

test('log tail is capped at the last 18 lines', () => {
  const many = Array.from({ length: 60 }, (_, i) => `L${i}`);
  const out = formatDiagnostic({ ...fullPayload(), log_tail: many, error_log: [] });
  assert.ok(out.includes('L59') && out.includes('L42'));
  assert.ok(!out.includes('\nL41\n'), 'lines older than the last 18 are trimmed');
});
