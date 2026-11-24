import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { transformSync } from 'esbuild';
import { Fragment } from 'react';
import { createElement, render } from '../../../tests/support/mountJsx.mjs';

// Render the real components in a given state; hooks are supplied by the test
// and effects stay inert, so nothing is fetched or installed.
function load(file, name, context) {
  const source = readFileSync(new URL(file, import.meta.url), 'utf8')
    .replace(/^import .*\r?\n/gm, '')
    .replace(/^export default function /m, 'function ')
    .replace(/^export function /gm, 'function ');
  const compiled = transformSync(source, { loader: 'jsx', jsxFactory: 'createElement', jsxFragment: 'Fragment' }).code;
  vm.runInContext(`${compiled}\nglobalThis.Component = ${name};`, context);
  return context.Component;
}

function withState(values) {
  let next = 0;
  return () => [values[next++], () => {}];
}

const DETAIL_DAEMON = {
  class_type: 'DetailDaemonSamplerNode', pack: 'ComfyUI-Detail-Daemon',
  url: 'https://github.com/Jonseed/ComfyUI-Detail-Daemon', search: 'Detail Daemon',
  install_action: 'detail_daemon_nodes',
};

test('the node repair panel offers the Detail Daemon install where the SDXL node is missing', () => {
  const runners = [];
  const InstallRunner = (props) => { runners.push(props); return createElement('button', null, props.buttonLabel); };
  const context = vm.createContext({
    createElement, Fragment, InstallRunner, HelpBadge: () => null, SettingsLink: () => null,
    // expanded, method, comfyUrl, addressError, checking, error, result
    useState: withState([true, 'portable', '', '', false, '', null]),
    useRef: () => ({ current: null }), useId: () => 'repair', useEffect: () => {},
    apiFetch: () => { throw new Error('Render tests must not fetch.'); },
    URL,
  });
  const ComfyNodeRepair = load('./ComfyNodeRepair.jsx', 'ComfyNodeRepair', context);
  const html = render(ComfyNodeRepair, { nodes: ['DetailDaemonSamplerNode'], nodePacks: [DETAIL_DAEMON] });
  assert.deepEqual(runners.map((r) => r.action), ['detail_daemon_nodes']);
  assert.equal(runners[0].successMessage, 'ComfyUI-Detail-Daemon installed. Restart ComfyUI to load its nodes.');
  assert.match(html, />Install ComfyUI-Detail-Daemon</);
  assert.match(html, /restart ComfyUI to load its nodes/);
  assert.doesNotMatch(html, /In ComfyUI-Manager, search/);
});

test('an install that needs Python packages LDS does not manage shows its exact steps, not a failure', () => {
  const steps = ['Download https://codeload.github.com/Jonseed/ComfyUI-Detail-Daemon/zip/3394e44',
    'If ComfyUI’s Python does not have matplotlib yet, add it: "C:\\ComfyUI\\.venv\\Scripts\\python.exe" -m pip install "matplotlib==3.11.2"',
    'Restart ComfyUI.'];
  const context = vm.createContext({
    createElement, Fragment,
    // state, cancelling, log, returncode, progress, step, runtimeNotice, manualSteps, pythonPicker
    useState: withState(['error', false, ['This node pack adds Python packages.'], 1, null, null, null, steps, null]),
    useRef: () => ({ current: null }), useEffect: () => {}, useToast: () => ({}),
    installRuntimeNotice: () => null, installCompletion: () => null, fmtSize: String,
    ScoringPythonDialog: () => null, PICKER_PROFILES: {},
    apiFetch: () => { throw new Error('Render tests must not fetch.'); }, postJson: () => {},
  });
  const InstallRunner = load('./InstallRunner.jsx', 'InstallRunner', context);
  const html = render(InstallRunner, { action: 'detail_daemon_nodes', buttonLabel: 'Install ComfyUI-Detail-Daemon' });
  assert.match(html, /Install it yourself/);
  assert.match(html, /pip install &quot;matplotlib==3\.11\.2&quot;/);
  assert.equal((html.match(/<li /g) || []).length, 3);
  assert.doesNotMatch(html, /Install failed/);
});
