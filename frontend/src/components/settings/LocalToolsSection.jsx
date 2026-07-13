import { Card, TextField, TestResult, TestButton, SecretField } from './primitives'

/* HF token unlocks the auto-download of license-gated models (Klein fp8) that
   the ComfyUI setup step offers — which is why it lives with the ComfyUI card
   rather than in the engines' API-keys list. */
const HF_SECRET = {
  key: 'HF_TOKEN', label: 'Hugging Face token', testTarget: null,
  help: 'Only needed to auto-download license-gated models (the Klein fp8 model). Read token from hf.co/settings/tokens, after accepting the model license.',
}

export default function LocalToolsSection(props) {
  const { config, setField, testResults, recordTestResult } = props
  return (
    <div className="space-y-6">
      <Card
        title="ComfyUI"
        help="Local (Klein) generation and the Test Studio. The API URL is where a running ComfyUI answers; the install directory is scanned for checkpoints and LoRAs."
      >
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <TextField
              id="comfyui-api-url"
              label="ComfyUI API URL"
              value={config.comfyui.api_url}
              onChange={(v) => setField('comfyui', 'api_url', v)}
              placeholder="http://127.0.0.1:8188"
            />
            <TestResult result={testResults.comfyui} />
          </div>
          <TestButton target="comfyui" onResult={(r) => recordTestResult('comfyui', r)} />
        </div>
        <TextField
          id="comfyui-base-dir"
          label="ComfyUI install directory"
          value={config.comfyui.base_dir}
          onChange={(v) => setField('comfyui', 'base_dir', v)}
          placeholder="C:\ComfyUI"
          help="Used to derive the output/input/models/loras folders unless overridden."
        />
        <SecretField field={HF_SECRET} {...props} />
      </Card>

      <Card
        title="Ollama"
        help="Lightweight local vision backend — captioning, framing auto-classify and head-crop."
      >
        <div className="flex items-end gap-3">
          <div className="flex-1 space-y-4">
            <TextField
              id="ollama-url"
              label="Ollama URL"
              value={config.ollama.url}
              onChange={(v) => setField('ollama', 'url', v)}
              placeholder="http://127.0.0.1:11434"
            />
            <TextField
              id="ollama-vision-model"
              label="Ollama vision model"
              value={config.ollama.vision_model}
              onChange={(v) => setField('ollama', 'vision_model', v)}
              placeholder="huihui_ai/qwen3-vl-abliterated:8b"
            />
            <TestResult result={testResults.ollama} />
          </div>
          <TestButton target="ollama" onResult={(r) => recordTestResult('ollama', r)} />
        </div>
      </Card>

      <Card
        title="ai-toolkit"
        help="The training engine. Point at the folder containing run.py — its venv/ or .venv/ is detected automatically."
      >
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <TextField
              id="aitoolkit-dir"
              label="ai-toolkit directory"
              value={config.aitoolkit.dir}
              onChange={(v) => setField('aitoolkit', 'dir', v)}
              placeholder="C:\ai-toolkit"
            />
            <TestResult result={testResults.aitoolkit} />
          </div>
          <TestButton target="aitoolkit" onResult={(r) => recordTestResult('aitoolkit', r)} />
        </div>
        <TextField
          id="aitoolkit-python"
          label="Python interpreter (optional)"
          value={config.aitoolkit.python}
          onChange={(v) => setField('aitoolkit', 'python', v)}
          placeholder="Auto — only needed when ai-toolkit has no venv/.venv (conda, uv, system Python)"
          help="Full path to the python executable ai-toolkit should run with, e.g. C:\miniconda3\envs\aitk\python.exe."
        />

        <details className="rounded-lg border border-border p-3">
          <summary className="cursor-pointer text-sm font-medium text-content-muted">
            Advanced: ai-toolkit overrides
          </summary>
          <div className="mt-3 space-y-4">
            <TextField
              id="aitoolkit-datasets-dir"
              label="Datasets directory override"
              value={config.aitoolkit.datasets_dir}
              onChange={(v) => setField('aitoolkit', 'datasets_dir', v)}
              placeholder="Defaults to <ai-toolkit>/datasets"
            />
            <TextField
              id="aitoolkit-output-dir"
              label="Output directory override"
              value={config.aitoolkit.output_dir}
              onChange={(v) => setField('aitoolkit', 'output_dir', v)}
              placeholder="Defaults to <ai-toolkit>/output"
            />
            <TextField
              id="aitoolkit-hf-home"
              label="Hugging Face cache override"
              value={config.aitoolkit.hf_home}
              onChange={(v) => setField('aitoolkit', 'hf_home', v)}
              placeholder="Defaults to <ai-toolkit>/hf-cache/huggingface"
            />
          </div>
        </details>
      </Card>
    </div>
  )
}
