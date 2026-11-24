import CopyCommand from './CopyCommand'

const linkClass = 'text-primary underline'

export default function ComfyManualInstall({ platform }) {
  return (
    <div className="space-y-3 text-sm text-content-muted">
      {(!platform || platform === 'portable') && (
        <div className="space-y-1">
          <p><strong>Windows with NVIDIA.</strong> Download the Windows preparation script and open PowerShell in its folder.
            It verifies the portable runtime from LDS, skips the optional preview models and creates a new ComfyUI-portable folder.</p>
          <a className={linkClass} href="/assets/prepare-comfyui.ps1" download>Download Windows preparation script</a>
          <CopyCommand command={'Unblock-File -LiteralPath .\\prepare-comfyui.ps1\npowershell -NoProfile -ExecutionPolicy RemoteSigned -File .\\prepare-comfyui.ps1'} />
          <p>Open <code>run_nvidia_gpu.bat</code> in the new folder.</p>
        </div>
      )}
      {(!platform || platform === 'manual') && (
        <div className="space-y-1">
          <p><strong>Linux with NVIDIA or Apple Silicon.</strong> Download the preparation script and open a terminal
            in its folder. The script supplies Python 3.12 on x86-64 Linux with glibc 2.35 or later and a CUDA 12.8 driver,
            or on an Apple Silicon Mac running macOS 14 or later.</p>
          <a className={linkClass} href="/assets/prepare-comfyui.sh" download>Download Linux and macOS preparation script</a>
          <CopyCommand command={'sh prepare-comfyui.sh\ncd ComfyUI-0.46.0\n.venv/bin/python main.py'} />
          <p>For a separately managed Python environment, use <code>--source-only</code> to extract the verified ComfyUI source.</p>
        </div>
      )}
      <p>When ComfyUI is running, select its folder in LDS and enter its API address. On the same computer,
        the default address is <code>http://127.0.0.1:8188</code>.</p>
    </div>
  )
}
