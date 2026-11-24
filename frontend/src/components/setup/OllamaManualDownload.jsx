const ARCHIVES = [
  ['Windows x64', 'ollama-windows-amd64.zip', 'ca97292ac1e9acc77fcf473bb2741fa6acd12e3798f261b419680068b7cc4c16', '.\\ollama.exe serve'],
  ['Windows ARM64', 'ollama-windows-arm64.zip', '671d27e9f93e14662f3941d705e4b3626fc05bd120055f160489c80c9c552912', '.\\ollama.exe serve'],
  ['Linux x64', 'ollama-linux-amd64.tar.zst', 'b82649006cb5b54ed1237fa2d1a26a7f939a4fd03150033759acbff7ff315adf', './bin/ollama serve'],
  ['Linux ARM64', 'ollama-linux-arm64.tar.zst', '63f01dae3b65641866991e75e2aecb7aed8a4041f4c2c3c7d92862c5a47ebed6', './bin/ollama serve'],
  ['macOS', 'ollama-darwin.tgz', '015261f434d31d7f0781f8ba6b59cc8e1f70ecb562a4e6db44dfdd5e02f7e562', './ollama serve'],
]

export default function OllamaManualDownload() {
  return (
    <details className="space-y-2 text-sm text-content-muted">
      <summary className="cursor-pointer text-primary underline">Download Ollama 0.40.18 from LDS</summary>
      <p>Select your operating system and processor. Verify the downloaded archive's SHA-256, extract it into a new folder,
        then run the command shown from that folder. Select your vision model in Setup when Ollama is running.</p>
      <ul className="space-y-2">
        {ARCHIVES.map(([label, file, sha, command]) => (
          <li key={file}>
            <a className="text-primary underline" href={'https://lds.perfectgf.fr/models/tools/' + sha + '/' + file}>{label}</a>
            {' · '}<code>{command}</code>
            <div className="break-all text-xs">SHA-256 <code>{sha}</code></div>
          </li>
        ))}
      </ul>
    </details>
  )
}
