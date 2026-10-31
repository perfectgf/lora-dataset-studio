import { Link, useParams } from 'react-router'
import { membershipBlocked } from '../plugins/registry.js'

// Old bookmarks stay understandable when their owner is absent, disabled or
// failed to load. Reaching a bookmark never installs or enables its plugin.
export default function UnavailablePluginPage({ label = 'This workspace' }) {
  const { pluginId } = useParams()
  const membersOnly = membershipBlocked(pluginId)
  return <section className="space-y-4 rounded-xl border border-border bg-surface p-5">
    <h1 className="text-xl font-semibold">{label} is unavailable</h1>
    <p role="status" className="text-sm text-content-muted">{membersOnly
      ? 'An active Patreon membership is required. Connect Patreon in Plugins to reopen this workspace. Your files are kept.'
      : 'Its plugin is not active in this session. Open Plugins to check whether it is installed, enabled, or needs a reload.'}</p>
    <div className="flex flex-wrap gap-4">
      <Link to="/plugins?tab=installed" className="inline-flex min-h-10 items-center text-sm text-primary hover:underline">Plugins</Link>
      <Link to="/datasets" className="inline-flex min-h-10 items-center text-sm text-primary hover:underline">Open datasets</Link>
    </div>
  </section>
}
